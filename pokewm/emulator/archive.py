"""Go-Explore style frontier archive of emulator save states.

The single hardest fact about this task is the horizon. A Hall of Fame run is on the
order of 10^5 agent steps. Any policy-gradient or value-learning method has an effective
credit-assignment horizon far shorter than that, and the probability that undirected
exploration stumbles onto the Silph Co. card key is, for practical purposes, zero.

Go-Explore (Ecoffet et al., *Nature* 2021) resolves this by *detaching* the exploration
problem from the control problem: remember the states you reached, return to a promising
one directly (here: by restoring a save state, which a deterministic emulator makes
exact), and explore from there. The horizon the learner faces is then the length of one
milestone, not the length of the game.

Cell definition
---------------
A cell is keyed by irreversible progress only -- badges, story flags, party size, the
set of maps visited -- deliberately *not* by (x, y). Position-keyed cells would produce
tens of thousands of near-identical cells per town and dilute the sampling distribution.
The key comes from `GameState.progress_key()`.

Cell selection
--------------
Cells are sampled by softmax over a score that trades off depth against novelty:

    score(c) = milestone_index(c) + w_novel / sqrt(1 + visits(c))

The first term drives the frontier forward; the second is the standard count-based
exploration bonus (Strehl & Littman 2008) and prevents the archive from collapsing onto
a single cell that happens to be deepest but is a dead end.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..agent.milestones import NUM_MILESTONES, map_rank
from .maps import POKECENTER_MAPS, UTILITY_MAPS
from ..config import ArchiveConfig

log = logging.getLogger(__name__)


@dataclass
class Cell:
    key: str
    blob: bytes = field(repr=False)
    milestone: int
    map_id: int
    badges: int
    events: int
    seen_maps: frozenset[int]
    # Party health when the state was saved. A launch pad the agent cannot fight from
    # is not a launch pad; see `score`.
    hp_frac: float = 1.0
    # Party strength when the state was saved. `progress_key` hashes badges, dex and
    # party *size* -- not levels -- so a party that trained from level 6 to level 15
    # lands in the cell it started from. Until this was tracked, the `better` test below
    # had no level term, that stronger state never replaced the stored blob, and the
    # next restore threw the experience away: measured flat at level_sum ~5 for 54M
    # steps, with all 205 Viridian Forest cells holding one level-6 Pokemon.
    level_sum: int = 0
    # Total party experience. `level_sum` only moves in whole levels and a level costs
    # several wins, so ratcheting on levels alone throws away every partial gain: all six
    # sampled cells held exactly 327 XP after 82M env steps, i.e. not one point had ever
    # been banked, and `reward/level` had never once fired.
    exp: int = 0
    visits: int = 0
    chosen: int = 0
    created_at: float = field(default_factory=time.time)
    # Best shaped return observed on a trajectory that *departed* from this cell. Used
    # only for diagnostics; selection is deliberately return-agnostic so the archive
    # cannot be captured by a reward-hacking loop.
    best_return: float = -np.inf

    def score(
        self,
        novelty_weight: float,
        map_rank_weight: float = 0.0,
        target_maps: frozenset[int] = frozenset(),
        target_bonus: float = 0.0,
        hp_weight: float = 0.0,
        visit_weight: float = 0.0,
    ) -> float:
        """Depth along the critical path, plus a count-based novelty bonus.

        `milestone` alone is too coarse to rank cells. It is a monotone *history*
        counter, so a state that reached Route 1 and then wandered back into Red's
        House still scores milestone 6 -- identical to a state actually standing on
        Route 1. Measured: of 62 cells at the frontier level, only 12 were on Route 1
        and 50 were backtracked into the starting area, so most restores landed behind
        the frontier and it never advanced.

        `map_rank` is a property of the stored state rather than of the trajectory that
        produced it, so it separates "has been far" from "is far".

        `hp_frac` matters for the same reason. Measured on the milestone-11 frontier,
        the archived Viridian Forest cells sat at 10-40% party HP, so every restore
        dropped the agent into a state where fighting risked a wipe and fleeing was the
        correct play -- a scripted "mash A" policy won 60% of battles from those same
        cells and levelled up, while the learned policy declined, rationally. Preferring
        healthy states is what makes a frontier actionable rather than merely deep.

        `visits` is what ratchets progress *within* a single map. Every other term is
        constant across a map, so in a maze the size of Viridian Forest nothing preferred
        the cells nearest the far exit. `visits` counts how often the agent reached a
        cell -- a property of the world, unlike `chosen`, which counts how often this
        sampler picked it. Measured in the forest: cells near the entrance averaged 85.9
        visits against 25.5 for cells deep inside, corr(y, visits) = +0.73. Rarely
        reached means hard to get to, which in a maze means further in.

        It enters as a log rather than through the novelty term's 1/sqrt, which
        compresses that 3.4x difference down to a 0.09 score gap -- invisible under the
        softmax.

        Health as a *gate* is deliberately not here -- see `FrontierArchive.viable`,
        which filters the candidate set instead. A score penalty has to be tuned against
        the depth terms and cannot win cleanly, because selection weight depends on cell
        *count* as much as on score.
        """
        return (
            self.milestone
            + map_rank_weight * max(map_rank(self.map_id), 0)
            + (target_bonus if self.map_id in target_maps else 0.0)
            + hp_weight * self.hp_frac
            - visit_weight * float(np.log1p(self.visits))
            + novelty_weight / np.sqrt(1.0 + self.chosen)
        )

    def meta(self) -> dict:
        return {
            "key": self.key,
            "milestone": self.milestone,
            "map_id": self.map_id,
            "badges": self.badges,
            "events": self.events,
            "seen_maps": sorted(self.seen_maps),
            "hp_frac": self.hp_frac,
            "level_sum": self.level_sum,
            "exp": self.exp,
            "visits": self.visits,
            "chosen": self.chosen,
            "created_at": self.created_at,
            "best_return": None if self.best_return == -np.inf else self.best_return,
        }


class FrontierArchive:
    """Thread-safe. The trainer writes from the collector thread and reads when
    resetting workers, so every mutation takes the lock."""

    def __init__(self, cfg: ArchiveConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or ArchiveConfig()
        self._cells: dict[str, Cell] = {}
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(seed)
        self.max_milestone = 0
        # Maps where the work for the *next* milestone lives. Set by the trainer; see
        # `set_target_maps`.
        self._target_maps: frozenset[int] = frozenset()
        # Fingerprint of the milestone chain the stored labels were computed under.
        # Cell milestones (and cell keys, which embed them) are only meaningful relative
        # to a chain, so this travels with the archive rather than with the checkpoint.
        self.chain: str = ""
        self.chain_matches: bool = True
        self.num_inserts = 0
        self.num_updates = 0
        self.root = Path(self.cfg.directory)

    # ------------------------------------------------------------------ mutation

    def __len__(self) -> int:
        with self._lock:
            return len(self._cells)

    def insert(
        self,
        key: str,
        blob: bytes,
        milestone: int,
        map_id: int,
        badges: int,
        events: int,
        seen_maps: frozenset[int] | set[int],
        episode_return: float = -np.inf,
        hp_frac: float = 1.0,
        level_sum: int = 0,
        exp: int = 0,
    ) -> bool:
        """Add or refresh a cell. Returns True if this was a *new* cell."""
        with self._lock:
            existing = self._cells.get(key)
            if existing is None:
                if len(self._cells) >= self.cfg.max_cells:
                    self._evict_locked()
                self._cells[key] = Cell(
                    key=key,
                    blob=blob,
                    milestone=milestone,
                    map_id=map_id,
                    badges=badges,
                    events=events,
                    seen_maps=frozenset(seen_maps),
                    hp_frac=float(hp_frac),
                    level_sum=int(level_sum),
                    exp=int(exp),
                    visits=1,
                    best_return=episode_return,
                )
                self.num_inserts += 1
                self.max_milestone = max(self.max_milestone, milestone)
                self._enforce_level_caps_locked()
                return True

            existing.visits += 1
            existing.best_return = max(existing.best_return, episode_return)
            # Replace the stored blob when we find a strictly better representative of
            # the same progress cell: further along, or the same depth with more of the
            # map explored (which makes the cell a better launch point).
            better = milestone > existing.milestone or (
                milestone == existing.milestone
                and (len(seen_maps) > len(existing.seen_maps)
                     # A healthier version of the same progress cell is strictly the
                     # better launch pad: same position, more room to act.
                     or hp_frac > existing.hp_frac + 0.05
                     # ...and so is a stronger one. Levels are monotone within a life
                     # (a blackout costs money, never experience), so preferring the
                     # higher level_sum cannot oscillate, and it is the only way
                     # training survives a restore -- see `Cell.level_sum`.
                     #
                     # `exp` is the same argument at the granularity that actually
                     # accumulates: a level costs several wins, so ratcheting only on
                     # whole levels discards the progress toward the next one and
                     # levelling never completes.
                     or level_sum > existing.level_sum
                     or exp > existing.exp)
            )
            if better:
                existing.blob = blob
                existing.milestone = milestone
                existing.map_id = map_id
                existing.badges = badges
                existing.events = events
                existing.seen_maps = frozenset(seen_maps)
                existing.hp_frac = float(hp_frac)
                existing.level_sum = int(level_sum)
                existing.exp = int(exp)
                self.num_updates += 1
                self.max_milestone = max(self.max_milestone, milestone)
            return False

    def _enforce_level_caps_locked(self) -> None:
        """Keep superseded milestone levels from crowding out the frontier.

        Without this, shallow levels grow without bound until the global `max_cells` cap
        of 4000 -- in practice never. A measured run held 83 Pallet Town cells against 11
        on Route 1, dropping frontier-selection probability to ~3% (docs/PROOF.md §3.2).

        Two rules:

        * The **deepest level is exempt.** It is the frontier, and spatial coverage there
          is exactly what lets the agent ratchet forward along a route. Capping it would
          also evict the *most-chosen* cells, which at the frontier are the ones actively
          being used to make progress.
        * Every **other** level is trimmed, and trimmed on every insert rather than only
          when that level is written to. A level inserted while it *was* the frontier
          would otherwise stay exempt forever once a deeper level appeared -- which is
          how the live run accumulated its shallow backlog.
        * Victims are chosen **least-far-along first**, not most-chosen first.

        That last rule is a correction of a bad one. Evicting the most-chosen cell is the
        intuitive "we have already exploited this" heuristic, but combined with a hard cap
        it is self-destructive: the cells the archive relies on are by definition the ones
        it picks most often, so they accrue the highest `chosen` count and are deleted
        first. In a live run it deleted **every single Viridian City cell** -- the entire
        launch pad for the next milestone -- because Viridian had been the restore target
        for hours. Restores then fell back to Route 1, one map further behind, and the
        agent had to re-walk ground it had already covered.

        Ranking by `map_rank` keeps whatever spatial progress a level represents, and
        `chosen` survives only as a tie-break among equally-advanced cells.
        """
        cap = getattr(self.cfg, "max_cells_per_level", 0)
        if cap <= 0 or not self._cells:
            return
        deepest = max(c.milestone for c in self._cells.values())
        by_level: dict[int, list[Cell]] = {}
        for cell in self._cells.values():
            by_level.setdefault(cell.milestone, []).append(cell)
        keep = max(0, getattr(self.cfg, "utility_cells_per_map", 0))
        # Strength is global, not per level: the strongest party in the archive is
        # usually *behind* the frontier, because it got strong by grinding somewhere
        # safe. Ranking victims by `map_rank` therefore deletes it first. See
        # `ArchiveConfig.strongest_cells_kept`.
        n_strong = max(0, getattr(self.cfg, "strongest_cells_kept", 0))
        strongest: set[str] = set()
        if n_strong:
            strongest = {
                c.key
                for c in sorted(self._cells.values(), key=lambda c: -c.exp)[:n_strong]
                if c.exp > 0
            }
        for level, cells in by_level.items():
            if level >= deepest:
                continue
            # Shield a few cells inside each Pokemon Center and Poke Mart.
            #
            # Victims are ranked by `map_rank`, and no milestone names a shop or a
            # Center, so every one of them scores -1 and is evicted before anything on
            # the critical path. Measured at 83.5M env steps: all 7 Viridian Mart cells
            # were gone, so the agent had no archived state inside the only building
            # that sells Poke Balls -- while carrying 0 balls in all 607 cells and
            # enough money to buy them. These are the launch pads for the resource
            # behaviours (`ball`, `heal_visit`), which no amount of depth substitutes
            # for. Bounded per map so they cannot themselves crowd the archive.
            shielded: set[str] = set()
            if keep:
                per_map: dict[int, int] = {}
                for c in sorted(cells, key=lambda c: (-c.hp_frac, c.chosen)):
                    if c.map_id in UTILITY_MAPS and per_map.get(c.map_id, 0) < keep:
                        per_map[c.map_id] = per_map.get(c.map_id, 0) + 1
                        shielded.add(c.key)
            while len(cells) > cap:
                pool = [c for c in cells
                        if c.key not in shielded and c.key not in strongest] or cells
                victim = min(
                    pool, key=lambda c: (map_rank(c.map_id), -c.chosen, len(c.seen_maps))
                )
                del self._cells[victim.key]
                cells.remove(victim)

    def _evict_locked(self) -> None:
        """Drop the least useful cell: shallowest, then most-frequently chosen.

        Never evicts a cell that is the unique deepest one, so the frontier cannot
        regress under memory pressure -- this is what the proof's monotonicity
        assumption (docs/PROOF.md §3, A2) requires.
        """
        if not self._cells:
            return
        deepest = max(c.milestone for c in self._cells.values())
        n_deepest = sum(1 for c in self._cells.values() if c.milestone == deepest)
        candidates = [
            c
            for c in self._cells.values()
            if not (c.milestone == deepest and n_deepest == 1)
        ]
        if not candidates:
            return
        victim = min(candidates, key=lambda c: (c.milestone, -c.chosen))
        del self._cells[victim.key]

    # ------------------------------------------------------------------ selection

    def _frontier_cells_locked(self) -> list[Cell]:
        """Cells at the deepest milestone level, widened until there are enough of them.

        A brand-new milestone level starts with exactly one cell, and launching 80% of
        episodes from a single save state is fragile: if that one state happens to be a
        bad launch pad -- mid-battle, one tile from a ledge, low HP -- the whole run
        inherits it. Widening to the next level down until `frontier_min_cells` are
        available costs a little depth and buys back the variety that makes the frontier
        recoverable.
        """
        cells = list(self._cells.values())
        if not cells:
            return cells
        want = max(int(getattr(self.cfg, "frontier_min_cells", 1)), 1)
        min_hp = float(getattr(self.cfg, "frontier_min_hp", 0.0))
        chosen: list[Cell] = []
        for level in sorted({c.milestone for c in cells}, reverse=True):
            chosen.extend(c for c in cells if c.milestone == level)
            enough = len(chosen) >= want
            # A frontier you cannot survive is not a frontier.
            #
            # Health was already a preference *within* the chosen set, but the frontier
            # restriction picks that set first, so when every cell at the deepest level
            # is nearly dead the preference has nothing to choose between. Measured: all
            # 82 Viridian Forest cells held a single level-6 Pokemon at 10-40% HP, while
            # the forest's trainers field level 6-9 teams. Every restore was a loss, the
            # party never levelled, and the region beyond those trainers stayed sealed
            # for 24M steps. Widening to a shallower level offers a healthy state to
            # start from and walk in with.
            healthy = any(c.hp_frac >= min_hp for c in chosen) if min_hp > 0 else True
            if enough and healthy:
                break
        return chosen

    def sample(self) -> Cell | None:
        """Draw a launch cell, or None to start from the ROM boot state.

        Most draws are restricted to cells at the deepest milestone level, because
        milestone is not just "further along" -- it encodes *irreversible world state*:
        gates opened, key items held, NPCs moved. A cell one level back can make the
        next milestone physically impossible, not merely slower to reach.

        This was measured, not assumed. After Oak's Parcel was delivered, 22 of the 24
        archived Route 1 cells were *pre*-delivery states, in which the old man still
        blocks Viridian's north exit and Route 2 cannot be entered at all. The score
        weights milestone at 1.0 against map_rank + target_bonus summing to 11, so
        roughly 80% of on-target restores launched into a world where the next milestone
        was unreachable; the run held milestone 9 for 2.0M steps.

        The remaining `1 - frontier_prob` of draws stay unrestricted. Go-Explore's
        diversity is what recovers from a frontier that turns out to be a dead end, and
        a strictly frontier-only archive cannot back out of one.
        """
        with self._lock:
            if not self._cells:
                return None
            if self._rng.random() > self.cfg.restore_prob:
                return None
            cells = list(self._cells.values())
            # A reserved share of restores goes to the strongest party available,
            # whatever map it is on. See `ArchiveConfig.strength_prob`.
            if (self.cfg.strength_prob > 0.0
                    and self._rng.random() < self.cfg.strength_prob):
                pool = [c for c in self._viable_cells(cells) if c.exp > 0]
                if pool:
                    # Chosen uniformly among the strongest, *not* by `score`. Scoring
                    # inside the pool would just re-run the depth comparison that
                    # created the problem: the strong cells sit at map_rank 1-5 and the
                    # frontier at 12, an 11-point gap the softmax turns into odds of
                    # roughly e^-11. Restricting the pool is not enough; the draw itself
                    # has to be on experience.
                    pool.sort(key=lambda c: -c.exp)
                    pool = pool[: max(1, self.cfg.strength_pool)]
                    pick = pool[int(self._rng.integers(len(pool)))]
                    pick.chosen += 1
                    return pick
            if self._rng.random() < self.cfg.frontier_prob:
                cells = self._frontier_cells_locked()
            cells = self._viable_cells(cells)
            scores = np.array(
                [
                    c.score(self.cfg.novelty_weight, self.cfg.map_rank_weight,
                            self._target_maps, self.cfg.target_bonus,
                            self.cfg.hp_weight, self.cfg.visit_weight)
                    for c in cells
                ],
                dtype=np.float64,
            )
            t = max(self.cfg.temperature, 1e-6)
            logits = (scores - scores.max()) / t
            p = np.exp(logits)
            p /= p.sum()
            idx = int(self._rng.choice(len(cells), p=p))
            cells[idx].chosen += 1
            return cells[idx]

    def viable(self, cell: Cell) -> bool:
        """Whether an episode launched from this cell can actually do anything.

        A party one hit from a blackout cannot fight, level, or survive the walk to a
        Pokemon Center, so the episode is spent regardless of how deep the cell is.
        Measured on the live archive at 79.8M env steps: 75% of restores began below
        0.3 HP and 55% began essentially dead, 47% of them in Pewter City where not one
        of 24 cells was above 0.33 HP. Over the 20.8M steps that followed, `level_sum`
        never left 8 and the heal, ball and party-member rewards never fired once.

        An *empty* party is viable: it reads 0.0 HP because there is nothing to heal,
        not because it is dying. `level_sum == 0` identifies those exactly -- verified
        76 of 76 against party size on the live archive.

        A hurt party *inside a Pokemon Center* is viable too, and this exception is the
        point rather than a concession: the remedy is about six steps and an A press
        away, so the state is not a dead end but the one place the heal can be practised
        at all. Without it the filter excluded 7 of the 9 archived Center cells -- every
        Pewter one, all at or below 0.25 HP -- which are exactly the launch pads
        `heal_visit` needs. Two of this file's fixes were quietly cancelling each other.
        """
        return (cell.level_sum == 0
                or cell.hp_frac >= self.cfg.frontier_min_hp
                or cell.map_id in POKECENTER_MAPS)

    def _viable_cells(self, cells: list[Cell]) -> list[Cell]:
        """Drop unusable cells, unless doing so would leave too little to choose from.

        The fallback matters early, before the agent has banked any healthy states: an
        archive that refuses to restore anything is worse than one that restores a hurt
        cell.
        """
        if not self.cfg.require_viable:
            return cells
        viable = [c for c in cells if self.viable(c)]
        if len(viable) >= min(self.cfg.frontier_min_cells, len(cells)):
            return viable
        return cells

    def relabel(self, evaluate) -> int:
        """Recompute every cell's milestone against the current chain.

        Cell milestones are computed by whichever chain was live when the cell was
        stored, and the cell *key* embeds that number. After a chain change the stored
        labels are simply wrong -- a run carried 242 cells claiming milestone 8 under a
        chain where those states are milestone 7. Wrong labels are not cosmetic here:
        they outrank everything, and the deepest level is exempt from trimming, so they
        would never age out on their own.

        `evaluate(cell) -> int` returns the corrected milestone; the caller supplies it
        because scoring a cell requires booting its save state in an emulator, which the
        archive deliberately knows nothing about.
        """
        with self._lock:
            cells = list(self._cells.values())
            changed = 0
            self._cells.clear()
            for cell in cells:
                try:
                    new_ms = int(evaluate(cell))
                except Exception:  # a corrupt blob must not take the archive with it
                    continue
                if new_ms != cell.milestone:
                    changed += 1
                    # The key embeds the milestone, so it has to be rewritten too.
                    parts = cell.key.split(":")
                    if len(parts) >= 2:
                        parts[1] = str(new_ms)
                        cell.key = ":".join(parts)
                    cell.milestone = new_ms
                self._cells[cell.key] = cell
            self.max_milestone = max(
                (c.milestone for c in self._cells.values()), default=0
            )
            self._enforce_level_caps_locked()
        return changed

    def set_target_maps(self, maps) -> None:
        """Bias restores towards the maps the next milestone actually needs.

        Depth heuristics alone assume the critical path only ever runs forwards. It does
        not: Oak's Parcel must be carried from Viridian *back* to Pallet Town, and the
        Viridian north exit stays blocked until it is. With a forward-only bias the
        archive sent 97% of restores to Viridian and 0.5% to Oak's Lab, so the one
        required action was effectively unreachable and the run sat at the same milestone
        for 9.1M steps.

        **Fallback for unreached targets.** A "reach map X" milestone names a map the
        agent has by definition never visited, so no cell sits on it and the bonus
        applies to nothing -- measured at 0.0% on-target with restores scattering 46.6%
        into Oak's Lab. In that case the useful launch pad is the deepest map actually
        reached, so the target falls back to the highest-`map_rank` maps present in the
        archive. That keeps the agent starting from the edge of known territory, which is
        where a new map can be discovered from.
        """
        with self._lock:
            requested = frozenset(maps or ())
            # Fall back over *frontier* maps only. A shallower level may hold cells on a
            # deeper map (Route 1 cells from the outbound parcel trip outrank anything
            # the post-delivery frontier has reached), and pointing the bonus there
            # rewards cells that `sample` mostly no longer draws.
            cells = self._frontier_cells_locked() or list(self._cells.values())
            present = {c.map_id for c in cells}
            if requested & present:
                self._target_maps = requested
                return
            if not present:
                self._target_maps = requested
                return
            best = max(map_rank(m) for m in present)
            if best < 0:
                self._target_maps = requested
                return
            self._target_maps = frozenset(
                m for m in present if map_rank(m) == best
            )

    @property
    def target_maps(self) -> frozenset[int]:
        """The *effective* target after any fallback -- what scoring actually uses."""
        with self._lock:
            return self._target_maps

    def deepest(self) -> Cell | None:
        """The most advanced *state* in the archive.

        Ranking by `milestone` alone is misleading, because milestone is a worker-lifetime
        monotone counter rather than a property of the stored state: a worker that reached
        Route 1 and then wandered back into Oak's Lab still records milestone 5, so the
        "deepest" cell could be spatially behind several shallower ones. That is exactly
        what `play.py --from-frontier` hit -- every milestone-5 cell was saved inside
        Oak's Lab, so the viewer always launched there.

        `map_rank` breaks the tie by how far along the critical path the cell's *map*
        actually is, which is a property of the saved state and cannot be inflated by
        backtracking.

        Health enters as a *threshold*, not as a raw ordering. A cell you cannot survive
        in is not the most advanced state, it is a blackout with extra steps: measured,
        this returned a Pewter City cell at 0.21 HP while 29 of the 62 cells at the same
        milestone sat at full health, and `play --from-frontier` fainted out of it in
        ~100 steps -- back to Pallet Town, because the run has never healed at a Pokemon
        Center and `wLastBlackoutMap` is still the player's house. Using the threshold
        rather than `hp_frac` itself keeps the existing tie-breaks meaningful among
        viable cells instead of reordering them by trivial health differences, and the
        trailing `hp_frac` only separates cells that are otherwise identical.
        """
        floor = self.cfg.frontier_min_hp
        with self._lock:
            if not self._cells:
                return None
            return max(
                self._cells.values(),
                key=lambda c: (c.milestone, map_rank(c.map_id), c.hp_frac >= floor,
                               len(c.seen_maps), c.events, c.hp_frac),
            )

    def stats(self) -> dict[str, float]:
        with self._lock:
            if not self._cells:
                return {
                    "archive/cells": 0.0,
                    "archive/max_milestone": 0.0,
                    "archive/mean_milestone": 0.0,
                    "archive/frontier_frac": 0.0,
                }
            ms = np.array([c.milestone for c in self._cells.values()], dtype=np.float64)
            return {
                "archive/cells": float(len(self._cells)),
                "archive/max_milestone": float(ms.max()),
                "archive/mean_milestone": float(ms.mean()),
                "archive/frontier_frac": float((ms == ms.max()).mean()),
                "archive/progress": float(ms.max() / max(NUM_MILESTONES - 1, 1)),
            }

    # ------------------------------------------------------------------ persistence

    def save(self, directory: str | Path | None = None) -> Path:
        """Atomic-ish snapshot: states as individual files plus a JSON index.

        A 24 h run must survive being killed, so this is written to a temp directory and
        moved into place rather than mutated in situ.
        """
        root = Path(directory or self.root)
        tmp = root.with_name(root.name + ".tmp")
        if tmp.exists():
            for p in tmp.glob("*"):
                p.unlink()
        (tmp / "states").mkdir(parents=True, exist_ok=True)
        with self._lock:
            index = []
            for cell in self._cells.values():
                (tmp / "states" / f"{cell.key}.state").write_bytes(cell.blob)
                index.append(cell.meta())
            meta = {
                "chain": self.chain,
                "cells": index,
                "max_milestone": self.max_milestone,
                "num_inserts": self.num_inserts,
                "num_updates": self.num_updates,
            }
        (tmp / "index.json").write_text(json.dumps(meta))
        backup = root.with_name(root.name + ".old")
        if root.exists():
            if backup.exists():
                import shutil

                shutil.rmtree(backup)
            root.rename(backup)
        tmp.rename(root)
        if backup.exists():
            import shutil

            shutil.rmtree(backup, ignore_errors=True)
        return root

    def load(self, directory: str | Path | None = None) -> int:
        root = Path(directory or self.root)
        index_path = root / "index.json"
        if not index_path.exists():
            return 0
        meta = json.loads(index_path.read_text())
        loaded = 0
        with self._lock:
            self._cells.clear()
            for entry in meta.get("cells", []):
                blob_path = root / "states" / f"{entry['key']}.state"
                if not blob_path.exists():
                    continue
                self._cells[entry["key"]] = Cell(
                    key=entry["key"],
                    blob=blob_path.read_bytes(),
                    milestone=int(entry["milestone"]),
                    map_id=int(entry["map_id"]),
                    badges=int(entry["badges"]),
                    events=int(entry["events"]),
                    seen_maps=frozenset(entry.get("seen_maps", [])),
                    # Archives written before health was tracked default to 1.0, which
                    # would make every stale cell look like a perfect launch pad. Assume
                    # the opposite so they lose to any cell whose health is known.
                    hp_frac=float(entry.get("hp_frac", 0.0)),
                    level_sum=int(entry.get("level_sum", 0)),
                    exp=int(entry.get("exp", 0)),
                    visits=int(entry.get("visits", 0)),
                    chosen=int(entry.get("chosen", 0)),
                    created_at=float(entry.get("created_at", time.time())),
                    best_return=(
                        -np.inf
                        if entry.get("best_return") is None
                        else float(entry["best_return"])
                    ),
                )
                loaded += 1
            self.max_milestone = int(meta.get("max_milestone", 0))
            self.num_inserts = int(meta.get("num_inserts", loaded))
            self.num_updates = int(meta.get("num_updates", 0))
            stored_chain = meta.get("chain", "")
            self.chain_matches = bool(stored_chain) and stored_chain == self.chain
        log.info("archive: loaded %d cells (max milestone %d)", loaded, self.max_milestone)
        return loaded
