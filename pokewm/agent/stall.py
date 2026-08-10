"""Stall detection, and an active probe that tells you *why*.

Motivation
----------
Every stall this project has hit was visible in the metrics long before anyone noticed
it, and in most cases the diagnosis was counter-intuitive:

* the archive froze at 28 cells for 2.7M steps (cell key could not change within a phase)
* imagined reward went negative and the agent began blacking out on purpose
* policy entropy pinned at exactly ln|A| for 400k steps, then later collapsed to 0.24
* a milestone sat unchanged for 9.1M steps because the *measurement* was broken -- the
  agent was completing it in ~25 steps and a wrong RAM base made it invisible

The last one is the important one. Watching metrics alone cannot distinguish "the agent
cannot do X" from "X is being measured wrongly" or "X is gated behind something else".
Only an active experiment can, which is what `ReachabilityProbe` is for: it takes the
frontier save states and drives them with a deliberately dumb high-entropy policy. If
that reaches the next milestone, the milestone is reachable and the *learned policy* is
the bottleneck. If it never does, the problem is upstream of learning -- a gate, a wrong
target, or a broken predicate -- and no amount of RL tuning will help.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class StallReport:
    stalled: bool
    reasons: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        head = "STALLED" if self.stalled else "healthy"
        parts = [f"{head}: " + ("; ".join(self.reasons) if self.reasons else "progressing")]
        parts += [f"  hint: {h}" for h in self.hints]
        return "\n".join(parts)


class StallDetector:
    """Flags a stall when nothing that represents *progress* has moved.

    Deliberately narrow about what counts. Throughput, losses and reward are all poor
    proxies -- a run can hold 350 steps/s with a falling world-model loss while making no
    game progress whatsoever, which is exactly what several stalls here looked like.
    Progress means: the milestone advanced, the archive grew, new ground was covered, or
    a story flag fired.
    """

    def __init__(
        self,
        window_steps: int = 750_000,
        min_samples: int = 8,
        action_space: int = 7,
        hard_window_steps: int = 3_000_000,
    ) -> None:
        self.window_steps = window_steps
        # Separate, much longer window for *game* progress -- milestones and story flags.
        #
        # `any(...)` over the full signal set was too permissive to catch the failure it
        # was written for. `unique_coords` climbs indefinitely (interiors, re-covered
        # ground after a worker restart) and the archive keeps minting cells, so a run
        # that had not advanced a milestone in 2.0M steps still read "healthy:
        # progressing". Secondary counters show the machinery is alive; only the
        # milestone shows the *game* moving, and it needs its own test.
        self.hard_window_steps = hard_window_steps
        self.min_samples = min_samples
        self.max_entropy = float(np.log(action_space))
        self._hist: deque[tuple[int, dict[str, float]]] = deque(maxlen=4096)

    def update(self, env_steps: int, signals: dict[str, float]) -> None:
        self._hist.append((int(env_steps), dict(signals)))

    def state_dict(self) -> list:
        """History, for checkpointing.

        Without this the detector is blind to exactly the stalls it exists to catch. Its
        hard-progress window is 3M env steps, but the history lived only in memory, so
        every restart reset it to empty -- and a night of frequent restarts meant the
        window never once filled. Milestone 11 was held for 3.1M steps while the detector
        reported "healthy: progressing" throughout, because from its point of view the
        run was always a few minutes old.
        """
        return [[int(s), dict(d)] for s, d in self._hist]

    def load_state_dict(self, state) -> None:
        self._hist.clear()
        for entry in state or []:
            try:
                step, signals = entry
                self._hist.append((int(step), dict(signals)))
            except (TypeError, ValueError):
                continue

    # Counters that only ever increase within one continuous stretch of a run. A drop in
    # any of them means the stretch was interrupted, not that progress reversed.
    _MONOTONE = ("milestone", "archive_cells", "unique_coords", "max_events")

    def _discontinuity(self) -> int:
        """Index just after the last counter reset, or 0.

        Persisting history across restarts made the detector blind to its own seams.
        Worker-lifetime counters restart at zero, so a window straddling a restart
        computes deltas like `dunique_coords=-2264` and reports STALLED on a run that is
        simply young again. A monotone counter going backwards is proof of a seam, and
        everything before it is not comparable to what follows.
        """
        cut = 0
        hist = list(self._hist)
        for i in range(1, len(hist)):
            prev, cur = hist[i - 1][1], hist[i][1]
            for k in self._MONOTONE:
                if k in prev and k in cur and cur[k] < prev[k] - 1e-9:
                    cut = i
                    break
        return cut

    def _window(self, span: int | None = None) -> list[tuple[int, dict[str, float]]]:
        if not self._hist:
            return []
        hist = list(self._hist)[self._discontinuity():]
        if not hist:
            return []
        latest = hist[-1][0]
        limit = self.window_steps if span is None else span
        return [(s, d) for s, d in hist if latest - s <= limit]

    def check(self) -> StallReport:
        win = self._window()
        if len(win) < self.min_samples:
            return StallReport(False, ["warming up"], [], {})

        span = win[-1][0] - win[0][0]
        first, last = win[0][1], win[-1][1]

        def delta(key: str) -> float:
            return float(last.get(key, 0.0) - first.get(key, 0.0))

        progress = {
            "milestone": delta("milestone"),
            "archive_cells": delta("archive_cells"),
            "unique_coords": delta("unique_coords"),
            "max_events": delta("max_events"),
        }
        moved = any(v > 0 for v in progress.values())

        reasons: list[str] = []
        hints: list[str] = []
        if not moved:
            reasons.append(
                f"no progress in {span:,} env steps "
                + ", ".join(f"d{k}={v:+.0f}" for k, v in progress.items())
            )

        # Hard progress: the milestone or a story flag, over a much longer window.
        hard_win = self._window(self.hard_window_steps)
        hard_span = hard_win[-1][0] - hard_win[0][0] if hard_win else 0
        hard_stalled = False
        if hard_span >= self.hard_window_steps and len(hard_win) >= self.min_samples:
            hf, hl = hard_win[0][1], hard_win[-1][1]
            d_ms = float(hl.get("milestone", 0.0) - hf.get("milestone", 0.0))
            d_ev = float(hl.get("max_events", 0.0) - hf.get("max_events", 0.0))
            if d_ms <= 0 and d_ev <= 0:
                hard_stalled = True
                reasons.append(
                    f"no milestone or story flag in {hard_span:,} env steps "
                    f"(cells and coords may still be growing -- that is the machinery "
                    f"working, not the game advancing)"
                )
                hints.append(
                    "check that restores land at the *frontier milestone*, not merely on "
                    "the right map: a stale level can put the agent somewhere the next "
                    "milestone is sealed off (`python -m pokewm.diagnose --logdir ...`)"
                )

        # Known failure signatures. Each is a stall *cause* seen in a real run here, so
        # they are reported whether or not the stall test above has tripped -- they are
        # early warnings on their own.
        h = float(last.get("entropy", 0.0))
        if h >= 0.995 * self.max_entropy:
            hints.append(
                f"policy entropy {h:.3f} is at the ln|A|={self.max_entropy:.3f} ceiling: "
                "the entropy bonus is swamping the policy gradient (advantages too "
                "small, or ActorCriticConfig.entropy too high)"
            )
        elif h <= 0.15 * self.max_entropy:
            hints.append(
                f"policy entropy {h:.3f} has collapsed: the policy will not stumble into "
                "scripted interactions (face an NPC, press A). Raise "
                "ActorCriticConfig.entropy or --reset-policy"
            )

        imag = float(last.get("imag_reward", 0.0))
        if imag < 0:
            hints.append(
                f"imagined reward {imag:+.4f} is negative: every state has negative "
                "value, so ending the episode is optimal. Per-step penalties must stay "
                "below the intrinsic earning rate (docs/PROOF.md 6.1b)"
            )

        frac = float(last.get("frontier_frac", 1.0))
        if frac < 0.15:
            hints.append(
                f"only {frac:.1%} of archive cells are at the frontier: restores land "
                "behind it. Check ArchiveConfig.max_cells_per_level and target_bonus"
            )

        if progress["archive_cells"] <= 0 and span > 0:
            hints.append(
                "archive stopped growing: cell keys may be unable to change within the "
                "current phase (this froze a run at 28 cells for 2.7M steps)"
            )

        if progress["max_events"] <= 0 and progress["milestone"] <= 0:
            hints.append(
                "no story flag has fired: if the agent looks like it is doing the right "
                "thing, suspect the *measurement* before the policy -- a wrong "
                "wEventFlags base hid a milestone the agent was achieving in ~25 steps"
            )

        return StallReport(
            stalled=(not moved) or hard_stalled,
            reasons=reasons,
            hints=hints,
            signals={**{f"d_{k}": v for k, v in progress.items()},
                     "span": float(span), "entropy": h, "imag_reward": imag},
        )


@dataclass
class ProbeResult:
    milestone_key: str
    reachable: bool
    cells_tried: int
    steps_used: int
    detail: str
    # Did the probe reach any position bucket the launch cells did not already cover?
    #
    # Without this the verdict cannot separate "blocked" from "too far", and it defaults
    # to blaming the environment. Measured: Route 2's accessible southern section is
    # 10 tiles wide by 24 tall, which at position_bucket=8 is six cells -- the archive
    # held five, so the grid was *saturated*, not stalled, and the corridor north was
    # wide open. The old verdict said "check the RAM/predicate before touching the
    # policy" for what was purely a random-walk range limit.
    new_ground: bool = False

    def describe(self) -> str:
        if self.reachable:
            return (
                f"PROBE: '{self.milestone_key}' IS reachable from the frontier "
                f"({self.detail}). The milestone, its target maps and its predicate are "
                "all fine -- the learned policy is the bottleneck."
            )
        if self.new_ground:
            return (
                f"PROBE: '{self.milestone_key}' not reached from {self.cells_tried} "
                f"frontier cells in {self.steps_used:,} probe steps, but the probe did "
                f"cover new ground ({self.detail}). INCONCLUSIVE, and it does not "
                "implicate the environment: a uniform random walk covers ~sqrt(n) "
                "distance, so a corridor tens of tiles long defeats it even when wide "
                "open. Treat this as 'too far for a random walk', not as a gate."
            )
        return (
            f"PROBE: '{self.milestone_key}' NOT reached from {self.cells_tried} frontier "
            f"cells in {self.steps_used:,} probe steps, and it never left the ground the "
            f"cells already cover ({self.detail}). This one does point upstream of "
            "learning: a game gate, the wrong target maps, or a predicate that cannot "
            "fire. Check the RAM/predicate before touching the policy."
        )


class ReachabilityProbe:
    """Actively test whether the next milestone can be reached at all.

    Drives the frontier save states with a fixed, deliberately unsophisticated policy
    (uniform movement with frequent A presses). That is enough for scripted interactions
    and short traversals, which is exactly the class of obstacle that has blocked this
    run, and it depends on none of the learned components -- so a negative result
    implicates the environment, the predicate or the targets rather than the agent.
    """

    def __init__(self, cfg, num_cells: int = 6, steps_per_cell: int = 4_000) -> None:
        self.cfg = cfg
        self.num_cells = num_cells
        self.steps_per_cell = steps_per_cell

    def run(self, archive, next_milestone, seen_maps_hint=None) -> ProbeResult:
        from dataclasses import replace as _replace

        from ..emulator.env import NUM_ACTIONS, PokemonRedEnv
        from ..llm.subgoals import NUM_SUBGOALS

        if next_milestone is None:
            return ProbeResult("(complete)", True, 0, 0, "chain finished")

        targets = next_milestone.targets()
        cells = sorted(
            archive._cells.values(),
            key=lambda c: (c.map_id in targets, c.milestone),
            reverse=True,
        )[: self.num_cells]
        if not cells:
            return ProbeResult(next_milestone.key, False, 0, 0, "archive empty")

        env = PokemonRedEnv(
            _replace(self.cfg.env, render_gui=False, max_episode_steps=10**9),
            self.cfg.reward, num_subgoals=NUM_SUBGOALS,
        )
        rng = np.random.default_rng(0)
        used = 0
        bucket = max(int(getattr(self.cfg.archive, "position_bucket", 8)), 1)
        covered: set[tuple[int, int, int]] = set()
        try:
            for cell in cells:
                _, info = env.reset(options={"state_blob": cell.blob})
                seen = set(cell.seen_maps) | set(seen_maps_hint or ())
                start = info["state"]
                home = (start.map_id, start.x // bucket, start.y // bucket)
                for i in range(self.steps_per_cell):
                    # Uniform movement, A pressed a third of the time: enough for a
                    # scripted trigger or a short walk, and independent of the policy.
                    act = 4 if i % 3 == 0 else int(rng.integers(NUM_ACTIONS))
                    _, _, _, _, info = env.step(act)
                    used += 1
                    gs = info["state"]
                    seen.add(gs.map_id)
                    here = (gs.map_id, gs.x // bucket, gs.y // bucket)
                    if here != home:
                        covered.add(here)
                    if next_milestone.satisfied(gs, seen):
                        return ProbeResult(
                            next_milestone.key, True, len(cells), used,
                            f"fired after {i} steps from a {gs.map_name} cell",
                            new_ground=True,
                        )
        finally:
            env.close()
        archived = {
            tuple(int(p) for p in c.key.split(":")[2:5])
            for c in archive._cells.values()
            if len(c.key.split(":")) >= 5
        }
        fresh = covered - archived
        return ProbeResult(
            next_milestone.key, False, len(cells), used,
            "targets=" + ", ".join(sorted(str(t) for t in targets))
            + f"; {len(fresh)} position buckets not already archived",
            new_ground=bool(fresh),
        )
