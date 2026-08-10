"""Gymnasium environment wrapping PyBoy running Pokemon Red.

Observation is deliberately *multi-modal*, following the tokenisation argument in
Simulus (2025): a pixel-only observation forces the world model to spend capacity
re-deriving facts (badge count, party HP, whether a text box is open) that are exactly
representable as a handful of numbers, while a symbolic-only observation throws away the
spatial structure the agent needs to navigate. We give it both:

  frame     (frame_stack + 1, H, W) uint8  -- downsampled luminance + a visited-tile map
  symbolic  (SYMBOLIC_DIM,)         float32 -- decoded RAM
  subgoal   (NUM_SUBGOALS,)         float32 -- one-hot subgoal from the LLM proposer

The visited-tile channel is a memory prosthesis: Game Boy screens are locally
ambiguous (one patch of grass looks like any other), so without it the POMDP requires
the recurrent state to carry a full map. PokeRL (2026) measured +40.6% unique tiles
visited from exactly this channel, and it is cheap.
"""

from __future__ import annotations

import hashlib
import io
import os
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..agent.milestones import map_rank
from .maps import POKECENTER_MAPS
from ..config import ROM_SHA1, EnvConfig, RewardConfig
from . import ram_map as RM
from .ram_map import SYMBOLIC_DIM, GameState, RamReader, encode_symbolic

# Action set. START is included (unlike PokeRL, which drops it to stop menu spam)
# because the late game is unreachable without it: HM field moves, healing items and
# the bicycle are all behind the START menu. Menu spam is handled by the shaped step
# cost and the epistemic bonus instead, not by amputating the action space.
ACTIONS: tuple[str, ...] = ("down", "left", "right", "up", "a", "b", "start")
NUM_ACTIONS = len(ACTIONS)

# Local window (in tiles) of the per-map visited overlay, centred on the player.
VISIT_WIN_H = 36
VISIT_WIN_W = 40
MAP_GRID = 160  # lazily allocated per-map visited bitmap edge, in tiles


def rom_sha1(path: str | os.PathLike) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_luma(rgba: np.ndarray) -> np.ndarray:
    """(144,160,C) uint8 -> (144,160) uint8 luminance."""
    if rgba.ndim == 2:
        return rgba.astype(np.uint8)
    rgb = rgba[:, :, :3].astype(np.uint16)
    # Integer BT.601 weights; avoids a float round-trip on the hot path.
    return ((rgb[:, :, 0] * 77 + rgb[:, :, 1] * 150 + rgb[:, :, 2] * 29) >> 8).astype(
        np.uint8
    )


def downsample(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Box-average downsample by an integer factor, else nearest-neighbour."""
    h, w = frame.shape
    if h % out_h == 0 and w % out_w == 0:
        fh, fw = h // out_h, w // out_w
        return frame.reshape(out_h, fh, out_w, fw).mean(axis=(1, 3)).astype(np.uint8)
    ys = (np.arange(out_h) * h // out_h).clip(0, h - 1)
    xs = (np.arange(out_w) * w // out_w).clip(0, w - 1)
    return frame[np.ix_(ys, xs)]


class PokemonRedEnv(gym.Env):
    """One PyBoy instance, one agent.

    Thread/process model: this class is *not* thread-safe and owns a native emulator, so
    the vector env runs one instance per subprocess.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        cfg: EnvConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        num_subgoals: int = 1,
        worker_id: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.worker_id = worker_id
        self.num_subgoals = max(1, num_subgoals)
        self._rng = np.random.default_rng(seed if seed is not None else worker_id)

        rom = Path(self.cfg.rom_path)
        if not rom.exists():
            raise FileNotFoundError(f"ROM not found at {rom}")
        if self.cfg.check_rom_hash:
            got = rom_sha1(rom)
            if got != ROM_SHA1:
                raise ValueError(
                    f"ROM sha1 mismatch: expected {ROM_SHA1}, got {got}. Every RAM "
                    "offset in pokewm.emulator.ram_map is specific to the USA/Europe "
                    "revision; set EnvConfig.check_rom_hash=False to override."
                )

        # Derive the post-intro state from the ROM on first use. Idempotent and cheap
        # (~0.5 s); every worker after the first just reads the cached file.
        if not self.cfg.init_state:
            from .bootstrap import ensure_init_state

            self.cfg.init_state = ensure_init_state(self.cfg)

        from pyboy import PyBoy  # imported lazily so tests can run without a display

        self._pyboy = PyBoy(
            str(rom),
            window="SDL2" if self.cfg.render_gui else "null",
            sound_emulated=False,
            log_level="ERROR",
        )
        self._pyboy.set_emulation_speed(self.cfg.emulation_speed)
        self.ram = RamReader(self._pyboy.memory)

        c = self.cfg.frame_stack + self.cfg.seen_map_channels
        self.observation_space = spaces.Dict(
            {
                "frame": spaces.Box(
                    0, 255, (c, self.cfg.frame_h, self.cfg.frame_w), np.uint8
                ),
                "symbolic": spaces.Box(-4.0, 4.0, (SYMBOLIC_DIM,), np.float32),
                "subgoal": spaces.Box(0.0, 1.0, (self.num_subgoals,), np.float32),
            }
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # -- persistent (cross-episode) exploration memory -------------------------
        # These intentionally survive `reset()`: novelty is defined per *run*, not per
        # episode, otherwise an agent is paid repeatedly for rediscovering Route 1.
        self.visited_tiles: dict[int, np.ndarray] = {}
        self.seen_coords: set[tuple[int, int, int]] = set()
        self.seen_maps: set[int] = set()
        self.max_event_bits = 0
        self.max_badges = 0
        self.max_level_sum = 0
        self.max_party_size = 0
        self.max_dex_owned = 0
        self.max_dex_seen = 0

        self._frames: list[np.ndarray] = []
        self._subgoal_id = 0
        # What the proposer last asked for, kept separate from what is currently active:
        # a hurt party overrides it with HEAL. See `_active_subgoal`.
        self._llm_subgoal_id = 0
        self._forced_subgoal = False
        self._subgoal_paid = False
        # Short-term position memory backing the anti-dither penalty. Episodic, unlike
        # `seen_coords`, which is a run-level novelty record.
        self._recent_positions: deque[tuple[int, int, int]] = deque(
            maxlen=max(1, self.cfg.dither_window)
        )
        self._steps = 0
        # Last on-path map rank, the reference point for `map_progress`. Re-anchored on
        # reset; off-path maps leave it unchanged.
        self._map_rank_ref = 0
        # Set when the party wipes. Separate flags because the teleport and the refill
        # are separate events consumed by separate reward terms.
        self._blackout_map_pending = False
        self._blackout_hp_pending = False
        # Paid once per bout of damage, when the agent actually reaches a Pokecenter.
        self._heal_trip_paid = True
        self._prev_statused = 0.0
        # Battle bookkeeping: last seen opponent HP (for the damage term), whether the
        # opponent has fainted (for `battle_won`), and the no-progress counter behind
        # `battle_stall`.
        self._enemy_fainted = False
        self._prev_balls = 0
        self._prev_enemy_hp = 0
        self._prev_enemy_max_hp = 0
        self._enemy_fainted = False
        self._battle_stall_steps = 0
        self._battle_stall_charged = 0.0
        self._stall_enemy_hp = -1
        self._stall_hp = -1.0
        self._prev_state: GameState | None = None
        self._prev_hp_frac = 0.0
        self._episode_reward = 0.0
        self._pending_epistemic = 0.0
        self._last_state: GameState | None = None

    # ---------------------------------------------------------------- emulator I/O

    @property
    def pyboy(self):
        return self._pyboy

    def _raw_frame(self) -> np.ndarray:
        return to_luma(np.asarray(self._pyboy.screen.ndarray))

    def _tick(self, n: int, render: bool = False) -> None:
        self._pyboy.tick(n, render)

    def _press(self, action: int) -> None:
        """Send one button for `button_hold_frames`, then idle out the rest.

        Headless, only the final tick of an action is rendered: PyBoy skips the LCD
        pipeline otherwise, which is where most of the speedup comes from, and the agent
        only ever looks at the last frame anyway.

        With a visible SDL window that same trick makes the display unwatchable. One
        agent action is `action_frames` (24) emulator frames, so presenting only the last
        one shows ~2.5 frames per second of a 60 fps console -- it reads as flicker, not
        as slow motion. When a window is open we present every frame; the emulator is
        speed-limited to real time in that mode anyway, so it costs nothing.
        """
        btn = ACTIONS[action]
        hold = min(self.cfg.button_hold_frames, self.cfg.action_frames)
        render_all = bool(self.cfg.render_gui)
        self._pyboy.button_press(btn)
        self._tick(hold, render_all)
        self._pyboy.button_release(btn)
        remaining = self.cfg.action_frames - hold
        if remaining > 0:
            self._tick(remaining - 1, render_all)
        self._tick(1, True)

    def save_state(self) -> bytes:
        buf = io.BytesIO()
        self._pyboy.save_state(buf)
        return buf.getvalue()

    def load_state(self, blob: bytes) -> None:
        self._pyboy.load_state(io.BytesIO(blob))
        self._pyboy.tick(1, True)

    # ---------------------------------------------------------------- observation

    def _visit_grid(self, map_id: int) -> np.ndarray:
        grid = self.visited_tiles.get(map_id)
        if grid is None:
            grid = np.zeros((MAP_GRID, MAP_GRID), dtype=np.uint8)
            self.visited_tiles[map_id] = grid
        return grid

    def _visited_channel(self, gs: GameState) -> np.ndarray:
        """Local crop of the visited overlay, upsampled to frame size."""
        grid = self._visit_grid(gs.map_id)
        cy, cx = int(gs.y), int(gs.x)
        top, left = cy - VISIT_WIN_H // 2, cx - VISIT_WIN_W // 2
        win = np.zeros((VISIT_WIN_H, VISIT_WIN_W), dtype=np.uint8)
        ys0, xs0 = max(0, top), max(0, left)
        ys1 = min(MAP_GRID, top + VISIT_WIN_H)
        xs1 = min(MAP_GRID, left + VISIT_WIN_W)
        if ys1 > ys0 and xs1 > xs0:
            win[ys0 - top : ys1 - top, xs0 - left : xs1 - left] = grid[ys0:ys1, xs0:xs1]
        win = win * 255
        # Mark the player's own tile with a mid-grey so the agent can localise itself.
        win[VISIT_WIN_H // 2, VISIT_WIN_W // 2] = 128
        reps_y = max(1, self.cfg.frame_h // VISIT_WIN_H)
        reps_x = max(1, self.cfg.frame_w // VISIT_WIN_W)
        up = np.kron(win, np.ones((reps_y, reps_x), dtype=np.uint8))
        return downsample(up, self.cfg.frame_h, self.cfg.frame_w)

    def _observe(self, gs: GameState) -> dict[str, np.ndarray]:
        stack = np.stack(self._frames, axis=0)
        if self.cfg.seen_map_channels:
            stack = np.concatenate([stack, self._visited_channel(gs)[None]], axis=0)
        subgoal = np.zeros(self.num_subgoals, dtype=np.float32)
        active = self._active_subgoal(gs.party_hp_frac, gs.party_size,
                                      gs.in_battle, gs.party_statused,
                                      gs.ball_count, gs.money)
        self._forced_subgoal = active != self._llm_subgoal_id
        self._subgoal_id = active
        subgoal[active % self.num_subgoals] = 1.0
        return {
            "frame": stack.astype(np.uint8),
            "symbolic": encode_symbolic(gs),
            "subgoal": subgoal,
        }

    def _push_frame(self) -> None:
        f = downsample(self._raw_frame(), self.cfg.frame_h, self.cfg.frame_w)
        if not self._frames:
            self._frames = [f.copy() for _ in range(self.cfg.frame_stack)]
        else:
            self._frames.pop(0)
            self._frames.append(f)

    # ---------------------------------------------------------------- subgoals

    def set_subgoal(self, subgoal_id: int) -> None:
        """Called by the async LLM proposer. Cheap and lock-free by construction."""
        new_id = int(subgoal_id) % self.num_subgoals
        if new_id != self._subgoal_id:
            # A fresh assignment is payable again; without this reset the bonus would
            # only ever fire once per run.
            self._subgoal_paid = False
        self._subgoal_id = new_id
        self._llm_subgoal_id = new_id

    def _heal_trip_owed(self, gs: GameState) -> bool:
        """Whether a restored state starts out needing a trip to a Pokemon Center.

        An empty party is the opening of the game, not an injury -- it reads 0.0 HP with
        nothing to heal, and treating it as hurt would owe a heal trip from the first
        step of every run.
        """
        if gs.party_size <= 0:
            return False
        return (gs.party_hp_frac < self.reward_cfg.heal_subgoal_hp
                or gs.party_statused > 0.0)

    def _active_subgoal(self, hp_frac: float, party_size: int = 1,
                        in_battle: int = 0, statused: float = 0.0,
                        ball_count: int = 1, money: int = 0) -> int:
        """Override the LLM while the party is hurt: get healed first.

        The proposer decides well but slowly -- single-flight, a 30 s cooldown, and
        round-robin over 8 workers, so a given worker's suggestion is minutes old. "You
        are at 10% HP" cannot wait minutes, and it was the state the agent spent nearly
        all of its time in: every archived Viridian Forest cell held one level-6 Pokemon
        at 10-40% health, losing 72 of 90 wild encounters.

        This changes only what the agent *sees*. The subgoal bonus is deliberately not
        paid for the override (see `_reward`), because a bonus for healing would make a
        damage-then-heal loop profitable. The reward for recovering HP stays
        `hp_potential`, which is symmetric and therefore unfarmable; this just points the
        policy at the Pokecenter while it is in trouble.
        """
        # A wild battle is the only moment catching is possible, and it lasts seconds.
        # It takes precedence over HEAL because a Pokecenter is unreachable mid-battle.
        #
        # `ball_count` is not a refinement, it is the difference between a goal and a
        # fiction: with an empty bag there is no action sequence that catches anything,
        # and CATCH fired on every wild encounter for tens of millions of steps while
        # the agent carried a Town Map and a Potion. Pointing at the Mart instead makes
        # the subgoal reachable, and `RewardConfig.ball` pays for arriving.
        if (in_battle == 1 and ball_count > 0
                and 0 < party_size < self.reward_cfg.catch_subgoal_party):
            from ..llm.subgoals import CATCH_SUBGOAL_ID

            return CATCH_SUBGOAL_ID % self.num_subgoals
        # Affordability is part of the condition for the same reason `ball_count` is:
        # forcing BUY_ITEMS at a Mart the agent cannot pay at would replace one
        # unsatisfiable subgoal with another. A Poke Ball is 200.
        if (ball_count == 0
                and 0 < party_size < self.reward_cfg.catch_subgoal_party
                and money >= self.reward_cfg.ball_price
                and hp_frac >= self.reward_cfg.heal_subgoal_hp
                and statused <= 0.0):
            from ..llm.subgoals import BUY_BALLS_SUBGOAL_ID

            return BUY_BALLS_SUBGOAL_ID % self.num_subgoals
        if self.reward_cfg.heal_subgoal_hp <= 0.0:
            return self._llm_subgoal_id
        # An empty party reads as 0.0 HP, which is not an injury -- it is the opening of
        # the game, before Oak hands over a starter. Forcing HEAL there would pin the
        # subgoal from the first step of the run and never release it.
        # A status ailment is treated exactly like being hurt. Poison in particular
        # costs HP every few steps in the overworld, so waiting for HP to fall past the
        # threshold means bleeding out on the way there; and only a Pokecenter (or an
        # item the agent does not carry) clears it.
        if party_size > 0 and (hp_frac < self.reward_cfg.heal_subgoal_hp
                               or statused > 0.0):
            from ..llm.subgoals import HEAL_SUBGOAL_ID

            return HEAL_SUBGOAL_ID % self.num_subgoals
        return self._llm_subgoal_id

    @property
    def subgoal_id(self) -> int:
        return self._subgoal_id

    def set_epistemic_bonus(self, value: float) -> None:
        """World-model disagreement for the *previous* transition.

        Injected by the trainer rather than computed here, because it needs the model.
        Applied on the next step so the env stays a pure function of the emulator.
        """
        self._pending_epistemic = float(value)

    # ---------------------------------------------------------------- gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        blob: bytes | None = options.get("state_blob")
        if blob is not None:
            self.load_state(blob)
        elif self.cfg.init_state:
            with open(self.cfg.init_state, "rb") as fh:
                self.load_state(fh.read())
        else:
            raise RuntimeError(
                "No initial state. Run scripts/make_init_state.py once to script "
                "through the title/naming screens, or pass options['state_blob']."
            )

        self._frames = []
        self._push_frame()
        self._steps = 0
        self._episode_reward = 0.0
        self._pending_epistemic = 0.0
        self._subgoal_paid = False
        self._recent_positions.clear()

        gs = self.ram.read()
        self._prev_state = gs
        self._last_state = gs
        self._prev_hp_frac = gs.party_hp_frac
        # Re-anchor the map-progress reference. An archive restore can drop the agent
        # anywhere, and without this the first step of an episode that began on Pallet
        # Town after one that ended on Route 1 would be charged the whole 4-rank drop.
        r0 = map_rank(gs.map_id)
        if r0 >= 0:
            self._map_rank_ref = r0
        self._blackout_map_pending = False
        self._blackout_hp_pending = False
        # Owe a heal trip if the *restored* state is already hurt.
        #
        # Hardcoding True here made `heal_visit` unearnable in precisely the situation
        # it exists for. The flag is only cleared by a downward crossing of
        # `heal_subgoal_hp` within an episode, but 85% of episodes restore an archived
        # cell and every deep cell is captured hurt -- measured, all 20 Pewter City
        # cells sat below 0.38 HP. Starting below the threshold means the crossing never
        # happens, so the flag stayed True for the whole episode: `heal_visit` fired
        # zero times in 58M env steps, and the run reached Pewter having never once
        # completed a heal. That is why `wLastBlackoutMap` is still Pallet Town and
        # every faint costs the entire journey back.
        self._heal_trip_paid = not self._heal_trip_owed(gs)
        self._prev_statused = gs.party_statused
        self._prev_enemy_hp = 0
        self._prev_enemy_max_hp = 0
        self._enemy_fainted = False
        # Anchored to what the restored state actually carries, so a restore is never
        # itself paid (or charged) for the balls that came with the save.
        self._prev_balls = min(gs.ball_count, self.reward_cfg.ball_cap)
        self._battle_stall_steps = 0
        self._battle_stall_charged = 0.0
        self._stall_enemy_hp, self._stall_hp = -1, -1.0
        self._mark_visited(gs)
        return self._observe(gs), {"state": gs, "info_kind": "reset"}

    def export_exploration(self) -> dict:
        """Novelty memory that should outlive the process, not just the episode.

        These are documented as *run*-level records: an agent must not be paid twice for
        discovering Route 1. They survived `reset()` but not a restart, so a long run
        punctuated by restarts kept re-earning `new_tile` over ground it had already
        covered -- rewarding a return to the easy, well-trodden part of a map rather than
        the push into the unknown.
        """
        return {
            "seen_coords": [list(c) for c in self.seen_coords],
            "seen_maps": sorted(self.seen_maps),
            "max_event_bits": self.max_event_bits,
            "max_badges": self.max_badges,
            "max_level_sum": self.max_level_sum,
            "max_party_size": self.max_party_size,
            "max_dex_owned": self.max_dex_owned,
            "max_dex_seen": self.max_dex_seen,
        }

    def import_exploration(self, state: dict | None) -> None:
        if not state:
            return
        self.seen_coords = {tuple(c) for c in state.get("seen_coords", [])}
        self.seen_maps = set(state.get("seen_maps", []))
        # The monotone maxima matter as much as the coordinates: restoring coverage but
        # not these would let the agent re-earn `event`, `level` and dex credit too.
        self.max_event_bits = int(state.get("max_event_bits", 0))
        self.max_badges = int(state.get("max_badges", 0))
        self.max_level_sum = int(state.get("max_level_sum", 0))
        self.max_party_size = int(state.get("max_party_size", 0))
        self.max_dex_owned = int(state.get("max_dex_owned", 0))
        self.max_dex_seen = int(state.get("max_dex_seen", 0))

    def _mark_visited(self, gs: GameState) -> None:
        grid = self._visit_grid(gs.map_id)
        if 0 <= gs.y < MAP_GRID and 0 <= gs.x < MAP_GRID:
            grid[gs.y, gs.x] = 1

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(int(action)), f"bad action {action}"
        self._press(int(action))
        self._push_frame()
        gs = self.ram.read()

        reward, breakdown = self._reward(gs)
        self._mark_visited(gs)

        self._steps += 1
        self._episode_reward += reward

        # Do not cut an episode off in the middle of a battle.
        #
        # Every episode ends by truncation at `max_episode_steps`, and when that landed
        # mid-fight the battle was simply abandoned: the worker restored from a different
        # archive cell, the outcome never happened, and whatever the agent had learned
        # about that fight was split across an episode boundary. Trainer battles are the
        # ones that matter -- they cannot be fled, so an abandoned one is a fight the
        # agent never has to resolve. `battle_grace_steps` bounds the extension so a
        # genuinely stuck battle cannot hold an episode open forever.
        over_budget = self._steps - self.cfg.max_episode_steps
        truncated = self._steps >= self.cfg.max_episode_steps
        # ...but a battle that has already been charged the full stall penalty is not
        # going to resolve, and holding the episode open for it just burns the worker.
        # That is the whole failure above: an unwinnable, unfleeable gym battle
        # consuming episodes end to end. Once the stall charge is exhausted the fight
        # has demonstrably stopped progressing, so let the truncation land.
        stalled_out = self._battle_stall_charged >= abs(self.reward_cfg.faint) * 1.5
        if (truncated and gs.in_battle != 0 and not stalled_out
                and over_budget < self.cfg.battle_grace_steps):
            truncated = False
        # A blackout is a setback, not an ending: the game teleports the player to a
        # Pokecenter and continues. Ending the episode here would let the agent escape
        # any negative per-step reward by dying on purpose, which it demonstrably
        # learned to do. See EnvConfig.terminate_on_wipe.
        terminated = bool(gs.party_wiped) and self.cfg.terminate_on_wipe
        info: dict[str, Any] = {
            "state": gs,
            "reward_breakdown": breakdown,
            "episode_reward": self._episode_reward,
            "steps": self._steps,
            "badges": gs.badge_count,
            "events": gs.event_flag_bits,
            "map_id": gs.map_id,
            "hp_frac": gs.party_hp_frac,
            "position": gs.position,
            "unique_coords": len(self.seen_coords),
            "unique_maps": len(self.seen_maps),
            "progress_key": gs.progress_key(),
            "hall_of_fame": gs.map_id == RM.MAP_HALL_OF_FAME,
        }
        self._prev_state = gs
        self._last_state = gs
        return self._observe(gs), reward, terminated, truncated, info

    # ---------------------------------------------------------------- reward

    def _reward(self, gs: GameState) -> tuple[float, dict[str, float]]:
        """First-visit-credit shaping.

        Each term pays only when a *monotone* progress statistic reaches a new maximum
        for this run. That makes the shaped reward a bounded, non-recurring bonus, which
        is the condition under which the optimal policy set is preserved
        (docs/PROOF.md §4).
        """
        rc = self.reward_cfg
        b: dict[str, float] = {}

        if gs.event_flag_bits > self.max_event_bits:
            b["event"] = rc.event * (gs.event_flag_bits - self.max_event_bits)
            self.max_event_bits = gs.event_flag_bits
        if gs.badge_count > self.max_badges:
            b["badge"] = rc.badge * (gs.badge_count - self.max_badges)
            self.max_badges = gs.badge_count
        if gs.party_size > self.max_party_size:
            b["party_member"] = rc.party_member * (gs.party_size - self.max_party_size)
            self.max_party_size = gs.party_size
        if gs.party_level_sum > self.max_level_sum:
            b["level"] = rc.level * (gs.party_level_sum - self.max_level_sum)
            self.max_level_sum = gs.party_level_sum
        if rc.ball:
            balls = min(gs.ball_count, rc.ball_cap)
            if balls != self._prev_balls:
                b["ball"] = rc.ball * (balls - self._prev_balls)
            self._prev_balls = balls
        if gs.dex_owned > self.max_dex_owned:
            b["dex_owned"] = rc.dex_owned * (gs.dex_owned - self.max_dex_owned)
            self.max_dex_owned = gs.dex_owned
        if gs.dex_seen > self.max_dex_seen:
            b["dex_seen"] = rc.dex_seen * (gs.dex_seen - self.max_dex_seen)
            self.max_dex_seen = gs.dex_seen

        # Directed progress along the critical path.
        #
        # Every other exploration term here is first-visit-only, so on ground the agent
        # has already covered there is *no* signal at all: `new_tile` has been paid,
        # `new_map` has been paid, and the epistemic bonus has decayed to nothing exactly
        # where the world model is well fit. Measured consequence: restored onto Route 1
        # with Viridian to the north and Pallet to the south, the agent kept re-running
        # the delivery route it had spent millions of steps learning -- it walked south.
        # 38 archived frontier cells accumulated in Pallet Town against 2 on Route 1.
        #
        # Paid on *transitions* between on-path maps, which telescopes: any round trip
        # nets exactly zero, so this cannot be farmed by oscillating, and there is no
        # per-step component. The latter matters more than it looks -- a standing cost of
        # even -0.002/step previously made blacking out the optimal policy (see
        # RewardConfig.dither). Discounting the potential by gamma would reintroduce one
        # of ~-0.0018/step on the deepest maps, so this deliberately uses gamma = 1 and
        # accepts an O(1-gamma) departure from exact policy invariance.
        #
        # Off-path maps (rank -1: shops, Pokecenters, houses) carry the previous rank
        # forward rather than scoring -1, so stepping into a building is free. Charging
        # for it would have penalised entering the Viridian Mart, which is where the
        # parcel is.
        # A blackout teleports the player home. That is a map change, but it is not
        # travel, and charging it double-counts a setback `faint` already prices -- at
        # weight 0.5 a Route 2 -> Pallet Town wipe billed -4.0 on top of the -10. It also
        # corrupted the diagnostic: the mean payout per transition looked like the agent
        # choosing to walk backwards when it was being *carried* backwards, which is a
        # very different fault with a very different fix.
        if gs.party_wiped:
            # Two independent flags, consumed by two independent terms. A wipe produces
            # both a teleport home and a full refill, and with one shared flag whichever
            # term ran first swallowed it -- the map term did, so the refill was paid
            # after all. Measured live as hp_potential pinned at exactly +3.000 (a full
            # 0 -> 1.0 swing) alongside faint, which made a blackout cost -5 rather than
            # the intended -8.
            self._blackout_map_pending = True
            self._blackout_hp_pending = True
        if rc.map_progress:
            rank_now = map_rank(gs.map_id)
            if rank_now >= 0 and rank_now != self._map_rank_ref:
                if self._blackout_map_pending:
                    self._blackout_map_pending = False   # re-anchor, do not charge
                else:
                    b["map_progress"] = rc.map_progress * (rank_now - self._map_rank_ref)
                self._map_rank_ref = rank_now

        if gs.map_id not in self.seen_maps:
            self.seen_maps.add(gs.map_id)
            b["new_map"] = rc.new_map
        if gs.position not in self.seen_coords:
            self.seen_coords.add(gs.position)
            b["new_tile"] = rc.new_tile
        elif gs.position in self._recent_positions:
            # Short-term revisit: dithering rather than travelling.
            b["dither"] = rc.dither
        self._recent_positions.append(gs.position)

        # Walking to a Pokecenter while hurt.
        #
        # `hp_potential` is symmetric, so healing only refunds what the damage cost --
        # there is no net gain, and therefore no pull towards the trip itself. The forced
        # HEAL subgoal says where to go but pays nothing for going. This is the directed
        # part: a bounded bonus for arriving at a Pokecenter while actually hurt.
        #
        # Paid once per bout of damage -- re-armed only when HP drops below the threshold
        # again -- so walking in and out of the building cannot farm it.
        if rc.heal_visit and gs.map_id in POKECENTER_MAPS:
            if not self._heal_trip_paid and (gs.party_hp_frac < rc.heal_subgoal_hp
                                             or gs.party_statused > 0.0):
                b["heal_visit"] = rc.heal_visit
                self._heal_trip_paid = True

        # Curing a status ailment. Symmetric like `hp_potential`: contracting one costs
        # what curing it pays, so there is no loop in deliberately getting poisoned.
        if rc.status_potential and abs(gs.party_statused - self._prev_statused) > 1e-6:
            if self._blackout_hp_pending or gs.party_wiped:
                pass                              # the free clear on waking is not earned
            else:
                b["status_potential"] = rc.status_potential * (
                    self._prev_statused - gs.party_statused
                )

        # Party HP as a symmetric potential. Paid on losses as well as gains, so a
        # damage-then-heal round trip telescopes to zero and cannot be farmed, while
        # sitting at low HP stays strictly worse than sitting at full HP. A blackout is
        # excluded: the free refill on waking would otherwise pay the agent back for
        # everything it just lost, which is the opposite of the intended lesson, and
        # `faint` already prices the wipe.
        hp = gs.party_hp_frac
        ailing = gs.party_statused > 0.0
        if hp >= rc.heal_subgoal_hp and not ailing:
            self._heal_trip_paid = True          # healthy and clean: nothing owed
        elif (self._prev_hp_frac >= rc.heal_subgoal_hp and hp < rc.heal_subgoal_hp) \
                or (ailing and not self._prev_statused):
            self._heal_trip_paid = False         # newly hurt or newly ailing
        if rc.hp_potential and abs(hp - self._prev_hp_frac) > 1e-6:
            if self._blackout_hp_pending or gs.party_wiped:
                self._blackout_hp_pending = False     # the free refill is not earned
            else:
                b["hp_potential"] = rc.hp_potential * (hp - self._prev_hp_frac)
        self._prev_hp_frac = hp
        self._prev_statused = gs.party_statused

        # Damage dealt: the only dense signal for actually fighting. Without it "attack"
        # and "reopen the bag" score the same, and a trainer battle -- which cannot be
        # fled -- has nothing objecting to an agent that spends it cycling menus.
        if rc.enemy_damage and gs.in_battle != 0 and gs.enemy_max_hp > 0:
            if self._prev_enemy_max_hp == gs.enemy_max_hp:
                dealt = self._prev_enemy_hp - gs.enemy_hp
                if dealt > 0:
                    b["enemy_damage"] = rc.enemy_damage * (dealt / gs.enemy_max_hp)
            self._prev_enemy_hp = gs.enemy_hp
            self._prev_enemy_max_hp = gs.enemy_max_hp
        elif gs.in_battle == 0:
            self._prev_enemy_hp = 0
            self._prev_enemy_max_hp = 0

        # Winning the fight, as distinct from damaging the opponent.
        #
        # The flag is latched while the battle is up and cashed when it ends, because at
        # the moment `in_battle` clears the enemy party struct is already stale. Fleeing
        # never sets it -- that is the whole point: the two outcomes have to be
        # distinguishable or fleeing remains the safe play. See `RewardConfig.battle_won`.
        if gs.in_battle != 0:
            if gs.enemy_max_hp > 0 and gs.enemy_hp == 0:
                self._enemy_fainted = True
        elif self._enemy_fainted:
            if rc.battle_won and not gs.party_wiped:
                b["battle_won"] = rc.battle_won
            self._enemy_fainted = False

        # Stalling in a battle there is no way out of.
        #
        # Deliberately keyed on "nothing is happening" rather than on a RUN cursor index:
        # the battle menu is nested and its indices shift as submenus open, so a
        # hardcoded index would be both ROM-specific and wrong the moment a submenu is up.
        if gs.in_battle == 2:
            moved = (gs.enemy_hp != self._stall_enemy_hp
                     or abs(hp - self._stall_hp) > 1e-6)
            if moved:
                self._battle_stall_steps = 0
                self._battle_stall_charged = 0.0
                self._stall_enemy_hp, self._stall_hp = gs.enemy_hp, hp
            else:
                self._battle_stall_steps += 1
                # Cap the cumulative charge below what a wipe costs. The penalty applies
                # every step after the grace with no bound, so a long stall could exceed
                # `faint` -- and then deliberately losing the battle becomes the cheaper
                # way out, which is precisely the failure that `dither` once produced.
                # The cap used to sit *below* `faint`, to stop deliberately losing from
                # becoming the cheaper way out of a stall. That produced the mirror
                # failure instead, and it is the one that actually happened: with the cap
                # at 2.5 against a wipe at 5.0, **stalling was the cheaper way out**, and
                # the agent took it. Measured over the 530k steps after it first engaged
                # Brock -- a level-14 Onix against a level-8 Charmander, a trainer battle
                # it can neither win nor flee -- `battle_stall` fired in 48 of 52 metric
                # rows while `battle_won` and `faint` fired in *none*. It was not losing;
                # it was sitting there until the episode ran out.
                #
                # Above `faint`, so the ordering is: win (+3.0) > lose (-5.0) > stall
                # (-7.5). Losing at least ends the fight and hands back a healthy state;
                # stalling burns the whole episode and teaches nothing. Deliberately
                # losing is still discouraged by `faint` itself.
                cap = abs(rc.faint) * 1.5
                if (self._battle_stall_steps > rc.battle_stall_grace
                        and self._battle_stall_charged < cap):
                    b["battle_stall"] = rc.battle_stall
                    self._battle_stall_charged += abs(rc.battle_stall)
        else:
            self._battle_stall_steps = 0
            self._battle_stall_charged = 0.0
            self._stall_enemy_hp, self._stall_hp = -1, -1.0

        if gs.party_wiped:
            b["faint"] = rc.faint

        # The LLM's suggestion pays out only when its machine-checkable predicate fires,
        # and only once per assignment. An unhelpful suggestion therefore costs nothing.
        # Not paid while HEAL is being forced: a bonus for healing would make a
        # damage-then-heal cycle profitable. `hp_potential` already pays for recovery and
        # is symmetric, so it cannot be farmed.
        if not self._subgoal_paid and self._prev_state is not None \
                and not self._forced_subgoal:
            from ..llm.subgoals import satisfied as _sg_satisfied

            if _sg_satisfied(self._subgoal_id, self._prev_state, gs):
                b["subgoal"] = rc.subgoal
                self._subgoal_paid = True

        if self._pending_epistemic:
            b["epistemic"] = rc.epistemic * self._pending_epistemic
            self._pending_epistemic = 0.0

        b["step_cost"] = rc.step_cost
        total = float(np.clip(sum(b.values()), -rc.clip, rc.clip))
        return total, b

    # ---------------------------------------------------------------- misc

    def render(self):
        return np.asarray(self._pyboy.screen.ndarray)[:, :, :3]

    @property
    def state(self) -> GameState | None:
        return self._last_state

    def exploration_stats(self) -> dict[str, float]:
        return {
            "unique_coords": float(len(self.seen_coords)),
            "unique_maps": float(len(self.seen_maps)),
            "max_badges": float(self.max_badges),
            "max_events": float(self.max_event_bits),
            "max_level_sum": float(self.max_level_sum),
        }

    def close(self) -> None:
        if getattr(self, "_pyboy", None) is not None:
            self._pyboy.stop(save=False)
            self._pyboy = None
