"""Subprocess vector environment.

PyBoy holds native emulator state and releases the GIL inconsistently, so threads do not
scale. One OS process per emulator does, and the observation payload is small
(5x72x80 uint8 = 28 KB) relative to the ~1.5 ms of emulation per step.

Save-state blobs (~164 KB) are the one large payload. A worker only ships one when it
enters a progress cell it has not reported before, so the archive gets its candidates
without the pipe carrying a state on every step.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import traceback
from dataclasses import replace
from typing import Any

import numpy as np

from ..agent.milestones import MilestoneTracker
from ..config import EnvConfig, RewardConfig

log = logging.getLogger(__name__)

# Commands
_RESET = "reset"
_STEP = "step"
_SUBGOAL = "subgoal"
_EPISTEMIC = "epistemic"
_STATS = "stats"
_SNAPSHOT = "snapshot"
_TEXTSTATE = "textstate"
_CLOSE = "close"
_GET_EXPLORED = "get_explored"
_SET_EXPLORED = "set_explored"


def _worker(
    remote,
    parent_remote,
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    num_subgoals: int,
    worker_id: int,
    seed: int,
    position_bucket: int = 8,
) -> None:
    parent_remote.close()
    try:
        from .env import PokemonRedEnv

        env = PokemonRedEnv(
            replace(env_cfg, render_gui=False),
            reward_cfg,
            num_subgoals=num_subgoals,
            worker_id=worker_id,
            seed=seed,
        )
        tracker = MilestoneTracker()
        reported_keys: set[str] = set()
        # Guard against snapshotting a state whose coordinates have not settled after a
        # map change; see the `controllable` check below.
        last_map_id, steps_on_map = -1, 0
        last_obs: dict[str, np.ndarray] | None = None
        # Health and strength captured with `blob`, so the archive stores metadata that
        # describes the state it actually holds. See `_pack_info`.
        blob_hp_frac, blob_level_sum, blob_exp = 1.0, 0, 0

        while True:
            cmd, data = remote.recv()

            if cmd == _RESET:
                blob, seed_maps = data
                obs, info = env.reset(options={"state_blob": blob})
                # Restore the milestone history along with the emulator state.
                #
                # Without this the tracker restarts from scratch on every archive
                # restore, so a worker relaunched deep in the game reports milestone 1.
                # Since a cell's archive score *is* its milestone, deep cells were being
                # filed as shallow -- 36 of 131 cells in the first long run sat in Oak's
                # Lab labelled milestone 1. That is self-defeating: the archive restores
                # into a deep cell, records the result as shallow, and its own frontier
                # estimate stops advancing.
                tracker = MilestoneTracker()
                if seed_maps:
                    tracker.seen_maps.update(seed_maps)
                tracker.update(info["state"])
                reported_keys.clear()
                last_map_id, steps_on_map = -1, 0
                blob_hp_frac, blob_level_sum, blob_exp = 1.0, 0, 0
                last_obs = obs
                remote.send(("ok", (obs, _pack_info(info, tracker, None))))

            elif cmd == _STEP:
                # The epistemic bonus rides along with the action rather than taking its
                # own round trip; at ~1 ms of emulation per step the extra synchronous
                # exchange was costing more than the emulation itself.
                action, bonus = data
                if bonus:
                    env.set_epistemic_bonus(bonus)
                obs, reward, term, trunc, info = env.step(int(action))
                newly = tracker.update(info["state"])
                blob = None
                gs = info["state"]
                key = cell_key(gs, tracker.index, position_bucket)
                info["cell_key"] = key
                # Never archive a state the agent does not control.
                #
                # While a script owns the joypad (`wJoyIgnore` nonzero: map transitions,
                # text boxes, cutscenes) or a battle is up, the position in RAM is a
                # transient -- and it is the position that becomes the cell key, and so
                # the bucket. One such cell was stored at Viridian Forest (5,0), a tile
                # with no walkable neighbour, and pressing a direction from it did not
                # move the agent but dropped it at (17,43) or (15,47) wherever the script
                # was headed. Restoring there also wastes the episode's opening steps,
                # because input is ignored until the script releases it. Measured: 46 of
                # 778 archived cells (6%) were captured this way.
                # Also require the position to have settled after a map change.
                #
                # `joy_ignore` clears a step or two before the coordinates do, so a state
                # captured in that window reads a transient position -- and the position
                # is the cell key, hence the bucket. Entering Viridian Forest from the
                # south gate reliably produced a cell claiming (5,0), a tile on the far
                # side of the map from where the agent actually was: stepping from it
                # teleported to (17,47). Purging those cells did not help because the
                # transition regenerates them every time the agent walks in, and they
                # went on to corrupt two separate reachability analyses.
                if gs.map_id != last_map_id:
                    last_map_id = gs.map_id
                    steps_on_map = 0
                else:
                    steps_on_map += 1
                # A wiped party is not a launch pad either: restoring there blacks out
                # on the next step and teleports to `wLastBlackoutMap`, spending the
                # episode somewhere the archive did not choose. HP reaching zero in the
                # overworld (poison, most often) leaves `joy_ignore` clear for a step or
                # two before the blackout script takes the joypad, which is the window
                # these were captured in -- 9 of 378 cells, 7 of them in Oak's Lab.
                #
                # An *empty* party is not a wipe: that is the opening of the game, and
                # those cells are legitimate. Hence the party_size check.
                alive = gs.party_size == 0 or gs.party_hp_frac > 0.0
                controllable = (gs.joy_ignore == 0 and gs.in_battle == 0
                                and steps_on_map >= 2 and alive)
                # Re-snapshot when the party gets materially better, not only when the
                # cell key is new.
                #
                # `progress_key` hashes badges, dex, party size and the story flags.
                # Winning a *wild* battle changes none of them, so the key stays fixed
                # for the whole episode, the snapshot keeps the state as it was around
                # step 2, and every point of experience earned afterwards is thrown away
                # when the episode ends. Measured at 86.4M env steps: `battle_won` firing
                # in 145 of 146 metric rows while no archived cell had ever exceeded
                # 327 XP and `level_sum` sat at 8 -- the agent was winning constantly and
                # banking none of it. The historical cells at 589 and 722 XP came from
                # forest *trainer* battles, which set a story flag and so happened to
                # change the key.
                #
                # Both triggers are rare (a win, or a heal), so this costs a handful of
                # extra 164 KB snapshots per episode rather than one per step.
                stronger = gs.party_exp_sum > blob_exp
                healthier = gs.party_hp_frac > blob_hp_frac + 0.05
                if controllable and (key not in reported_keys or stronger or healthier):
                    reported_keys.add(key)
                    blob = env.save_state()
                    blob_hp_frac = gs.party_hp_frac
                    blob_level_sum = gs.party_level_sum
                    blob_exp = gs.party_exp_sum
                packed = _pack_info(info, tracker, blob, newly=newly,
                                    blob_hp_frac=blob_hp_frac,
                                    blob_level_sum=blob_level_sum,
                                    blob_exp=blob_exp)
                if term or trunc:
                    packed["terminal_observation"] = obs
                last_obs = obs
                remote.send(("ok", (obs, float(reward), bool(term), bool(trunc), packed)))

            elif cmd == _SUBGOAL:
                env.set_subgoal(int(data))
                remote.send(("ok", None))

            elif cmd == _EPISTEMIC:
                env.set_epistemic_bonus(float(data))
                remote.send(("ok", None))

            elif cmd == _STATS:
                remote.send(("ok", env.exploration_stats()))

            elif cmd == _GET_EXPLORED:
                remote.send(("ok", env.export_exploration()))

            elif cmd == _SET_EXPLORED:
                env.import_exploration(data)
                remote.send(("ok", None))

            elif cmd == _SNAPSHOT:
                gs = env.state
                remote.send(
                    (
                        "ok",
                        {
                            "blob": env.save_state(),
                            "text": gs.to_text() if gs else "",
                            "screen": env.render(),
                            "obs": last_obs,
                        },
                    )
                )

            elif cmd == _TEXTSTATE:
                gs = env.state
                remote.send(
                    (
                        "ok",
                        {
                            "text": gs.to_text() if gs else "",
                            "screen": env.render(),
                            "milestone": tracker.index,
                            "frontier": tracker.frontier_label,
                        },
                    )
                )

            elif cmd == _CLOSE:
                env.close()
                remote.send(("ok", None))
                break

            else:  # pragma: no cover - programming error
                raise RuntimeError(f"unknown command {cmd!r}")

    except EOFError:
        pass
    except Exception:  # pragma: no cover - surfaced to the parent
        remote.send(("error", traceback.format_exc()))
    finally:
        try:
            remote.close()
        except Exception:
            pass


def cell_key(gs, milestone: int, position_bucket: int) -> str:
    """Archive cell identity: irreversible progress + milestone + coarse position.

    Each component earns its place:

    * `progress_key` -- badges, story flags, party size. Distinguishes genuinely
      different game states.
    * `milestone` -- lets the frontier advance during a phase that sets no story flag
      (e.g. leaving the bedroom before receiving a starter).
    * coarse `(map, x//b, y//b)` -- lets the frontier advance *within* a milestone.
      Without it the archive freezes solid across long routes: the first long training
      run sat at 28 cells for 2.7M steps on Route 1 because nothing else in the key ever
      changed. Bucketing keeps this from exploding into one cell per tile.
    """
    return (
        f"{gs.progress_key()}:{milestone}:"
        f"{gs.map_id}:{gs.x // position_bucket}:{gs.y // position_bucket}"
    )


def default_start_method() -> str:
    """`forkserver` where available, else `spawn`.

    This is a memory decision, and a large one. With `spawn`, each worker re-imports
    `__main__` -- which for a training run is `pokewm.train`, and therefore pulls in
    torch and the CUDA runtime. Measured at **527 MB RSS per worker**: 16 emulators cost
    8.4 GB on a 16 GB machine, which is most of the way to an OOM kill overnight.

    `forkserver` starts one lean server process that imports only the modules named in
    `set_forkserver_preload` (none of which touch torch), and forks each worker from it,
    so the interpreter and library pages are shared copy-on-write.

    Plain `fork` would share even more, but forking a process that has already
    initialised a CUDA context is unsafe; forkserver sidesteps that because the server
    is a fresh process that never touches CUDA.
    """
    if sys.platform.startswith("linux") and "forkserver" in mp.get_all_start_methods():
        return "forkserver"
    return "spawn"


def _pack_info(
    info: dict[str, Any],
    tracker: MilestoneTracker,
    blob: bytes | None,
    newly: list[str] | None = None,
    blob_hp_frac: float = 1.0,
    blob_level_sum: int = 0,
    blob_exp: int = 0,
) -> dict[str, Any]:
    """Strip the un-picklable / oversized parts of `info` before it crosses the pipe."""
    gs = info["state"]
    return {
        "reward_breakdown": info.get("reward_breakdown", {}),
        "episode_reward": info.get("episode_reward", 0.0),
        "steps": info.get("steps", 0),
        "badges": gs.badge_count,
        "events": gs.event_flag_bits,
        "map_id": gs.map_id,
        "position": gs.position,
        "party_size": gs.party_size,
        "level_sum": gs.party_level_sum,
        "unique_coords": info.get("unique_coords", 0),
        "unique_maps": info.get("unique_maps", 0),
        "progress_key": info.get("progress_key", ""),
        "cell_key": info.get("cell_key", info.get("progress_key", "")),
        "milestone": tracker.index,
        "frontier": tracker.frontier_label,
        "seen_maps": frozenset(tracker.seen_maps),
        "newly_satisfied": newly or [],
        "hall_of_fame": info.get("hall_of_fame", False),
        "state_blob": blob,
        # Party health and strength *of the snapshot*, not of the current step.
        #
        # Two distinct defects, both fixed here.
        #
        # `hp_frac` was never in this payload at all, so the trainer's
        # `info.get("hp_frac", 1.0)` took its default on every insert. Measured on a
        # live archive: recorded health took exactly two values, 1.0 for the 242 cells
        # the trainer had inserted and 0.0 for the 132 inherited from an older archive.
        # It was never a measurement, so `hp_weight` and `frontier_min_hp` were ranking
        # cells by a constant -- and `--from-frontier` handed the viewer a "healthy"
        # Pewter cell that restored at 0.42 HP and blacked out 111 steps later.
        #
        # `level_sum` *is* in the payload but is read fresh each step, while `blob` is
        # captured once, when the cell key first appears, and rides along unchanged
        # afterwards. Pairing the two recorded a strength the stored state does not have.
        "blob_hp_frac": blob_hp_frac,
        "blob_level_sum": blob_level_sum,
        "blob_exp": blob_exp,
        "text": gs.to_text(),
    }


class VecPokemonRed:
    """Synchronous-step, parallel-execution vector env.

    Autoreset semantics match Gymnasium's `SyncVectorEnv`: when a sub-env terminates or
    truncates, it is reset immediately and the *new* observation is returned, with the
    final observation available under `info["terminal_observation"]`.
    """

    def __init__(
        self,
        num_envs: int,
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        num_subgoals: int,
        seed: int = 0,
        start_method: str | None = None,
        position_bucket: int = 8,
    ) -> None:
        self.num_envs = num_envs
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.num_subgoals = num_subgoals
        self.position_bucket = max(1, int(position_bucket))
        # Materialise the post-intro state once in the parent so N workers do not race
        # to write the same file.
        if not env_cfg.init_state:
            from .bootstrap import ensure_init_state

            env_cfg.init_state = ensure_init_state(env_cfg)
        self._boot_blob = open(env_cfg.init_state, "rb").read()

        start_method = start_method or default_start_method()
        ctx = mp.get_context(start_method)
        if start_method == "forkserver":
            # Only what a worker actually needs. Deliberately excludes anything that
            # reaches torch -- that is the entire point (see default_start_method).
            try:
                ctx.set_forkserver_preload(["pokewm.emulator.vec_env"])
            except Exception:  # pragma: no cover - non-fatal optimisation
                pass
        self.start_method = start_method
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(num_envs)]
        )
        self.procs = []
        for i, (wr, r) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(
                target=_worker,
                args=(wr, r, env_cfg, reward_cfg, num_subgoals, i, seed + i,
                      self.position_bucket),
                daemon=True,
            )
            p.start()
            self.procs.append(p)
            wr.close()
        self.closed = False

    # ------------------------------------------------------------------ plumbing

    def _recv(self, remote):
        status, payload = remote.recv()
        if status == "error":
            self.close()
            raise RuntimeError(f"env worker failed:\n{payload}")
        return payload

    @property
    def boot_blob(self) -> bytes:
        return self._boot_blob

    # ------------------------------------------------------------------ api

    def reset(self, blobs: list[bytes | None] | None = None, seen_maps=None):
        blobs = blobs or [None] * self.num_envs
        seen_maps = seen_maps or [None] * self.num_envs
        for remote, blob, seen in zip(self.remotes, blobs, seen_maps):
            remote.send(
                (_RESET, (blob if blob is not None else self._boot_blob, seen))
            )
        results = [self._recv(r) for r in self.remotes]
        obs = _stack_obs([o for o, _ in results])
        infos = [i for _, i in results]
        return obs, infos

    def reset_one(self, index: int, blob: bytes | None = None, seen_maps=None):
        """`seen_maps` seeds the milestone tracker so a restored worker does not
        report itself back to milestone 1 -- see the `_RESET` handler."""
        self.remotes[index].send(
            (_RESET, (blob if blob is not None else self._boot_blob, seen_maps))
        )
        obs, info = self._recv(self.remotes[index])
        return obs, info

    def step(self, actions, epistemic=None):
        acts = np.asarray(actions).reshape(-1)
        bonus = (
            np.zeros(self.num_envs, dtype=np.float32)
            if epistemic is None
            else np.asarray(epistemic, dtype=np.float32).reshape(-1)
        )
        for remote, a, e in zip(self.remotes, acts, bonus):
            remote.send((_STEP, (int(a), float(e))))
        results = [self._recv(r) for r in self.remotes]
        obs = _stack_obs([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        terms = np.array([r[2] for r in results], dtype=bool)
        truncs = np.array([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]
        return obs, rewards, terms, truncs, infos

    def export_exploration(self) -> list[dict]:
        """Per-worker novelty memory, for checkpointing.

        `seen_coords` is documented as a *run*-level record -- novelty should not pay
        twice for the same tile -- but it only ever lived in the worker process, so every
        restart silently reset it. Across a night of restarts that turned `new_tile` into
        a repeating payment for re-walking known ground instead of a bounty on new
        ground, which is the opposite of what it is for.
        """
        for r in self.remotes:
            r.send((_GET_EXPLORED, None))
        return [self._recv(r) for r in self.remotes]

    def import_exploration(self, states) -> None:
        for r, st in zip(self.remotes, states or []):
            r.send((_SET_EXPLORED, st))
            self._recv(r)

    def set_subgoals(self, subgoal_ids) -> None:
        """Broadcast the proposer's current assignment to every worker."""
        for remote, sg in zip(self.remotes, np.asarray(subgoal_ids).reshape(-1)):
            remote.send((_SUBGOAL, int(sg)))
        for remote in self.remotes:
            self._recv(remote)

    def set_epistemic(self, values):
        """Standalone injection. `step(actions, epistemic)` is the fast path; this
        exists for tests and for callers that are not stepping."""
        for remote, v in zip(self.remotes, np.asarray(values).reshape(-1)):
            remote.send((_EPISTEMIC, float(v)))
        for remote in self.remotes:
            self._recv(remote)

    def exploration_stats(self) -> list[dict[str, float]]:
        for remote in self.remotes:
            remote.send((_STATS, None))
        return [self._recv(r) for r in self.remotes]

    def snapshot(self, index: int) -> dict[str, Any]:
        self.remotes[index].send((_SNAPSHOT, None))
        return self._recv(self.remotes[index])

    def text_state(self, index: int) -> dict[str, Any]:
        self.remotes[index].send((_TEXTSTATE, None))
        return self._recv(self.remotes[index])

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send((_CLOSE, None))
            except (BrokenPipeError, OSError):
                pass
        for remote in self.remotes:
            try:
                remote.recv()
            except Exception:
                pass
            try:
                remote.close()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _stack_obs(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}
