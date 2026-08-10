"""The training loop.

Structure:

    collector thread  --->  SequenceReplay  --->  main thread
    (8 emulators,           (200k steps)          (world model + actor-critic
     actor forward,                                gradient steps on the GPU)
     archive writes)

They run concurrently. The emulator farm is CPU-bound and the learner is GPU-bound, so
serialising them would waste ~40% of wall clock -- measured 1.43 vs 2.30 updates/s on
this machine. The collector self-paces against a target *replay ratio* (replayed steps
per collected step) so the two never drift apart.

Everything that defines a run -- model weights, optimiser state, archive, milestone
tracker, RNG -- is checkpointed together, because the intended use is a multi-day run
that gets interrupted.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..emulator.archive import FrontierArchive
from ..emulator.env import NUM_ACTIONS
from ..emulator.ram_map import SYMBOLIC_DIM
from ..emulator.vec_env import VecPokemonRed
from ..llm.proposer import build_proposer
from ..llm.subgoals import NUM_SUBGOALS
from ..wm.actor_critic import ImaginationActorCritic
from ..wm.replay import SequenceReplay
from ..wm.rssm import RSSMState, flatten_state
from ..wm.world_model import WorldModel
from .stall import ReachabilityProbe, StallDetector
from .milestones import (
    NUM_MILESTONES,
    achieved_milestone,
    chain_fingerprint,
    next_milestone,
)

log = logging.getLogger(__name__)


def _purge_stale_optimizer_state(opt: torch.optim.Optimizer) -> int:
    """Drop per-parameter state left over from a differently-shaped checkpoint.

    Optimiser `load_state_dict` restores moments by position without checking shapes, so
    a parameter that changed width loads a stale `exp_avg` and the mismatch only surfaces
    inside `step()`, far from the cause. Returns the number of parameters cleared.
    """
    cleared = 0
    for param, state in list(opt.state.items()):
        for value in state.values():
            if torch.is_tensor(value) and value.dim() and value.shape != param.shape:
                state.clear()
                cleared += 1
                break
    if cleared:
        log.warning("cleared optimiser moments for %d reshaped parameter(s)", cleared)
    return cleared


class Metrics:
    """Windowed mean of every scalar anyone reports."""

    def __init__(self, window: int = 200) -> None:
        self._d: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def add(self, name: str, value) -> None:
        self._d[name].append(float(value))

    def update(self, d: dict) -> None:
        for k, v in d.items():
            self.add(k, v)

    def mean(self) -> dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self._d.items() if len(v)}

    def clear(self) -> None:
        self._d.clear()


class PokeWorldTrainer:
    def __init__(self, cfg: Config, resume: bool = True,
                 reset_policy: bool = False, force_relabel: bool = False) -> None:
        self.cfg = cfg
        self.device = torch.device(
            cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu"
            else "cpu"
        )
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        if cfg.train.torch_threads > 0:
            torch.set_num_threads(cfg.train.torch_threads)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.logdir = Path(cfg.train.logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        (self.logdir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

        # -- environment ------------------------------------------------------------
        self.envs = VecPokemonRed(
            cfg.train.num_envs,
            cfg.env,
            cfg.reward,
            num_subgoals=NUM_SUBGOALS,
            seed=cfg.train.seed,
            position_bucket=cfg.archive.position_bucket,
        )
        self.frame_channels = cfg.env.frame_stack + cfg.env.seen_map_channels
        self.frame_hw = (cfg.env.frame_h, cfg.env.frame_w)

        # -- model ------------------------------------------------------------------
        self.wm = WorldModel(
            self.frame_channels,
            self.frame_hw,
            SYMBOLIC_DIM,
            NUM_ACTIONS,
            NUM_SUBGOALS,
            cfg.wm,
        ).to(self.device)
        self.ac = ImaginationActorCritic(
            self.wm.rssm.feat_dim, NUM_ACTIONS, cfg.ac, cfg.wm
        ).to(self.device)
        self.opt_wm = torch.optim.AdamW(
            self.wm.parameters(), lr=cfg.wm.lr, eps=cfg.wm.eps,
            weight_decay=cfg.wm.weight_decay,
        )
        # `log_alpha` is a dual variable, not a policy weight: it needs its own (much
        # larger) step size, and it must stay out of the actor/critic gradient clip --
        # otherwise a big policy-gradient norm silently scales the temperature update
        # down to nothing.
        self._ac_params = [
            p for n, p in self.ac.named_parameters() if n != "log_alpha"
        ]
        self.opt_ac = torch.optim.AdamW(
            self._ac_params, lr=cfg.ac.lr, eps=cfg.ac.eps
        )
        self.opt_alpha = torch.optim.Adam(
            [self.ac.log_alpha], lr=cfg.ac.entropy_lr
        )

        # -- data -------------------------------------------------------------------
        self.replay = SequenceReplay(
            cfg.replay,
            num_streams=cfg.train.num_envs,
            frame_hw=self.frame_hw,
            symbolic_dim=SYMBOLIC_DIM,
            num_actions=NUM_ACTIONS,
            num_subgoals=NUM_SUBGOALS,
            extra_planes=cfg.env.seen_map_channels,
            frame_stack=cfg.env.frame_stack,
            seed=cfg.train.seed,
        )
        self.archive = FrontierArchive(cfg.archive, seed=cfg.train.seed)
        self.archive.chain = chain_fingerprint()
        self.proposer = build_proposer(cfg.llm, cfg.train.num_envs)

        # -- counters ---------------------------------------------------------------
        self.env_steps = 0
        self.updates = 0
        self.episodes = 0
        self.best_milestone = 0
        self.best_badges = 0
        self.completed = False
        self.start_time = time.time()
        self.metrics = Metrics()
        # Effective archive target last written to the log; used to log only changes.
        self._logged_targets: frozenset[int] = frozenset({-1})
        self.stall = StallDetector(
            window_steps=cfg.train.stall_window,
            hard_window_steps=cfg.train.stall_hard_window,
            action_space=NUM_ACTIONS,
        )
        # Milestone indices already announced this run. Reset when the chain changes,
        # because an index names a different milestone under a different chain.
        self._milestone_events: set[int] = set()
        # Latest novelty memory published by the collector thread; see save_checkpoint.
        self._exploration_snapshot: list[dict] = []
        self._last_exploration_snapshot = 0
        self._last_stall_check = 0
        self._last_target_refresh = 0
        self._last_probe_milestone = -1
        # Reasons of the last stall written, so an unchanged report is not rewritten.
        self._last_stall_fingerprint: tuple = ()
        self._stop = threading.Event()
        self._shutdown = False
        self._collector: threading.Thread | None = None
        self._collector_error: BaseException | None = None
        self._model_lock = threading.Lock()
        self._zero_bonus = np.zeros(cfg.train.num_envs, dtype=np.float32)

        # Replayed steps per collected step. Governs how hard the collector is throttled.
        self.steps_per_update = max(
            1, int(cfg.wm.batch_size * cfg.wm.batch_length / max(cfg.train.replay_ratio, 1e-6))
        )

        if resume:
            self.load_checkpoint(reset_policy=reset_policy)
        # Baseline for rate metrics, taken after any resume so throughput reflects this
        # process rather than the whole run history.
        if (getattr(self, "_chain_changed", False)
                or not self.archive.chain_matches
                or force_relabel):
            self._relabel_archive_for_current_chain()
        self._session_env_steps = self.env_steps
        self._session_updates = self.updates
        self._refresh_archive_targets()

    # ================================================================== collection

    def _init_rssm(self) -> RSSMState:
        return self.wm.rssm.initial(self.cfg.train.num_envs, self.device)

    @torch.no_grad()
    def _policy_step(
        self, obs: dict[str, np.ndarray], state: RSSMState, prev_action: torch.Tensor,
        is_first: np.ndarray, greedy: bool = False,
    ) -> tuple[np.ndarray, RSSMState, np.ndarray]:
        """Filter one observation and choose actions for every worker."""
        t = {
            "frame": torch.as_tensor(obs["frame"], device=self.device),
            "symbolic": torch.as_tensor(obs["symbolic"], device=self.device),
            "subgoal": torch.as_tensor(obs["subgoal"], device=self.device),
        }
        first = torch.as_tensor(is_first, device=self.device, dtype=torch.float32)
        with self._model_lock:
            state = self.wm.observe_step(state, t, prev_action, first)
            feat = state.feature()
            idx = self.ac.policy(feat, greedy=greedy)
            # The ensemble forward is skipped entirely unless the extrinsic reward is
            # configured to carry the bonus. On the collector's hot path that saved four
            # extra MLP forwards per step for a term that is normally applied only in
            # imagination.
            if self.cfg.reward.env_epistemic:
                onehot = torch.nn.functional.one_hot(idx, NUM_ACTIONS).float()
                bonus = self.wm.epistemic_bonus(state, onehot).float().cpu().numpy()
            else:
                bonus = self._zero_bonus
        return idx.cpu().numpy(), state, bonus

    def _collect_loop(self) -> None:
        try:
            self._collect_forever()
        except BaseException as exc:  # surfaced to the main thread
            self._collector_error = exc
            self._stop.set()
            log.exception("collector thread died")

    def _collect_forever(self) -> None:
        cfg = self.cfg
        n = cfg.train.num_envs
        obs, infos = self.envs.reset()
        state = self._init_rssm()
        prev_action = torch.zeros(n, NUM_ACTIONS, device=self.device)
        is_first = np.ones(n, dtype=bool)
        ep_return = np.zeros(n, dtype=np.float64)
        ep_len = np.zeros(n, dtype=np.int64)
        last_observe_step = np.zeros(n, dtype=np.int64)

        while not self._stop.is_set():
            # Pace against the learner so the replay ratio stays near target.
            if self.replay.ready:
                allowed = (self.updates + 1) * self.steps_per_update + cfg.train.prefill
                if self.env_steps >= allowed:
                    time.sleep(0.002)
                    continue

            actions, state, bonus = self._policy_step(obs, state, prev_action, is_first)
            next_obs, rewards, terms, truncs, infos = self.envs.step(actions, bonus)
            # `bonus` is all-zero unless reward.env_epistemic is set; the intrinsic
            # term is normally applied only to imagined rewards. See RewardConfig.

            done = terms | truncs
            cont = ~terms  # truncation is not a real terminal; bootstrap through it
            self.replay.add(
                frame=obs["frame"],
                symbolic=obs["symbolic"],
                subgoal=obs["subgoal"],
                action=actions,
                reward=rewards,
                cont=cont,
                is_first=is_first,
            )
            self.env_steps += n
            ep_return += rewards
            ep_len += 1

            self._ingest_infos(infos)

            prev_action = torch.nn.functional.one_hot(
                torch.as_tensor(actions, device=self.device), NUM_ACTIONS
            ).float()
            is_first = done.copy()
            obs = next_obs

            # -- episode boundaries: relaunch from the frontier ---------------------
            for i in np.nonzero(done)[0]:
                self.episodes += 1
                self.metrics.add("episode/return", ep_return[i])
                self.metrics.add("episode/length", ep_len[i])
                self.metrics.add("episode/terminated", float(terms[i]))
                ep_return[i] = 0.0
                ep_len[i] = 0
                blob, seen_maps = None, None
                if cfg.archive.enabled:
                    cell = self.archive.sample()
                    if cell is not None:
                        blob = cell.blob
                        # Hand the cell's map history back so the restored worker's
                        # milestone tracker continues from the cell's progress instead
                        # of restarting at 1.
                        seen_maps = cell.seen_maps
                        self.metrics.add("archive/restore_milestone", cell.milestone)
                new_obs, _ = self.envs.reset_one(int(i), blob, seen_maps)
                for k in obs:
                    obs[k][i] = new_obs[k]
                # Zero this worker's recurrent state; `is_first` already marks it.
                state = _reset_row(state, int(i))

            # -- LLM: publish state, apply whatever it has decided -------------------
            for i in range(n):
                if self.env_steps - last_observe_step[i] >= cfg.llm.refresh_steps:
                    last_observe_step[i] = self.env_steps
                    self.proposer.observe(
                        i, self.env_steps, infos[i]["text"], infos[i]["milestone"]
                    )
            if self.env_steps % 256 < n:
                self.envs.set_subgoals(self.proposer.all_subgoals())

            # Publish novelty memory for the checkpoint. Done here because this thread
            # owns the worker pipes; the main thread must not touch them.
            if self.env_steps - self._last_exploration_snapshot >= 50_000:
                self._last_exploration_snapshot = self.env_steps
                self._exploration_snapshot = self.envs.export_exploration()

    def _ingest_infos(self, infos: list[dict]) -> None:
        for i, info in enumerate(infos):
            ms = int(info["milestone"])
            if ms > self.best_milestone:
                self.best_milestone = ms
                self._refresh_archive_targets()
                # `ms` counts satisfied milestones, so the achievement is the one
                # *before* it. Indexing with the count names the next target instead.
                done = achieved_milestone(ms)
                label = done.label if done else "(none)"
                nxt = next_milestone(ms)
                # Log a milestone once per run, not once per rediscovery.
                #
                # Editing the milestone chain resets `best_milestone` so archive targets
                # are recomputed, and the counter is then rediscovered from live workers
                # -- which re-fired a "COMPLETED" event for every index it walked back
                # through. The events file is the record of *when each milestone was
                # first reached*, and the repeats made it unreadable: index 1 appeared
                # four times, and the history showed milestone 8 at 1.8M steps against
                # milestone 7 at 11.6M. That is the file I steer by, so it has to mean
                # what it says.
                first_time = ms not in self._milestone_events
                if first_time:
                    self._milestone_events.add(ms)
                    log.info(
                        "[%d steps] MILESTONE %d/%d COMPLETED: %s  (next: %s)",
                        self.env_steps, ms, NUM_MILESTONES - 1, label,
                        nxt.label if nxt else "GAME COMPLETE",
                    )
                    self._append_event({
                        "kind": "milestone", "index": ms, "label": label,
                        "next": nxt.label if nxt else None,
                        # The index only means something relative to a chain, and the
                        # chain has been edited mid-run.
                        "chain": chain_fingerprint(),
                    })
                else:
                    log.debug(
                        "milestone %d re-reached after a chain change; not re-logged", ms
                    )
            if info["badges"] > self.best_badges:
                self.best_badges = int(info["badges"])
                log.info("[%d steps] BADGE %d obtained", self.env_steps, self.best_badges)
                self._append_event({"kind": "badge", "count": self.best_badges})
            if info["hall_of_fame"]:
                self.completed = True
                log.info("HALL OF FAME REACHED at %d env steps", self.env_steps)
                self._append_event({"kind": "hall_of_fame"})

            blob = info.get("state_blob")
            if blob is not None and self.cfg.archive.enabled:
                self.archive.insert(
                    key=info["cell_key"],
                    blob=blob,
                    milestone=ms,
                    map_id=int(info["map_id"]),
                    badges=int(info["badges"]),
                    events=int(info["events"]),
                    seen_maps=info["seen_maps"],
                    episode_return=float(info["episode_reward"]),
                    # Of the *snapshot*, not of this step -- `state_blob` was captured
                    # when the cell key first appeared and rides along unchanged
                    # afterwards. See the note in `vec_env._pack_info`.
                    hp_frac=float(info.get("blob_hp_frac", 1.0)),
                    level_sum=int(info.get("blob_level_sum", 0)),
                    exp=int(info.get("blob_exp", 0)),
                )
            for k, v in info["reward_breakdown"].items():
                self.metrics.add(f"reward/{k}", v)
            self.metrics.add("progress/milestone", ms)
            self.metrics.add("progress/badges", info["badges"])
            self.metrics.add("progress/events", info["events"])
            self.metrics.add("progress/unique_coords", info["unique_coords"])
            self.metrics.add("progress/unique_maps", info["unique_maps"])
            self.metrics.add("progress/level_sum", info["level_sum"])

    def _relabel_archive_for_current_chain(self) -> None:
        """Re-score archived cells after a milestone-chain change.

        Boots each stored save state once in a throwaway emulator and recomputes its
        milestone. Costs a few seconds for a few hundred cells, and is the alternative
        to discarding the archive outright -- which would throw away every route the
        agent has already learned to walk.
        """
        if not self.cfg.archive.enabled or len(self.archive) == 0:
            return
        from dataclasses import replace as _replace

        from ..emulator.env import PokemonRedEnv
        from .milestones import milestone_index_of

        env = PokemonRedEnv(
            _replace(self.cfg.env, render_gui=False, max_episode_steps=10**9),
            self.cfg.reward, num_subgoals=NUM_SUBGOALS,
        )
        try:
            def evaluate(cell):
                _, info = env.reset(options={"state_blob": cell.blob})
                return milestone_index_of(info["state"], cell.seen_maps)

            t0 = time.time()
            changed = self.archive.relabel(evaluate)
            log.info(
                "relabelled %d/%d archive cells for the current milestone chain "
                "in %.1fs (max milestone now %d)",
                changed, len(self.archive), time.time() - t0,
                self.archive.max_milestone,
            )
        finally:
            env.close()

    def _check_for_stall(self) -> None:
        """Watch progress signals, and on a stall actively test reachability.

        The passive half cannot tell "cannot learn it" from "cannot be measured"; the
        probe can, and that distinction is what several rounds of misdirected tuning
        here cost.
        """
        m = self.metrics.mean()
        self.stall.update(self.env_steps, {
            "milestone": float(self.best_milestone),
            "archive_cells": float(len(self.archive)),
            "unique_coords": m.get("progress/unique_coords", 0.0),
            "max_events": m.get("progress/events", 0.0),
            "entropy": m.get("policy/entropy", 0.0),
            "imag_reward": m.get("policy/imag_reward", 0.0),
            "frontier_frac": self.archive.stats().get("archive/frontier_frac", 1.0),
        })
        report = self.stall.check()
        for hint in report.hints:
            log.warning("stall-watch: %s", hint)
        if not report.stalled:
            self._last_stall_fingerprint = ()   # a recurrence is worth recording again
            return
        log.warning("stall-watch: %s", "; ".join(report.reasons))
        # Record a stall when it *starts* or when what it says changes -- not once per
        # check. The same report repeated every 100k steps is not new information, and
        # it drowns everything else in the file.
        fingerprint = tuple(report.reasons)
        if fingerprint != self._last_stall_fingerprint:
            self._last_stall_fingerprint = fingerprint
            self._append_event(
                {"kind": "stall", "reasons": report.reasons, "hints": report.hints,
                 "signals": report.signals},
                diagnostic=True,
            )

        if not self.cfg.train.stall_probe:
            return
        nxt = next_milestone(self.best_milestone)
        if nxt is None or self._last_probe_milestone == self.best_milestone:
            return  # only probe once per milestone; it costs an emulator
        self._last_probe_milestone = self.best_milestone
        probe = ReachabilityProbe(
            self.cfg, num_cells=self.cfg.train.stall_probe_cells,
            steps_per_cell=self.cfg.train.stall_probe_steps,
        )
        result = probe.run(self.archive, nxt)
        log.warning("stall-watch: %s", result.describe())
        self._append_event({"kind": "probe", "milestone": result.milestone_key,
                            "reachable": result.reachable, "detail": result.detail},
                           diagnostic=True)

    def _refresh_archive_targets(self) -> None:
        """Point the archive at the maps the next milestone needs.

        Recomputed whenever the frontier advances, because the required maps are not
        monotone along the route -- the parcel run doubles back to Pallet Town.

        Also recomputed *periodically*, which is not a refinement but a fix. A "reach
        map X" milestone names a map with no cells, so the archive falls back to the
        deepest map it currently holds -- and that is a function of the archive, which
        keeps changing, not of the milestone. Refreshing only on milestone transitions
        froze the fallback at whatever existed in the instant the milestone fired, which
        is the worst possible instant to sample it: the milestone fires the step the
        agent *enters* the new map, and cells are inserted at episode end, so the new
        map is reliably absent. Measured after milestone 10: the target stuck on
        Viridian City, the five Route 2 cells were drawn once or twice each while
        Viridian accumulated 26, and `pokewm.diagnose` reported a healthy 99.7%
        on-target because it recomputes the fallback before sampling -- so the
        diagnostic and the live trainer were describing different archives.
        """
        if not self.cfg.archive.enabled:
            return
        nxt = next_milestone(self.best_milestone)
        targets = nxt.targets() if nxt else frozenset()
        self.archive.set_target_maps(targets)
        effective = self.archive.target_maps
        if effective != self._logged_targets:
            self._logged_targets = effective
            from ..emulator import maps as _M
            names = ", ".join(sorted(_M.map_name(m) for m in effective)) or "(none)"
            req = ", ".join(sorted(_M.map_name(m) for m in targets)) or "(none)"
            # Log the *effective* target. Logging only the request hid the fallback
            # entirely -- "archive targets -> Viridian Forest" was true and useless.
            log.info(
                "archive targets -> %s (for: %s%s)",
                names,
                nxt.label if nxt else "-",
                "" if effective == targets else f"; requested {req}, unreached",
            )

    def _append_event(self, payload: dict, *, diagnostic: bool = False) -> None:
        """Append to the run's ledgers.

        Two files, because they answer different questions and have wildly different
        rates. `events.jsonl` is the progress ledger -- what the run *achieved* -- and is
        read to reconstruct when each milestone fell. `diagnostics.jsonl` is the watch
        log: stall reports and reachability probes.

        Mixing them made the progress ledger unreadable. A stall report is written on
        every check, so at `stall_check_every` = 100k steps a plateau adds ~10 entries
        per million steps; the 23M-step milestone-11 plateau would have buried its 11
        milestones under 200+ status lines saying the same thing.
        """
        payload |= {"env_steps": self.env_steps, "wall_s": time.time() - self.start_time}
        name = "diagnostics.jsonl" if diagnostic else "events.jsonl"
        with open(self.logdir / name, "a") as fh:
            fh.write(json.dumps(payload) + "\n")

    # ================================================================== learning

    def train_step(self) -> dict[str, float]:
        cfg = self.cfg
        sampled = self.replay.sample(
            cfg.wm.batch_size, cfg.wm.batch_length, device=self.device
        )
        if sampled is None:
            return {}
        batch, streams, positions = sampled
        amp = torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=cfg.train.amp and self.device.type == "cuda"
        )

        # The model lock guards *writes* to parameters only -- the optimiser steps and
        # the slow-critic EMA. Forward and backward merely read weights (backward writes
        # .grad, which the collector never touches), so holding the lock across the whole
        # update is unnecessary and catastrophic for throughput: it pins the lock for
        # ~430 ms of every ~435 ms cycle, leaving the collector almost no time to run and
        # collapsing collection from ~1200 to ~36 env-steps/s. Guarding only the writes
        # takes the hold time to a few milliseconds.
        # -- world model ------------------------------------------------------------
        with amp:
            losses, post = self.wm.loss(batch)
        self.opt_wm.zero_grad(set_to_none=True)
        losses.total.backward()
        wm_grad = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), cfg.wm.grad_clip)
        with self._model_lock:
            self.opt_wm.step()

        # -- actor-critic on imagined rollouts --------------------------------------
        start = flatten_state(post.detach())
        n_start = start.deter.shape[0]
        if cfg.wm.imag_batch < n_start:
            sel = torch.randperm(n_start, device=self.device)[: cfg.wm.imag_batch]
            start = start[sel]
        with amp:
            feat, actions, logps, reward, cont = self.wm.imagine(
                start, self.ac.imagination_policy(), cfg.wm.horizon
            )
            epistemic = self.wm.imagine_epistemic(feat, actions)
            ac_losses = self.ac.losses(
                feat.float(), logps.float(), reward.float(), cont.float(),
                epistemic.float(), cfg.reward.epistemic,
            )
        self.opt_ac.zero_grad(set_to_none=True)
        self.opt_alpha.zero_grad(set_to_none=True)
        (ac_losses.actor + ac_losses.critic + ac_losses.alpha_loss).backward()
        ac_grad = torch.nn.utils.clip_grad_norm_(self._ac_params, cfg.ac.grad_clip)
        with self._model_lock:
            self.opt_ac.step()
            if cfg.ac.entropy_adaptive:
                self.opt_alpha.step()
                self.ac.clamp_log_alpha()
            self.ac.critic.update_slow()

        self.replay.update_priorities(
            streams, positions, losses.per_sequence.float().cpu().numpy()
        )
        self.updates += 1
        return {
            "loss/world_model": float(losses.total.detach()),
            "loss/frame": float(losses.frame),
            "loss/symbolic": float(losses.symbolic),
            "loss/reward": float(losses.reward),
            "loss/cont": float(losses.cont),
            "loss/kl": float(losses.kl),
            "loss/kl_dyn": float(losses.dyn),
            "loss/kl_rep": float(losses.rep),
            "loss/ensemble": float(losses.ensemble),
            "loss/actor": float(ac_losses.actor.detach()),
            "loss/critic": float(ac_losses.critic.detach()),
            "policy/entropy": float(ac_losses.entropy),
            "policy/entropy_coef": float(ac_losses.alpha),
            "policy/entropy_target": float(self.ac.target_entropy),
            "policy/value": float(ac_losses.value_mean),
            "policy/imag_return": float(ac_losses.return_mean),
            "policy/imag_reward": float(ac_losses.imag_reward),
            "policy/adv_std": float(ac_losses.adv_std),
            "policy/epistemic": float(epistemic.mean()),
            "grad/world_model": float(wm_grad),
            "grad/actor_critic": float(ac_grad),
        }

    # ================================================================== driver

    def _install_signal_handlers(self) -> None:
        """Deterministic shutdown on SIGINT/SIGTERM.

        Relying on KeyboardInterrupt alone is unreliable here: the main thread spends
        most of its time inside long C-level torch calls, and SIGTERM (what process
        managers actually send) does not raise it at all. Setting a flag the loop polls
        makes shutdown prompt and identical for both signals, so the run always
        checkpoints before exiting.
        """
        import signal

        def handler(signum, _frame):
            if not self._shutdown:
                log.info("received signal %d; finishing current step then saving", signum)
            self._shutdown = True
            self._stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:
                pass  # not on the main thread (e.g. under pytest); fall back to except

    def run(self) -> None:
        cfg = self.cfg
        self._install_signal_handlers()
        self.proposer.start()
        self._collector = threading.Thread(
            target=self._collect_loop, name="collector", daemon=True
        )
        self._collector.start()
        log.info(
            "training on %s | %d envs | wm %.1fM params | llm: %s",
            self.device, cfg.train.num_envs,
            sum(p.numel() for p in self.wm.parameters()) / 1e6,
            getattr(self.proposer, "reason", "disabled"),
        )

        last_log = time.time()
        last_ckpt = self.env_steps
        last_ckpt_t = time.time()
        try:
            while (
                self.env_steps < cfg.train.total_steps
                and not self.completed
                and not self._shutdown
            ):
                if self._collector_error is not None:
                    raise self._collector_error
                if not self.replay.ready:
                    time.sleep(0.25)
                    continue
                # Pace the learner against the replay ratio, symmetrically with the
                # collector.
                #
                # Only the collector was throttled, so the learner ran flat out and the
                # ratio was set by whatever the GPU could manage rather than by the
                # config. Measured: 1.6 updates/s x 1024 replayed steps against 253
                # collected steps/s is a replay ratio of 6.5, not the configured 2.0 --
                # three times as much re-training per fresh sample as intended. It is
                # also self-defeating, because the learner's kernels saturate the GPU and
                # the collector's 1.79 ms policy forward then queues behind them: the
                # emulators measured 1483 steps/s standalone but 253 in the live loop,
                # with the CPU 2/3 idle.
                #
                # Deltas are session-relative. Cumulative totals carry the whole run's
                # history -- 77k updates against 19.4M env steps is already a lifetime
                # ratio above target -- so pacing on them would simply stop training.
                replayed = ((self.updates - self._session_updates)
                            * cfg.wm.batch_size * cfg.wm.batch_length)
                collected = self.env_steps - self._session_env_steps
                if replayed > cfg.train.replay_ratio * collected:
                    time.sleep(0.002)
                    continue
                stats = self.train_step()
                self.metrics.update(stats)

                if time.time() - last_log >= 20.0:
                    self._log()
                    last_log = time.time()
                if self.env_steps - self._last_stall_check >= cfg.train.stall_check_every:
                    self._last_stall_check = self.env_steps
                    self._check_for_stall()
                if (self.env_steps - self._last_target_refresh
                        >= cfg.train.target_refresh_every):
                    self._last_target_refresh = self.env_steps
                    self._refresh_archive_targets()

                due_steps = self.env_steps - last_ckpt >= cfg.train.checkpoint_every
                due_time = time.time() - last_ckpt_t >= cfg.train.checkpoint_every_s
                if due_steps or due_time:
                    self.save_checkpoint()
                    last_ckpt = self.env_steps
                    last_ckpt_t = time.time()
        except KeyboardInterrupt:
            log.info("interrupted; checkpointing before exit")
        finally:
            self._stop.set()
            if self._collector is not None:
                self._collector.join(timeout=15)
            self.save_checkpoint()
            self.proposer.stop()
            self.envs.close()
            self._log()
            log.info(
                "finished: %d env steps, %d updates, milestone %d/%d, %d badges",
                self.env_steps, self.updates, self.best_milestone,
                NUM_MILESTONES - 1, self.best_badges,
            )

    def _log(self) -> None:
        m = self.metrics.mean()
        m |= self.archive.stats()
        m |= self.proposer.stats()
        elapsed = time.time() - self.start_time
        # Rates are measured over *this process's* contribution. Dividing the cumulative
        # counters by the elapsed time of a resumed run reports the whole history against
        # a few minutes of wall clock, which inflated throughput by ~20x after a restart.
        m |= {
            "run/env_steps": float(self.env_steps),
            "run/updates": float(self.updates),
            "run/episodes": float(self.episodes),
            "run/best_milestone": float(self.best_milestone),
            "run/best_badges": float(self.best_badges),
            "run/hours": elapsed / 3600.0,
            "run/steps_per_s": (self.env_steps - self._session_env_steps)
            / max(elapsed, 1e-9),
            "run/updates_per_s": (self.updates - self._session_updates)
            / max(elapsed, 1e-9),
            "run/replay_size": float(len(self.replay)),
        }
        with open(self.logdir / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps({k: round(v, 6) for k, v in sorted(m.items())}) + "\n")
        log.info(
            "step %8d | upd %6d | %5.0f sps %4.2f ups | milestone %2d/%d (%s) | "
            "badges %d | events %.0f | maps %.0f | coords %.0f | cells %d | "
            "wm %.1f actor %.3f H %.2f",
            self.env_steps, self.updates, m.get("run/steps_per_s", 0),
            m.get("run/updates_per_s", 0), self.best_milestone, NUM_MILESTONES - 1,
            (achieved_milestone(self.best_milestone).key
             if achieved_milestone(self.best_milestone) else "none"),
            self.best_badges, m.get("progress/events", 0), m.get("progress/unique_maps", 0),
            m.get("progress/unique_coords", 0), len(self.archive),
            m.get("loss/world_model", float("nan")), m.get("loss/actor", float("nan")),
            m.get("policy/entropy", float("nan")),
        )

    # ================================================================== checkpoints

    @property
    def ckpt_path(self) -> Path:
        return self.logdir / "checkpoint.pt"

    def save_checkpoint(self) -> None:
        tmp = self.ckpt_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "world_model": self.wm.state_dict(),
                "actor_critic": self.ac.state_dict(),
                "opt_wm": self.opt_wm.state_dict(),
                "opt_ac": self.opt_ac.state_dict(),
                "opt_alpha": self.opt_alpha.state_dict(),
                "ret_norm": self.ac.ret_norm.state_dict(),
                "stall": self.stall.state_dict(),
                "milestone_events": sorted(self._milestone_events),
                # Per-worker novelty memory. Without it every restart re-opens ground the
                # agent has already covered, so `new_tile` pays again for walking back
                # into the easy part of a map instead of bounties on the frontier.
                #
                # Read from a snapshot the *collector* publishes, never queried here: the
                # worker pipes are an unsynchronised request/response protocol and this
                # runs on the main thread, so querying them directly raced the collector
                # and handed it someone else's reply.
                "exploration": self._exploration_snapshot,
                "env_steps": self.env_steps,
                "updates": self.updates,
                "episodes": self.episodes,
                "best_milestone": self.best_milestone,
                "best_badges": self.best_badges,
                "completed": self.completed,
                "config": asdict(self.cfg),
                "chain": chain_fingerprint(),
            },
            tmp,
        )
        tmp.replace(self.ckpt_path)
        if self.cfg.archive.enabled:
            self.archive.save(self.logdir / "archive")
        log.info("checkpoint written at %d env steps", self.env_steps)

    def load_archive_only(self) -> int:
        """Restore the frontier archive without the network weights.

        The archive holds emulator save states, which are independent of the model
        architecture. That makes exploration progress transferable in ways the
        checkpoint is not: switching preset (cpu -> laptop), resizing the RSSM, or
        starting a fresh policy all keep whatever the previous run discovered about the
        game. Without this, the archive was only loadable alongside a matching
        checkpoint and every architecture change threw away the frontier.
        """
        if not self.cfg.archive.enabled:
            return 0
        loaded = self.archive.load(self.logdir / "archive")
        if loaded:
            self.best_milestone = max(self.best_milestone, self.archive.max_milestone)
        return loaded

    def load_checkpoint(self, reset_policy: bool = False) -> bool:
        """Restore a run. `reset_policy` keeps the world model but reinitialises the
        actor, critic and their optimiser state.

        Useful when the reward function has changed materially or the policy has
        collapsed: the world model is the expensive part and stays valid (it models
        dynamics, not rewards), while a stale critic trained against different reward
        weights is worse than no critic at all.
        """
        if not self.ckpt_path.exists():
            # Still adopt any archive lying in the logdir -- see load_archive_only.
            n = self.load_archive_only()
            if n:
                log.info(
                    "no checkpoint, but adopted %d archived cells (max milestone %d)",
                    n, self.archive.max_milestone,
                )
            return False
        ck = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        # Tolerant load. Adding a symbolic feature changes the encoder's input width, so
        # exactly one weight matrix stops matching -- refusing to resume 20M steps of a
        # world model over one widened Linear would be the wrong trade. Mismatched
        # tensors are dropped and reinitialised; the conv stack, RSSM and heads all carry
        # over, and the symbolic projection relearns in a few thousand updates.
        wm_sd = ck["world_model"]
        own = self.wm.state_dict()
        dropped = [k for k, v in wm_sd.items()
                   if k in own and own[k].shape != v.shape]
        for k in dropped:
            wm_sd.pop(k)
        missing, unexpected = self.wm.load_state_dict(wm_sd, strict=False)
        if dropped or missing or unexpected:
            log.warning(
                "world-model schema drift: reshaped=%s missing=%s unexpected=%s",
                dropped, list(missing), list(unexpected),
            )
        try:
            self.opt_wm.load_state_dict(ck["opt_wm"])
        except ValueError as exc:
            log.warning("opt_wm state discarded (%s); restarting its moments", exc)
        # Purge Adam moments whose shape no longer matches their parameter.
        #
        # `load_state_dict` does not validate shapes, so a stale exp_avg survives the
        # load and only explodes later inside `opt.step()`:
        #     RuntimeError: The size of tensor a (22) must match the size of tensor b (23)
        # That crashed the run in a supervisor restart loop, having done zero updates.
        # Clearing the affected entries costs only those tensors' moments, which re-warm
        # in a few hundred steps.
        _purge_stale_optimizer_state(self.opt_wm)
        if reset_policy:
            log.info("resetting actor-critic; keeping the world model")
        else:
            # Tolerant on both counts: checkpoints written before the entropy
            # coefficient became a dual variable have no `log_alpha` tensor, and their
            # `opt_ac` state was built over a parameter list of a different length.
            # Refusing to resume a 15 M-step run over a one-scalar schema change would
            # be the worst possible trade.
            missing, unexpected = self.ac.load_state_dict(
                ck["actor_critic"], strict=False
            )
            if missing or unexpected:
                log.warning(
                    "actor-critic checkpoint schema drift: missing=%s unexpected=%s",
                    list(missing), list(unexpected),
                )
            try:
                self.opt_ac.load_state_dict(ck["opt_ac"])
            except ValueError as exc:
                log.warning("opt_ac state discarded (%s); restarting its moments", exc)
            _purge_stale_optimizer_state(self.opt_ac)
            if "opt_alpha" in ck:
                self.opt_alpha.load_state_dict(ck["opt_alpha"])
            self.ac.ret_norm.load_state_dict(ck.get("ret_norm", {}))
        # Carry the stall history across restarts; see StallDetector.state_dict.
        self.stall.load_state_dict(ck.get("stall", []))
        self._milestone_events = set(ck.get("milestone_events", []))
        if ck.get("exploration"):
            self.envs.import_exploration(ck["exploration"])
            log.info(
                "restored novelty memory for %d workers", len(ck["exploration"])
            )
        self.env_steps = int(ck["env_steps"])
        self.updates = int(ck["updates"])
        self.episodes = int(ck.get("episodes", 0))
        self.best_milestone = int(ck.get("best_milestone", 0))
        if ck.get("chain") != chain_fingerprint():
            # The milestone chain changed since this checkpoint, so the stored counter
            # names a different milestone. Rediscover it from live workers, which
            # recompute their own index every step.
            log.warning(
                "milestone chain changed since checkpoint; resetting best_milestone "
                "from %d so archive targets are recomputed", self.best_milestone,
            )
            self.best_milestone = 0
            # Indices are only meaningful relative to a chain, so the "already
            # announced" record does not carry over either -- index 12 named the badge
            # before the gym rungs were added and names the gym door after.
            self._milestone_events.clear()
            self._chain_changed = True
        self.best_badges = int(ck.get("best_badges", 0))
        self.completed = bool(ck.get("completed", False))
        if self.cfg.archive.enabled:
            self.archive.load(self.logdir / "archive")
        log.info(
            "resumed from %s at %d env steps (milestone %d)",
            self.ckpt_path, self.env_steps, self.best_milestone,
        )
        return True


def _reset_row(state: RSSMState, i: int) -> RSSMState:
    deter = state.deter.clone()
    logits = state.logits.clone()
    stoch = state.stoch.clone()
    deter[i] = 0
    logits[i] = 0
    stoch[i] = 0
    return RSSMState(deter, logits, stoch)


def configure_logging(logdir: Path, level: int = logging.INFO) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    fh = logging.FileHandler(logdir / "train.log")
    fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
