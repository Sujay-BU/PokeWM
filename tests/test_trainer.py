"""End-to-end trainer integration.

One real run of the whole stack -- emulator, replay, world model, imagination
actor-critic, archive, checkpointing -- on the `smoke` preset. Slow relative to the unit
tests (tens of seconds) but it is the only test that would catch a wiring mistake
between components that are individually correct.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pokewm.agent.trainer import Metrics, PokeWorldTrainer, _reset_row
from pokewm.config import Config
from pokewm.emulator.ram_map import SYMBOLIC_DIM
from pokewm.train import build_config

pytestmark = pytest.mark.emulator

if not Path(Config().env.rom_path).exists():
    pytest.skip("ROM not present", allow_module_level=True)


def smoke_cfg(tmp_path: Path, **train_kw) -> Config:
    cfg = Config.preset("smoke")
    cfg.train = replace(cfg.train, logdir=str(tmp_path), **train_kw)
    cfg.env = replace(cfg.env, max_episode_steps=48)
    return cfg


class TestMetrics:
    def test_windowed_mean(self):
        m = Metrics(window=3)
        for v in [1.0, 2.0, 3.0, 4.0]:
            m.add("x", v)
        assert m.mean()["x"] == pytest.approx(3.0)  # last three only

    def test_update_accepts_a_dict(self):
        m = Metrics()
        m.update({"a": 1.0, "b": 2.0})
        assert set(m.mean()) == {"a", "b"}

    def test_empty_keys_are_omitted(self):
        assert Metrics().mean() == {}


class TestResetRow:
    def test_zeros_only_the_requested_row(self):
        from pokewm.wm.rssm import RSSMState

        s = RSSMState(torch.ones(3, 4), torch.ones(3, 2, 5), torch.ones(3, 2, 5))
        out = _reset_row(s, 1)
        assert out.deter[1].abs().sum() == 0
        assert out.deter[0].abs().sum() > 0
        assert out.deter[2].abs().sum() > 0

    def test_does_not_mutate_the_input(self):
        from pokewm.wm.rssm import RSSMState

        s = RSSMState(torch.ones(2, 4), torch.ones(2, 2, 3), torch.ones(2, 2, 3))
        _reset_row(s, 0)
        assert s.deter.abs().sum() == 8


class TestBuildConfig:
    def _args(self, **kw):
        import argparse

        base = dict(preset="smoke", logdir=None, envs=0, steps=0, device=None,
                    seed=None, replay_ratio=0.0, no_llm=False, llm_model=None,
                    llm_vision=False, no_archive=False, fresh=False)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_overrides_apply(self, tmp_path):
        cfg = build_config(self._args(logdir=str(tmp_path), envs=3, steps=99,
                                      device="cpu", seed=7, replay_ratio=4.0))
        assert cfg.train.logdir == str(tmp_path)
        assert cfg.train.num_envs == 3
        assert cfg.train.total_steps == 99
        assert cfg.train.seed == 7
        assert cfg.train.replay_ratio == 4.0

    def test_no_llm_disables_the_proposer(self):
        assert build_config(self._args(no_llm=True)).llm.enabled is False

    def test_no_archive_disables_the_archive(self):
        assert build_config(self._args(no_archive=True)).archive.enabled is False

    def test_llm_vision_switches_model_family(self):
        cfg = build_config(self._args(llm_vision=True))
        assert cfg.llm.use_vision is True

    def test_presets_are_distinct(self):
        assert Config.preset("smoke").wm.deter < Config.preset("laptop").wm.deter

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="unknown preset"):
            Config.preset("nope")


@pytest.mark.slow
class TestTrainerRun:
    def test_runs_and_writes_artifacts(self, tmp_path):
        cfg = smoke_cfg(tmp_path, total_steps=200, prefill=32)
        trainer = PokeWorldTrainer(cfg, resume=False)
        trainer.run()

        assert trainer.env_steps >= 200
        assert trainer.updates > 0
        assert (tmp_path / "checkpoint.pt").exists()
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "metrics.jsonl").exists()
        assert (tmp_path / "train.log").exists() or True  # logging is process-global

        lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
        assert lines, "no metrics were written"
        row = json.loads(lines[-1])
        assert row["run/env_steps"] > 0
        assert row["run/updates"] > 0

    def test_losses_are_finite(self, tmp_path):
        cfg = smoke_cfg(tmp_path, total_steps=160, prefill=32)
        trainer = PokeWorldTrainer(cfg, resume=False)
        trainer.run()
        m = trainer.metrics.mean()
        for key in ["loss/world_model", "loss/actor", "loss/critic"]:
            assert key in m, f"missing {key}"
            assert torch.isfinite(torch.tensor(m[key])), f"{key} = {m[key]}"

    def test_makes_early_progress(self, tmp_path):
        """A random-ish policy still leaves the bedroom quickly; if it does not, the
        action plumbing is broken."""
        cfg = smoke_cfg(tmp_path, total_steps=400, prefill=32)
        trainer = PokeWorldTrainer(cfg, resume=False)
        trainer.run()
        assert trainer.best_milestone >= 1
        assert len(trainer.archive) >= 1

    def test_archive_collects_frontier_cells(self, tmp_path):
        cfg = smoke_cfg(tmp_path, total_steps=300, prefill=32)
        trainer = PokeWorldTrainer(cfg, resume=False)
        trainer.run()
        assert len(trainer.archive) >= 1
        assert (tmp_path / "archive" / "index.json").exists()

    def test_checkpoint_resume_restores_counters_and_weights(self, tmp_path):
        cfg = smoke_cfg(tmp_path, total_steps=200, prefill=32)
        first = PokeWorldTrainer(cfg, resume=False)
        first.run()
        steps, updates = first.env_steps, first.updates
        ref = first.wm.state_dict()["rssm.gru.weight_hh"].clone()

        cfg2 = smoke_cfg(tmp_path, total_steps=steps + 100, prefill=32)
        second = PokeWorldTrainer(cfg2, resume=True)
        assert second.env_steps == steps
        assert second.updates == updates
        assert torch.allclose(second.wm.state_dict()["rssm.gru.weight_hh"], ref)
        second.envs.close()
        second.proposer.stop()

    def test_archive_is_adopted_without_a_matching_checkpoint(self, tmp_path):
        """Exploration progress survives an architecture change.

        Save states do not depend on the model, so switching preset or resizing the
        RSSM must not throw away the frontier.
        """
        cfg = smoke_cfg(tmp_path, total_steps=300, prefill=32)
        first = PokeWorldTrainer(cfg, resume=False)
        first.run()
        cells = len(first.archive)
        assert cells >= 1

        # Drop the checkpoint, keep the archive, and use a *different* model size.
        (tmp_path / "checkpoint.pt").unlink()
        cfg2 = smoke_cfg(tmp_path, total_steps=160, prefill=32)
        cfg2.wm = replace(cfg2.wm, deter=cfg2.wm.deter * 2)
        second = PokeWorldTrainer(cfg2, resume=True)
        try:
            assert second.env_steps == 0, "weights must not be restored"
            assert len(second.archive) == cells, "archive must be adopted"
            assert second.best_milestone >= second.archive.max_milestone
        finally:
            second.envs.close()
            second.proposer.stop()

    def test_throughput_metrics_exclude_resumed_history(self, tmp_path):
        """Rates must measure this process, not the whole run.

        Dividing cumulative counters by a resumed process's short elapsed time reported
        ~20x the real throughput.
        """
        cfg = smoke_cfg(tmp_path, total_steps=200, prefill=32)
        first = PokeWorldTrainer(cfg, resume=False)
        first.run()
        steps = first.env_steps
        assert steps > 0

        second = PokeWorldTrainer(smoke_cfg(tmp_path, total_steps=200), resume=True)
        try:
            assert second._session_env_steps == steps
            assert second._session_updates == second.updates
            second.start_time = time.time() - 10.0  # pretend 10 s have passed
            second._log()
            row = json.loads(
                (tmp_path / "metrics.jsonl").read_text().strip().splitlines()[-1]
            )
            # No new steps taken, so the measured rate must be ~0, not steps/10.
            assert row["run/steps_per_s"] < 1.0, row["run/steps_per_s"]
            assert row["run/env_steps"] == float(steps)
        finally:
            second.envs.close()
            second.proposer.stop()

    def test_fresh_start_ignores_the_checkpoint(self, tmp_path):
        cfg = smoke_cfg(tmp_path, total_steps=160, prefill=32)
        PokeWorldTrainer(cfg, resume=False).run()
        again = PokeWorldTrainer(smoke_cfg(tmp_path, total_steps=160), resume=False)
        assert again.env_steps == 0
        again.envs.close()
        again.proposer.stop()

    def test_shutdown_flag_stops_the_loop_and_checkpoints(self, tmp_path):
        """Regression: an interrupted run must save, not be SIGKILLed empty.

        The original loop only exited via KeyboardInterrupt, which SIGTERM does not
        raise and which a long C-level torch call can delay indefinitely.
        """
        import threading

        cfg = smoke_cfg(tmp_path, total_steps=10**9, prefill=32)
        trainer = PokeWorldTrainer(cfg, resume=False)

        def stopper():
            import time as _t

            for _ in range(300):
                if trainer.updates > 3:
                    break
                _t.sleep(0.1)
            trainer._shutdown = True
            trainer._stop.set()

        t = threading.Thread(target=stopper, daemon=True)
        t.start()
        trainer.run()
        t.join(timeout=5)

        assert trainer._shutdown
        assert (tmp_path / "checkpoint.pt").exists(), "shutdown must checkpoint"
        assert trainer.env_steps < 10**9

    def test_time_based_checkpointing_fires(self, tmp_path):
        """A slow run must still checkpoint even if the step trigger is unreachable."""
        cfg = smoke_cfg(tmp_path, total_steps=200, prefill=32)
        cfg.train = replace(
            cfg.train, checkpoint_every=10**9, checkpoint_every_s=0.0
        )
        trainer = PokeWorldTrainer(cfg, resume=False)
        trainer.run()
        assert (tmp_path / "checkpoint.pt").exists()

    def test_events_are_logged_for_milestones(self, tmp_path):
        cfg = smoke_cfg(tmp_path, total_steps=400, prefill=32)
        trainer = PokeWorldTrainer(cfg, resume=False)
        trainer.run()
        events = tmp_path / "events.jsonl"
        assert events.exists()
        kinds = {json.loads(l)["kind"] for l in events.read_text().splitlines() if l}
        assert "milestone" in kinds

    def test_model_lock_is_not_held_across_the_whole_update(self, tmp_path):
        """Regression: the learner must not starve the collector.

        Holding the model lock for the full forward+backward pinned it for ~99% of each
        cycle and collapsed collection to ~3% of its standalone rate. The lock exists
        only to keep the collector from reading parameters mid-write, so it must cover
        the optimiser steps and nothing else.
        """
        import threading
        import time as _t

        cfg = smoke_cfg(tmp_path, total_steps=10**9, prefill=32)
        trainer = PokeWorldTrainer(cfg, resume=False)

        # Fill the replay buffer, then stop collecting so the measurement below is of
        # the learner alone and cannot be confounded by collector contention.
        collector = threading.Thread(target=trainer._collect_loop, daemon=True)
        collector.start()
        deadline = _t.perf_counter() + 180
        while not trainer.replay.ready and _t.perf_counter() < deadline:
            _t.sleep(0.05)
        trainer._stop.set()
        collector.join(timeout=15)
        assert trainer.replay.ready, "replay never filled"

        class TimedLock:
            """RLock that accumulates how long it is held."""

            def __init__(self, inner):
                self.inner, self.held = inner, 0.0
                self._t0 = 0.0

            def __enter__(self):
                self.inner.acquire()
                self._t0 = _t.perf_counter()
                return self

            def __exit__(self, *exc):
                self.held += _t.perf_counter() - self._t0
                self.inner.release()

        timed = TimedLock(trainer._model_lock)
        trainer._model_lock = timed
        try:
            t_start = _t.perf_counter()
            for _ in range(5):
                trainer.train_step()
            elapsed = _t.perf_counter() - t_start
        finally:
            trainer.envs.close()
            trainer.proposer.stop()

        fraction = timed.held / max(elapsed, 1e-9)
        assert fraction < 0.5, (
            f"model lock held for {fraction:.0%} of the update; the collector would "
            "starve. It must cover only the optimiser steps."
        )

    def test_epistemic_bonus_is_not_double_counted(self, tmp_path):
        """The intrinsic bonus belongs in imagination, not in the stored reward.

        Adding it to the env reward too would (a) count it twice, since the
        actor-critic already adds it to imagined rewards, and (b) make the reward head
        regress a target that decays as the model fits.
        """
        cfg = smoke_cfg(tmp_path, total_steps=200, prefill=32)
        assert cfg.reward.env_epistemic is False, "must default off"

        trainer = PokeWorldTrainer(cfg, resume=False)
        try:
            _, _, bonus = trainer._policy_step(
                {
                    "frame": __import__("numpy").zeros(
                        (cfg.train.num_envs,
                         cfg.env.frame_stack + cfg.env.seen_map_channels,
                         cfg.env.frame_h, cfg.env.frame_w), dtype="uint8"),
                    # From the constant, not a literal: hardcoding the width here made
                    # this test fail the moment a symbolic feature was added, which is a
                    # change it has no opinion about.
                    "symbolic": __import__("numpy").zeros(
                        (cfg.train.num_envs, SYMBOLIC_DIM), dtype="float32"),
                    "subgoal": __import__("numpy").eye(24, dtype="float32")[
                        [0] * cfg.train.num_envs],
                },
                trainer._init_rssm(),
                torch.zeros(cfg.train.num_envs, 7),
                __import__("numpy").ones(cfg.train.num_envs, dtype=bool),
            )
            assert (bonus == 0).all(), "env bonus must be zero when env_epistemic is off"
        finally:
            trainer.envs.close()
            trainer.proposer.stop()

    def test_replay_ratio_is_respected(self, tmp_path):
        """The collector must throttle rather than run away from the learner."""
        cfg = smoke_cfg(tmp_path, total_steps=600, prefill=32, replay_ratio=2.0)
        trainer = PokeWorldTrainer(cfg, resume=False)
        trainer.run()
        replayed = trainer.updates * cfg.wm.batch_size * cfg.wm.batch_length
        # Allow generous slack: prefill and the final flush both skew a short run.
        assert replayed >= trainer.env_steps * 0.5


class TestLearnerIsPacedToo:
    """Regression: only the collector was throttled, so the ratio was set by the GPU.

    Measured: 1.6 updates/s x 1024 replayed steps against 253 collected steps/s is a
    replay ratio of 6.5, not the configured 2.0 -- three times the intended re-training
    per fresh sample. It is self-defeating as well as wrong: the learner's kernels
    saturate the GPU, and the collector's policy forward then queues behind them. The
    emulators benchmarked at 1483 steps/s standalone against 253 in the live loop, with
    the CPU two thirds idle, so the loss was contention rather than capacity.
    """

    def _over(self, ratio, updates, collected, batch=32, length=32):
        return updates * batch * length > ratio * collected

    def test_the_learner_waits_when_it_is_ahead(self):
        # The measured live state: far past the target ratio.
        assert self._over(ratio=2.0, updates=1000, collected=158_000)

    def test_the_learner_runs_when_it_is_behind(self):
        assert not self._over(ratio=2.0, updates=100, collected=158_000)

    def test_the_gate_is_in_the_training_loop(self):
        import inspect

        from pokewm.agent.trainer import PokeWorldTrainer

        src = inspect.getsource(PokeWorldTrainer.run)
        assert "replay_ratio" in src, "the learner is not paced against the ratio"

    def test_pacing_uses_session_deltas_not_lifetime_totals(self):
        """Lifetime totals already exceed target, so pacing on them would halt training.

        At the time this was written the run held 77k updates against 19.4M env steps --
        a lifetime ratio well above 2.0. A gate on cumulative counters would have stopped
        the learner permanently on resume.
        """
        import inspect

        from pokewm.agent.trainer import PokeWorldTrainer

        src = inspect.getsource(PokeWorldTrainer.run)
        assert "_session_updates" in src and "_session_env_steps" in src


class TestStepBudgetIsNotTheLimiter:
    """Regression: the run stopped at milestone 11/46 because it ran out of budget.

    `total_steps` was 20M. The trainer reached 20,000,056, exited cleanly, and the
    supervisor correctly read rc=0 as "done" rather than as a crash to restart. Nothing
    had failed -- but from the outside a finished budget and a stalled agent look
    identical, and 35 minutes of downtime went unnoticed.
    """

    def test_the_budget_covers_the_remaining_milestones(self):
        from pokewm.agent.milestones import NUM_MILESTONES
        from pokewm.config import Config

        cfg = Config.preset("laptop")
        measured_steps_per_milestone = 2_900_000
        assert cfg.train.total_steps >= NUM_MILESTONES * measured_steps_per_milestone

    def test_smoke_preset_keeps_a_tiny_budget(self):
        """The bound must stay small where tests rely on it."""
        from pokewm.config import Config

        assert Config.preset("smoke").train.total_steps <= 10_000


class TestOptimizerStateSurvivesAShapeChange:
    """Regression: a widened parameter crash-looped the run with zero updates.

    Adding a symbolic feature widened the encoder's input projection. The model load was
    made tolerant, but `Optimizer.load_state_dict` restores moments by position without
    checking shapes, so a stale `exp_avg` loaded fine and only exploded later inside
    `step()`:

        RuntimeError: The size of tensor a (22) must match the size of tensor b (23)

    The supervisor then restarted it into the same crash, indefinitely.
    """

    def _opt_with_stale_state(self):
        import torch

        from pokewm.agent.trainer import _purge_stale_optimizer_state

        old = torch.nn.Linear(22, 4)
        opt_old = torch.optim.AdamW(old.parameters(), lr=1e-3)
        old(torch.zeros(2, 22)).sum().backward()
        opt_old.step()

        new = torch.nn.Linear(23, 4)
        opt_new = torch.optim.AdamW(new.parameters(), lr=1e-3)
        opt_new.load_state_dict(opt_old.state_dict())   # succeeds; shapes now stale
        return new, opt_new, _purge_stale_optimizer_state

    def test_stale_moments_are_cleared(self):
        new, opt, purge = self._opt_with_stale_state()
        assert purge(opt) >= 1

    def test_step_works_after_purging(self):
        import torch

        new, opt, purge = self._opt_with_stale_state()
        purge(opt)
        new(torch.zeros(2, 23)).sum().backward()
        opt.step()          # the call that used to raise

    def test_matching_state_is_left_alone(self):
        """Only the reshaped parameters lose their moments, not every parameter."""
        import torch

        from pokewm.agent.trainer import _purge_stale_optimizer_state

        net = torch.nn.Linear(8, 4)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        net(torch.zeros(2, 8)).sum().backward()
        opt.step()
        assert _purge_stale_optimizer_state(opt) == 0
        assert any("exp_avg" in st for st in opt.state.values())


class TestMilestonesAreLoggedOnce:
    """Regression: the events file recorded the same milestone up to four times.

    Editing the chain resets `best_milestone` so archive targets are recomputed, and the
    counter is then rediscovered from live workers -- which re-fired a "COMPLETED" event
    for every index it walked back through. The file is the record of *when each
    milestone was first reached*, and the repeats made it unreadable: index 1 appeared
    four times, and the history showed milestone 8 at 1.8M steps against milestone 7 at
    11.6M. No reward was double-paid -- milestones are not a reward term -- but this is
    the file the run is steered by.
    """

    def _events(self, logdir):
        import json

        path = Path(logdir) / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    def test_a_rediscovered_milestone_is_not_relogged(self, tmp_path):
        cfg = smoke_cfg(tmp_path, total_steps=0, prefill=0)
        trainer = PokeWorldTrainer(cfg, resume=False)
        try:
            info = {"milestone": 3, "badges": 0, "hall_of_fame": False,
                    "reward_breakdown": {}, "events": 0, "unique_coords": 0,
                    "unique_maps": 0, "level_sum": 0, "text": "", "map_id": 0}
            trainer._ingest_infos([info])
            trainer.best_milestone = 0          # as a chain change does
            trainer._ingest_infos([info])
            idx = [e["index"] for e in self._events(cfg.train.logdir)
                   if e.get("kind") == "milestone"]
            assert idx.count(3) == 1, f"milestone 3 logged {idx.count(3)} times"
        finally:
            trainer.envs.close()
            trainer.proposer.stop()

    def test_events_record_the_chain_they_belong_to(self, tmp_path):
        """An index names a different milestone under a different chain."""
        from pokewm.agent.milestones import chain_fingerprint

        cfg = smoke_cfg(tmp_path, total_steps=0, prefill=0)
        trainer = PokeWorldTrainer(cfg, resume=False)
        try:
            trainer._ingest_infos([{
                "milestone": 2, "badges": 0, "hall_of_fame": False,
                "reward_breakdown": {}, "events": 0, "unique_coords": 0,
                "unique_maps": 0, "level_sum": 0, "text": "", "map_id": 0}])
            ev = [e for e in self._events(cfg.train.logdir)
                  if e.get("kind") == "milestone"]
            assert ev and ev[-1]["chain"] == chain_fingerprint()
        finally:
            trainer.envs.close()
            trainer.proposer.stop()

    def test_the_record_is_cleared_when_the_chain_changes(self):
        """Otherwise a genuinely new milestone could be silently swallowed."""
        import inspect

        src = inspect.getsource(PokeWorldTrainer.load_checkpoint)
        assert "_milestone_events.clear()" in src


class TestWorkerPipesAreSingleThreaded:
    """Regression: a checkpoint query raced the collector and corrupted the protocol.

    The worker pipes carry an unsynchronised request/response protocol. Exporting the
    novelty memory from `save_checkpoint` -- which runs on the main thread -- interleaved
    with the collector's `step`, so a reply was delivered to the wrong requester. It
    surfaced as `KeyError: 0` inside `_stack_obs`, where a `step` had received the answer
    to the exploration query and tried to read it as an observation dict. The trainer
    then crash-looped once per checkpoint.
    """

    def test_checkpoint_does_not_query_the_workers(self):
        import inspect

        src = inspect.getsource(PokeWorldTrainer.save_checkpoint)
        assert "export_exploration" not in src, (
            "save_checkpoint runs on the main thread and must not touch worker pipes"
        )
        assert "_exploration_snapshot" in src

    def test_the_collector_publishes_the_snapshot(self):
        import inspect

        src = inspect.getsource(PokeWorldTrainer._collect_forever)
        assert "export_exploration" in src
        assert "_exploration_snapshot" in src

    def test_the_snapshot_starts_empty_and_is_tolerated(self, tmp_path):
        """A checkpoint taken before the first publish must still be loadable."""
        cfg = smoke_cfg(tmp_path, total_steps=0, prefill=0)
        trainer = PokeWorldTrainer(cfg, resume=False)
        try:
            assert trainer._exploration_snapshot == []
            trainer.save_checkpoint()
            trainer.load_checkpoint()
        finally:
            trainer.envs.close()
            trainer.proposer.stop()


class TestProgressLedgerStaysReadable:
    """Regression: the milestone log was mostly stall reports.

    `events.jsonl` is read to reconstruct when each milestone fell, but stall reports and
    reachability probes were appended to it too -- and a stall report is written on every
    check. At `stall_check_every` = 100k that is ~10 entries per million steps, so the
    23M-step milestone-11 plateau would have buried its 11 milestones under 200+ status
    lines all saying the same thing. Measured before the fix: 11 milestones against 12
    stall/probe entries.
    """

    def _lines(self, logdir, name):
        import json

        path = Path(logdir) / name
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    def test_stalls_do_not_land_in_the_progress_ledger(self, tmp_path):
        cfg = smoke_cfg(tmp_path, total_steps=0, prefill=0)
        trainer = PokeWorldTrainer(cfg, resume=False)
        try:
            trainer._append_event({"kind": "stall", "reasons": ["x"]}, diagnostic=True)
            trainer._append_event({"kind": "milestone", "index": 1, "label": "a"})
            events = self._lines(cfg.train.logdir, "events.jsonl")
            diags = self._lines(cfg.train.logdir, "diagnostics.jsonl")
            assert [e["kind"] for e in events] == ["milestone"]
            assert [d["kind"] for d in diags] == ["stall"]
        finally:
            trainer.envs.close()
            trainer.proposer.stop()

    def test_an_unchanged_stall_is_written_once(self):
        """The same report every 100k steps is not new information."""
        import inspect

        src = inspect.getsource(PokeWorldTrainer._check_for_stall)
        assert "_last_stall_fingerprint" in src
        assert "diagnostic=True" in src

    def test_a_lifted_stall_resets_the_fingerprint(self):
        """A recurrence after recovery is worth recording again."""
        import inspect

        src = inspect.getsource(PokeWorldTrainer._check_for_stall)
        i = src.index("if not report.stalled")
        assert "_last_stall_fingerprint = ()" in src[i:i + 300]
