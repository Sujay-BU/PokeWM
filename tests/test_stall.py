"""Stall detection.

Each case here is a stall that actually happened in this project. The detector exists so
they are caught in minutes rather than after millions of steps, and so the *hint* points
at the real cause -- in several cases the intuitive diagnosis was wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokewm.agent.stall import ProbeResult, StallDetector, StallReport


def feed(det: StallDetector, n: int, step0=0, dstep=100_000, **series):
    """Push n samples; each kwarg is either a constant or a per-sample callable."""
    for i in range(n):
        sig = {}
        for k, v in series.items():
            sig[k] = float(v(i)) if callable(v) else float(v)
        det.update(step0 + i * dstep, sig)


class TestNoStall:
    def test_warming_up_is_not_a_stall(self):
        det = StallDetector(window_steps=500_000, min_samples=8)
        feed(det, 3, milestone=1, archive_cells=10, unique_coords=5, max_events=1)
        r = det.check()
        assert not r.stalled
        assert "warming up" in r.reasons[0]

    def test_growing_archive_counts_as_progress(self):
        det = StallDetector(window_steps=500_000, min_samples=4)
        feed(det, 10, milestone=5, archive_cells=lambda i: 10 + i,
             unique_coords=100, max_events=3, entropy=1.0, imag_reward=0.01)
        assert not det.check().stalled

    def test_advancing_milestone_counts_as_progress(self):
        det = StallDetector(window_steps=500_000, min_samples=4)
        feed(det, 10, milestone=lambda i: 5 + i // 5, archive_cells=50,
             unique_coords=100, max_events=3, entropy=1.0, imag_reward=0.01)
        assert not det.check().stalled

    def test_new_story_flags_count_as_progress(self):
        det = StallDetector(window_steps=500_000, min_samples=4)
        feed(det, 10, milestone=5, archive_cells=50, unique_coords=100,
             max_events=lambda i: 3 + i * 0.5, entropy=1.0, imag_reward=0.01)
        assert not det.check().stalled

    def test_throughput_alone_is_not_progress(self):
        """A run can hold 350 steps/s with a falling loss and make no game progress.

        That is exactly what several stalls here looked like, so the detector must
        ignore anything that is not a progress signal.
        """
        det = StallDetector(window_steps=500_000, min_samples=4)
        feed(det, 10, milestone=5, archive_cells=50, unique_coords=100, max_events=3,
             entropy=1.0, imag_reward=0.01)
        assert det.check().stalled


class TestHardProgress:
    """Regression: a 2.0M-step milestone plateau read as "healthy: progressing".

    `unique_coords` climbs indefinitely -- interiors, and ground re-covered after a
    worker restart resets its novelty memory -- and the archive keeps minting cells.
    Under an `any(...)` rule over all progress signals, those two alone were enough to
    mask the fact that no milestone or story flag had fired in millions of steps.
    """

    def _det(self, **series):
        det = StallDetector(window_steps=500_000, min_samples=4,
                            hard_window_steps=3_000_000)
        base = dict(milestone=9, archive_cells=264, unique_coords=380, max_events=10,
                    entropy=0.45, imag_reward=0.03, frontier_frac=0.9)
        base.update(series)
        feed(det, 40, dstep=100_000, **base)
        return det

    def test_growing_coords_do_not_mask_a_milestone_plateau(self):
        det = self._det(unique_coords=lambda i: 100 + 8 * i,
                        archive_cells=lambda i: 250 + i)
        r = det.check()
        assert r.stalled
        assert any("no milestone or story flag" in x for x in r.reasons)

    def test_a_milestone_inside_the_long_window_clears_it(self):
        det = self._det(unique_coords=lambda i: 100 + 8 * i,
                        milestone=lambda i: 9 + (i >= 20))
        assert not det.check().stalled

    def test_a_story_flag_inside_the_long_window_clears_it(self):
        det = self._det(unique_coords=lambda i: 100 + 8 * i,
                        max_events=lambda i: 10 + (i >= 30))
        assert not det.check().stalled

    def test_a_short_history_does_not_trip_the_long_window(self):
        """The rule must not fire before it has seen its own window's worth of data."""
        det = StallDetector(window_steps=500_000, min_samples=4,
                            hard_window_steps=3_000_000)
        feed(det, 10, dstep=100_000, milestone=9, archive_cells=264,
             unique_coords=lambda i: 100 + 8 * i, max_events=10,
             entropy=0.45, imag_reward=0.03)
        assert not det.check().stalled


class TestStallSignatures:
    def _stalled(self, **extra):
        det = StallDetector(window_steps=500_000, min_samples=4)
        base = dict(milestone=5, archive_cells=50, unique_coords=100, max_events=3,
                    entropy=1.0, imag_reward=0.01, frontier_frac=0.9)
        base.update(extra)
        feed(det, 10, **base)
        return det.check()

    def test_flat_everything_is_a_stall(self):
        r = self._stalled()
        assert r.stalled
        assert "no progress" in r.reasons[0]

    def test_entropy_at_ceiling_is_flagged(self):
        """400k steps sat at exactly ln(7) because the entropy bonus swamped the
        policy gradient."""
        r = self._stalled(entropy=float(np.log(7)))
        assert any("ceiling" in h for h in r.hints)

    def test_collapsed_entropy_is_flagged(self):
        """Entropy 0.24 stopped the policy ever trying the A press an NPC needs."""
        r = self._stalled(entropy=0.2)
        assert any("collapsed" in h for h in r.hints)

    def test_negative_imagined_reward_is_flagged(self):
        """Negative value everywhere made deliberate blackouts optimal."""
        r = self._stalled(imag_reward=-0.013)
        assert any("negative" in h for h in r.hints)

    def test_starved_frontier_is_flagged(self):
        r = self._stalled(frontier_frac=0.03)
        assert any("frontier" in h for h in r.hints)

    def test_frozen_archive_is_flagged(self):
        r = self._stalled()
        assert any("archive stopped growing" in h for h in r.hints)

    def test_no_story_flags_points_at_measurement(self):
        """The parcel bug: the agent was succeeding and the RAM read was wrong."""
        r = self._stalled()
        assert any("measurement" in h for h in r.hints)

    def test_hints_fire_even_before_a_full_stall(self):
        """Early warning: a collapsing policy is worth reporting while coords still move."""
        det = StallDetector(window_steps=500_000, min_samples=4)
        feed(det, 10, milestone=5, archive_cells=lambda i: 50 + i,
             unique_coords=100, max_events=3, entropy=0.1, imag_reward=0.01)
        r = det.check()
        assert not r.stalled
        assert any("collapsed" in h for h in r.hints)


class TestReport:
    def test_describe_is_readable(self):
        r = StallReport(True, ["no progress in 750,000 env steps"], ["entropy collapsed"])
        text = r.describe()
        assert "STALLED" in text and "entropy collapsed" in text

    def test_healthy_describe(self):
        assert "healthy" in StallReport(False).describe()


class TestProbeResult:
    def test_reachable_blames_the_policy(self):
        text = ProbeResult("got_parcel", True, 6, 900, "fired after 25 steps").describe()
        assert "IS reachable" in text
        assert "policy is the bottleneck" in text

    def test_unreachable_blames_the_environment(self):
        text = ProbeResult("got_parcel", False, 6, 24000, "targets=42").describe()
        assert "NOT reached" in text
        # The whole point: send the reader upstream, not to the RL hyperparameters.
        assert "before touching the policy" in text


@pytest.mark.emulator
class TestProbeIntegration:
    def test_probe_runs_against_a_real_archive(self, tmp_path):
        from pathlib import Path

        from pokewm.agent.milestones import MILESTONES
        from pokewm.agent.stall import ReachabilityProbe
        from pokewm.config import Config
        from pokewm.emulator.archive import FrontierArchive
        from pokewm.emulator.bootstrap import make_init_state

        cfg = Config.preset("laptop")
        if not Path(cfg.env.rom_path).exists():
            pytest.skip("ROM not present")
        from dataclasses import replace

        state = make_init_state(cfg.env.rom_path, tmp_path / "init.state")
        cfg.env = replace(cfg.env, init_state=str(state))

        arch = FrontierArchive(cfg.archive, seed=0)
        arch.insert(key="k:1:38:0:0", blob=state.read_bytes(), milestone=1,
                    map_id=38, badges=0, events=0, seen_maps=frozenset({38}))

        # `leave_room` is trivially reachable from the bedroom by walking.
        target = MILESTONES[1]
        probe = ReachabilityProbe(cfg, num_cells=1, steps_per_cell=1500)
        result = probe.run(arch, target)
        assert result.milestone_key == "leave_room"
        assert result.reachable, result.describe()

    def test_probe_reports_unreachable_rather_than_hanging(self, tmp_path):
        from pathlib import Path

        from pokewm.agent.milestones import MILESTONES, MILESTONE_INDEX
        from pokewm.agent.stall import ReachabilityProbe
        from pokewm.config import Config
        from pokewm.emulator.archive import FrontierArchive
        from pokewm.emulator.bootstrap import make_init_state

        cfg = Config.preset("laptop")
        if not Path(cfg.env.rom_path).exists():
            pytest.skip("ROM not present")
        from dataclasses import replace

        state = make_init_state(cfg.env.rom_path, tmp_path / "init.state")
        cfg.env = replace(cfg.env, init_state=str(state))
        arch = FrontierArchive(cfg.archive, seed=0)
        arch.insert(key="k:1:38:0:0", blob=state.read_bytes(), milestone=1,
                    map_id=38, badges=0, events=0, seen_maps=frozenset({38}))

        # The Boulder Badge is obviously not reachable from the starting bedroom.
        target = MILESTONES[MILESTONE_INDEX["badge_1"]]
        result = ReachabilityProbe(cfg, num_cells=1, steps_per_cell=300).run(arch, target)
        assert not result.reachable
        # It walked around the bedroom, so the honest verdict is inconclusive rather
        # than an accusation against the environment -- see TestProbeVerdict.
        assert "not reached" in result.describe()


class TestProbeVerdict:
    """Regression: a negative probe blamed the environment for a distance problem.

    A uniform random walk covers ~sqrt(n) distance, so a corridor tens of tiles long
    defeats it even when wide open. Route 2's accessible southern section measured 10
    tiles wide by 24 tall -- six cells at position_bucket=8, of which the archive held
    five, so its grid was *saturated*, not stalled, and the way north was clear. The
    probe nonetheless returned "check the RAM/predicate before touching the policy",
    which is an expensive place to send someone for what was a range limit.
    """

    def _result(self, **kw):
        base = dict(milestone_key="viridian_forest", reachable=False, cells_tried=4,
                    steps_used=12_000, detail="targets=51")
        base.update(kw)
        return ProbeResult(**base)

    def test_new_ground_makes_the_verdict_inconclusive(self):
        text = self._result(new_ground=True).describe()
        assert "INCONCLUSIVE" in text
        assert "not as a gate" in text
        assert "before touching the policy" not in text

    def test_no_new_ground_still_implicates_the_environment(self):
        text = self._result(new_ground=False).describe()
        assert "before touching the policy" in text
        assert "INCONCLUSIVE" not in text

    def test_a_reachable_probe_blames_the_policy(self):
        text = self._result(reachable=True, new_ground=True).describe()
        assert "the learned policy is the bottleneck" in text


class TestHistorySurvivesARestart:
    """Regression: the detector was blind to the stalls it exists to catch.

    Its hard-progress window is 3M env steps, but the history lived only in memory and
    reset to empty on every restart. Across a night of frequent restarts the window never
    once filled, so milestone 11 was held for 3.1M steps while the detector reported
    "healthy: progressing" -- from its point of view the run was always a few minutes old.
    """

    def _fill(self, det, n=40, step0=0):
        feed(det, n, step0=step0, dstep=100_000, milestone=11, archive_cells=500,
             unique_coords=lambda i: 100 + 8 * i, max_events=14,
             entropy=0.8, imag_reward=0.03)

    def test_a_restart_no_longer_forgets(self):
        old = StallDetector(window_steps=500_000, min_samples=4,
                            hard_window_steps=3_000_000)
        self._fill(old)
        assert old.check().stalled, "precondition: this history is a stall"

        fresh = StallDetector(window_steps=500_000, min_samples=4,
                              hard_window_steps=3_000_000)
        assert not fresh.check().stalled, "an empty detector cannot know"
        fresh.load_state_dict(old.state_dict())
        assert fresh.check().stalled, "the stall must survive the restart"

    def test_round_trips_through_json(self):
        """It travels in the checkpoint, so it has to survive serialisation."""
        import json

        det = StallDetector(window_steps=500_000, min_samples=4)
        self._fill(det, n=10)
        clone = StallDetector(window_steps=500_000, min_samples=4)
        clone.load_state_dict(json.loads(json.dumps(det.state_dict())))
        assert clone.check().signals == det.check().signals

    def test_a_corrupt_entry_is_skipped_not_fatal(self):
        det = StallDetector()
        det.load_state_dict([[1, {"milestone": 1.0}], "junk", None])
        assert len(det._hist) == 1


class TestCounterResetsAreNotStalls:
    """Regression: persisting history made the detector blind to its own seams.

    Worker-lifetime counters (`unique_coords`, `max_events`) restart at zero, so a window
    straddling a restart computed deltas like `dunique_coords=-2264` and reported
    STALLED on a run that was merely young again. It fired three times in a row on a
    healthy trainer. A monotone counter going backwards proves a seam; samples before it
    are not comparable to those after.
    """

    def _det(self):
        return StallDetector(window_steps=500_000, min_samples=4,
                             hard_window_steps=3_000_000)

    def test_a_reset_is_not_reported_as_a_stall(self):
        det = self._det()
        # a healthy stretch, then a restart, then a healthy stretch
        feed(det, 12, step0=0, dstep=100_000, milestone=11, archive_cells=700,
             unique_coords=lambda i: 2000 + 30 * i, max_events=14,
             entropy=0.9, imag_reward=0.03)
        feed(det, 12, step0=1_300_000, dstep=100_000, milestone=11, archive_cells=700,
             unique_coords=lambda i: 100 + 30 * i, max_events=9,
             entropy=0.9, imag_reward=0.03)
        r = det.check()
        assert not r.stalled, r.reasons

    def test_a_genuine_stall_after_a_reset_still_fires(self):
        """Trimming must not make the detector permanently deaf."""
        det = self._det()
        feed(det, 8, step0=0, dstep=100_000, milestone=11, archive_cells=700,
             unique_coords=lambda i: 2000 + 30 * i, max_events=14,
             entropy=0.9, imag_reward=0.03)
        feed(det, 12, step0=1_300_000, dstep=100_000, milestone=11, archive_cells=700,
             unique_coords=100, max_events=9, entropy=0.9, imag_reward=0.03)
        assert det.check().stalled

    def test_an_uninterrupted_run_is_unaffected(self):
        det = self._det()
        feed(det, 12, dstep=100_000, milestone=11, archive_cells=700,
             unique_coords=lambda i: 2000 + 30 * i, max_events=14,
             entropy=0.9, imag_reward=0.03)
        assert not det.check().stalled

    def test_the_seam_is_found_at_the_right_place(self):
        det = self._det()
        feed(det, 5, step0=0, dstep=100_000, milestone=11, archive_cells=700,
             unique_coords=lambda i: 2000 + i, max_events=14)
        feed(det, 5, step0=600_000, dstep=100_000, milestone=11, archive_cells=700,
             unique_coords=lambda i: 10 + i, max_events=2)
        assert det._discontinuity() == 5
