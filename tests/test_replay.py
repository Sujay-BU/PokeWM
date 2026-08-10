"""Sequence replay.

The subtle part is frame-stack reconstruction. Frames are stored unstacked to save
memory, so `sample` rebuilds the stack by looking backwards -- and must not look back
across an episode boundary, or the model is trained on a transition that never happened.
"""

from __future__ import annotations

import numpy as np
import torch

from pokewm.config import ReplayConfig
from pokewm.wm.replay import SequenceReplay

H, W = 8, 8
SD, NA, NSG = 6, 7, 24
STACK = 4
STREAMS = 2


def make_replay(capacity=4096, prioritized=0.5, min_size=8) -> SequenceReplay:
    cfg = ReplayConfig(
        capacity=capacity, min_size=min_size, prioritized_fraction=prioritized
    )
    return SequenceReplay(
        cfg, num_streams=STREAMS, frame_hw=(H, W), symbolic_dim=SD,
        num_actions=NA, num_subgoals=NSG, extra_planes=1, frame_stack=STACK, seed=0,
    )


def push(replay: SequenceReplay, n: int, first_at: set[int] | None = None,
         frame_value=None) -> None:
    first_at = first_at or {0}
    for t in range(n):
        frame = np.zeros((STREAMS, STACK + 1, H, W), dtype=np.uint8)
        val = t if frame_value is None else frame_value(t)
        frame[:, STACK - 1] = val % 256  # newest plane tagged with the timestep
        frame[:, STACK] = 7  # the visited-overlay plane
        replay.add(
            frame=frame,
            symbolic=np.full((STREAMS, SD), t, dtype=np.float32),
            subgoal=np.eye(NSG, dtype=np.float32)[np.full(STREAMS, t % NSG)],
            action=np.full(STREAMS, t % NA),
            reward=np.full(STREAMS, float(t)),
            cont=np.ones(STREAMS, dtype=bool),
            is_first=np.array([t in first_at] * STREAMS),
        )


class TestBasics:
    def test_starts_empty_and_not_ready(self):
        r = make_replay()
        assert len(r) == 0 and not r.ready

    def test_length_counts_every_stream(self):
        r = make_replay()
        push(r, 10)
        assert len(r) == 10 * STREAMS

    def test_becomes_ready_at_min_size(self):
        r = make_replay(min_size=20)
        push(r, 9)
        assert not r.ready
        push(r, 1)
        assert r.ready

    def test_sample_returns_none_when_too_short(self):
        r = make_replay()
        push(r, 3)
        assert r.sample(2, length=16) is None

    def test_capacity_is_split_across_streams(self):
        r = make_replay(capacity=8192)
        assert r.per_stream >= 1024


class TestSampling:
    def test_shapes_and_dtypes(self):
        r = make_replay()
        push(r, 64)
        batch, streams, positions = r.sample(5, length=12)
        assert batch["frame"].shape == (5, 12, STACK + 1, H, W)
        assert batch["symbolic"].shape == (5, 12, SD)
        assert batch["subgoal"].shape == (5, 12, NSG)
        assert batch["action"].shape == (5, 12, NA)
        assert batch["reward"].shape == (5, 12)
        assert batch["cont"].shape == (5, 12)
        assert batch["is_first"].shape == (5, 12)
        assert streams.shape == (5,) and positions.shape == (5,)
        assert batch["frame"].dtype == torch.uint8

    def test_actions_are_one_hot(self):
        r = make_replay()
        push(r, 48)
        batch, _, _ = r.sample(4, length=8)
        assert torch.allclose(batch["action"].sum(-1), torch.ones(4, 8))

    def test_subgoals_are_one_hot(self):
        r = make_replay()
        push(r, 48)
        batch, _, _ = r.sample(4, length=8)
        assert torch.allclose(batch["subgoal"].sum(-1), torch.ones(4, 8))

    def test_sequences_are_temporally_contiguous(self):
        r = make_replay()
        push(r, 100)
        batch, _, _ = r.sample(6, length=10)
        # `symbolic` was written as the timestep index, so deltas must all be +1.
        deltas = torch.diff(batch["symbolic"][:, :, 0], dim=1)
        assert torch.allclose(deltas, torch.ones_like(deltas))

    def test_extra_plane_is_preserved(self):
        r = make_replay()
        push(r, 40)
        batch, _, _ = r.sample(3, length=8)
        assert (batch["frame"][:, :, STACK] == 7).all()

    def test_ring_buffer_wraps_without_stale_reads(self):
        r = make_replay(capacity=2 * 1024)  # 1024 per stream
        push(r, 1500)  # forces a wrap
        batch, _, _ = r.sample(4, length=16)
        deltas = torch.diff(batch["symbolic"][:, :, 0], dim=1)
        assert torch.allclose(deltas, torch.ones_like(deltas))


class TestFrameStack:
    def test_stack_is_reconstructed_in_order(self):
        r = make_replay()
        push(r, 64)
        batch, _, _ = r.sample(4, length=8)
        # Plane k should be the frame from (t - (STACK-1-k)); within a window away from
        # any reset the planes must increase by 1 across the stack axis.
        planes = batch["frame"][:, -1, :STACK, 0, 0].to(torch.int64)
        assert torch.all(torch.diff(planes, dim=-1) == 1)

    def test_stack_does_not_read_across_an_episode_boundary(self):
        """The property this test exists for: no frames from the previous episode."""
        r = make_replay()
        push(r, 80, first_at={0, 40})
        for _ in range(40):  # sample enough to hit the boundary
            batch, _, _ = r.sample(8, length=16)
            first = batch["is_first"]
            frames = batch["frame"][:, :, :STACK, 0, 0].to(torch.int64)
            sym = batch["symbolic"][:, :, 0].to(torch.int64)
            rows, cols = torch.nonzero(first > 0, as_tuple=True)
            for b, t in zip(rows.tolist(), cols.tolist()):
                # At a reset every stack plane must equal the current timestep's frame.
                assert (frames[b, t] == sym[b, t] % 256).all(), (
                    f"stack leaked across reset at row {b} step {t}: {frames[b, t]}"
                )

    def test_steps_after_a_reset_only_use_post_reset_frames(self):
        r = make_replay()
        push(r, 80, first_at={0, 40})
        for _ in range(40):
            batch, _, _ = r.sample(8, length=16)
            first = batch["is_first"]
            frames = batch["frame"][:, :, :STACK, 0, 0].to(torch.int64)
            sym = batch["symbolic"][:, :, 0].to(torch.int64)
            for b in range(first.shape[0]):
                idx = torch.nonzero(first[b] > 0).flatten().tolist()
                if not idx:
                    continue
                reset = idx[0]
                for t in range(reset, first.shape[1]):
                    lowest = int(frames[b, t].min())
                    assert lowest >= int(sym[b, reset]) % 256


class TestPriorities:
    def test_update_and_reuse(self):
        r = make_replay()
        push(r, 64)
        _, streams, positions = r.sample(4, length=8)
        r.update_priorities(streams, positions, np.array([9.0, 9.0, 9.0, 9.0]))
        ring = positions % r.per_stream
        assert np.allclose(r.priority[streams, ring], 9.0)

    def test_prioritized_sampling_prefers_high_loss_positions(self):
        r = make_replay(prioritized=1.0)
        push(r, 400)
        r.priority[:] = 0.0
        hot = 200
        r.priority[:, hot] = 50.0  # one very high-loss position per stream
        counts = 0
        trials = 60
        for _ in range(trials):
            _, streams, positions = r.sample(8, length=8)
            counts += int(((positions % r.per_stream) == hot).sum())
        # Uniform would give ~8*60/392 ~= 1.2 hits; prioritised must do far better.
        assert counts > 20, f"prioritised sampling barely helped: {counts} hits"

    def test_uniform_fraction_still_covers_the_buffer(self):
        r = make_replay(prioritized=0.5)
        push(r, 400)
        r.priority[:] = 0.0
        r.priority[:, 200] = 100.0
        seen = set()
        for _ in range(40):
            _, _, positions = r.sample(8, length=8)
            seen.update((positions % r.per_stream).tolist())
        assert len(seen) > 30, "uniform half must keep coverage broad"

    def test_degenerate_priorities_do_not_crash(self):
        r = make_replay(prioritized=1.0)
        push(r, 64)
        r.priority[:] = np.nan
        assert r.sample(4, length=8) is not None


class TestPersistence:
    def test_state_dict_roundtrip_invalidates_data(self):
        r = make_replay()
        push(r, 64)
        d = r.state_dict()
        other = make_replay()
        other.load_state_dict(d)
        # Arrays are not persisted, so the buffer must report itself as empty rather
        # than serve uninitialised frames.
        assert len(other) == 0
        assert other.sample(2, length=8) is None
