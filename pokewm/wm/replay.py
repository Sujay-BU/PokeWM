"""Sequence replay with Simulus-style prioritisation.

Storage layout
--------------
One ring buffer per collector stream (i.e. per parallel emulator). Windows are sampled
contiguously *within* a stream so the RSSM sees genuine temporal structure.

Frames are stored **unstacked** -- one 72x80 luma plane plus one visited-overlay plane
per transition -- and the frame stack is rebuilt at sample time. Storing the stacked
observation directly would multiply memory by 4 for zero information, which on a 16 GB
machine is the difference between a 200k-step buffer and a 50k-step one.

Prioritisation
--------------
`prioritized_fraction` of each batch is drawn proportional to softmax(per-sequence world
model loss), the rest uniformly. Purely prioritised sampling on a nonstationary loss
collapses onto a handful of pathological sequences; the uniform half bounds the
importance ratio and keeps the model's coverage honest.
"""

from __future__ import annotations

import threading

import numpy as np
import torch

from ..config import ReplayConfig


class SequenceReplay:
    def __init__(
        self,
        cfg: ReplayConfig,
        num_streams: int,
        frame_hw: tuple[int, int],
        symbolic_dim: int,
        num_actions: int,
        num_subgoals: int,
        extra_planes: int = 1,
        frame_stack: int = 4,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.num_streams = num_streams
        self.frame_stack = frame_stack
        self.extra_planes = extra_planes
        self.symbolic_dim = symbolic_dim
        self.num_actions = num_actions
        self.num_subgoals = num_subgoals
        h, w = frame_hw
        self.frame_hw = frame_hw
        self.per_stream = max(cfg.capacity // num_streams, 1024)

        shape = (num_streams, self.per_stream)
        self.frame = np.zeros((*shape, h, w), dtype=np.uint8)
        self.planes = np.zeros((*shape, extra_planes, h, w), dtype=np.uint8)
        self.symbolic = np.zeros((*shape, symbolic_dim), dtype=np.float32)
        self.subgoal = np.zeros(shape, dtype=np.uint8)
        self.action = np.zeros(shape, dtype=np.uint8)
        self.reward = np.zeros(shape, dtype=np.float32)
        self.cont = np.zeros(shape, dtype=np.uint8)
        self.is_first = np.zeros(shape, dtype=np.uint8)
        self.priority = np.full(shape, 1.0, dtype=np.float32)

        self.written = np.zeros(num_streams, dtype=np.int64)  # monotone counter
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ properties

    def __len__(self) -> int:
        return int(np.minimum(self.written, self.per_stream).sum())

    @property
    def ready(self) -> bool:
        return len(self) >= self.cfg.min_size

    # ------------------------------------------------------------------ writing

    def add(
        self,
        frame: np.ndarray,  # (N, C, H, W) -- stacked observation as returned by the env
        symbolic: np.ndarray,  # (N, D)
        subgoal: np.ndarray,  # (N, S) one-hot
        action: np.ndarray,  # (N,)
        reward: np.ndarray,  # (N,)
        cont: np.ndarray,  # (N,) bool -- False on true termination
        is_first: np.ndarray,  # (N,) bool
    ) -> None:
        """Append one synchronous timestep across all streams."""
        with self._lock:
            idx = (self.written % self.per_stream).astype(np.int64)
            rows = np.arange(self.num_streams)
            # The newest plane of the stack, plus whatever extra planes follow it.
            self.frame[rows, idx] = frame[:, self.frame_stack - 1]
            if self.extra_planes:
                self.planes[rows, idx] = frame[:, self.frame_stack :]
            self.symbolic[rows, idx] = symbolic
            self.subgoal[rows, idx] = subgoal.argmax(axis=-1).astype(np.uint8)
            self.action[rows, idx] = action.astype(np.uint8)
            self.reward[rows, idx] = reward
            self.cont[rows, idx] = cont.astype(np.uint8)
            self.is_first[rows, idx] = is_first.astype(np.uint8)
            # New data is maximally interesting until proven otherwise.
            self.priority[rows, idx] = self.priority.max() if self.written.max() else 1.0
            self.written += 1

    # ------------------------------------------------------------------ sampling

    def _valid_range(self, stream: int, length: int) -> tuple[int, int]:
        """Half-open [lo, hi) of window start positions in absolute coordinates."""
        written = int(self.written[stream])
        lookback = self.frame_stack - 1
        lo = max(0, written - self.per_stream + lookback)
        hi = written - length
        return lo, hi

    def sample(
        self, batch_size: int, length: int, device: torch.device | str = "cpu"
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray] | None:
        with self._lock:
            starts: list[tuple[int, int]] = []
            n_prior = int(round(batch_size * self.cfg.prioritized_fraction))
            for i in range(batch_size):
                stream = int(self._rng.integers(self.num_streams))
                lo, hi = self._valid_range(stream, length)
                if hi <= lo:
                    return None
                if i < n_prior:
                    pos = self._sample_prioritized(stream, lo, hi)
                else:
                    pos = int(self._rng.integers(lo, hi))
                starts.append((stream, pos))
            batch = self._gather(starts, length)

        streams = np.array([s for s, _ in starts], dtype=np.int64)
        positions = np.array([p for _, p in starts], dtype=np.int64)
        tensors = {
            k: torch.as_tensor(v).to(device, non_blocking=True) for k, v in batch.items()
        }
        return tensors, streams, positions

    def _sample_prioritized(self, stream: int, lo: int, hi: int) -> int:
        span = hi - lo
        # Subsample candidates rather than softmaxing over the whole buffer; with
        # 25k valid starts per stream the full softmax would dominate step time.
        n = min(span, 256)
        cand = self._rng.integers(lo, hi, size=n)
        pri = self.priority[stream, cand % self.per_stream]
        t = max(self.cfg.priority_temperature, 1e-6)
        logits = (pri - pri.max()) / t
        p = np.exp(logits)
        total = p.sum()
        if not np.isfinite(total) or total <= 0:
            return int(self._rng.integers(lo, hi))
        return int(cand[self._rng.choice(n, p=p / total)])

    def _gather(self, starts: list[tuple[int, int]], length: int) -> dict[str, np.ndarray]:
        b = len(starts)
        h, w = self.frame_hw
        stack = self.frame_stack
        c = stack + self.extra_planes

        out_frame = np.empty((b, length, c, h, w), dtype=np.uint8)
        out_sym = np.empty((b, length, self.symbolic_dim), dtype=np.float32)
        out_sg = np.empty((b, length, self.num_subgoals), dtype=np.float32)
        out_act = np.zeros((b, length, self.num_actions), dtype=np.float32)
        out_rew = np.empty((b, length), dtype=np.float32)
        out_cont = np.empty((b, length), dtype=np.float32)
        out_first = np.empty((b, length), dtype=np.float32)

        for i, (stream, pos) in enumerate(starts):
            abs_idx = pos + np.arange(length)
            ring = abs_idx % self.per_stream

            first = self.is_first[stream, ring]
            out_first[i] = first
            out_rew[i] = self.reward[stream, ring]
            out_cont[i] = self.cont[stream, ring]
            out_sym[i] = self.symbolic[stream, ring]
            out_sg[i] = np.eye(self.num_subgoals, dtype=np.float32)[
                self.subgoal[stream, ring]
            ]
            out_act[i, np.arange(length), self.action[stream, ring]] = 1.0

            # Rebuild the frame stack, never reading across an episode boundary: the
            # earliest in-episode frame is repeated instead.
            reset_at = np.where(first == 1, abs_idx, -1)
            last_reset = np.maximum.accumulate(reset_at)
            # Positions before the first in-window reset may still look back into the
            # previous episode's tail, which is correct -- the window simply started
            # mid-episode.
            floor = np.where(last_reset >= 0, last_reset, abs_idx - (stack - 1))
            for k in range(stack):
                src = np.maximum(abs_idx - (stack - 1 - k), floor)
                out_frame[i, :, k] = self.frame[stream, src % self.per_stream]
            if self.extra_planes:
                out_frame[i, :, stack:] = self.planes[stream, ring]

        return {
            "frame": out_frame,
            "symbolic": out_sym,
            "subgoal": out_sg,
            "action": out_act,
            "reward": out_rew,
            "cont": out_cont,
            "is_first": out_first,
        }

    # ------------------------------------------------------------------ priorities

    def update_priorities(
        self, streams: np.ndarray, positions: np.ndarray, losses: np.ndarray
    ) -> None:
        with self._lock:
            ring = positions % self.per_stream
            self.priority[streams, ring] = np.asarray(losses, dtype=np.float32)

    # ------------------------------------------------------------------ persistence

    def state_dict(self) -> dict:
        """Only counters. The arrays are multi-GB; checkpoints stay small and a resumed
        run refills the buffer from fresh interaction, which is also healthier for a
        nonstationary policy."""
        return {"written": self.written.copy()}

    def load_state_dict(self, d: dict) -> None:
        self.written = np.asarray(d["written"], dtype=np.int64).copy()
        # The arrays are empty on resume; invalidate so we do not sample garbage.
        self.written[:] = 0
