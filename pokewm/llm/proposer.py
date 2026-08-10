"""Asynchronous subgoal proposer.

Contract with the training loop, in one line: **the RL loop never waits for the LLM.**

A single background thread issues one request at a time, round-robining over the
workers that have gone longest without a refresh. Whatever throughput the local model
achieves is the refresh rate; if it achieves zero (daemon down, model evicted, machine
busy) every worker simply keeps its last subgoal and training is unaffected. This is the
only design that survives an 8-billion-parameter model running on CPU next to an
emulator farm.

Single-flight is deliberate. Concurrent requests to one Ollama daemon serialise anyway,
and queueing them would only build unbounded latency between the state a subgoal was
chosen for and the state it is applied to.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import LLMConfig
from .ollama_client import OllamaClient, OllamaUnavailable, extract_json
from .subgoals import DEFAULT_SUBGOAL, NUM_SUBGOALS, parse_subgoal, vocabulary_prompt

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    """You are a strategy adviser for an agent learning to play Pokemon Red.

You do NOT press buttons. You pick ONE high-level subgoal from the fixed list below; a
reinforcement-learning policy carries it out over the next few thousand frames.

Pick the MOST SPECIFIC subgoal that applies right now. Reserve MAIN_QUEST for when
genuinely nothing more specific fits -- it is the fallback, not the default.

Priority rules, highest first:
1. If any party Pokemon has fainted (0 HP) or the party is below ~35% HP, choose HEAL.
2. If a battle is in progress, choose WIN_BATTLE (or FLEE_BATTLE if badly outmatched).
3. If the party is smaller than 3, or the strongest Pokemon is far below the next Gym
   Leader's level, choose CATCH_POKEMON or TRAIN_LEVELS.
4. If the current city has a Gym you have not beaten and the party looks ready,
   choose CHALLENGE_GYM.
5. If the way forward is blocked by a tree, water, or boulder, choose USE_FIELD_MOVE.
6. If the current map is fully explored and the story points elsewhere, choose
   REACH_NEXT_CITY or LEAVE_AREA.
7. If standing in an unexplored area, choose EXPLORE.

Reply with compact JSON only, no prose:
{"subgoal": "<NAME>", "why": "<one short sentence>"}

Valid subgoal names:
"""
    + vocabulary_prompt()
)


@dataclass
class ProposalRecord:
    worker: int
    step: int
    subgoal_id: int
    subgoal_name: str
    why: str
    latency_s: float
    state_text: str
    milestone: int = 0
    ok: bool = True
    error: str = ""

    def to_json(self) -> str:
        d = dict(self.__dict__)
        d["t"] = time.time()
        return json.dumps(d)


@dataclass
class _WorkerSlot:
    subgoal_id: int = DEFAULT_SUBGOAL
    last_refresh_step: int = -(10**9)
    pending_text: str = ""
    pending_step: int = 0
    pending_screen: np.ndarray | None = None
    milestone: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class SubgoalProposer:
    """Owns the LLM thread and the per-worker subgoal assignment."""

    def __init__(self, cfg: LLMConfig, num_workers: int) -> None:
        self.cfg = cfg
        self.num_workers = num_workers
        self.slots = [_WorkerSlot() for _ in range(num_workers)]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: OllamaClient | None = None
        self.available = False
        self.reason = "disabled"
        self.num_calls = 0
        self.num_failures = 0
        self.total_latency = 0.0
        self._history: list[ProposalRecord] = []
        self._cache_fh = None

        if not cfg.enabled:
            return
        model = cfg.vision_model if cfg.use_vision else cfg.model
        self._client = OllamaClient(
            cfg.host,
            model,
            num_ctx=cfg.num_ctx,
            num_gpu=cfg.num_gpu,
            num_thread=cfg.num_thread,
            keep_alive=cfg.keep_alive,
            temperature=cfg.temperature,
            timeout=cfg.request_timeout,
        )
        try:
            self._client.check()
            self.available = True
            self.reason = f"ok ({model})"
        except OllamaUnavailable as exc:
            self.reason = str(exc)
            if cfg.hard_fail:
                raise
            log.warning("LLM proposer disabled: %s", exc)
            self._client = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        path = Path(self.cfg.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_fh = open(path, "a", buffering=1)
        self._thread = threading.Thread(
            target=self._loop, name="subgoal-proposer", daemon=True
        )
        self._thread.start()
        log.info("subgoal proposer started (%s)", self.reason)

    def stop(self) -> None:
        self._stop.set()
        # Drop the HTTP session first. The worker may be parked in a socket read with
        # the full `request_timeout` still to run, and joining politely just waits it
        # out; closing underneath it makes the read fail fast.
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._cache_fh is not None:
            self._cache_fh.close()
            self._cache_fh = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------------ interface

    def observe(
        self,
        worker: int,
        step: int,
        state_text: str,
        milestone: int = 0,
        screen: np.ndarray | None = None,
    ) -> None:
        """Publish a worker's latest state for the proposer thread. Non-blocking."""
        slot = self.slots[worker]
        slot.pending_text = state_text
        slot.pending_step = step
        slot.milestone = milestone
        if self.cfg.use_vision:
            slot.pending_screen = screen

    def subgoal_for(self, worker: int) -> int:
        return self.slots[worker].subgoal_id

    def all_subgoals(self) -> np.ndarray:
        return np.array([s.subgoal_id for s in self.slots], dtype=np.int64)

    def stats(self) -> dict[str, float]:
        return {
            "llm/calls": float(self.num_calls),
            "llm/failures": float(self.num_failures),
            "llm/mean_latency_s": float(
                self.total_latency / self.num_calls if self.num_calls else 0.0
            ),
            "llm/available": float(self.available),
        }

    def recent(self, n: int = 5) -> list[ProposalRecord]:
        return self._history[-n:]

    # ------------------------------------------------------------------ thread

    def _pick_worker(self, global_step: int) -> int | None:
        """Least-recently-refreshed worker that is eligible."""
        best, best_age = None, -1
        for i, slot in enumerate(self.slots):
            if not slot.pending_text:
                continue
            age = slot.pending_step - slot.last_refresh_step
            if age >= self.cfg.refresh_steps and age > best_age:
                best, best_age = i, age
        return best

    def _loop(self) -> None:
        backoff = 1.0
        last_request = 0.0
        while not self._stop.is_set():
            # Rate limit in wall clock, not in env steps. Eligibility is step-based and
            # is essentially always satisfied with 8 workers, so without this the loop
            # keeps an 8B model resident on the CPU continuously and starves the
            # emulators it is supposed to be helping.
            wait = self.cfg.min_interval_s - (time.time() - last_request)
            if wait > 0 and self._stop.wait(min(wait, 1.0)):
                break
            if wait > 0:
                continue
            worker = self._pick_worker(0)
            if worker is None:
                self._stop.wait(1.0)
                continue
            last_request = time.time()
            slot = self.slots[worker]
            text, step = slot.pending_text, slot.pending_step
            screen = slot.pending_screen
            try:
                record = self._request(worker, step, text, slot.milestone, screen)
                slot.subgoal_id = record.subgoal_id
                slot.last_refresh_step = step
                self.num_calls += 1
                self.total_latency += record.latency_s
                backoff = 1.0
            except Exception as exc:  # network, timeout, malformed -> keep last subgoal
                self.num_failures += 1
                slot.last_refresh_step = step  # do not hot-loop on a broken worker
                record = ProposalRecord(
                    worker, step, slot.subgoal_id, "(unchanged)", "", 0.0, text,
                    ok=False, error=repr(exc)[:200],
                )
                log.warning("subgoal request failed (%s); backing off %.0fs", exc, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
            self._record(record)

    def _request(
        self,
        worker: int,
        step: int,
        state_text: str,
        milestone: int,
        screen: np.ndarray | None,
    ) -> ProposalRecord:
        assert self._client is not None
        user = (
            f"Current game state:\n{state_text}\n\n"
            f"Story progress: milestone {milestone}.\n\n"
            "Which single subgoal should the agent pursue next?"
        )
        t0 = time.perf_counter()
        result = self._client.chat(
            SYSTEM_PROMPT,
            user,
            images=[screen] if (screen is not None and self.cfg.use_vision) else None,
            max_tokens=self.cfg.max_tokens,
        )
        latency = time.perf_counter() - t0
        obj = extract_json(result.content)
        name = obj.get("subgoal") if isinstance(obj.get("subgoal"), str) else None
        sg_id = parse_subgoal(name)
        why = str(obj.get("why", ""))[:200]
        return ProposalRecord(
            worker=worker,
            step=step,
            subgoal_id=sg_id,
            subgoal_name=name or "(unparsed)",
            why=why,
            latency_s=latency,
            state_text=state_text,
            milestone=milestone,
        )

    def _record(self, record: ProposalRecord) -> None:
        self._history.append(record)
        if len(self._history) > 256:
            del self._history[:128]
        if self._cache_fh is not None:
            try:
                self._cache_fh.write(record.to_json() + "\n")
            except Exception:
                pass


class NullProposer:
    """Stand-in used when the LLM is disabled. Same surface, zero behaviour."""

    available = False
    reason = "disabled"

    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def observe(self, *a, **k) -> None: ...
    def subgoal_for(self, worker: int) -> int:
        return DEFAULT_SUBGOAL

    def all_subgoals(self) -> np.ndarray:
        return np.full(self.num_workers, DEFAULT_SUBGOAL, dtype=np.int64)

    def stats(self) -> dict[str, float]:
        return {"llm/calls": 0.0, "llm/failures": 0.0, "llm/available": 0.0}

    def recent(self, n: int = 5) -> list:
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc): ...


def build_proposer(cfg: LLMConfig, num_workers: int):
    if not cfg.enabled:
        return NullProposer(num_workers)
    proposer = SubgoalProposer(cfg, num_workers)
    return proposer if proposer.available else NullProposer(num_workers)


__all__ = [
    "SubgoalProposer",
    "NullProposer",
    "build_proposer",
    "NUM_SUBGOALS",
    "ProposalRecord",
]
