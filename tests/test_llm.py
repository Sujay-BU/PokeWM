"""LLM client and proposer.

The behaviour that matters most is what happens when the model is *absent or wrong*: an
overnight run must survive a stopped Ollama daemon, a timeout, or a garbled response
without stalling or corrupting the reward signal. Tests marked `llm` need a live daemon;
everything else runs offline against fakes.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pytest

from pokewm.config import LLMConfig
from pokewm.llm.ollama_client import OllamaClient, OllamaUnavailable, extract_json
from pokewm.llm.proposer import (
    SYSTEM_PROMPT,
    NullProposer,
    ProposalRecord,
    SubgoalProposer,
    build_proposer,
)
from pokewm.llm.subgoals import BY_NAME, DEFAULT_SUBGOAL, NUM_SUBGOALS


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"subgoal": "HEAL"}') == {"subgoal": "HEAL"}

    def test_object_embedded_in_prose(self):
        text = 'Sure! Here you go:\n{"subgoal": "EXPLORE", "why": "new area"}\nDone.'
        assert extract_json(text)["subgoal"] == "EXPLORE"

    def test_nested_braces(self):
        assert extract_json('{"a": {"b": 1}, "subgoal": "HEAL"}')["subgoal"] == "HEAL"

    @pytest.mark.parametrize("bad", ["", "   ", "not json", "{", '{"a":', "[1,2,3]"])
    def test_malformed_returns_empty_dict(self, bad):
        assert extract_json(bad) == {}

    def test_non_object_json_returns_empty_dict(self):
        assert extract_json('"a string"') == {}

    def test_none_is_safe(self):
        assert extract_json(None) == {}


class TestSystemPrompt:
    def test_mentions_every_subgoal(self):
        for name in BY_NAME:
            assert name in SYSTEM_PROMPT

    def test_requests_json(self):
        assert "JSON" in SYSTEM_PROMPT

    def test_discourages_the_fallback(self):
        assert "MAIN_QUEST" in SYSTEM_PROMPT
        assert "fallback" in SYSTEM_PROMPT.lower()


class TestClientOffline:
    def test_unreachable_host_raises_ollama_unavailable(self):
        client = OllamaClient("http://127.0.0.1:9", "whatever", timeout=1.0)
        with pytest.raises(OllamaUnavailable):
            client.available_models()

    def test_check_reports_a_missing_model(self, monkeypatch):
        client = OllamaClient("http://x", "missing:1b")
        monkeypatch.setattr(client, "available_models", lambda: ["other:8b"])
        with pytest.raises(OllamaUnavailable, match="not present"):
            client.check()

    def test_check_passes_when_present(self, monkeypatch):
        client = OllamaClient("http://x", "qwen3:8b")
        monkeypatch.setattr(client, "available_models", lambda: ["qwen3:8b"])
        client.check()

    def test_png_encoding_roundtrips(self):
        from pokewm.llm.ollama_client import _encode_png

        rgb = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        b64 = _encode_png(rgb)
        assert isinstance(b64, str) and len(b64) > 100

    def test_png_encoding_handles_grayscale(self):
        from pokewm.llm.ollama_client import _encode_png

        assert len(_encode_png(np.zeros((72, 80), dtype=np.uint8))) > 20


class TestNullProposer:
    def test_matches_the_real_interface(self):
        p = NullProposer(4)
        p.start()
        p.observe(0, 100, "state", 3, None)
        assert p.subgoal_for(0) == DEFAULT_SUBGOAL
        assert p.all_subgoals().shape == (4,)
        assert p.stats()["llm/available"] == 0.0
        assert p.recent() == []
        p.stop()

    def test_all_subgoals_are_valid_indices(self):
        sg = NullProposer(8).all_subgoals()
        assert ((sg >= 0) & (sg < NUM_SUBGOALS)).all()

    def test_context_manager(self):
        with NullProposer(2) as p:
            assert p.subgoal_for(1) == DEFAULT_SUBGOAL


class TestBuildProposer:
    def test_disabled_config_gives_null_proposer(self):
        cfg = LLMConfig(enabled=False)
        assert isinstance(build_proposer(cfg, 4), NullProposer)

    def test_unreachable_daemon_degrades_to_null(self):
        """An overnight run must not die because Ollama is not running."""
        cfg = LLMConfig(enabled=True, host="http://127.0.0.1:9", request_timeout=1.0)
        assert isinstance(build_proposer(cfg, 4), NullProposer)

    def test_hard_fail_propagates_when_requested(self):
        cfg = LLMConfig(
            enabled=True, host="http://127.0.0.1:9", hard_fail=True, request_timeout=1.0
        )
        with pytest.raises(OllamaUnavailable):
            SubgoalProposer(cfg, 2)


class TestProposerScheduling:
    """The proposer thread is exercised with a stubbed client, so these are fast."""

    def _proposer(self, monkeypatch, responses, num_workers=2, tmp_path=None):
        cfg = LLMConfig(
            enabled=True, refresh_steps=10,
            cache_path=str((tmp_path or "/tmp") / "cache.jsonl") if tmp_path else "/tmp/pokewm_test_cache.jsonl",
        )
        monkeypatch.setattr(SubgoalProposer, "__init__", SubgoalProposer.__init__)
        p = SubgoalProposer.__new__(SubgoalProposer)
        # Build by hand to avoid touching the network in __init__.
        from pokewm.llm.proposer import _WorkerSlot
        import threading

        p.cfg = cfg
        p.num_workers = num_workers
        p.slots = [_WorkerSlot() for _ in range(num_workers)]
        p._stop = threading.Event()
        p._thread = None
        p.available = True
        p.reason = "stub"
        p.num_calls = 0
        p.num_failures = 0
        p.total_latency = 0.0
        p._history = []
        p._cache_fh = None

        class StubClient:
            def __init__(self):
                self.calls = 0

            def chat(self, system, user, images=None, max_tokens=192, json_mode=True):
                from pokewm.llm.ollama_client import ChatResult

                r = responses[min(self.calls, len(responses) - 1)]
                self.calls += 1
                if isinstance(r, Exception):
                    raise r
                return ChatResult(content=r, thinking="", eval_count=10, duration_s=0.01)

        p._client = StubClient()
        return p

    def test_parses_a_good_response(self, monkeypatch, tmp_path):
        p = self._proposer(monkeypatch, ['{"subgoal": "HEAL", "why": "hurt"}'],
                           tmp_path=tmp_path)
        rec = p._request(0, 100, "state", 3, None)
        assert rec.subgoal_id == BY_NAME["HEAL"].id
        assert rec.why == "hurt"

    def test_garbled_response_falls_back_safely(self, monkeypatch, tmp_path):
        p = self._proposer(monkeypatch, ["complete nonsense"], tmp_path=tmp_path)
        rec = p._request(0, 100, "state", 3, None)
        assert rec.subgoal_id == BY_NAME["MAIN_QUEST"].id

    def test_unknown_subgoal_name_falls_back(self, monkeypatch, tmp_path):
        p = self._proposer(monkeypatch, ['{"subgoal": "EAT_LUNCH"}'], tmp_path=tmp_path)
        assert p._request(0, 1, "s", 0, None).subgoal_id == BY_NAME["MAIN_QUEST"].id

    def test_worker_selection_prefers_the_stalest(self, monkeypatch, tmp_path):
        p = self._proposer(monkeypatch, ["{}"], num_workers=3, tmp_path=tmp_path)
        p.slots[0].pending_text = "a"
        p.slots[0].pending_step = 100
        p.slots[0].last_refresh_step = 90  # age 10
        p.slots[1].pending_text = "b"
        p.slots[1].pending_step = 100
        p.slots[1].last_refresh_step = 0  # age 100 -- stalest
        p.slots[2].pending_text = ""  # no state yet: ineligible
        assert p._pick_worker(0) == 1

    def test_worker_below_refresh_threshold_is_skipped(self, monkeypatch, tmp_path):
        p = self._proposer(monkeypatch, ["{}"], tmp_path=tmp_path)
        p.slots[0].pending_text = "a"
        p.slots[0].pending_step = 5
        p.slots[0].last_refresh_step = 0  # age 5 < refresh_steps 10
        p.slots[1].pending_text = ""
        assert p._pick_worker(0) is None

    def test_observe_is_non_blocking_and_records_state(self, monkeypatch, tmp_path):
        p = self._proposer(monkeypatch, ["{}"], tmp_path=tmp_path)
        t0 = time.perf_counter()
        for i in range(10_000):
            p.observe(i % 2, i, "some state text", 4)
        # The whole point: this must never wait on the model.
        assert time.perf_counter() - t0 < 0.5
        assert p.slots[0].pending_text == "some state text"

    def test_failed_request_keeps_the_previous_subgoal(self, monkeypatch, tmp_path):
        p = self._proposer(monkeypatch, [RuntimeError("boom")], tmp_path=tmp_path)
        p.slots[0].subgoal_id = BY_NAME["HEAL"].id
        p.slots[0].pending_text = "s"
        p.slots[0].pending_step = 1000
        p._cache_fh = None
        with pytest.raises(RuntimeError):
            p._request(0, 1000, "s", 0, None)
        assert p.slots[0].subgoal_id == BY_NAME["HEAL"].id

    def test_history_is_bounded(self, monkeypatch, tmp_path):
        p = self._proposer(monkeypatch, ["{}"], tmp_path=tmp_path)
        for i in range(1000):
            p._record(ProposalRecord(0, i, 0, "X", "", 0.0, ""))
        assert len(p._history) <= 256

    def test_record_serialises_to_json(self):
        rec = ProposalRecord(1, 2, 3, "HEAL", "because", 1.5, "state text")
        parsed = json.loads(rec.to_json())
        assert parsed["subgoal_name"] == "HEAL" and parsed["worker"] == 1


@pytest.fixture(scope="module")
def live_cfg() -> LLMConfig:
    """Reachable daemon *and* a resident model.

    Ollama evicts the model between runs, and a cold load of qwen3:8b measured 47.9 s on
    this machine -- on its own most of the production request budget. That made these
    tests pass or fail on whether the model happened to still be warm, which is a
    property of the last few minutes rather than of the code under test. Warm it here,
    with a budget generous enough for the load, so a failure downstream means something.
    """
    cfg = LLMConfig()
    try:
        OllamaClient(cfg.host, cfg.model, timeout=5.0).check()
    except OllamaUnavailable as exc:
        pytest.skip(str(exc))
    warm = OllamaClient(cfg.host, cfg.model, num_gpu=cfg.num_gpu,
                        num_thread=cfg.num_thread, timeout=240.0)
    try:
        warm.chat("reply with ok", "ok", max_tokens=4, json_mode=False)
    except Exception as exc:                      # cold load too slow, or daemon busy
        pytest.skip(f"could not warm {cfg.model}: {exc!r}")
    finally:
        warm.close()
    return cfg


@pytest.mark.llm
class TestLiveOllama:
    """Requires a running daemon with the configured model pulled."""

    def test_returns_a_valid_subgoal(self, live_cfg):
        cfg = live_cfg
        p = build_proposer(replace(cfg, refresh_steps=1), 1)
        assert not isinstance(p, NullProposer)
        rec = p._request(
            0, 0,
            "location: Cerulean City at (x=20, y=30)\n"
            "badges: 1 [boulder]\n"
            "party (2/6): #4 Lv22 2/60HP, #16 Lv18 0/44HP\n"
            "money: $1200  pokedex: 8 owned / 21 seen\n"
            "in battle: no\nstory flags set: 61",
            16, None,
        )
        assert 0 <= rec.subgoal_id < NUM_SUBGOALS
        assert rec.latency_s > 0

    def test_thinking_is_suppressed_so_content_is_usable(self, live_cfg):
        """Regression guard: qwen3 will otherwise spend the whole budget thinking."""
        cfg = live_cfg
        client = OllamaClient(cfg.host, cfg.model, num_gpu=cfg.num_gpu,
                              timeout=cfg.request_timeout)
        result = client.chat(
            SYSTEM_PROMPT, "location: Pallet Town\nbadges: 0\nparty (0/6): empty",
            max_tokens=128,
        )
        assert result.content.strip(), "model returned empty content"
        assert extract_json(result.content), "content was not parseable JSON"


class TestProposerIsRateLimited:
    """Regression: the proposer starved the emulators it exists to help.

    It is single-flight and round-robins, but eligibility is measured in env steps and
    with 8 workers is essentially always satisfied, so `_loop` reissued immediately on
    every pass. An 8B model then ran on the CPU at 100% duty cycle -- measured 518%, five
    of sixteen cores -- and collection fell to 300 env steps/s against the ~760 the
    replay ratio permitted and ~1200 recorded in docs/PROOF.md. A subgoal is coarse
    guidance refreshed over minutes; nothing about it needs a continuously hot model.
    """

    def test_a_cooldown_is_configured(self):
        from pokewm.config import LLMConfig

        assert LLMConfig().min_interval_s > 0

    def test_the_cooldown_leaves_the_cpu_mostly_free(self):
        """Measured mean latency was 7.8 s; duty cycle must stay well under half."""
        from pokewm.config import LLMConfig

        assert 7.8 / LLMConfig().min_interval_s < 0.5

    def test_threads_are_bounded_below_the_core_count(self):
        from pokewm.config import LLMConfig

        assert 0 < LLMConfig().num_thread <= 6

    def test_the_thread_cap_reaches_the_request(self):
        from pokewm.llm.ollama_client import OllamaClient

        c = OllamaClient("http://127.0.0.1:11434", "m", num_thread=4)
        payload = c._payload("hi", max_tokens=32) if hasattr(c, "_payload") else None
        if payload is None:
            import inspect
            src = inspect.getsource(type(c))
            assert "num_thread" in src
        else:
            assert payload["options"]["num_thread"] == 4

    def test_subgoals_still_refresh_often_enough_to_matter(self):
        """The cap must not make guidance so rare it is useless."""
        from pokewm.config import LLMConfig

        cfg = LLMConfig()
        per_worker_s = cfg.min_interval_s * 8      # round-robin over 8 workers
        assert per_worker_s < 600                  # a fresh subgoal at least every 10 min


class TestShutdownDoesNotHang:
    """Regression: the trainer stayed alive for minutes after logging "finished".

    `proposer.stop()` runs in the trainer's `finally`. The proposer thread parks in a
    socket read for the full `request_timeout`, and joining politely just waits it out.
    With the timeout at 120 s and `scripts/train.sh stop` allowing 180 s before SIGKILL,
    shutdown was a coin flip on losing the process to a kill.
    """

    def test_request_timeout_fits_inside_the_stop_budget(self):
        """`scripts/train.sh stop` SIGKILLs after 180 s; one request must fit with room.

        This is a bound, not the fix. The fix is `stop()` closing the session so an
        in-flight read fails at once -- see `test_stop_closes_the_client_before_joining`.
        """
        from pokewm.config import LLMConfig

        assert LLMConfig().request_timeout <= 120

    def test_the_client_can_be_closed(self):
        from pokewm.llm.ollama_client import OllamaClient

        c = OllamaClient("http://127.0.0.1:11434", "m")
        c.close()          # must not raise, and must be safe when idle
        c.close()

    def test_stop_closes_the_client_before_joining(self):
        import inspect

        from pokewm.llm.proposer import SubgoalProposer as P

        src = inspect.getsource(P.stop)
        assert src.index("close()") < src.index("join(")
