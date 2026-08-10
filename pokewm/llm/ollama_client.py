"""Minimal Ollama chat client.

Deliberately not the `ollama` python package: we need exactly one endpoint, and a thin
wrapper over `requests` means the failure modes (daemon down, model not pulled, timeout)
are visible and individually recoverable inside a 24-hour run.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass

import numpy as np
import requests

log = logging.getLogger(__name__)


class OllamaUnavailable(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    thinking: str
    eval_count: int
    duration_s: float


class OllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        *,
        num_ctx: int = 4096,
        num_gpu: int | None = None,
        num_thread: int | None = None,
        keep_alive: str = "30m",
        temperature: float = 0.6,
        timeout: float = 120.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.num_gpu = num_gpu
        self.num_thread = num_thread
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.timeout = timeout
        self._session = requests.Session()

    def close(self) -> None:
        """Drop the connection pool, failing any in-flight request immediately."""
        self._session.close()

    # ------------------------------------------------------------------ health

    def available_models(self) -> list[str]:
        try:
            r = self._session.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception as exc:  # daemon down, wrong port, ...
            raise OllamaUnavailable(f"cannot reach ollama at {self.host}: {exc}") from exc

    def check(self) -> None:
        models = self.available_models()
        if self.model not in models:
            raise OllamaUnavailable(
                f"model {self.model!r} not present; `ollama pull {self.model}`. "
                f"Available: {models}"
            )

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        system: str,
        user: str,
        *,
        images: list[np.ndarray] | None = None,
        max_tokens: int = 192,
        json_mode: bool = True,
    ) -> ChatResult:
        message: dict = {"role": "user", "content": user}
        if images:
            message["images"] = [_encode_png(img) for img in images]

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, message],
            "stream": False,
            # Qwen3 emits chain-of-thought into a separate `thinking` field and will
            # otherwise consume the whole token budget there, returning empty content.
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
            },
        }
        if json_mode:
            payload["format"] = "json"
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        if self.num_thread is not None:
            payload["options"]["num_thread"] = self.num_thread

        r = self._session.post(
            f"{self.host}/api/chat", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        return ChatResult(
            content=msg.get("content", "") or "",
            thinking=msg.get("thinking", "") or "",
            eval_count=int(data.get("eval_count", 0) or 0),
            duration_s=float(data.get("total_duration", 0) or 0) / 1e9,
        )


def _encode_png(arr: np.ndarray) -> str:
    from PIL import Image

    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")
    # The Game Boy screen is 160x144; upscale so the model's patch tokeniser has
    # something to work with.
    img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction.

    Ollama's `format: "json"` is usually exact, but a model that stops early can emit a
    truncated object; falling back to a brace scan keeps one bad response from killing
    the proposer thread.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}
