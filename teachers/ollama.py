"""Local Ollama teacher — uses the official `ollama` Python AsyncClient.

We deliberately use the SDK (not raw aiohttp) so `genai-otel-instrument`'s
built-in OllamaInstrumentor picks it up automatically and emits proper
`gen_ai.*` spans with model / token-count / latency attributes — which the
cost + eval enrichers then key off.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

import prompts
from opponents import Opponent

from .base import (
    ChoiceResult,
    RetryableError,
    Teacher,
    build_messages,
    temperature_for,
)


DEFAULT_OLLAMA_PORT = 11434


def _normalize_host(raw: str) -> str:
    """Coerce a possibly-misshapen OLLAMA_HOST into a valid client URL.

    Same rules as before: prepend scheme, rewrite 0.0.0.0 → 127.0.0.1,
    append :11434 if no port. The `ollama` Python client accepts the same
    URL shape aiohttp did, so the existing normalization carries over.
    """
    s = (raw or "").strip()
    if not s:
        return f"http://127.0.0.1:{DEFAULT_OLLAMA_PORT}"
    if "://" not in s:
        s = "http://" + s
    scheme, _, rest = s.partition("://")
    if "/" in rest:
        authority, _, tail = rest.partition("/")
        tail = "/" + tail
    else:
        authority, tail = rest, ""
    if authority.startswith("0.0.0.0"):
        authority = authority.replace("0.0.0.0", "127.0.0.1", 1)
    if ":" not in authority:
        authority = f"{authority}:{DEFAULT_OLLAMA_PORT}"
    return f"{scheme}://{authority}{tail}".rstrip("/")


class OllamaTeacher(Teacher):
    name = "ollama"
    # See aiohttp version's note: cold-load + retry burn doesn't pay back here.
    max_retries = 1

    def __init__(self, model: str, host: Optional[str] = None) -> None:
        # Lazy import so `mock`/`gemini`-only runs don't pay for ollama install.
        from ollama import AsyncClient  # type: ignore

        self.model = model
        raw_host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.host = _normalize_host(raw_host)
        # 600s timeout matches the previous aiohttp value — covers cold model
        # loads (qwen3:4b ~100s first call) and long reasoning generations.
        self._client = AsyncClient(host=self.host, timeout=600)

    async def aclose(self) -> None:
        # ollama.AsyncClient owns an httpx.AsyncClient under the hood; expose
        # its close for clean shutdown.
        try:
            inner = getattr(self._client, "_client", None)
            if inner is not None and hasattr(inner, "aclose"):
                await inner.aclose()
        except Exception:
            pass

    async def _choose_async(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        messages = build_messages(opp, state)
        try:
            resp = await self._client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options={
                    "temperature": temperature_for(opp),
                    "top_p": 0.9,
                    "num_predict": opp.think_tokens + 80,
                },
            )
        except Exception as exc:
            # The ollama SDK raises ResponseError for HTTP failures and
            # httpx.ConnectError / TimeoutException for transport issues.
            # Treat all of them as retryable so the harness backs off.
            raise RetryableError(f"ollama client error: {exc}") from exc

        # Response is a Pydantic model on ollama>=0.4. `model_dump` handles
        # both that and the older dict-style return.
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        msg = data.get("message") or {}
        content = msg.get("content") or ""
        thinking = msg.get("thinking") or ""
        # qwen3 et al. surface reasoning under `message.thinking`. Stitch
        # into <think>…</think>{json} so prompts.parse_reply finds it.
        if thinking and "<think>" not in content:
            raw = f"<think>{thinking.strip()}</think>\n{content.lstrip()}"
        else:
            raw = content
        parsed = prompts.parse_reply(raw, legal)
        return ChoiceResult(
            parsed=parsed,
            raw=raw,
            meta={
                "backend": "ollama",
                "model": self.model,
                "host": self.host,
                "eval_count": data.get("eval_count"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "finish_reason": data.get("done_reason"),
            },
        )
