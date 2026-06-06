"""Local Ollama teacher — async HTTP client to 127.0.0.1:11434 (no SDK)."""

from __future__ import annotations
import asyncio
import os
from typing import Any, Dict, List, Optional

import aiohttp

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

    Ollama's documented `OLLAMA_HOST` env var is used by both the server
    (to bind) and the CLI (to dial). Users frequently set it to
    `0.0.0.0`, `0.0.0.0:11434`, or just `localhost` — none of which work
    cleanly when handed to aiohttp:

      * `0.0.0.0` is a server bind address, not a client dial address
      * a host without scheme becomes a relative URL aiohttp rejects
      * a URL without explicit port defaults to :80, where Ollama isn't

    Rules applied in order:
      1. prepend `http://` if no scheme present
      2. rewrite `0.0.0.0` host part to `127.0.0.1`
      3. append `:11434` if no port is specified on the authority
      4. strip trailing slash
    """
    s = (raw or "").strip()
    if not s:
        return f"http://127.0.0.1:{DEFAULT_OLLAMA_PORT}"
    if "://" not in s:
        s = "http://" + s
    scheme, _, rest = s.partition("://")
    # split rest into authority (host[:port]) and the remaining path/query
    if "/" in rest:
        authority, slash, tail = rest.partition("/")
        tail = "/" + tail
    else:
        authority, tail = rest, ""
    if authority.startswith("0.0.0.0"):
        authority = authority.replace("0.0.0.0", "127.0.0.1", 1)
    if ":" not in authority:  # no port — assume Ollama default
        authority = f"{authority}:{DEFAULT_OLLAMA_PORT}"
    return f"{scheme}://{authority}{tail}".rstrip("/")


class OllamaTeacher(Teacher):
    name = "ollama"
    # Ollama's cold-load of a fresh model (especially after a swap on a
    # tight-VRAM machine) can easily eat 60-120s before any tokens land. The
    # 4-retry exponential backoff inherited from Teacher base would then
    # burn 15s of fruitless sleeping on a doomed local call. Override:
    max_retries = 1

    def __init__(self, model: str, host: Optional[str] = None) -> None:
        self.model = model
        raw_host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.host = _normalize_host(raw_host)
        self._session: Optional[aiohttp.ClientSession] = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _choose_async(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        messages = build_messages(opp, state)
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature_for(opp),
                "top_p": 0.9,
                "num_predict": opp.think_tokens + 80,
            },
        }
        session = self._ensure_session()
        try:
            async with session.post(
                f"{self.host.rstrip('/')}/api/chat",
                json=body,
                # 600s = 10 min — covers cold model loads (qwen3:4b takes
                # ~100s to load on first call) AND long reasoning generations
                # (qwen3's default `think:true` can emit 2k+ thinking tokens
                # before the visible answer).
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status in (429, 502, 503, 504):
                    raise RetryableError(f"ollama HTTP {resp.status}")
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"ollama HTTP {resp.status}: {text[:200]}")
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise RetryableError(f"ollama client error: {exc}") from exc

        # Ollama returns model output in two fields when the underlying model
        # has reasoning support (qwen3 etc.): `message.thinking` for the
        # hidden chain-of-thought, `message.content` for the visible answer.
        # Our prompt asks the model to put its reasoning inside <think>...
        # </think> in the VISIBLE output — but qwen3 often hides it in
        # `thinking` instead. Concatenate them so parse_reply sees a single
        # <think>...</think>\n{json} blob no matter which mode the model used.
        msg = data.get("message") or {}
        content = msg.get("content") or ""
        thinking = msg.get("thinking") or ""
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
