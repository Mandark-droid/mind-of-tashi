"""OpenAI-compatible teacher — covers OpenRouter, Mistral, and Sarvam.

All three expose `/v1/chat/completions` with the same request/response shape,
so one class with a provider-keyed config does the job.
"""

from __future__ import annotations
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


# Update if a provider's base_url changes. Free-tier endpoints are the same as
# paid — quota is enforced by the API key.
PROVIDER_CONFIG: Dict[str, Dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
    },
    "sarvam": {
        # Sarvam's chat-completions endpoint is OpenAI-compatible.
        "base_url": "https://api.sarvam.ai/v1",
        "api_key_env": "SARVAM_API_KEY",
    },
}


class OpenAICompatTeacher(Teacher):
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key_env: str,
    ) -> None:
        self.name = provider
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} is not set; cannot use {provider}:{model} teacher"
            )
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
            "temperature": temperature_for(opp),
            "top_p": 0.9,
            # API providers can afford a generous budget; the local llama.cpp
            # path uses tighter caps. Without slack here the model gets cut
            # off mid-<think> and never emits the JSON line — parse_reply
            # would silently fall back to GUARD.
            "max_tokens": opp.think_tokens * 3 + 200,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter encourages a referer + app title.
        if self.provider == "openrouter":
            headers.setdefault("HTTP-Referer", "https://github.com/mind-of-tashi")
            headers.setdefault("X-Title", "Mind of Tashi self-play")

        session = self._ensure_session()
        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status in (408, 429, 500, 502, 503, 504):
                    body_text = await resp.text()
                    raise RetryableError(
                        f"{self.provider} HTTP {resp.status}: {body_text[:200]}"
                    )
                if resp.status >= 400:
                    body_text = await resp.text()
                    raise RuntimeError(
                        f"{self.provider} HTTP {resp.status}: {body_text[:300]}"
                    )
                data = await resp.json()
        except aiohttp.ClientError as exc:
            raise RetryableError(f"{self.provider} client error: {exc}") from exc

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        # Some providers (DeepSeek-R1 family, OpenAI o1) split <think> into a
        # separate `reasoning_content` field. Concatenate so parse_reply finds it.
        reasoning_extra = msg.get("reasoning_content") or msg.get("reasoning") or ""
        content = msg.get("content", "") or ""
        if reasoning_extra and "<think>" not in content:
            raw = f"<think>{reasoning_extra}</think>\n{content}"
        else:
            raw = content
        # Repair providers that emit <think> but drop the closing tag and
        # run straight into the JSON line. Seen heavily on gpt-oss-120b:
        # the reasoning is rich but the tag is unbalanced, which breaks
        # downstream split-on-</think> and silently turns valuable rows
        # into zero-think SFT examples. Insert </think> immediately
        # before the JSON object so the rest of the pipeline can parse.
        if "<think>" in raw and "</think>" not in raw:
            j = raw.rfind("{")
            if j > raw.find("<think>"):
                raw = raw[:j] + "</think>\n" + raw[j:]
        parsed = prompts.parse_reply(raw, legal)

        usage = data.get("usage", {}) or {}
        return ChoiceResult(
            parsed=parsed,
            raw=raw,
            meta={
                "backend": self.provider,
                "model": self.model,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
            },
        )
