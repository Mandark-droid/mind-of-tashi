"""OpenAI-compatible teacher — covers OpenRouter, Mistral, and Sarvam.

All three speak `/v1/chat/completions` with the OpenAI request/response shape,
so we use the official `openai` AsyncOpenAI client with a per-provider
`base_url`. Going through the SDK (rather than raw aiohttp) means
`genai-otel-instrument`'s OpenAIInstrumentor auto-emits `gen_ai.*` spans
with token counts and latency, which the cost + eval enrichers consume.
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
        # Lazy import so non-openai-compat runs don't pay the import cost.
        from openai import AsyncOpenAI  # type: ignore

        self.name = provider
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} is not set; cannot use {provider}:{model} teacher"
            )
        # OpenRouter expects a Referer + Title header for attribution.
        default_headers: Dict[str, str] = {}
        if provider == "openrouter":
            default_headers["HTTP-Referer"] = "https://github.com/mind-of-tashi"
            default_headers["X-Title"] = "Mind of Tashi self-play"
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=120.0,
            max_retries=0,  # we own retry in Teacher.choose
            default_headers=default_headers or None,
        )

    async def aclose(self) -> None:
        try:
            await self._client.close()
        except Exception:
            pass

    async def _choose_async(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        messages = build_messages(opp, state)

        # APIError + status-code import is local: openai.APIStatusError covers
        # HTTP errors and APIConnectionError covers transport.
        from openai import APIConnectionError, APIStatusError, RateLimitError  # type: ignore

        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature_for(opp),
                top_p=0.9,
                # API providers can afford a generous budget; the local llama.cpp
                # path uses tighter caps. Without slack here the model gets cut
                # off mid-<think> and never emits the JSON line — parse_reply
                # would silently fall back to GUARD.
                max_tokens=opp.think_tokens * 3 + 200,
            )
        except RateLimitError as exc:
            raise RetryableError(f"{self.provider} rate-limited: {exc}") from exc
        except APIStatusError as exc:
            if exc.status_code in (408, 429, 500, 502, 503, 504):
                raise RetryableError(
                    f"{self.provider} HTTP {exc.status_code}: {str(exc)[:200]}"
                ) from exc
            raise RuntimeError(
                f"{self.provider} HTTP {exc.status_code}: {str(exc)[:300]}"
            ) from exc
        except APIConnectionError as exc:
            raise RetryableError(f"{self.provider} connection error: {exc}") from exc

        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        # Some providers (DeepSeek-R1 family, OpenAI o1) split <think> into a
        # separate `reasoning_content` field. The openai SDK exposes unknown
        # fields via model_extra (Pydantic). Pull both names defensively.
        extras = (msg.model_extra or {}) if hasattr(msg, "model_extra") else {}
        reasoning_extra = extras.get("reasoning_content") or extras.get("reasoning") or ""
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

        usage = resp.usage
        return ChoiceResult(
            parsed=parsed,
            raw=raw,
            meta={
                "backend": self.provider,
                "model": self.model,
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "finish_reason": choice.finish_reason,
            },
        )
