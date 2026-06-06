"""Google Gemini teacher — uses the google-genai async client.

Gemini's chat shape differs from OpenAI's: system instructions go into a
top-level `system_instruction`, the user turn is a `contents` list, and the
response holds `text` + (optionally) thought parts. We coerce to the same
<think>...</think>{json} raw output the rest of the pipeline expects.
"""

from __future__ import annotations
import asyncio
import os
from typing import Any, Dict, List, Optional

import prompts
from opponents import Opponent

from .base import (
    ChoiceResult,
    RetryableError,
    Teacher,
    legal_moves,
    temperature_for,
)


# Per-request hard timeout. google-genai's async client has no default timeout
# and will block forever on a stalled connection — that's the root cause of the
# 2026-05-28 harvest hang. The pool already wraps each teacher in its own retry
# loop, so a 90s cap here is generous enough for slow models (gemini-2.5-pro on
# a long context) without letting a single bad call wedge the harness.
_REQUEST_TIMEOUT_S = 90.0


class GeminiTeacher(Teacher):
    name = "gemini"

    def __init__(self, model: str) -> None:
        self.model = model
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; cannot use gemini teacher")
        # Lazy import so projects without google-genai installed can still use
        # other backends.
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-genai package not installed. "
                "Add `google-genai` to requirements.txt or `pip install google-genai`."
            ) from exc
        self._genai = genai
        self._client = genai.Client(api_key=self.api_key)

    async def _choose_async(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        system_text = prompts.build_system(opp)
        user_text = prompts.build_user(opp, state, legal)

        genai_types = self._genai.types
        # IMPORTANT — Gemini 2.5 Flash has implicit hidden "thinking" enabled by
        # default that consumes tokens BEFORE any visible output, and the
        # max_output_tokens cap includes those hidden thinking tokens. If we
        # leave the persona's think_tokens budget as-is, the visible <think>
        # block gets truncated mid-sentence and the trailing JSON is never
        # emitted — parse_reply silently falls back to GUARD. We disable
        # hidden thinking entirely (thinking_budget=0) and give a generous
        # visible budget instead, so the persona's full mind-scroll + JSON
        # land within the cap.
        try:
            thinking_config = genai_types.ThinkingConfig(thinking_budget=0)
        except (AttributeError, TypeError):
            # older google-genai versions
            thinking_config = None
        config_kwargs = dict(
            temperature=temperature_for(opp),
            top_p=0.9,
            max_output_tokens=opp.think_tokens * 3 + 200,
            system_instruction=system_text,
        )
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        config = genai_types.GenerateContentConfig(**config_kwargs)

        try:
            resp = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.model,
                    contents=user_text,
                    config=config,
                ),
                timeout=_REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            # Treat client-side timeout as retryable so the pool can fail over
            # to the next teacher rather than crash the whole match.
            raise RetryableError(
                f"gemini {self.model} timed out after {_REQUEST_TIMEOUT_S}s"
            ) from exc
        except Exception as exc:
            # Gemini SDK raises various provider errors; treat 429/5xx-like as retryable.
            msg = str(exc).lower()
            if any(s in msg for s in ("429", "quota", "rate", "503", "unavailable", "timeout")):
                raise RetryableError(f"gemini transient: {exc}") from exc
            raise

        # The SDK exposes either a flat `.text` or a candidate parts list. If a
        # thinking part exists (Flash with thinking enabled), pull it out so the
        # raw blob keeps the <think> shape.
        thought_text = ""
        body_text = ""
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", None) or []
            for p in parts:
                if getattr(p, "thought", False):
                    thought_text += getattr(p, "text", "") or ""
                else:
                    body_text += getattr(p, "text", "") or ""
        if not body_text:
            body_text = getattr(resp, "text", "") or ""

        if thought_text and "<think>" not in body_text:
            raw = f"<think>{thought_text}</think>\n{body_text}"
        else:
            raw = body_text

        parsed = prompts.parse_reply(raw, legal)

        usage = getattr(resp, "usage_metadata", None)
        meta: Dict[str, Any] = {
            "backend": "gemini",
            "model": self.model,
        }
        if usage is not None:
            meta["prompt_tokens"] = getattr(usage, "prompt_token_count", None)
            meta["completion_tokens"] = getattr(usage, "candidates_token_count", None)
            meta["thoughts_tokens"] = getattr(usage, "thoughts_token_count", None)

        return ChoiceResult(parsed=parsed, raw=raw, meta=meta)
