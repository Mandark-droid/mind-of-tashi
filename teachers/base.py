"""
Shared abstraction across teacher backends.
"""

from __future__ import annotations
import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import prompts
from engine import MOVES
from opponents import Opponent


@dataclass
class ChoiceResult:
    parsed: Dict[str, Any]      # {"reasoning", "move", "taunt"} — the parsed move
    raw: str                    # full text the model emitted (or synthesised) — SFT target
    meta: Dict[str, Any] = field(default_factory=dict)
    # meta typically: provider, model, latency_ms, prompt_tokens, completion_tokens,
    #                 retries, finish_reason


def synthesize_raw(parsed: Dict[str, Any]) -> str:
    """Build a <think>...</think>{json} string from a parsed dict.

    Used by backends (mock, llamacpp) where we got the parsed dict directly
    rather than a single text blob. Keeps the harness output uniform.
    """
    obj = {"move": parsed["move"], "taunt": parsed["taunt"]}
    return f"<think>{parsed['reasoning']}</think>\n{json.dumps(obj, ensure_ascii=False)}"


def temperature_for(opp: Opponent) -> float:
    """Same scaling as the live llm._llm_choose path — brawlers run hotter."""
    return 0.7 + 0.05 * (5 - opp.difficulty)


def legal_moves(prana: int) -> List[str]:
    return [m for m in MOVES if prana >= MOVES[m]["cost"]]


def build_messages(opp: Opponent, state: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build the chat-style messages list. Identical to live llm.Reasoner."""
    legal = legal_moves(state["ai_prana"])
    return [
        {"role": "system", "content": prompts.build_system(opp)},
        {"role": "user", "content": prompts.build_user(opp, state, legal)},
    ]


class Teacher(ABC):
    """Backend-agnostic move generator.

    Subclasses implement either `_choose_async` (preferred for API backends) or
    `_choose_sync` (for in-process backends). The default `choose` dispatches
    appropriately and handles retries + latency tracking.
    """

    name: str = "teacher"
    max_retries: int = 4
    retry_base_delay: float = 1.0  # seconds; exponential backoff

    async def choose(self, opp: Opponent, state: Dict[str, Any]) -> ChoiceResult:
        legal = legal_moves(state["ai_prana"])
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                result = await self._dispatch(opp, state, legal)
                result.meta.setdefault("latency_ms", int((time.time() - t0) * 1000))
                result.meta.setdefault("retries", attempt)
                result.meta.setdefault("provider", self.name)
                return result
            except RetryableError as exc:
                last_err = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)
            except Exception as exc:  # non-retryable
                last_err = exc
                break
        # All retries exhausted → fall back to a safe GUARD so the match doesn't stall.
        fallback = {
            "reasoning": f"(teacher {self.name} failed: {type(last_err).__name__})",
            "move": "GUARD" if "GUARD" in legal else (legal[0] if legal else "GUARD"),
            "taunt": "...",
        }
        return ChoiceResult(
            parsed=fallback,
            raw=synthesize_raw(fallback),
            meta={
                "provider": self.name,
                "fallback": True,
                "error": f"{type(last_err).__name__}: {last_err}",
            },
        )

    async def _dispatch(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        """Default: if subclass overrode _choose_async use it; else _choose_sync."""
        if type(self)._choose_async is not Teacher._choose_async:  # overridden
            return await self._choose_async(opp, state, legal)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._choose_sync, opp, state, legal)

    async def _choose_async(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        raise NotImplementedError

    def _choose_sync(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        """Override to release resources (sessions, sockets) when the harness exits."""
        return None


class RetryableError(Exception):
    """Signals that the call should be retried with backoff (e.g. 429, 503, timeout)."""
