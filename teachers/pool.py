"""
pool.py — a Teacher that rotates across several underlying teachers based on
per-spec daily quota.

Usage:

    pool = make_pool(
        specs=[
            "gemini:gemini-2.5-flash",
            "gemini:gemini-2.5-flash-lite",
            "gemini:gemini-2.5-pro",
            "mistral:mistral-large-latest",
            "openrouter:meta-llama/llama-3.3-70b-instruct:free",
        ],
        quota_path=Path("data/selfplay/.quota.json"),
    )
    result = await pool.choose(opp, state)

The pool calls `quota.reserve(spec)` before each underlying call; on 429 or a
fallback whose error mentions quota, it marks the spec exhausted for the day
and tries the next one. When every spec is unavailable, `PoolExhausted` is
raised — the harness exits cleanly, the JSONL is intact, and tomorrow's
midnight UTC rollover lets the harvest resume.
"""

from __future__ import annotations
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from opponents import Opponent

from .base import ChoiceResult, RetryableError, Teacher
from .quota import QuotaState


class PoolExhausted(RuntimeError):
    """All teachers in the pool are out of quota or have failed for the day."""


class TeacherPool(Teacher):
    """Rotates across a list of teachers, respecting per-spec daily quotas.

    Members are constructed once at pool-creation time; the pool sets
    `max_retries = 0` on each so they fail fast on quota errors (the pool
    handles failover, not per-teacher backoff — that would burn 30+ seconds
    per dead spec).
    """

    name = "pool"
    max_retries = 0  # we handle failover across teachers ourselves

    def __init__(
        self,
        teachers: Sequence[Teacher],
        specs: Sequence[str],
        quota: QuotaState,
    ) -> None:
        if len(teachers) != len(specs):
            raise ValueError("teachers and specs must align by index")
        if not teachers:
            raise ValueError("pool needs at least one teacher")
        self.teachers: List[Teacher] = list(teachers)
        self.specs: List[str] = list(specs)
        self.quota = quota
        self._cursor = itertools.cycle(range(len(self.teachers)))
        # Fast-fail underlying teachers; the pool does the multi-spec retry.
        for t in self.teachers:
            t.max_retries = 0

    async def _dispatch(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        tried: List[str] = []
        n = len(self.teachers)
        # Try at most one full rotation per call so we never loop forever on
        # a fully-exhausted pool.
        for _ in range(n):
            i = next(self._cursor)
            spec = self.specs[i]
            if spec in tried:
                continue
            tried.append(spec)

            reserved = await self.quota.reserve(spec)
            if not reserved:
                continue

            teacher = self.teachers[i]
            try:
                result = await teacher.choose(opp, state)
            except RetryableError as exc:
                # underlying teacher signalled a transient error; pool treats
                # quota-flavoured ones as a kill switch for the day.
                if _looks_like_quota_error(str(exc)):
                    await self.quota.mark_exhausted(spec)
                else:
                    await self.quota.refund(spec)
                continue
            except Exception as exc:
                # non-retryable — refund and move on
                await self.quota.refund(spec)
                continue

            # base.Teacher swallows retry-exhausted as a fallback; inspect it.
            if result.meta.get("fallback"):
                err = str(result.meta.get("error", ""))
                if _looks_like_quota_error(err):
                    await self.quota.mark_exhausted(spec)
                else:
                    await self.quota.refund(spec)
                continue

            result.meta["pool_spec"] = spec
            return result

        raise PoolExhausted(
            f"all pool specs unavailable today (tried {tried})"
        )

    async def aclose(self) -> None:
        for t in self.teachers:
            try:
                await t.aclose()
            except Exception:
                pass


def _looks_like_quota_error(text: str) -> bool:
    lo = text.lower()
    return any(
        s in lo
        for s in ("429", "quota", "rate", "exhausted", "resource_exhausted")
    )


def make_pool(
    specs: Sequence[str],
    quota_path: Path,
    custom_limits: Optional[Dict[str, int]] = None,
) -> TeacherPool:
    """Build a pool from a list of teacher specs. Imports `make_teacher`
    lazily to avoid a circular import."""
    from . import make_teacher
    quota = QuotaState(quota_path, custom_limits=custom_limits)
    teachers = [make_teacher(s) for s in specs]
    return TeacherPool(teachers, list(specs), quota)
