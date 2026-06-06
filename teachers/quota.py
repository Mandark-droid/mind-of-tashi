"""
quota.py — file-backed daily-quota tracker for teacher pools.

The Gemini free tier is 50 requests/day **per model ID** — so the way to
sustain a multi-day SFT harvest is to rotate across several Gemini models
(2.5-flash, 2.5-flash-lite, 2.5-pro, 3.0-flash, 3.5-flash, ...) plus
OpenRouter / Mistral / Sarvam free tiers. This module keeps a per-spec
counter on disk so the harness can stop cold on a spec when its bucket is
empty and switch to another, and can resume the next day after the UTC
midnight reset.

Concurrency: a single asyncio.Lock guards the in-memory state and the
atomic file write. Multi-process safety is NOT provided — run one harvest
at a time.
"""

from __future__ import annotations
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


# Conservative defaults — we'd rather under-spend than hit a 429 mid-match.
# Override per-spec via QuotaState(custom_limits=...).
DEFAULT_DAILY_LIMITS: Dict[str, int] = {
    "gemini":     50,        # per Gemini model id
    "openrouter": 200,       # :free variants vary; pessimistic default
    "mistral":    100,
    "sarvam":     100,
    "ollama":     10**9,     # effectively unlimited
    "llamacpp":   10**9,
    "mock":       10**9,
}


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class QuotaState:
    """File-backed per-spec daily counter.

    `spec` is the same string used by `teachers.make_teacher` — e.g.
    "gemini:gemini-2.5-flash" or "openrouter:meta-llama/llama-3.3-70b-instruct:free".
    The provider key (text before the first colon) determines the default
    daily limit unless overridden in `custom_limits`.
    """

    def __init__(
        self,
        path: Path,
        custom_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        self.path = Path(path)
        self.custom_limits = dict(custom_limits or {})
        self._lock = asyncio.Lock()
        self._state = self._load()

    # ----- internal ------------------------------------------------------ #

    def _fresh(self) -> dict:
        return {
            "date_utc": _today_utc(),
            "used": {},               # spec -> int
            "exhausted_until_eod": [], # specs marked dead for the day on 429
        }

    def _load(self) -> dict:
        if not self.path.exists():
            return self._fresh()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self._fresh()
        if data.get("date_utc") != _today_utc():
            return self._fresh()
        # normalise
        data.setdefault("used", {})
        data.setdefault("exhausted_until_eod", [])
        return data

    def _refresh_if_stale(self) -> None:
        if self._state.get("date_utc") != _today_utc():
            self._state = self._fresh()
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _limit_for(self, spec: str) -> int:
        if spec in self.custom_limits:
            return self.custom_limits[spec]
        provider = spec.split(":", 1)[0]
        return DEFAULT_DAILY_LIMITS.get(provider, 50)

    # ----- public -------------------------------------------------------- #

    async def reserve(self, spec: str) -> bool:
        """Reserve one call against `spec`. Returns True if reserved, False if
        the bucket is empty or the spec is exhausted-until-EOD."""
        async with self._lock:
            self._refresh_if_stale()
            if spec in self._state["exhausted_until_eod"]:
                return False
            used = self._state["used"].get(spec, 0)
            if used >= self._limit_for(spec):
                return False
            self._state["used"][spec] = used + 1
            self._save()
            return True

    async def refund(self, spec: str) -> None:
        """Give back a previously-reserved call (used when a downstream error
        means the call never actually billed)."""
        async with self._lock:
            self._refresh_if_stale()
            cur = self._state["used"].get(spec, 0)
            if cur > 0:
                self._state["used"][spec] = cur - 1
                self._save()

    async def mark_exhausted(self, spec: str) -> None:
        """Mark a spec dead for the rest of the UTC day (e.g. on 429 even though
        our counter said budget remained — the provider's truth wins)."""
        async with self._lock:
            self._refresh_if_stale()
            if spec not in self._state["exhausted_until_eod"]:
                self._state["exhausted_until_eod"].append(spec)
                self._save()

    def snapshot(self) -> dict:
        """Read-only view of current state (date, used per spec, exhausted)."""
        # caller doesn't need the lock for a snapshot read
        used = dict(self._state.get("used", {}))
        exhausted = list(self._state.get("exhausted_until_eod", []))
        return {
            "date_utc": self._state.get("date_utc", _today_utc()),
            "used": used,
            "exhausted_until_eod": exhausted,
            "remaining": {
                spec: max(0, self._limit_for(spec) - used.get(spec, 0))
                for spec in set(used) | set(exhausted) | set(self.custom_limits)
            },
        }
