"""Heuristic teacher — wraps llm._mock_choose so the harness runs offline."""

from __future__ import annotations
from typing import Any, Dict, List

import llm
from opponents import Opponent

from .base import ChoiceResult, Teacher, synthesize_raw


class MockTeacher(Teacher):
    name = "mock"

    def _choose_sync(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        parsed = llm._mock_choose(opp, state, legal)
        return ChoiceResult(
            parsed=parsed,
            raw=synthesize_raw(parsed),
            meta={"backend": "mock", "model": "heuristic"},
        )
