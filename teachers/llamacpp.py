"""Local llama.cpp teacher — wraps the live-game Reasoner for self-play use."""

from __future__ import annotations
import os
from typing import Any, Dict, List

from opponents import Opponent

from .base import ChoiceResult, Teacher, synthesize_raw


class LlamaCppTeacher(Teacher):
    """Wraps llm.Reasoner. Sync underneath; the base class runs it in a thread.

    Configuration via env vars (same as the live game):
        MODEL_REPO, MODEL_FILE, MODEL_N_CTX, MODEL_N_THREADS, N_GPU_LAYERS
    """

    name = "llamacpp"
    max_retries = 0  # local; no transient errors worth retrying on

    def __init__(self) -> None:
        # imported lazily so requirements don't force llama_cpp at module load
        from llm import Reasoner
        self._reasoner = Reasoner()
        self._model = f"{os.environ.get('MODEL_REPO', '?')}/{os.environ.get('MODEL_FILE', '?')}"

    def _choose_sync(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        parsed, raw = self._reasoner.choose_with_raw(opp, state)
        return ChoiceResult(
            parsed=parsed,
            raw=raw,
            meta={
                "backend": "llamacpp",
                "model": self._model,
                "reasoner_backend": self._reasoner.backend,
            },
        )
