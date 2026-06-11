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

    def __init__(self, repo: str | None = None, filename: str | None = None) -> None:
        # imported lazily so requirements don't force llama_cpp at module load.
        # repo/filename let a self-play challenger load a different GGUF than
        # the house model; backend is pinned to llamacpp so a Space running
        # BACKEND=transformers doesn't route challengers through its singleton.
        from llm import Reasoner
        self._reasoner = Reasoner(repo=repo, filename=filename, backend="llamacpp")
        self._model = (f"{repo or os.environ.get('MODEL_REPO', '?')}"
                       f"/{filename or os.environ.get('MODEL_FILE', '?')}")

    def _choose_sync(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        parsed, raw = self._reasoner.choose_with_raw(opp, state)
        return ChoiceResult(
            parsed=parsed,
            raw=raw,
            meta={
                "backend": self.name,
                "model": self._model,
                "reasoner_backend": self._reasoner.backend,
                "reasoner_error": self._reasoner.load_error,
            },
        )


class TransformersTeacher(LlamaCppTeacher):
    """Same wrapper, transformers backend — safetensors on (Zero)GPU.

    The self-play challenger fallback when llama.cpp isn't available in the
    runtime (e.g. the wheel failed to install on a Space). Each repo gets its
    own entry in llm._TF_CACHE, so challengers never clobber the house model.
    """

    name = "transformers"

    def __init__(self, repo: str) -> None:
        from llm import Reasoner
        # tf_device='cpu': challengers are built at REQUEST time, and ZeroGPU
        # forbids cuda init outside @spaces.GPU after boot. llm._tf_generate
        # hops the model onto the GPU inside the decorated call instead.
        self._reasoner = Reasoner(backend="transformers", tf_repo=repo,
                                  tf_device="cpu")
        self._model = repo
