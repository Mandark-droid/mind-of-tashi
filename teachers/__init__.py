"""
teachers/ — backend-agnostic "give me a move" interface for self-play harvesting.

Two surfaces:

  - the live game (app.py) keeps using llm.Reasoner (sync, llama.cpp).
  - the self-play harness (tools/selfplay.py) uses a Teacher per side, async,
    so a 5x5 matchup sweep can fire both sides' blind-commit calls in parallel
    against API providers (Gemini / OpenRouter / Mistral / Sarvam) for 2x
    throughput on free-tier quotas.

The shared `Teacher.choose(opp, state)` returns a ChoiceResult dataclass with
parsed move + raw model text + per-call metadata. The raw text is the eventual
SFT target, so we preserve it verbatim across all backends — for backends that
don't natively emit a single text blob (mock, llamacpp), we synthesise an
equivalent.

Spec strings (the `--a-teacher` / `--b-teacher` CLI args):

    mock                                     — heuristic; no network, no model
    llamacpp                                 — local llama.cpp via env vars
    ollama:<model>                           — local Ollama, e.g. ollama:qwen3:14b
    gemini:<model>                           — Google Gemini, e.g. gemini:gemini-2.0-flash-exp
    openrouter:<author>/<model>[:<variant>]  — e.g. openrouter:meta-llama/llama-3.3-70b-instruct:free
    mistral:<model>                          — e.g. mistral:mistral-large-latest
    sarvam:<model>                           — e.g. sarvam:sarvam-m
"""

from __future__ import annotations
from typing import Optional

from .base import ChoiceResult, Teacher
from .pool import PoolExhausted, TeacherPool, make_pool
from .quota import QuotaState


def make_teacher(spec: str) -> Teacher:
    """Parse a spec string into a Teacher instance. See module docstring."""
    if not spec or not isinstance(spec, str):
        raise ValueError(f"teacher spec must be a non-empty string, got {spec!r}")

    head, _, tail = spec.partition(":")
    head = head.strip().lower()

    if head == "mock":
        from .mock import MockTeacher
        return MockTeacher()

    if head == "llamacpp":
        from .llamacpp import LlamaCppTeacher
        return LlamaCppTeacher()

    if head == "ollama":
        if not tail:
            raise ValueError("ollama spec needs a model: ollama:<model>")
        from .ollama import OllamaTeacher
        return OllamaTeacher(model=tail)

    if head == "gemini":
        if not tail:
            raise ValueError("gemini spec needs a model: gemini:<model>")
        from .gemini import GeminiTeacher
        return GeminiTeacher(model=tail)

    if head in ("openrouter", "mistral", "sarvam"):
        if not tail:
            raise ValueError(f"{head} spec needs a model: {head}:<model>")
        from .openai_compat import OpenAICompatTeacher, PROVIDER_CONFIG
        cfg = PROVIDER_CONFIG[head]
        return OpenAICompatTeacher(
            provider=head, model=tail,
            base_url=cfg["base_url"],
            api_key_env=cfg["api_key_env"],
        )

    raise ValueError(
        f"unknown teacher spec {spec!r}; "
        f"valid heads: mock, llamacpp, ollama, gemini, openrouter, mistral, sarvam"
    )


__all__ = [
    "Teacher", "ChoiceResult", "make_teacher",
    "TeacherPool", "PoolExhausted", "make_pool", "QuotaState",
]
