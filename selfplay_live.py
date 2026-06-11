"""
selfplay_live.py — opt-in "Tashi vs Tashi" mode where the LIVE game runs two
LLMs against each other through the UI. Useful for visual demos and for
sanity-checking the bilingual <think> register on local models without
burning API quota.

Enable via env (in .env or Spaces Variables):

    SELFPLAY_MODE=1
    SELFPLAY_PLAYER_TEACHER=ollama:qwen3:14b      # optional; see defaults
    SELFPLAY_OPPONENT_TEACHER=house               # optional; see defaults

Both specs go through `teachers.make_teacher` so any backend in the pool
works — `llamacpp:`, `ollama:`, `gemini:`, `openrouter:`, `mistral:`,
`sarvam:`, even `mock`. The special opponent spec `house` keeps the Space's
own Reasoner (the deployed mind of Tashi) on the opponent side — app.ai_turn
checks OPPONENT_IS_HOUSE and skips the teacher entirely, so watch-mode
matches keep the Conviction Meter / composure / grammar-locked Oath path.

OFF-THE-GRID GUARD: on a deployed Space (SPACE_ID set) cloud/API teacher
specs are refused and coerced to local llama.cpp — the Space never phones a
cloud provider, in self-play or otherwise. Same contract as the gym's
hard-refusal of `OPPONENT_BACKEND=api`.

CHALLENGER ROSTER: the watcher picks who climbs the ladder. Every entry is
a GGUF pulled from the Hub and run in-process via llama.cpp (CPU), so the
roster is Space-safe by construction — and it deliberately fields OpenBMB's
MiniCPM5-1B and NVIDIA's Nemotron-3-Nano-4B next to our own SFT/GRPO micro
models, so you can watch a 0.4B MoE out-read (or get out-read by) the
sponsor-class smalls.

In self-play matches `live_traces` capture is skipped — these are two-AI
play, not real player data, and shouldn't be confused at SFT-prep time. The
synthetic `tools/selfplay.py` harness is the right pipeline for SFT-bound
self-play; this module is purely for visual demo.
"""

from __future__ import annotations
import os
import threading
from typing import Any, Dict, Optional

from opponents import Opponent
from otel_bootstrap import init_otel


SELFPLAY_MODE = os.environ.get("SELFPLAY_MODE", "0") == "1"
ON_SPACE = bool(os.environ.get("SPACE_ID"))

# Instrument up front (cheap no-op when disabled) so the LLM teachers we
# lazily construct later get auto-traced. Safe to call regardless of
# SELFPLAY_MODE — init_otel is idempotent. NEVER on a Space: the default
# OTLP endpoint is the dev VM, and an Off-the-Grid Space exports nothing.
if SELFPLAY_MODE and not ON_SPACE:
    init_otel(service_name="mind-of-tashi-selfplay-live")

_LOCAL_HEADS = ("mock", "llamacpp", "transformers", "house")


def _space_safe(spec: str, fallback: str = "llamacpp") -> str:
    """Coerce cloud/API teacher specs to a local one on a deployed Space."""
    if not ON_SPACE:
        return spec
    head = spec.partition(":")[0].strip().lower()
    if head in _LOCAL_HEADS:
        return spec
    print(f"[selfplay] {spec!r} is a cloud/API teacher — refused on a Space "
          f"(Off-the-Grid). Using {fallback!r}.")
    return fallback


# The watcher-facing challenger roster: who plays the PLAYER side. Primary
# spec is a llama.cpp GGUF on CPU; tf_spec is the transformers/(Zero)GPU
# fallback used automatically when llama.cpp isn't importable in the runtime
# (e.g. the wheel failed to install on a Space). Both are local inference —
# Off-the-Grid either way. Order = UI order; first is default.
CHALLENGERS: Dict[str, Dict[str, str]] = {
    "tashi-grpo": {
        "label": "Tashi micro GRPO · 0.4B MoE (ours)",
        "spec": "llamacpp:build-small-hackathon/mind-of-tashi-micro-grpo-gguf:mind-of-tashi-micro-grpo-Q4_K_M.gguf",
        "tf_spec": "transformers:build-small-hackathon/mind-of-tashi-micro-grpo",
    },
    "tashi-sft": {
        "label": "Tashi micro SFT · 0.4B MoE (ours)",
        "spec": "llamacpp:build-small-hackathon/mind-of-tashi-micro-sft-gguf:mind-of-tashi-micro-sft-Q4_K_M.gguf",
        "tf_spec": "transformers:build-small-hackathon/mind-of-tashi-micro-sft",
    },
    "minicpm5-1b": {
        "label": "MiniCPM5 1B (OpenBMB)",
        "spec": "llamacpp:openbmb/MiniCPM5-1B-GGUF:MiniCPM5-1B-Q4_K_M.gguf",
        "tf_spec": "transformers:openbmb/MiniCPM5-1B",
    },
    "nemotron-4b": {
        "label": "Nemotron 3 Nano 4B (NVIDIA)",
        "spec": "llamacpp:unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf",
        "tf_spec": "transformers:unsloth/NVIDIA-Nemotron-3-Nano-4B",
    },
}
_DEFAULT_CHALLENGER = next(iter(CHALLENGERS))

# Defaults: on a Space the player side is the GRPO challenger and the
# opponent side is `house` (the Space's own deployed mind — ZeroGPU
# transformers or whatever BACKEND says). Locally the old Ollama pair is
# kept for the 6 GB VRAM (RTX 3060) dev box: qwen3.5:4b opponent for
# stronger reads, qwen3:1.7b player for fast moves.
PLAYER_TEACHER_SPEC = _space_safe(os.environ.get(
    "SELFPLAY_PLAYER_TEACHER",
    CHALLENGERS[_DEFAULT_CHALLENGER]["spec"] if ON_SPACE else "ollama:qwen3:1.7b",
), fallback=CHALLENGERS[_DEFAULT_CHALLENGER]["spec"])
OPPONENT_TEACHER_SPEC = _space_safe(os.environ.get(
    "SELFPLAY_OPPONENT_TEACHER",
    "house" if ON_SPACE else "ollama:qwen3.5:4b",
), fallback="house")
# app.ai_turn checks this: house = use the live Reasoner for the opponent
# side of self-play matches (no second model, full conviction/oath path).
OPPONENT_IS_HOUSE = OPPONENT_TEACHER_SPEC.strip().lower() == "house"
# Player persona to role-play. "mirror" = use the same persona as the current
# opponent (mirror match, simplest). Any LADDER id (tashi/norbu/pema/drogpa/
# the-mountain) pins the player teacher to that persona for the whole run.
PLAYER_PERSONA = os.environ.get("SELFPLAY_PLAYER_PERSONA", "mirror")


_player_teacher = None
_opponent_teacher = None
_challenger_teachers: Dict[str, Any] = {}
_prewarmed: set = set()
_llamacpp_ok: Optional[bool] = None


def _llamacpp_available() -> bool:
    """True when llama_cpp actually imports (the wheel can be installed but
    fail at .so load — e.g. musl-linked builds on a glibc image)."""
    global _llamacpp_ok
    if _llamacpp_ok is None:
        try:
            import llama_cpp  # noqa: F401
            _llamacpp_ok = True
        except Exception as exc:
            print(f"[selfplay] llama_cpp unavailable: {exc}")
            _llamacpp_ok = False
    return _llamacpp_ok


def prewarm(challenger: Optional[str] = None) -> str:
    """Fire-and-forget weight download for a roster pick (default if None).

    The first watch-mode round otherwise stalls for the whole multi-hundred-MB
    fetch; warming on app boot + on picker change means the weights are
    already in the HF cache by the time the watcher clicks. Downloads the
    artifact the teacher will actually use: the GGUF when llama.cpp works in
    this runtime, the tf_spec safetensors snapshot otherwise."""
    if not SELFPLAY_MODE:
        return "selfplay off"
    name = challenger if challenger in CHALLENGERS else _DEFAULT_CHALLENGER
    entry = CHALLENGERS[name]
    spec = _space_safe(entry["spec"])
    head, _, tail = spec.partition(":")
    repo, _, fname = tail.partition(":")
    if head == "llamacpp" and repo and fname and not _llamacpp_available():
        tf_spec = entry.get("tf_spec", "")
        if tf_spec.startswith("transformers:"):
            repo, fname = tf_spec.partition(":")[2], None  # snapshot, not one file
        else:
            return "nothing to prewarm (llama.cpp unavailable, no tf_spec)"
    elif head != "llamacpp" or not repo or not fname:
        return "nothing to prewarm"
    key = f"{repo}/{fname or '*'}"
    if key in _prewarmed:
        return f"already warm: {key}"

    def _dl():
        try:
            from huggingface_hub import hf_hub_download, snapshot_download  # lazy
            if fname:
                hf_hub_download(repo_id=repo, filename=fname)
            else:
                snapshot_download(repo_id=repo)
            print(f"[selfplay] prewarmed {key}")
        except Exception as exc:
            _prewarmed.discard(key)  # let a later request retry
            print(f"[selfplay] prewarm failed for {key}: {exc}")

    _prewarmed.add(key)
    threading.Thread(target=_dl, daemon=True).start()
    return f"warming: {key}"


def status() -> Dict[str, Any]:
    """Surfaced into the page meta so the UI can decide whether to show
    the "Watch self-play" button (and which challengers to offer)."""
    return {
        "enabled": bool(SELFPLAY_MODE),
        "player_teacher": PLAYER_TEACHER_SPEC if SELFPLAY_MODE else None,
        "opponent_teacher": OPPONENT_TEACHER_SPEC if SELFPLAY_MODE else None,
        "player_persona": PLAYER_PERSONA if SELFPLAY_MODE else None,
        "challengers": (
            [{"id": k, "label": v["label"]} for k, v in CHALLENGERS.items()]
            if SELFPLAY_MODE else []
        ),
    }


def _ensure_teachers():
    """Lazy-construct on first call so the module imports cheaply when
    self-play is disabled (the teachers/* imports pull in aiohttp, etc.)."""
    global _player_teacher, _opponent_teacher
    if not SELFPLAY_MODE:
        return
    if _player_teacher is None:
        from teachers import make_teacher  # lazy
        _player_teacher = make_teacher(PLAYER_TEACHER_SPEC)
    if _opponent_teacher is None and not OPPONENT_IS_HOUSE:
        from teachers import make_teacher  # lazy
        _opponent_teacher = make_teacher(OPPONENT_TEACHER_SPEC)


def _challenger_teacher(challenger: Optional[str]):
    """Teacher for a roster pick; None -> fall through to PLAYER_TEACHER_SPEC.
    Built once per challenger and cached — each is its own model context, so
    switching challengers mid-session never reloads an earlier pick.

    If the GGUF/llama.cpp build silently degraded to mock (wheel missing,
    fetch failed) and the entry has a tf_spec, retry on the transformers/
    (Zero)GPU path so the picker stays honest."""
    if not challenger or challenger not in CHALLENGERS:
        return None
    t = _challenger_teachers.get(challenger)
    if t is None:
        from teachers import make_teacher  # lazy
        entry = CHALLENGERS[challenger]
        t = make_teacher(_space_safe(entry["spec"]))
        tf_spec = entry.get("tf_spec")
        degraded = (getattr(getattr(t, "_reasoner", None), "backend", None)
                    == "mock")
        if degraded and tf_spec and os.environ.get("FORCE_MOCK", "0") != "1":
            print(f"[selfplay] challenger {challenger}: llama.cpp unavailable "
                  f"-> transformers fallback ({tf_spec})")
            try:
                t = make_teacher(_space_safe(tf_spec))
            except Exception as exc:
                print(f"[selfplay] transformers fallback failed too: {exc}")
        _challenger_teachers[challenger] = t
    return t


def _player_opponent(opp: Opponent) -> Opponent:
    """The persona the PLAYER teacher should role-play. Defaults to mirror."""
    if PLAYER_PERSONA == "mirror":
        return opp
    from opponents import BY_ID, LADDER
    return BY_ID.get(PLAYER_PERSONA, LADDER[0])


def _flip_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror the state into the player teacher's POV. The live game stores
    state from the AI/opponent perspective (ai_hp = opponent's HP, history
    rows say `ai_move` = opponent's prior moves). The player teacher needs
    its own moves under `ai_move` and the opponent's under `player_move`."""
    return {
        "round": state["round"],
        "ai_hp": state["player_hp"],
        "ai_prana": state["player_prana"],
        "player_hp": state["ai_hp"],
        "player_prana": state["ai_prana"],
        "history": [
            {
                "round": h["round"],
                "player_move": h.get("ai_move"),   # opponent's prior move
                "ai_move":     h.get("player_move"),  # our prior move
                "outcome": h.get("outcome", ""),
            }
            for h in state.get("history", [])
        ],
    }


async def player_choose(state: Dict[str, Any], opp: Opponent,
                        challenger: Optional[str] = None):
    """Ask the player teacher for a blind-commit move on its turn.
    `challenger` is a CHALLENGERS roster id (from the UI picker); unset or
    unknown falls back to PLAYER_TEACHER_SPEC. Returns a teachers.ChoiceResult
    so the caller has parsed + raw + meta."""
    if not SELFPLAY_MODE:
        raise RuntimeError("SELFPLAY_MODE is off")
    teacher = _challenger_teacher(challenger)
    if teacher is None:
        _ensure_teachers()
        teacher = _player_teacher
    return await teacher.choose(_player_opponent(opp), _flip_state(state))


async def opponent_choose(state: Dict[str, Any], opp: Opponent):
    """Ask the opponent teacher for its move — replaces llm.Reasoner.choose
    for self-play matches, so both sides are LLM-driven. Not called when
    OPPONENT_IS_HOUSE (app.ai_turn keeps the live Reasoner instead)."""
    if not SELFPLAY_MODE:
        raise RuntimeError("SELFPLAY_MODE is off")
    if OPPONENT_IS_HOUSE:
        raise RuntimeError("opponent teacher is 'house' — use the live Reasoner")
    _ensure_teachers()
    return await _opponent_teacher.choose(opp, state)


async def aclose() -> None:
    """Best-effort cleanup of underlying teacher sessions (used on shutdown)."""
    for t in (_player_teacher, _opponent_teacher, *_challenger_teachers.values()):
        if t is not None:
            try:
                await t.aclose()
            except Exception:
                pass
