"""
selfplay_live.py — opt-in mode where the LIVE game runs two LLMs against
each other through the UI. Useful for visual demos and for sanity-checking
the bilingual <think> register on local models without burning API quota.

Enable via env (in .env or Spaces Variables):

    SELFPLAY_MODE=1
    SELFPLAY_PLAYER_TEACHER=ollama:qwen3:14b
    SELFPLAY_OPPONENT_TEACHER=ollama:deepseek-r1:14b

Both specs go through `teachers.make_teacher` so any backend in the pool
works — `ollama:`, `gemini:`, `openrouter:`, `mistral:`, `sarvam:`, even
`mock`. The local game keeps its normal Reasoner (llama.cpp / mock) when
SELFPLAY_MODE is off.

In self-play mode `live_traces` capture is skipped — these matches are
two-AI play, not real player data, and shouldn't be confused at SFT-prep
time. The synthetic `tools/selfplay.py` harness is the right pipeline for
SFT-bound self-play; this module is purely for visual demo.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional

from opponents import Opponent


SELFPLAY_MODE = os.environ.get("SELFPLAY_MODE", "0") == "1"
# Defaults tuned for 6 GB VRAM (RTX 3060): qwen3.5:4b on opponent for
# stronger reads (returns in ~3s on the user's box), qwen3:1.7b on
# player for fast moves. Different scales -> Ollama swaps cleanly,
# total wall per round is dominated by the opponent generation.
PLAYER_TEACHER_SPEC = os.environ.get(
    "SELFPLAY_PLAYER_TEACHER", "ollama:qwen3:1.7b"
)
OPPONENT_TEACHER_SPEC = os.environ.get(
    "SELFPLAY_OPPONENT_TEACHER", "ollama:qwen3.5:4b"
)
# Player persona to role-play. "mirror" = use the same persona as the current
# opponent (mirror match, simplest). Any LADDER id (tashi/norbu/pema/drogpa/
# the-mountain) pins the player teacher to that persona for the whole run.
PLAYER_PERSONA = os.environ.get("SELFPLAY_PLAYER_PERSONA", "mirror")


_player_teacher = None
_opponent_teacher = None


def status() -> Dict[str, Any]:
    """Surfaced into the page meta so the UI can decide whether to show
    the "Watch self-play" button."""
    return {
        "enabled": bool(SELFPLAY_MODE),
        "player_teacher": PLAYER_TEACHER_SPEC if SELFPLAY_MODE else None,
        "opponent_teacher": OPPONENT_TEACHER_SPEC if SELFPLAY_MODE else None,
        "player_persona": PLAYER_PERSONA if SELFPLAY_MODE else None,
    }


def _ensure_teachers():
    """Lazy-construct on first call so the module imports cheaply when
    self-play is disabled (the teachers/* imports pull in aiohttp, etc.)."""
    global _player_teacher, _opponent_teacher
    if not SELFPLAY_MODE:
        return
    if _player_teacher is None or _opponent_teacher is None:
        from teachers import make_teacher  # lazy
        _player_teacher = make_teacher(PLAYER_TEACHER_SPEC)
        _opponent_teacher = make_teacher(OPPONENT_TEACHER_SPEC)


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


async def player_choose(state: Dict[str, Any], opp: Opponent):
    """Ask the player teacher for a blind-commit move on its turn.
    Returns a teachers.ChoiceResult so the caller has parsed + raw + meta."""
    if not SELFPLAY_MODE:
        raise RuntimeError("SELFPLAY_MODE is off")
    _ensure_teachers()
    return await _player_teacher.choose(_player_opponent(opp), _flip_state(state))


async def opponent_choose(state: Dict[str, Any], opp: Opponent):
    """Ask the opponent teacher for its move — replaces llm.Reasoner.choose
    when SELFPLAY_MODE is on, so both sides are LLM-driven."""
    if not SELFPLAY_MODE:
        raise RuntimeError("SELFPLAY_MODE is off")
    _ensure_teachers()
    return await _opponent_teacher.choose(opp, state)


async def aclose() -> None:
    """Best-effort cleanup of underlying teacher sessions (used on shutdown)."""
    for t in (_player_teacher, _opponent_teacher):
        if t is not None:
            try:
                await t.aclose()
            except Exception:
                pass
