"""
mind_of_tashi_env — GRPO-facing environment that wraps the blind-commit duel.

The student model plays AS one of the LADDER personas. The opponent is
either a local mock (`OPPONENT_BACKEND=local`) or one of the API teachers
already used for self-play harvest (`OPPONENT_BACKEND=api`).

OFF-THE-GRID BOUNDARY (see ROADMAP.md §3)
-----------------------------------------
This env is training-rig code. When `OPPONENT_BACKEND=api` the env imports
`teachers.pool` which uses Gemini / Mistral / OpenRouter. That is fine for
training. It is NOT fine at runtime in the deployed Space — so `api` is
hard-refused when `SPACE_ID` is set (Hugging Face Spaces always sets this).

API USAGE
---------
    env = make_env(opponent_backend="api")
    obs = env.reset(seed=0)
    # obs = {"messages": [...], "state": {...}, "legal_moves": [...],
    #        "student_persona": "...", "opponent_persona": "..."}

    # student generates a completion (this is what GRPO does):
    completion = "<think>...</think>\\n{\\"move\\":\\"STRIKE\\",\\"taunt\\":\\"...\\"}"

    obs, reward, terminated, info = env.step(completion)
    # reward = {"turn": float, "outcome": float, "lexicon": float,
    #           "total": float}
    # info  = {"student_move", "opponent_move", "round_log",
    #          "teacher_meta", ...}

REWARD SHAPING
--------------
The engine's HP-delta and win/loss IS the rubric reward — no separate
reward model needed (ROADMAP C2).

    turn_reward    = +Δ(opponent_hp) − Δ(student_hp)      # dense, per-step
    outcome_reward = +10 win, −10 loss, 0 draw            # sparse, terminal
    lexicon_bonus  = +0.5 × (sanskrit_token_count / think_len)
                                                          # anti-anglicisation

`reward["total"]` is the trainer-facing scalar. Coefficients are tunable
via `MindOfTashiEnv(reward_coefficients=...)` — the defaults match ROADMAP.
"""

from __future__ import annotations
import asyncio
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project-internal imports — the env runs from mind-of-tashi/ as the working
# directory (same as app.py, selfplay.py).
from engine import Fighter, MOVES, apply, resolve
from opponents import LADDER, Opponent
from prompts import parse_reply
from teachers.base import Teacher, build_messages, legal_moves


# --- guard ----------------------------------------------------------------

class OpponentBackendForbidden(RuntimeError):
    """Raised when `OPPONENT_BACKEND=api` is requested inside a deployed Space.

    Off-the-Grid badge contract: the playable Space must not make outbound
    cloud-API calls. Training rigs are exempt because the deployed Space
    only ships the resulting GGUF.
    """


def _refuse_api_on_space(backend: str) -> None:
    if backend == "api" and os.getenv("SPACE_ID"):
        raise OpponentBackendForbidden(
            "OPPONENT_BACKEND=api is refused when SPACE_ID is set "
            "(Off-the-Grid badge contract — see ROADMAP §3). "
            "Use OPPONENT_BACKEND=local for any Space-deployed gym."
        )


# --- lexicon (for the bilingual bonus) ------------------------------------

_LEXICON_CACHE: Optional[set[str]] = None


def _load_lexicon() -> set[str]:
    """Load Sanskrit/Hindi lexicon tokens. Returns lowercased set."""
    global _LEXICON_CACHE
    if _LEXICON_CACHE is not None:
        return _LEXICON_CACHE
    p = Path(__file__).resolve().parents[2] / "assets" / "sanskrit_lexicon.txt"
    if not p.exists():
        _LEXICON_CACHE = set()
        return _LEXICON_CACHE
    tokens: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Each line may have multiple comma-separated forms.
        for tok in line.replace(",", " ").split():
            tokens.add(tok.lower())
    _LEXICON_CACHE = tokens
    return tokens


_WORD_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)


def _lexicon_hit_rate(think_text: str) -> float:
    """Fraction of think-tokens that match the Sanskrit/Hindi lexicon."""
    if not think_text:
        return 0.0
    lex = _load_lexicon()
    if not lex:
        return 0.0
    words = _WORD_RE.findall(think_text)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.lower() in lex)
    return hits / len(words)


def _extract_think(raw: str) -> str:
    m = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


# --- env ------------------------------------------------------------------

@dataclass
class RewardCoefficients:
    turn: float = 1.0       # per-HP scaling for dense reward
    outcome: float = 10.0   # win/loss bonus magnitude
    lexicon: float = 0.5    # bilingual bonus weight (0..0.5)


@dataclass
class _Episode:
    student_opp: Opponent
    opponent_opp: Opponent
    student_fighter: Fighter
    opponent_fighter: Fighter
    history_student: List[Dict] = field(default_factory=list)
    history_opponent: List[Dict] = field(default_factory=list)
    round_idx: int = 0
    terminated: bool = False


class MindOfTashiEnv:
    """One-episode gym wrapper around the blind-commit duel.

    Designed to be wrapped by `trl.GRPOTrainer` in environments mode.
    Multi-step: each `step()` is one blind-commit round; the episode
    terminates on KO or `max_rounds`.
    """

    def __init__(
        self,
        opponent_teacher: Teacher,
        backend_label: str,
        max_rounds: int = 30,
        reward_coefficients: Optional[RewardCoefficients] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.opponent_teacher = opponent_teacher
        self.backend_label = backend_label  # for info logging only
        self.max_rounds = max_rounds
        self.coef = reward_coefficients or RewardCoefficients()
        self.rng = rng or random.Random()
        self._ep: Optional[_Episode] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ----- lifecycle -----

    def reset(
        self,
        student_persona: Optional[str] = None,
        opponent_persona: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)
        s_opp = _pick_persona(student_persona, self.rng)
        # Default: pick a DIFFERENT persona for the opponent so the student
        # gets exposure across temperaments / strategies. Falls back to same
        # if the ladder has only one.
        o_opp = _pick_persona(
            opponent_persona,
            self.rng,
            exclude=s_opp.id if len(LADDER) > 1 else None,
        )
        self._ep = _Episode(
            student_opp=s_opp,
            opponent_opp=o_opp,
            student_fighter=Fighter(s_opp.name),
            opponent_fighter=Fighter(o_opp.name),
        )
        return self._observation()

    def step(self, student_completion: str) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        if self._ep is None or self._ep.terminated:
            raise RuntimeError("step() called on terminated or unreset env")
        ep = self._ep
        ep.round_idx += 1

        # 1. Parse the student's completion.
        legal_s = legal_moves(ep.student_fighter.prana)
        parsed_s = parse_reply(student_completion, legal_s)
        student_move = parsed_s["move"]

        # 2. Ask the opponent teacher (blind to student's pending move).
        opponent_state = self._state_for(ep.opponent_fighter, ep.student_fighter, ep.history_opponent)
        opp_result = self._sync_run(self.opponent_teacher.choose(ep.opponent_opp, opponent_state))
        opponent_move = opp_result.parsed["move"]

        # 3. Resolve the round.
        hp_s_before = ep.student_fighter.hp
        hp_o_before = ep.opponent_fighter.hp
        res = resolve(ep.student_fighter, ep.opponent_fighter, student_move, opponent_move)
        apply(ep.student_fighter, ep.opponent_fighter, res)

        # 4. Update each side's history (POV-correct).
        ep.history_student.append({
            "round": ep.round_idx,
            "player_move": opponent_move,   # from student's POV, opponent is "player"
            "ai_move": student_move,
            "outcome": _outcome_phrase(res),
        })
        ep.history_opponent.append({
            "round": ep.round_idx,
            "player_move": student_move,
            "ai_move": opponent_move,
            "outcome": _outcome_phrase(res),
        })

        # 5. Reward.
        student_dmg_dealt = max(0, hp_o_before - ep.opponent_fighter.hp)
        student_dmg_taken = max(0, hp_s_before - ep.student_fighter.hp)
        turn_reward_raw = float(student_dmg_dealt - student_dmg_taken)

        # 6. Termination + outcome.
        terminated = False
        outcome_reward_raw = 0.0
        if ep.opponent_fighter.hp <= 0 and ep.student_fighter.hp <= 0:
            terminated, outcome_reward_raw = True, 0.0  # double KO = draw
        elif ep.opponent_fighter.hp <= 0:
            terminated, outcome_reward_raw = True, +1.0  # win
        elif ep.student_fighter.hp <= 0:
            terminated, outcome_reward_raw = True, -1.0  # loss
        elif ep.round_idx >= self.max_rounds:
            terminated, outcome_reward_raw = True, 0.0  # round-cap draw
        ep.terminated = terminated

        # 7. Lexicon bonus on student's <think>.
        think = _extract_think(student_completion)
        lex_rate = _lexicon_hit_rate(think)

        reward = {
            "turn": self.coef.turn * turn_reward_raw,
            "outcome": self.coef.outcome * outcome_reward_raw,
            "lexicon": self.coef.lexicon * lex_rate,
        }
        reward["total"] = reward["turn"] + reward["outcome"] + reward["lexicon"]

        info = {
            "student_move": student_move,
            "opponent_move": opponent_move,
            "student_think": think,
            "student_taunt": parsed_s.get("taunt", ""),
            "opponent_think": opp_result.parsed.get("reasoning", ""),
            "round_log": res.log,
            "hp_after": {
                "student": ep.student_fighter.hp,
                "opponent": ep.opponent_fighter.hp,
            },
            "prana_after": {
                "student": ep.student_fighter.prana,
                "opponent": ep.opponent_fighter.prana,
            },
            "teacher_meta": opp_result.meta,
            "lexicon_hit_rate": lex_rate,
            "backend": self.backend_label,
        }

        obs = self._observation() if not terminated else self._terminal_obs()
        return obs, reward, terminated, info

    def close(self) -> None:
        if self.opponent_teacher is not None:
            try:
                self._sync_run(self.opponent_teacher.aclose())
            except Exception:
                pass
        if self._loop is not None:
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    # ----- observation helpers -----

    def _state_for(self, self_f: Fighter, other_f: Fighter, history: List[Dict]) -> Dict[str, Any]:
        """Build a `state` dict from `self_f`'s POV (matches selfplay shape)."""
        return {
            "round": (self._ep.round_idx if self._ep else 0) + 1,
            "ai_hp": self_f.hp,
            "ai_prana": self_f.prana,
            "player_hp": other_f.hp,
            "player_prana": other_f.prana,
            "history": list(history),
        }

    def _observation(self) -> Dict[str, Any]:
        ep = self._ep
        if ep is None:
            raise RuntimeError("env not reset")
        state = self._state_for(ep.student_fighter, ep.opponent_fighter, ep.history_student)
        legal = legal_moves(ep.student_fighter.prana)
        messages = build_messages(ep.student_opp, state)
        return {
            "messages": messages,
            "state": state,
            "legal_moves": legal,
            "student_persona": ep.student_opp.id,
            "opponent_persona": ep.opponent_opp.id,
            "round": ep.round_idx + 1,
        }

    def _terminal_obs(self) -> Dict[str, Any]:
        ep = self._ep
        return {
            "messages": [],
            "state": None,
            "legal_moves": [],
            "student_persona": ep.student_opp.id if ep else None,
            "opponent_persona": ep.opponent_opp.id if ep else None,
            "terminal": True,
        }

    # ----- async bridge -----

    def _sync_run(self, coro):
        """Run a coroutine on a persistent event loop owned by the env.

        Creating a new loop per step would discard connection pools the
        teacher backends hold. One loop per env survives the whole episode.
        """
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)


# --- factory --------------------------------------------------------------

DEFAULT_API_SPECS = [
    "gemini:gemini-2.5-flash-lite",
    "gemini:gemini-2.5-flash",
    "openrouter:meta-llama/llama-3.3-70b-instruct:free",
    "openrouter:qwen/qwen3-30b-a3b-instruct-2507:free",
]


def make_env(
    opponent_backend: str = "local",
    api_specs: Optional[List[str]] = None,
    quota_path: Optional[Path] = None,
    max_rounds: int = 30,
    reward_coefficients: Optional[RewardCoefficients] = None,
    rng: Optional[random.Random] = None,
) -> MindOfTashiEnv:
    """Build a MindOfTashiEnv with the requested opponent backend.

    `local` — uses the in-process mock teacher (no network, fast smoke tests).
    `api`   — uses TeacherPool over Gemini / OpenRouter / Mistral. Refused
              when SPACE_ID is set in env (Off-the-Grid badge contract).
    """
    backend = opponent_backend.lower()
    _refuse_api_on_space(backend)

    if backend == "local":
        from teachers.mock import MockTeacher
        teacher: Teacher = MockTeacher()
        label = "mock"
    elif backend == "api":
        from teachers.pool import make_pool
        specs = api_specs or DEFAULT_API_SPECS
        qpath = quota_path or (Path("data") / "grpo" / ".quota.json")
        qpath.parent.mkdir(parents=True, exist_ok=True)
        teacher = make_pool(specs=specs, quota_path=qpath)
        label = f"pool({len(specs)})"
    else:
        raise ValueError(f"unknown opponent_backend: {opponent_backend!r}")

    return MindOfTashiEnv(
        opponent_teacher=teacher,
        backend_label=label,
        max_rounds=max_rounds,
        reward_coefficients=reward_coefficients,
        rng=rng,
    )


# --- persona helper -------------------------------------------------------

def _pick_persona(
    persona_id: Optional[str],
    rng: random.Random,
    exclude: Optional[str] = None,
) -> Opponent:
    if persona_id:
        for opp in LADDER:
            if opp.id == persona_id:
                return opp
        raise ValueError(f"unknown persona id: {persona_id!r}")
    pool = [o for o in LADDER if o.id != exclude] if exclude else list(LADDER)
    return rng.choice(pool)


def _outcome_phrase(res) -> str:
    if res.a_dmg_taken and res.b_dmg_taken:
        return f"trade: student -{res.a_dmg_taken}, opponent -{res.b_dmg_taken}"
    if res.b_dmg_taken:
        return f"opponent -{res.b_dmg_taken}"
    if res.a_dmg_taken:
        return f"student -{res.a_dmg_taken}"
    return "no blood"
