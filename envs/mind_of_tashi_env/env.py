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
import json
import os
import random
import re
import time
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
    format_penalty: float = 3.0   # TOTAL MAGNITUDE of the granular format
                                  # signal; split across 6 structural elements
                                  # at coef.format_penalty/6 = 0.5 each.
                                  # Spread [-3.0..+3.0]; matches train/grpo.py.


# --- persona-tier routing (David-vs-Goliath stratification) ---------------

# Maps opponent difficulty (1..6) to a teacher tier. Tier name is opaque —
# only used to look up the right teacher in MindOfTashiEnv.opponent_teachers.
TIER_BY_DIFFICULTY: Dict[int, str] = {
    1: "low",
    2: "low",
    3: "mid",
    4: "mid",
    5: "high",
    6: "boss",
}

# Default pool composition per tier. The boss tier uses the paid Gemini 2.5
# Pro (sparingly — see C3 cost cap); lower tiers stay on cheap or free models
# so the bulk of GRPO rollouts don't burn money.
DEFAULT_TIER_SPECS: Dict[str, List[str]] = {
    "low": [
        "openrouter:meta-llama/llama-3.3-70b-instruct:free",
        "openrouter:qwen/qwen3-30b-a3b-instruct-2507:free",
    ],
    "mid": [
        "gemini:gemini-2.5-flash-lite",
        "mistral:mistral-small-latest",
    ],
    "high": [
        "gemini:gemini-2.5-flash",
        "mistral:mistral-large-latest",
    ],
    "boss": [
        "gemini:gemini-2.5-pro",
    ],
}


# Approximate per-million-token rates (USD) for cost tracking. These are
# ballpark Q3/Q4 2025 numbers; they're not used for billing, only for the
# in-script MAX_API_DOLLARS guardrail. Update as provider prices shift.
# OpenRouter ":free" tier and the mock teacher are zero-cost by definition.
TOKEN_RATES_USD_PER_M: Dict[str, Dict[str, float]] = {
    # Gemini
    "gemini-2.5-pro":         {"input": 1.25,  "output": 5.00},
    "gemini-2.5-flash":       {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-lite":  {"input": 0.04,  "output": 0.10},
    "gemini-3.0-flash":       {"input": 0.075, "output": 0.30},
    "gemini-3.0-flash-lite":  {"input": 0.04,  "output": 0.10},
    # Mistral
    "mistral-large-latest":   {"input": 2.00,  "output": 6.00},
    "mistral-small-latest":   {"input": 0.20,  "output": 0.60},
    # Sarvam (Indian languages)
    "sarvam-m":               {"input": 0.10,  "output": 0.30},
}


def _estimate_call_cost(meta: Dict[str, Any]) -> float:
    """Best-effort USD cost from teacher meta. Returns 0.0 for free/local."""
    provider = (meta.get("provider") or "").lower()
    model = meta.get("model") or ""
    if not model:
        return 0.0
    # OpenRouter ":free" suffix → no charge.
    if provider == "openrouter" and ":free" in model.lower():
        return 0.0
    # Mock / fallback / local: no cost.
    if provider in ("mock", "ollama") or meta.get("fallback"):
        return 0.0
    rate = TOKEN_RATES_USD_PER_M.get(model)
    if rate is None:
        # Strip any provider prefix and try again (some specs are "openrouter:gemini-2.5-flash" etc).
        rate = TOKEN_RATES_USD_PER_M.get(model.split("/")[-1].split(":")[0])
    if rate is None:
        return 0.0
    pt = float(meta.get("prompt_tokens") or 0)
    ct = float(meta.get("completion_tokens") or 0)
    return (pt / 1_000_000.0) * rate["input"] + (ct / 1_000_000.0) * rate["output"]


# --- format-validity detection -------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_JSON_MOVE_RE = re.compile(
    r"\{[^{}]*\"move\"\s*:\s*\"([A-Z_]+)\"[^{}]*\}",
    re.DOTALL,
)
# Granular structural-element regexes — each independent so the reward can
# give the model partial credit even when the full contract fails. Kept in
# sync with train/grpo.py.
_RE_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_RE_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_RE_JSON_BRACES = re.compile(r"\{[^{}]+\}", re.DOTALL)
_RE_MOVE_FIELD = re.compile(r'"move"\s*:')
_RE_TAUNT_FIELD = re.compile(r'"taunt"\s*:')
_RE_MOVE_VALUE = re.compile(r'"move"\s*:\s*"([A-Z_]+)"')

# Catalogue of valid move ids — used by _check_format_elements to verify the
# extracted "move" VALUE is one of the 6 known moves (catches
# {"move":"DFOCUS"}-style tokenisation bugs that the field-presence check
# alone misses). Kept in sync with engine.MOVES.
_CATALOGUE_MOVES = set(MOVES.keys())


def _check_format_elements(raw: str) -> Dict[str, bool]:
    """Return a per-element presence dict (6 independent structural checks).

    The keys are the six elements GRPO's reward function rewards
    independently. Even when the full <think>...</think>{move,taunt}
    contract isn't met, the model still gets credit for each piece it
    DID emit. Aligned with train/grpo.py's granular reward shape.

    Element #6 — `move_value_legal` — checks the VALUE of the move field
    against the catalogue. A completion with `"move":"DFOCUS"` (valid JSON,
    field present, but value not in the move list) is rewarded for
    `move_field` (the field itself exists) and PENALISED on
    `move_value_legal` (the value isn't one of STRIKE/GUARD/.../MIST_STEP).
    Found during 2026-05-28 ollama eval.
    """
    mv = _RE_MOVE_VALUE.search(raw)
    return {
        "think_open":       bool(_RE_THINK_OPEN.search(raw)),
        "think_close":      bool(_RE_THINK_CLOSE.search(raw)),
        "json_braces":      bool(_RE_JSON_BRACES.search(raw)),
        "move_field":       bool(_RE_MOVE_FIELD.search(raw)),
        "taunt_field":      bool(_RE_TAUNT_FIELD.search(raw)),
        "move_value_legal": bool(mv and mv.group(1) in _CATALOGUE_MOVES),
    }


def _check_format(raw: str, legal: List[str]) -> Tuple[bool, str]:
    """Return (is_valid, reason) — the strict all-or-nothing check.

    Valid means: full <think>...</think> block AND parseable {"move":..., ...}
    JSON whose move is in the legal set. Used to gate legality/combo rewards
    that presuppose a trustworthy move. The granular per-element credit
    (see _check_format_elements) is separate and continues to fire even when
    this strict check fails.
    """
    if not _THINK_RE.search(raw):
        return False, "no <think>...</think> block"
    m = _JSON_MOVE_RE.search(raw)
    if not m:
        return False, "no parseable {move,taunt} JSON"
    if m.group(1) not in legal:
        return False, f"move {m.group(1)!r} not in legal {legal}"
    return True, ""


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
        opponent_teachers: Dict[str, Teacher],
        backend_label: str,
        max_rounds: int = 30,
        reward_coefficients: Optional[RewardCoefficients] = None,
        rng: Optional[random.Random] = None,
        spend_log_path: Optional[Path] = None,
    ) -> None:
        """
        opponent_teachers — dict mapping tier name ("low" / "mid" / "high" /
            "boss") to a Teacher. The env picks one per step based on the
            opponent persona's difficulty. For backward compatibility with
            single-teacher backends (e.g. the mock), pass {"*": teacher} —
            the "*" key is the universal fallback.
        spend_log_path — JSONL file where each step appends a
            {ts, provider, model, prompt_tokens, completion_tokens, usd}
            row. Used by the trainer to enforce MAX_API_DOLLARS.
        """
        if not opponent_teachers:
            raise ValueError("opponent_teachers must not be empty")
        self.opponent_teachers = dict(opponent_teachers)
        self.backend_label = backend_label  # for info logging only
        self.max_rounds = max_rounds
        self.coef = reward_coefficients or RewardCoefficients()
        self.rng = rng or random.Random()
        self.spend_log_path = spend_log_path
        if spend_log_path is not None:
            spend_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._total_spend_usd: float = 0.0
        self._ep: Optional[_Episode] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ----- tier routing -----

    def _select_teacher(self, opp: Opponent) -> Tuple[Teacher, str]:
        """Pick the right teacher for this opponent's difficulty tier.

        Returns (teacher, tier_label). Falls back to "*" if the tiered dict
        doesn't have an exact match — this lets the local-mock backend
        (which has one MockTeacher under "*") work without modification.
        """
        tier = TIER_BY_DIFFICULTY.get(opp.difficulty, "mid")
        if tier in self.opponent_teachers:
            return self.opponent_teachers[tier], tier
        if "*" in self.opponent_teachers:
            return self.opponent_teachers["*"], tier
        # Last resort: any teacher in the dict.
        first_tier, first_teacher = next(iter(self.opponent_teachers.items()))
        return first_teacher, first_tier

    # ----- spend -----

    def get_total_spend(self) -> float:
        """Cumulative USD spent on API teacher calls in this env's lifetime."""
        return self._total_spend_usd

    def _log_spend(self, meta: Dict[str, Any], tier: str) -> float:
        """Estimate cost from teacher meta, accumulate, and append to JSONL.

        Returns the USD cost of this single call.
        """
        usd = _estimate_call_cost(meta)
        self._total_spend_usd += usd
        if self.spend_log_path is not None:
            row = {
                "ts": time.time(),
                "provider": meta.get("provider"),
                "model": meta.get("model"),
                "tier": tier,
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "thoughts_tokens": meta.get("thoughts_tokens"),
                "usd": usd,
                "cumulative_usd": self._total_spend_usd,
                "fallback": bool(meta.get("fallback")),
            }
            try:
                with self.spend_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except OSError:
                pass  # don't crash the env on a logging IO error
        return usd

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

        # 1. Parse the student's completion + check format validity.
        # parse_reply silently legalises any failure to FOCUS/GUARD — that's
        # fine for the game loop, but we ALSO record whether the raw output
        # was well-formed so the reward can penalise format drift. Two flavours
        # of check: (a) granular per-element presence (5 independent signals
        # for partial credit even when the full contract fails), and (b) the
        # strict all-or-nothing flag still used to gate combo/outcome rewards
        # that presuppose a trustworthy move.
        legal_s = legal_moves(ep.student_fighter.prana)
        parsed_s = parse_reply(student_completion, legal_s)
        student_move = parsed_s["move"]
        format_elements = _check_format_elements(student_completion)
        format_valid, format_reason = _check_format(student_completion, legal_s)

        # 2. Ask the opponent teacher (blind to student's pending move). Pick
        # the teacher tier matched to the opponent persona's difficulty —
        # the David-vs-Goliath stratification (bosses use Gemini 2.5 Pro,
        # low-tier uses free OpenRouter, etc.).
        teacher, tier = self._select_teacher(ep.opponent_opp)
        opponent_state = self._state_for(ep.opponent_fighter, ep.student_fighter, ep.history_opponent)
        opp_result = self._sync_run(teacher.choose(ep.opponent_opp, opponent_state))
        opponent_move = opp_result.parsed["move"]
        call_usd = self._log_spend(opp_result.meta, tier)

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

        # 8. Granular format-element score. Each of the 6 structural elements
        # (<think> open, </think> close, {...} JSON braces, "move": field,
        # "taunt": field, move-VALUE-in-catalogue) gets independent +/- credit.
        # The model has a gradient handle on EACH element instead of an
        # all-or-nothing penalty. Per-element magnitude =
        # coef.format_penalty / 6; total spread = [-coef.format_penalty ..
        # +coef.format_penalty].
        per_element = self.coef.format_penalty / 6.0
        granular_format = sum(
            per_element if present else -per_element
            for present in format_elements.values()
        )

        reward = {
            "turn": self.coef.turn * turn_reward_raw,
            "outcome": self.coef.outcome * outcome_reward_raw,
            "lexicon": self.coef.lexicon * lex_rate,
            "format_penalty": granular_format,
        }
        reward["total"] = (
            reward["turn"]
            + reward["outcome"]
            + reward["lexicon"]
            + reward["format_penalty"]
        )

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
            "tier": tier,
            "call_usd": call_usd,
            "cumulative_usd": self._total_spend_usd,
            "format_valid": format_valid,
            "format_reason": format_reason,
            "format_elements": format_elements,
            "lexicon_hit_rate": lex_rate,
            "backend": self.backend_label,
            # Hidden-combo triggers fired this round (None if no combo).
            # See engine.COMBOS / engine._detect_combo.
            "student_combo": res.a_combo,
            "opponent_combo": res.b_combo,
        }

        obs = self._observation() if not terminated else self._terminal_obs()
        return obs, reward, terminated, info

    def close(self) -> None:
        for teacher in self.opponent_teachers.values():
            try:
                self._sync_run(teacher.aclose())
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

def make_env(
    opponent_backend: str = "local",
    tier_specs: Optional[Dict[str, List[str]]] = None,
    quota_path: Optional[Path] = None,
    spend_log_path: Optional[Path] = None,
    max_rounds: int = 30,
    reward_coefficients: Optional[RewardCoefficients] = None,
    rng: Optional[random.Random] = None,
) -> MindOfTashiEnv:
    """Build a MindOfTashiEnv with the requested opponent backend.

    `local` — single in-process MockTeacher used regardless of tier. Fast
              smoke tests, no network, no spend.
    `api`   — one TeacherPool per tier built from `tier_specs`
              (defaults to DEFAULT_TIER_SPECS). Hard-refused when SPACE_ID
              is set in env (Off-the-Grid badge contract).

    `quota_path` — JSON file for daily-quota state used by the pool.
    `spend_log_path` — JSONL file where per-step API spend is appended
                       (used by the trainer for the MAX_API_DOLLARS cap).
    """
    backend = opponent_backend.lower()
    _refuse_api_on_space(backend)

    if backend == "local":
        from teachers.mock import MockTeacher
        # Single teacher under the universal "*" key — _select_teacher falls
        # back to this regardless of the opponent's difficulty tier.
        teachers: Dict[str, Teacher] = {"*": MockTeacher()}
        label = "mock"
    elif backend == "api":
        from teachers.pool import make_pool
        specs_by_tier = tier_specs or DEFAULT_TIER_SPECS
        qpath = quota_path or (Path("data") / "grpo" / ".quota.json")
        qpath.parent.mkdir(parents=True, exist_ok=True)
        teachers = {
            tier: make_pool(specs=tier_specs_list, quota_path=qpath)
            for tier, tier_specs_list in specs_by_tier.items()
            if tier_specs_list  # skip tiers with no specs
        }
        if not teachers:
            raise ValueError("api backend requires at least one tier with specs")
        label = f"tiered({','.join(teachers.keys())})"
    else:
        raise ValueError(f"unknown opponent_backend: {opponent_backend!r}")

    return MindOfTashiEnv(
        opponent_teachers=teachers,
        backend_label=label,
        max_rounds=max_rounds,
        reward_coefficients=reward_coefficients,
        rng=rng,
        spend_log_path=spend_log_path,
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
