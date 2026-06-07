"""ladder_eval.py — the gauntlet evaluation. How good is this checkpoint at
the actual game?

The student model plays through each of the 10 ladder personas as a MIRROR
MATCH (both sides use the same persona prompt — stays on the SFT-trained
distribution). The opponent at each level is drawn from the persona's
tier-matched teacher pool (see envs.mind_of_tashi_env.DEFAULT_TIER_SPECS):
low → free OpenRouter, mid → Gemini Flash-Lite + Mistral small,
high → Gemini Flash + Mistral large, boss → Gemini 2.5 Pro.

Scoring (per level):
  +10 points — student KOs the teacher
  +5  points — round-cap draw with student holding an HP advantage
  +0  points — loss, or draw with teacher holding HP advantage

Max possible: 100 points across 10 levels. This is the **canonical
gameplay-quality metric** for checkpoints — complements tools/format_gate.py
which only measures structural format adherence.

Usage (from mind-of-tashi/):
    # Real teachers (default tier specs, ~$0.50-2 per run)
    python -m tools.ladder_eval --model kshitijthakkar/mind-of-tashi-micro-sft

    # Quick smoke against the mock opponent (no API spend)
    python -m tools.ladder_eval --model <repo> --mock

    # Tighter time budget
    python -m tools.ladder_eval --model <repo> --max-rounds 20

Output:
    data/ladder_eval/<model_slug>_<utc_ts>.jsonl
        line 1: header
        lines 2-11: per-level result
        line 12: summary (total_score, wins/losses/draws, total spend)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env so the teacher pool finds GEMINI_API_KEY / MISTRAL_API_KEY / etc.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

from engine import Fighter, apply, resolve
from envs.mind_of_tashi_env.env import (
    DEFAULT_TIER_SPECS,
    TIER_BY_DIFFICULTY,
    _check_format_elements,
    _estimate_call_cost,
)
from opponents import LADDER, Opponent
from prompts import build_system, build_user, parse_reply
from teachers.base import (
    ChoiceResult,
    Teacher,
    legal_moves,
    synthesize_raw,
    temperature_for,
)


# --- TransformersTeacher --------------------------------------------------

class TransformersTeacher(Teacher):
    """Local-CUDA wrapper around an AutoModelForCausalLM that returns moves.

    Implements the Teacher protocol so the eval can pit the student against
    any other Teacher (API pool, mock, ollama, …) using the same async
    interface as tools/selfplay.py.

    The model is loaded ONCE in __init__ and reused across all 10 levels —
    the GPU load cost is paid once per run, not per match.
    """

    name = "transformers"

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_new_tokens: int = 400,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        # Lazy torch import so non-eval contexts that import this module
        # don't pay the cost.
        import torch  # noqa: F401

    def _choose_sync(
        self, opp: Opponent, state: Dict[str, Any], legal: List[str]
    ) -> ChoiceResult:
        import torch

        messages = [
            {"role": "system", "content": build_system(opp)},
            {"role": "user", "content": build_user(opp, state, legal)},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature_for(opp),
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = parse_reply(raw, legal)
        elements = _check_format_elements(raw)
        return ChoiceResult(
            parsed=parsed,
            raw=raw,
            meta={
                "backend": "transformers",
                "provider": "transformers",
                "model": str(getattr(self.model.config, "_name_or_path", "?")),
                "format_elements": elements,
                "format_valid": all(elements.values()),
            },
        )


# --- match loop -----------------------------------------------------------

def _outcome_phrase(res, s_name: str, o_name: str) -> str:
    if res.a_dmg_taken and res.b_dmg_taken:
        return f"trade: {s_name} -{res.a_dmg_taken}, {o_name} -{res.b_dmg_taken}"
    if res.b_dmg_taken:
        return f"{o_name} -{res.b_dmg_taken}"
    if res.a_dmg_taken:
        return f"{s_name} -{res.a_dmg_taken}"
    return "no blood"


async def run_level(
    student: Teacher,
    opponent: Teacher,
    persona: Opponent,
    max_rounds: int,
    max_history_rounds: int = 0,
) -> Dict[str, Any]:
    """Single mirror-match against one ladder persona.

    Both fighters are framed with the same persona prompt, but each sees
    state from their own POV (blind-commit contract). Returns a result row
    with all the analytics: score, moves played, format-element rates,
    teacher cost, etc.
    """
    s_name = f"student[{persona.id}]"
    o_name = f"teacher[{persona.id}]"
    s = Fighter(s_name)
    o = Fighter(o_name)
    hist_s: List[Dict[str, Any]] = []
    hist_o: List[Dict[str, Any]] = []
    moves_s: List[str] = []
    moves_o: List[str] = []
    student_combos: List[str] = []
    opponent_combos: List[str] = []
    format_element_hits = {
        k: 0 for k in (
            "think_open", "think_close", "json_braces",
            "move_field", "taunt_field", "move_value_legal",
        )
    }
    total_cost_usd = 0.0
    rnd = 0

    for rnd in range(1, max_rounds + 1):
        # Optionally cap the history block sent in the prompt to the most
        # recent N rounds. Engine state (HP, prana, Fighter.move_history for
        # combo detection) is always full-precision; this only affects what
        # the model SEES in its prompt's history block. Helps cap the O(N^2)
        # attention cost on long matches at modest off-distribution risk.
        if max_history_rounds and max_history_rounds > 0:
            hs = hist_s[-max_history_rounds:]
            ho = hist_o[-max_history_rounds:]
        else:
            hs, ho = list(hist_s), list(hist_o)
        state_s = {
            "round": rnd,
            "ai_hp": s.hp, "ai_prana": s.prana,
            "player_hp": o.hp, "player_prana": o.prana,
            "history": hs,
        }
        state_o = {
            "round": rnd,
            "ai_hp": o.hp, "ai_prana": o.prana,
            "player_hp": s.hp, "player_prana": s.prana,
            "history": ho,
        }

        # Blind commit, in parallel. Same per-round timeout as tools/selfplay.py
        # so a stalled teacher doesn't wedge the eval.
        try:
            res_s, res_o = await asyncio.wait_for(
                asyncio.gather(
                    student.choose(persona, state_s),
                    opponent.choose(persona, state_o),
                ),
                timeout=240.0,
            )
        except asyncio.TimeoutError:
            print(
                f"  [level {persona.id}] round {rnd} timed out — truncating",
                file=sys.stderr,
            )
            break

        s_move = res_s.parsed["move"]
        o_move = res_o.parsed["move"]
        moves_s.append(s_move)
        moves_o.append(o_move)

        # Cost: only the teacher (opponent) side counts; student is local GPU.
        total_cost_usd += _estimate_call_cost(res_o.meta)

        # Per-element format tally for the student side — diagnostic of how
        # well the student holds the format across a full match.
        elements = res_s.meta.get("format_elements") or {}
        for k, v in elements.items():
            if v and k in format_element_hits:
                format_element_hits[k] += 1

        result = resolve(s, o, s_move, o_move)
        apply(s, o, result)
        if result.a_combo:
            student_combos.append(result.a_combo)
        if result.b_combo:
            opponent_combos.append(result.b_combo)

        outcome = _outcome_phrase(result, s.name, o.name)
        hist_s.append({
            "round": rnd, "player_move": o_move, "ai_move": s_move, "outcome": outcome,
        })
        hist_o.append({
            "round": rnd, "player_move": s_move, "ai_move": o_move, "outcome": outcome,
        })

        if s.hp <= 0 or o.hp <= 0:
            break

    # --- score this level ---
    if s.hp <= 0 and o.hp <= 0:
        outcome_label, score = "double_ko", 0
    elif s.hp <= 0:
        outcome_label, score = "loss", 0
    elif o.hp <= 0:
        outcome_label, score = "win", 10
    elif s.hp > o.hp:
        outcome_label, score = "draw_hp_advantage", 5
    elif o.hp > s.hp:
        outcome_label, score = "draw_hp_disadvantage", 0
    else:
        outcome_label, score = "draw_even", 5

    rounds_played = max(rnd, 1)
    return {
        "_kind": "level",
        "persona_id": persona.id,
        "persona_name": persona.name,
        "difficulty": persona.difficulty,
        "tier": TIER_BY_DIFFICULTY.get(persona.difficulty, "mid"),
        "outcome": outcome_label,
        "score": score,
        "rounds_played": rounds_played,
        "final_hp_student": s.hp,
        "final_hp_opponent": o.hp,
        "moves_student": moves_s,
        "moves_opponent": moves_o,
        "student_combos_fired": student_combos,
        "opponent_combos_fired": opponent_combos,
        "format_element_rates": {
            k: v / rounds_played for k, v in format_element_hits.items()
        },
        "teacher_cost_usd": total_cost_usd,
    }


async def run_ladder(
    student: Teacher,
    mock: bool,
    max_rounds: int,
    quota_path: Path,
    max_history_rounds: int = 0,
) -> List[Dict[str, Any]]:
    """Drive the student through all 10 ladder personas in order."""
    if mock:
        from teachers.mock import MockTeacher
        teachers_by_tier: Dict[str, Teacher] = {"*": MockTeacher()}
    else:
        from teachers.pool import make_pool
        teachers_by_tier = {
            tier: make_pool(specs=list(specs), quota_path=quota_path)
            for tier, specs in DEFAULT_TIER_SPECS.items() if specs
        }

    rows: List[Dict[str, Any]] = []
    for persona in LADDER:
        tier = TIER_BY_DIFFICULTY.get(persona.difficulty, "mid")
        opp = teachers_by_tier.get(tier) or teachers_by_tier.get("*")
        if opp is None:
            raise RuntimeError(
                f"No teacher for persona {persona.id} (tier '{tier}') "
                f"and no '*' fallback in {list(teachers_by_tier.keys())}"
            )
        t0 = time.time()
        print(
            f"[level {persona.id} diff{persona.difficulty} tier={tier}] ... ",
            end="", flush=True,
        )
        result = await run_level(
            student, opp, persona, max_rounds,
            max_history_rounds=max_history_rounds,
        )
        result["elapsed_s"] = time.time() - t0
        print(
            f"-> {result['outcome']} (+{result['score']}) "
            f"in {result['rounds_played']} rounds "
            f"(${result['teacher_cost_usd']:.4f})"
        )
        rows.append(result)
        # Release PyTorch's cached VRAM between matches. Without this, the
        # caching allocator holds onto allocations from match N indefinitely;
        # with it, we hand the memory back to CUDA at match boundaries. Cheap
        # and safe — only does something when CUDA is the active device.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Best-effort cleanup of pool sockets.
    for t in teachers_by_tier.values():
        try:
            await t.aclose()
        except Exception:
            pass

    return rows


# --- main -----------------------------------------------------------------

def _slug(repo_id: str) -> str:
    return repo_id.replace("/", "_").replace(".", "_").lower()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Ladder gauntlet eval (mirror-match across 10 personas).")
    p.add_argument("--model", required=True, help="HF repo id of the student checkpoint to evaluate")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--max-rounds", type=int, default=30)
    p.add_argument("--max-history-rounds", type=int, default=10,
                   help="Cap the in-prompt history block to the most recent N "
                        "rounds (default 10). 0 disables capping (full history). "
                        "Only affects what the model sees; engine state is "
                        "always full-precision. Caps O(N^2) attention growth "
                        "on long matches at modest off-distribution risk.")
    p.add_argument("--mock", action="store_true",
                   help="Use the in-process mock opponent instead of API teachers (no $ spend)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quota-path", type=Path, default=Path("../data/ladder_eval/.quota.json"))
    p.add_argument("--out-dir", type=Path, default=Path("../data/ladder_eval"))
    args = p.parse_args(argv)

    # Resolve device.
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device
    if device == "cpu":
        print(
            "WARNING: running student inference on CPU — 30 rounds x 10 levels "
            "will take a long time. Consider --device cuda.",
            file=sys.stderr,
        )

    # Load student model.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"ERROR missing dep ({e}). Install: pip install torch transformers", file=sys.stderr)
        return 3

    print(f"[ladder_eval] loading student: {args.model}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    student = TransformersTeacher(
        model, tok, device=device, max_new_tokens=args.max_new_tokens,
    )

    # Prepare output paths.
    args.quota_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = args.out_dir / f"{_slug(args.model)}_{stamp}.jsonl"

    # Run the gauntlet.
    start = time.time()
    rows = asyncio.run(run_ladder(
        student=student,
        mock=args.mock,
        max_rounds=args.max_rounds,
        quota_path=args.quota_path,
        max_history_rounds=args.max_history_rounds,
    ))
    elapsed = time.time() - start

    # Aggregate.
    total_score = sum(r["score"] for r in rows)
    wins = sum(1 for r in rows if r["outcome"] == "win")
    losses = sum(1 for r in rows if r["outcome"] == "loss")
    draws = sum(1 for r in rows if r["outcome"].startswith("draw") or r["outcome"] == "double_ko")
    total_cost = sum(r["teacher_cost_usd"] for r in rows)

    header = {
        "_kind": "header",
        "model": args.model,
        "stamp_utc": stamp,
        "max_rounds": args.max_rounds,
        "max_history_rounds": args.max_history_rounds,
        "mock": args.mock,
        "device": device,
        "seed": args.seed,
    }
    summary = {
        "_kind": "summary",
        "total_score": total_score,
        "max_score": 100,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_cost_usd": total_cost,
        "total_elapsed_s": elapsed,
        "per_level_scores": {r["persona_id"]: r["score"] for r in rows},
    }

    # Write JSONL.
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # Print summary.
    print()
    print("=" * 60)
    print(f"LADDER EVAL: {total_score}/100  ({wins}W {losses}L {draws}D)")
    print(f"  model:        {args.model}")
    print(f"  device:       {device}")
    print(f"  mock:         {args.mock}")
    print(f"  elapsed:      {elapsed:.1f}s")
    print(f"  teacher cost: ${total_cost:.4f}")
    print(f"  output:       {out_path}")
    print("  per-level:")
    for r in rows:
        print(
            f"    {r['persona_id']:<14} diff{r['difficulty']} tier={r['tier']:<4}"
            f" -> {r['outcome']:<22} (+{r['score']:>2})"
            f" final HP s/o={r['final_hp_student']}/{r['final_hp_opponent']}"
        )
    print("=" * 60)
    return 0 if total_score > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
