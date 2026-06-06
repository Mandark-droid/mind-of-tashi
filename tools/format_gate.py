"""
format_gate.py — does the base student model follow the
<think>...</think>{"move":...,"taunt":...} contract?

This is the Day 3 (Sat 05-30) HARD GATE from ROADMAP §4 slot B:

    PASS    (>=15 / 20 valid)  -> continue with loggenix SFT
    PARTIAL (8-14 / 20 valid)  -> still proceed, tighten SFT format loss
    FAIL    (<8 / 20 valid)    -> swap base to tracegenix-mini-sft-clean-3ep

Grading per prompt:
  (a) emits <think>...</think>          (the mind-scroll wrapper)
  (b) emits parseable {"move":..., "taunt":...} JSON after the think
  (c) move is in the legal-move set for the prompt's state

A prompt counts as VALID only if all three hold.

Usage (from mind-of-tashi/):
    python -m tools.format_gate                                     # default model
    python -m tools.format_gate --model kshitijthakkar/tracegenix-mini-sft-clean-3ep
    python -m tools.format_gate --n 40                              # bigger sample
    python -m tools.format_gate --print-failures                    # show what went wrong
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project imports (env must allow `python -m tools.format_gate` from mind-of-tashi/).
from engine import MOVES
from opponents import LADDER, Opponent
from prompts import build_system, build_user
from teachers.base import legal_moves


DEFAULT_MODEL = "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1"
FALLBACK_MODEL = "kshitijthakkar/tracegenix-mini-sft-clean-3ep"

PASS_THRESHOLD = 15
PARTIAL_THRESHOLD = 8


# --- grading --------------------------------------------------------------

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
# Greedy {...} that allows a trailing taunt with non-ASCII chars.
JSON_RE = re.compile(r"\{[^{}]*\"move\"\s*:\s*\"([A-Z_]+)\"[^{}]*\}", re.DOTALL)


@dataclass
class GradedReply:
    persona_id: str
    legal: List[str]
    raw: str
    has_think: bool
    has_json: bool
    move: Optional[str]
    move_legal: Optional[bool]
    failure_reason: Optional[str]

    @property
    def valid(self) -> bool:
        return self.has_think and self.has_json and bool(self.move_legal)


def grade(raw: str, legal: List[str], persona_id: str) -> GradedReply:
    has_think = bool(THINK_RE.search(raw))
    json_match = JSON_RE.search(raw)
    has_json = json_match is not None
    move: Optional[str] = json_match.group(1) if json_match else None
    move_legal = (move in legal) if move else None

    reason: Optional[str] = None
    if not has_think:
        reason = "no <think>...</think> block"
    elif not has_json:
        reason = "no parseable {move,taunt} JSON"
    elif not move_legal:
        reason = f"move {move!r} not in legal {legal}"

    return GradedReply(
        persona_id=persona_id,
        legal=legal,
        raw=raw,
        has_think=has_think,
        has_json=has_json,
        move=move,
        move_legal=move_legal,
        failure_reason=reason,
    )


# --- prompt sampling ------------------------------------------------------

def sample_prompts(n: int, rng: random.Random) -> List[Tuple[Opponent, Dict[str, Any], List[str]]]:
    """Build `n` diverse (persona, state, legal_moves) tuples.

    Diversity dimensions:
      * persona: rotate through LADDER
      * round number: 1, 3, 6, 10 (early -> mid -> late)
      * prana: 0 -> 6 (cycles affordability of ART / MIST_STEP)
      * history: synthesised short trace so it's not always empty
    """
    out: List[Tuple[Opponent, Dict[str, Any], List[str]]] = []
    ladder = list(LADDER)
    for i in range(n):
        opp = ladder[i % len(ladder)]
        round_num = rng.choice([1, 2, 4, 6, 9])
        ai_prana = rng.randint(0, 6)
        player_prana = rng.randint(0, 6)
        ai_hp = rng.choice([100, 88, 67, 45, 22])
        player_hp = rng.choice([100, 90, 70, 50, 30])

        hist: List[Dict[str, Any]] = []
        # Synthesise round_num-1 plausible history entries from the AI's POV
        # (ai_move = what this opp did; player_move = what the human did).
        for r in range(1, round_num):
            p_move = rng.choice(list(MOVES.keys()))
            a_move = rng.choice(list(MOVES.keys()))
            hist.append({
                "round": r,
                "player_move": p_move,
                "ai_move": a_move,
                "outcome": rng.choice(["no blood", f"player -{rng.randint(5,18)}", f"ai -{rng.randint(5,18)}"]),
            })

        state = {
            "round": round_num,
            "ai_hp": ai_hp,
            "ai_prana": ai_prana,
            "player_hp": player_hp,
            "player_prana": player_prana,
            "history": hist,
        }
        legal = legal_moves(ai_prana)
        out.append((opp, state, legal))
    return out


# --- generation -----------------------------------------------------------

# --- ollama backend (for grading GGUF quantizations end-to-end) ----------

def generate_one_ollama(
    client, model_name: str, system_text: str, user_text: str,
    max_new_tokens: int,
) -> str:
    """Use the ollama Python client to grade a GGUF served by ollama.

    Forces greedy decoding (temperature=0) so the comparison vs the
    transformers-loaded baseline is apples-to-apples.
    """
    resp = client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        options={
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": max_new_tokens,
        },
        stream=False,
    )
    return resp["message"]["content"]


def load_model_and_tokenizer(model_id: str, device: str):
    """Lazy import torch + transformers so import errors surface at runtime
    with an actionable hint, not as a hard import failure."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(
            f"ERROR: missing dependency ({e}). Install with:\n"
            f"  pip install torch transformers accelerate\n",
            file=sys.stderr,
        )
        raise

    print(f"loading tokenizer for {model_id} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"loading model for {model_id} on {device} ...", file=sys.stderr)
    import torch
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tok


def generate_one(
    model, tok, system_text: str, user_text: str,
    max_new_tokens: int, device: str,
) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    # Most chat-tuned models on HF expose apply_chat_template; loggenix-s3.1
    # and tracegenix both do (they're qwen3 / qwen3.5 derivatives).
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    import torch
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # deterministic for grading
            temperature=1.0,            # ignored when do_sample=False
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=False)


# --- main -----------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Format-adherence gate for the base SFT student.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF repo id of the base model (default: {DEFAULT_MODEL})")
    p.add_argument("--ollama-model", default=None,
                   help="If set, grade this ollama model name instead of loading via transformers "
                        "(e.g. mind-of-tashi-micro:q4). Use to test GGUF deployment quality.")
    p.add_argument("--n", type=int, default=20, help="number of prompts to grade (default 20)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--print-failures", action="store_true",
                   help="print raw output for every prompt that didn't pass all three checks")
    p.add_argument("--print-all", action="store_true",
                   help="print raw output for every prompt")
    p.add_argument("--save-raw", type=Path, default=None,
                   help="optional JSONL path to dump every prompt+output for offline review")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)

    # Resolve device.
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device
    if args.ollama_model:
        try:
            import ollama  # type: ignore
        except ImportError:
            print("ERROR: --ollama-model requires `pip install ollama`.", file=sys.stderr)
            return 3
        ollama_client = ollama.Client()
        print(f"using ollama backend, model='{args.ollama_model}'", file=sys.stderr)
        model, tok = None, None
    else:
        if device == "cpu":
            print(
                "WARNING: running on CPU. A 0.4B MoE will take ~5-15s per prompt; "
                "with 20 prompts that's a few minutes. Consider --device cuda if a GPU is available.",
                file=sys.stderr,
            )
        model, tok = load_model_and_tokenizer(args.model, device)
        ollama_client = None

    prompts = sample_prompts(args.n, rng)
    graded: List[GradedReply] = []
    raw_rows: List[Dict[str, Any]] = []

    for i, (opp, state, legal) in enumerate(prompts, start=1):
        sys_text = build_system(opp)
        usr_text = build_user(opp, state, legal)
        if ollama_client is not None:
            raw = generate_one_ollama(ollama_client, args.ollama_model, sys_text, usr_text, args.max_new_tokens)
        else:
            raw = generate_one(model, tok, sys_text, usr_text, args.max_new_tokens, device)
        g = grade(raw, legal, opp.id)
        graded.append(g)
        verdict = "VALID" if g.valid else "INVALID"
        print(
            f"[{i:>2}/{args.n}] persona={opp.id:<14} prana={state['ai_prana']} "
            f"legal={len(legal):>1}  -> {verdict}"
            + (f"  ({g.failure_reason})" if not g.valid else "")
        )
        if args.print_all or (args.print_failures and not g.valid):
            print("    --- raw ---")
            for line in raw.splitlines():
                print(f"    {line}")
            print("    -----------")
        if args.save_raw is not None:
            raw_rows.append({
                "persona": opp.id, "state": state, "legal": legal,
                "raw": raw, "valid": g.valid, "failure_reason": g.failure_reason,
                "has_think": g.has_think, "has_json": g.has_json,
                "move": g.move, "move_legal": g.move_legal,
            })

    if args.save_raw is not None:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
        with args.save_raw.open("w", encoding="utf-8") as f:
            for row in raw_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(raw_rows)} rows to {args.save_raw}")

    # Summary.
    n_valid = sum(1 for g in graded if g.valid)
    n_has_think = sum(1 for g in graded if g.has_think)
    n_has_json = sum(1 for g in graded if g.has_json)
    n_legal_move = sum(1 for g in graded if g.move_legal)

    if n_valid >= PASS_THRESHOLD:
        verdict = "PASS"
        recommendation = f"proceed with {args.model} for SFT."
    elif n_valid >= PARTIAL_THRESHOLD:
        verdict = "PARTIAL"
        recommendation = (
            f"proceed with {args.model} but tighten format-only validation loss in SFT."
        )
    else:
        verdict = "FAIL"
        recommendation = (
            f"swap student to {FALLBACK_MODEL} (per ROADMAP §4 slot B fallback)."
        )

    print()
    print("=" * 60)
    print(f"FORMAT-ADHERENCE GATE: {verdict}  ({n_valid}/{args.n} valid)")
    print(f"  has <think>...</think>:    {n_has_think}/{args.n}")
    print(f"  has parseable JSON:        {n_has_json}/{args.n}")
    print(f"  emitted move is legal:     {n_legal_move}/{args.n}")
    print(f"  model:  {args.ollama_model or args.model}")
    print(f"  backend:{'ollama' if args.ollama_model else 'transformers'} device:{device}")
    print(f"=> {recommendation}")
    print("=" * 60)

    # Exit codes for scripting:
    #   0 = PASS, 1 = PARTIAL, 2 = FAIL, 3 = runtime error (handled by Python)
    if verdict == "PASS":
        return 0
    if verdict == "PARTIAL":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
