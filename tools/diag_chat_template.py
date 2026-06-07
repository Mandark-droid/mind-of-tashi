"""diag_chat_template.py — does GGUF runtime use the same chat template as
HF transformers used at SFT time?

Hypothesis: tokens-themselves are identical (proven by diag_tokenizer.py),
but the messages -> string formatting differs between paths. If we render
the same messages list via:

    A) HF: tokenizer.apply_chat_template(messages, ...)   ← SFT-time canonical
    B) llama-cpp-python: chat_format='chatml'             ← what bench used
    C) llama-cpp-python: chat_format=None (auto)          ← uses GGUF embedded

...the resulting prompt strings (or their token sequences) should all agree.
Any divergence -> the model is seeing a different prompt at inference than
it saw during SFT, and degenerate output is the natural consequence.

Usage (from mind-of-tashi/):
    python -m tools.diag_chat_template
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


# Realistic-shape messages — mirrors what app.py would send to Reasoner.
def sample_messages() -> List[Dict[str, str]]:
    from opponents import LADDER
    from prompts import build_system, build_user
    from teachers.base import legal_moves
    opp = LADDER[0]  # Tashi
    state = {
        "round": 1, "ai_hp": 100, "ai_prana": 1,
        "player_hp": 100, "player_prana": 1, "history": [],
    }
    legal = legal_moves(state["ai_prana"])
    return [
        {"role": "system", "content": build_system(opp)},
        {"role": "user",   "content": build_user(opp, state, legal)},
    ]


def render_hf(model_id: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    return {"prompt_text": prompt, "n_tokens": len(ids), "ids_head": ids[:30], "ids_tail": ids[-20:]}


def render_gguf(repo: str, fname: str, messages: List[Dict[str, str]], chat_format: Any) -> Dict[str, Any]:
    from llama_cpp import Llama
    llm = Llama.from_pretrained(
        repo_id=repo, filename=fname,
        n_gpu_layers=0, n_ctx=2048, verbose=False,
        chat_format=chat_format,
    )
    # Use the chat handler to render messages -> prompt the same way the
    # runtime would when create_chat_completion is called. Path A: rely on
    # the chat_format-resolved handler. Path B: dump messages via handler.
    handler = llm.chat_handler
    # llama-cpp-python's handlers expose ._convert_messages_to_prompt for
    # most formats; for the generic chatml format it builds a string the
    # same way it does during inference. Fall back to a manual path if the
    # private API isn't present.
    prompt_text = None
    try:
        if hasattr(handler, "_convert_messages_to_prompt"):
            prompt_text = handler._convert_messages_to_prompt(messages)
        elif hasattr(handler, "convert_messages_to_prompt"):
            prompt_text = handler.convert_messages_to_prompt(messages)
    except Exception as e:
        prompt_text = f"<<handler error: {e}>>"
    if prompt_text is None:
        # Fall back to the Jinja path: use the embedded template from GGUF
        # metadata, ignore whatever chat_format said.
        try:
            from jinja2 import Environment
            tpl_str = (llm.metadata or {}).get("tokenizer.chat_template")
            env = Environment()
            tpl = env.from_string(tpl_str)
            prompt_text = tpl.render(messages=messages, add_generation_prompt=True)
        except Exception as e:
            prompt_text = f"<<jinja fallback failed: {e}>>"

    ids = llm.tokenize(prompt_text.encode("utf-8"), add_bos=False, special=True)
    return {
        "chat_format_used": chat_format or "auto",
        "prompt_text": prompt_text,
        "n_tokens": len(ids),
        "ids_head": list(ids[:30]),
        "ids_tail": list(ids[-20:]),
    }


def diff_strings(a: str, b: str) -> Dict[str, Any]:
    """Return where two strings first diverge + a short window around it."""
    if a == b:
        return {"identical": True}
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return {
        "identical": False,
        "first_diff_index": i,
        "len_a": len(a),
        "len_b": len(b),
        "common_prefix": a[max(0, i-40):i],
        "a_window": a[i:i+80],
        "b_window": b[i:i+80],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model",  default="kshitijthakkar/mind-of-tashi-micro-sft-v2")
    p.add_argument("--gguf-repo", default="kshitijthakkar/mind-of-tashi-micro-sft-v2-gguf")
    p.add_argument("--gguf-file", default="mind-of-tashi-micro-sft-v2-Q4_K_M.gguf")
    p.add_argument("--out-dir",   type=Path, default=Path("../data/diag_tokenizer"))
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    messages = sample_messages()
    print(f"[diag] messages: {len(messages)} (system+user)", flush=True)

    print("[diag] rendering via HF apply_chat_template ...", flush=True)
    hf = render_hf(args.hf_model, messages)
    print(f"        HF: {hf['n_tokens']} tokens", flush=True)

    print("[diag] rendering via llama-cpp-python chat_format='chatml' ...", flush=True)
    gg_chatml = render_gguf(args.gguf_repo, args.gguf_file, messages, "chatml")
    print(f"        GGUF chatml: {gg_chatml['n_tokens']} tokens", flush=True)

    print("[diag] rendering via llama-cpp-python chat_format=None (GGUF embedded) ...", flush=True)
    gg_auto = render_gguf(args.gguf_repo, args.gguf_file, messages, None)
    print(f"        GGUF auto:   {gg_auto['n_tokens']} tokens", flush=True)

    # --- compare
    print()
    print("=" * 70)
    print(f"  HF                : {hf['n_tokens']} tokens")
    print(f"  GGUF (chatml fmt) : {gg_chatml['n_tokens']} tokens")
    print(f"  GGUF (auto/embed) : {gg_auto['n_tokens']} tokens")
    print("=" * 70)
    cmp1 = diff_strings(hf["prompt_text"], gg_chatml["prompt_text"])
    cmp2 = diff_strings(hf["prompt_text"], gg_auto["prompt_text"])
    cmp3 = diff_strings(gg_chatml["prompt_text"], gg_auto["prompt_text"])
    print()
    def show(label, c):
        if c["identical"]:
            print(f"  {label}: IDENTICAL")
        else:
            print(f"  {label}: DIFFER at offset {c['first_diff_index']} "
                  f"(lens {c['len_a']} vs {c['len_b']})")
            print(f"    common prefix ..…{c['common_prefix']!r}")
            print(f"    A window:   {c['a_window']!r}")
            print(f"    B window:   {c['b_window']!r}")
    show("HF  vs  GGUF.chatml", cmp1)
    show("HF  vs  GGUF.auto  ", cmp2)
    show("GGUF.chatml  vs  GGUF.auto", cmp3)

    payload = {
        "hf": hf, "gguf_chatml": gg_chatml, "gguf_auto": gg_auto,
        "cmp_hf_vs_chatml": cmp1, "cmp_hf_vs_auto": cmp2,
        "cmp_chatml_vs_auto": cmp3,
        "stamp": stamp,
    }
    out_path = args.out_dir / f"diag_chat_template_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0 if (cmp1["identical"] and cmp2["identical"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
