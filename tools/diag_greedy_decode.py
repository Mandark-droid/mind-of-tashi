"""diag_greedy_decode.py — compare HF and GGUF greedy decoding token-by-token.

Tokenization and chat-template rendering are proven identical (see
diag_tokenizer.py and diag_chat_template.py). So either:
  - the weights themselves are corrupted by conversion, or
  - llama-cpp-python's default sampling (e.g. repeat_penalty=1.1) is
    pushing it off-distribution.

This script runs greedy decoding on BOTH backends with the SAME prompt
and the SAME sampling rules (no penalty, no top_p truncation). If the
first 20 tokens still differ, it's the weights. If they match initially
and then diverge, it's a numerical precision issue downstream.

Usage:
    python -m tools.diag_greedy_decode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def sample_messages():
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


def greedy_hf(model_id: str, messages, n_tokens: int) -> Dict[str, Any]:
    """Greedy decode using transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        return {"error": f"torch/transformers not installed: {e}"}
    print(f"[hf] loading {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt_text, return_tensors="pt").to("cuda")
    print(f"[hf] prompt_tokens={inputs['input_ids'].shape[-1]}", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=False,             # greedy
            temperature=1.0,             # ignored
            repetition_penalty=1.0,      # NEUTRAL — match llama.cpp test below
            pad_token_id=tok.eos_token_id,
        )
    wall = time.perf_counter() - t0
    new_tokens = out[0, inputs["input_ids"].shape[1]:].cpu().tolist()
    text = tok.decode(new_tokens, skip_special_tokens=False)
    return {
        "ids": new_tokens,
        "tokens": [tok.decode([i], skip_special_tokens=False) for i in new_tokens],
        "text": text,
        "wall_s": wall,
    }


def greedy_gguf(repo: str, fname: str, messages, n_tokens: int) -> Dict[str, Any]:
    """Greedy decode using llama-cpp-python with NEUTRAL sampling (no
    repeat penalty, no top_p) so any divergence from HF is purely weights."""
    try:
        from llama_cpp import Llama
    except ImportError as e:
        return {"error": f"llama-cpp-python not installed: {e}"}
    print(f"[gguf] loading {repo}/{fname} ...", flush=True)
    llm = Llama.from_pretrained(
        repo_id=repo, filename=fname,
        n_gpu_layers=99, n_ctx=4096, verbose=False,
        chat_format="chatml",
    )
    print("[gguf] streaming greedy ...", flush=True)
    t0 = time.perf_counter()
    out_ids: List[int] = []
    out_pieces: List[str] = []
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=n_tokens,
        temperature=0.0,         # greedy
        top_p=1.0,               # no top-p truncation
        top_k=1,                 # tightest greedy
        repeat_penalty=1.0,      # NEUTRAL — match HF test above
        stream=True,
    )
    for chunk in stream:
        try:
            delta = chunk["choices"][0]["delta"].get("content", "")
        except (KeyError, IndexError, TypeError):
            delta = ""
        if delta:
            out_pieces.append(delta)
    wall = time.perf_counter() - t0
    text = "".join(out_pieces)
    # Re-tokenise the produced text for an ID-level comparison.
    ids = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
    return {
        "ids": list(ids),
        "tokens": [llm.detokenize([i]).decode("utf-8", errors="replace") for i in ids],
        "text": text,
        "wall_s": wall,
    }


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in a or "error" in b:
        return {"error": a.get("error") or b.get("error")}
    n = min(len(a["ids"]), len(b["ids"]))
    diff_at = None
    for i in range(n):
        if a["ids"][i] != b["ids"][i]:
            diff_at = i
            break
    return {
        "hf_len":  len(a["ids"]),
        "gg_len":  len(b["ids"]),
        "first_diff_at_token_index": diff_at,
        "identical_through_index":   diff_at if diff_at is not None else n,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model",  default="kshitijthakkar/mind-of-tashi-micro-sft-v2")
    p.add_argument("--gguf-repo", default="kshitijthakkar/mind-of-tashi-micro-sft-v2-gguf")
    p.add_argument("--gguf-file", default="mind-of-tashi-micro-sft-v2-Q4_K_M.gguf")
    p.add_argument("--n-tokens", type=int, default=80)
    p.add_argument("--out-dir",  type=Path, default=Path("../data/diag_tokenizer"))
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    messages = sample_messages()
    hf = greedy_hf(args.hf_model, messages, args.n_tokens)
    gg = greedy_gguf(args.gguf_repo, args.gguf_file, messages, args.n_tokens)
    cmp = compare(hf, gg)

    print()
    print("=" * 70)
    print("GREEDY DECODE — HF vs GGUF, both temp=0 / repeat=1 / top_p=1")
    print("-" * 70)
    if "error" in hf: print(f"  HF error:   {hf['error']}")
    else: print(f"  HF:   {len(hf['ids'])} tokens in {hf['wall_s']:.2f}s")
    if "error" in gg: print(f"  GGUF error: {gg['error']}")
    else: print(f"  GGUF: {len(gg['ids'])} tokens in {gg['wall_s']:.2f}s")
    if "error" not in cmp:
        if cmp["first_diff_at_token_index"] is None:
            print(f"  IDENTICAL across first {min(cmp['hf_len'], cmp['gg_len'])} tokens")
        else:
            print(f"  diverged at token {cmp['first_diff_at_token_index']}")
            print(f"  identical through index {cmp['identical_through_index']}")
    print("=" * 70)
    print()

    if "error" not in hf:
        print(f"--- HF text (first 400 chars) ---\n{hf['text'][:400]}\n")
    if "error" not in gg:
        print(f"--- GGUF text (first 400 chars) ---\n{gg['text'][:400]}\n")

    payload = {"hf": hf, "gguf": gg, "comparison": cmp, "stamp": stamp}
    out_path = args.out_dir / f"diag_greedy_decode_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
