"""diag_tokenizer.py — pinpoint the qwen3_moe GGUF quality bug.

Hypothesis: the SFT/safetensors path scores 20/20 on the format gate,
but THREE different llama.cpp-backed runtimes (ollama Q4, ollama BF16,
llama-cpp-python Q4) all produce mangled output (garbled Sanskrit
like "abhyāna" instead of "abhyāsa", broken JSON, unclosed think tags).
The model is identical; only the tokenizer round-trip differs. So this
script tokenises the same strings with BOTH backends and prints the
differences.

If they DIFFER on the same input → convert_hf_to_gguf.py is corrupting
the tokenizer. If they MATCH → the bug is in chat templating, special
tokens, or runtime decode.

Usage (from mind-of-tashi/):
    python -m tools.diag_tokenizer

Output: stdout side-by-side table + JSON dump in data/diag_tokenizer/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


# Test strings chosen to exercise the failure modes we observed:
#   1. The mangled bilingual register: "abhyāsa", "dṛṣṭi", "dhyāna"
#   2. The ChatML special tokens that the SFT'd model relies on
#   3. The JSON delimiter chars where the runtime broke (e.g. {"move":")
#   4. Plain ASCII baseline (should tokenise identically — sanity check)
TEST_STRINGS: List[str] = [
    # ASCII baselines
    "STRIKE",
    "GRAPPLE",
    '{"move":"STRIKE","taunt":"hello"}',
    # ChatML / persona markers
    "<think>",
    "</think>",
    "<|im_start|>",
    "<|im_end|>",
    # Sanskrit / Hindi IAST diacritics — the failure surface
    "abhyāsa",
    "dṛṣṭi",
    "dhyāna",
    "prāṇa",
    "vajra",
    "śiṣya",
    # Devanagari (the harvest data uses some of this too)
    "अभ्यास",
    "धर्म",
    "धुंध",
    # Full sample line from the SFT corpus
    "<think>They have struck six times in a row — abhyāsa, not nirṇay.</think>\n"
    '{"move": "GUARD", "taunt": "Your breath is a river. I am the mountain."}',
]


def hf_tokenize_all(repo_id: str, strings: List[str]) -> Dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        return {"error": f"transformers not installed: {e}"}
    print(f"[hf] loading tokenizer {repo_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    rows = []
    for s in strings:
        ids = tok.encode(s, add_special_tokens=False)
        # round-trip: encode -> decode should reproduce the original
        decoded = tok.decode(ids, skip_special_tokens=False)
        rows.append({
            "string": s,
            "n_tokens": len(ids),
            "ids": ids,
            "tokens": [tok.decode([i], skip_special_tokens=False) for i in ids],
            "roundtrip": decoded,
            "roundtrip_ok": decoded == s,
        })
    return {
        "vocab_size": tok.vocab_size if hasattr(tok, "vocab_size") else None,
        "model_max_length": tok.model_max_length if hasattr(tok, "model_max_length") else None,
        "rows": rows,
    }


def gguf_tokenize_all(repo_id: str, filename: str, strings: List[str]) -> Dict[str, Any]:
    try:
        from llama_cpp import Llama
    except ImportError as e:
        return {"error": f"llama-cpp-python not installed: {e}"}
    print(f"[gguf] loading {repo_id} / {filename} ...", flush=True)
    # Load on CPU here — we only need the tokenizer; no inference.
    llm = Llama.from_pretrained(
        repo_id=repo_id, filename=filename,
        n_gpu_layers=0, n_ctx=256, verbose=False,
    )
    rows = []
    for s in strings:
        ids = llm.tokenize(s.encode("utf-8"), add_bos=False, special=True)
        tokens = [llm.detokenize([i]).decode("utf-8", errors="replace") for i in ids]
        decoded = llm.detokenize(ids).decode("utf-8", errors="replace")
        rows.append({
            "string": s,
            "n_tokens": len(ids),
            "ids": list(ids),
            "tokens": tokens,
            "roundtrip": decoded,
            "roundtrip_ok": decoded == s,
        })
    # GGUF metadata snapshot — most useful: tokenizer model + special token IDs.
    meta = {}
    try:
        meta_raw = getattr(llm, "metadata", None)
        if isinstance(meta_raw, dict):
            interesting = [
                "general.architecture", "tokenizer.ggml.model",
                "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id",
                "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id",
                "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token",
                "tokenizer.chat_template",
                "qwen3moe.context_length", "qwen3moe.embedding_length",
            ]
            for k in interesting:
                if k in meta_raw:
                    meta[k] = meta_raw[k]
    except Exception as e:
        meta["error_reading_metadata"] = str(e)
    return {"metadata": meta, "rows": rows}


def compare(hf: Dict[str, Any], gg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "error" in hf:
        print(f"HF error: {hf['error']}")
        return []
    if "error" in gg:
        print(f"GGUF error: {gg['error']}")
        return []
    cmp_rows = []
    for h, g in zip(hf["rows"], gg["rows"]):
        match = h["ids"] == g["ids"]
        cmp_rows.append({
            "string": h["string"],
            "hf_n":   h["n_tokens"],
            "gg_n":   g["n_tokens"],
            "match":  match,
            "hf_ids": h["ids"],
            "gg_ids": g["ids"],
            "hf_tokens": h["tokens"],
            "gg_tokens": g["tokens"],
            "hf_roundtrip_ok": h["roundtrip_ok"],
            "gg_roundtrip_ok": g["roundtrip_ok"],
        })
    return cmp_rows


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 84)
    print(f"  {'string':<40}  {'HF':>6}  {'GGUF':>6}  match  HF→str  GGUF→str")
    print("-" * 84)
    differ = 0
    for r in rows:
        s_short = r["string"][:38] + ".." if len(r["string"]) > 40 else r["string"]
        s_short = s_short.replace("\n", "\\n")
        print(
            f"  {s_short:<40}  {r['hf_n']:>6}  {r['gg_n']:>6}  "
            f"{'OK' if r['match'] else 'X ':>5}  "
            f"{'OK' if r['hf_roundtrip_ok'] else 'X ':>6}  "
            f"{'OK' if r['gg_roundtrip_ok'] else 'X ':>7}"
        )
        if not r["match"]:
            differ += 1
    print("=" * 84)
    print(f"  diverged: {differ}/{len(rows)} strings")
    print()
    # Detail dump on mismatches
    for r in rows:
        if r["match"]: continue
        print(f"--- DIVERGED: {r['string']!r} ---")
        print(f"  HF  ({r['hf_n']:>3} tok):  ids={r['hf_ids']}")
        print(f"                  tokens={r['hf_tokens']}")
        print(f"  GG  ({r['gg_n']:>3} tok):  ids={r['gg_ids']}")
        print(f"                  tokens={r['gg_tokens']}")
        print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model",  default="kshitijthakkar/mind-of-tashi-micro-sft-v2",
                   help="HF repo id of the safetensors checkpoint (for the canonical tokenizer)")
    p.add_argument("--gguf-repo", default="kshitijthakkar/mind-of-tashi-micro-sft-v2-gguf")
    p.add_argument("--gguf-file", default="mind-of-tashi-micro-sft-v2-Q4_K_M.gguf")
    p.add_argument("--out-dir",   type=Path, default=Path("../data/diag_tokenizer"))
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = args.out_dir / f"diag_tokenizer_{stamp}.json"

    hf = hf_tokenize_all(args.hf_model, TEST_STRINGS)
    gg = gguf_tokenize_all(args.gguf_repo, args.gguf_file, TEST_STRINGS)
    rows = compare(hf, gg)

    payload = {
        "hf_model":  args.hf_model,
        "gguf":      {"repo": args.gguf_repo, "file": args.gguf_file},
        "stamp":     stamp,
        "hf":        hf,
        "gguf":      gg,
        "comparison": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print_summary(rows)
    print(f"wrote {out_path}")

    # Exit code carries the verdict for scripting:
    #   0  every string matched
    #   2  diverged on at least one string
    diverged = sum(1 for r in rows if not r["match"])
    return 0 if diverged == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
