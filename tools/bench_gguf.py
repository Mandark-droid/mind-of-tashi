"""bench_gguf.py — measure inference perf of a deployed GGUF.

Runs realistic-shape game prompts through llama-cpp-python with GPU
offload and reports:

  - cold-start load time (model load + first warmup)
  - prompt eval rate (tokens/sec, the prefill phase)
  - decode rate (tokens/sec, the autoregressive generation phase)
  - TTFT (time-to-first-token from generate() call)
  - per-call total latency (p50 / p90)
  - peak VRAM during the run (sampled via nvidia-smi every 0.5s)
  - quick output sanity (does it emit <think>...{move,taunt}?)

Usage (from mind-of-tashi/):
    # Default: v2 Q4 from Hub, GPU on
    python -m tools.bench_gguf

    # Compare v1 vs v2
    python -m tools.bench_gguf --model kshitijthakkar/mind-of-tashi-micro-sft-gguf \
                               --file mind-of-tashi-micro-sft-Q4_K_M.gguf

    # CPU baseline
    python -m tools.bench_gguf --n-gpu-layers 0

    # More calls for tighter numbers
    python -m tools.bench_gguf --calls 10

Output:
    data/bench_gguf/<model_slug>_<utc>.json   structured results
    plus a stdout summary table
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project-local imports for realistic prompt building.
from engine import MOVES
from opponents import LADDER
from prompts import build_system, build_user
from teachers.base import legal_moves as _legal_moves


DEFAULT_MODEL_REPO = "kshitijthakkar/mind-of-tashi-micro-sft-v2-gguf"
DEFAULT_MODEL_FILE = "mind-of-tashi-micro-sft-v2-Q4_K_M.gguf"


# --- VRAM sampler ---------------------------------------------------------

class VRAMSampler:
    """Periodically sample `nvidia-smi --query-gpu=memory.used` in a
    background thread. Peak + samples are exposed once stop() is called."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self._samples: List[int] = []
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    timeout=3, text=True,
                )
                # Multi-GPU systems return one number per line; take device 0.
                first = out.strip().splitlines()[0].strip()
                self._samples.append(int(first))
            except Exception:
                # nvidia-smi missing or returned non-int — skip the sample
                pass
            self._stop.wait(self.interval_s)

    def start(self):
        if self._thr is not None:
            return
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=2)
        if not self._samples:
            return {"peak_mb": None, "mean_mb": None, "n_samples": 0}
        return {
            "peak_mb":  max(self._samples),
            "mean_mb":  round(sum(self._samples) / len(self._samples), 1),
            "n_samples": len(self._samples),
        }


# --- prompt builder -------------------------------------------------------

def sample_prompts(n: int) -> List[Dict[str, Any]]:
    """Synthesize a small variety of in-game prompts that exercise the model
    across difficulties and prompt lengths (early empty-history rounds vs
    later rounds with longer history blocks)."""
    rows = []
    for i in range(n):
        opp = LADDER[i % len(LADDER)]
        # Alternate: round 1 (short prompt) and round 8 (longer history).
        round_n = 1 if (i % 2 == 0) else 8
        ai_prana = 1 if round_n == 1 else 4
        player_prana = 1 if round_n == 1 else 3
        ai_hp = 100 if round_n == 1 else 67
        player_hp = 100 if round_n == 1 else 78
        legal = _legal_moves(ai_prana)
        history = []
        if round_n > 1:
            for r in range(1, round_n):
                history.append({
                    "round": r,
                    "player_move": "STRIKE" if r % 2 == 0 else "GUARD",
                    "ai_move":     "FOCUS" if r % 2 == 0 else "GRAPPLE",
                    "outcome":     "no blood" if r % 2 == 0 else "ai -12",
                })
        state = {
            "round": round_n,
            "ai_hp": ai_hp, "ai_prana": ai_prana,
            "player_hp": player_hp, "player_prana": player_prana,
            "history": history,
        }
        rows.append({
            "system": build_system(opp),
            "user":   build_user(opp, state, legal),
            "persona_id": opp.id,
            "round": round_n,
        })
    return rows


# --- bench loop -----------------------------------------------------------

def run_bench(args) -> Dict[str, Any]:
    try:
        from llama_cpp import Llama, llama_supports_gpu_offload
    except ImportError as e:
        print(f"ERROR: llama-cpp-python is not installed in this venv: {e}", file=sys.stderr)
        print("Install with:\n  pip install llama-cpp-python --extra-index-url "
              "https://abetlen.github.io/llama-cpp-python/whl/cu124", file=sys.stderr)
        return {"error": "llama_cpp not installed"}

    cuda_available = bool(llama_supports_gpu_offload())
    print(f"[bench] llama_cpp CUDA support: {cuda_available}")
    print(f"[bench] model: {args.model}")
    print(f"[bench] file:  {args.file}")
    print(f"[bench] n_gpu_layers: {args.n_gpu_layers}  (0 = pure CPU)")
    print(f"[bench] n_ctx: {args.n_ctx}  max_new_tokens: {args.max_new_tokens}  calls: {args.calls}")

    # --- VRAM sampler around the load + warmup so peak load is captured.
    sampler = VRAMSampler(interval_s=0.4)
    sampler.start()

    load_t0 = time.perf_counter()
    llm = Llama.from_pretrained(
        repo_id=args.model,
        filename=args.file,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        verbose=False,
        chat_format="chatml",   # qwen3-moe is ChatML
    )
    load_s = time.perf_counter() - load_t0
    print(f"[bench] model loaded in {load_s:.2f}s")

    prompts = sample_prompts(args.calls)
    results: List[Dict[str, Any]] = []

    for i, p in enumerate(prompts, start=1):
        messages = [
            {"role": "system", "content": p["system"]},
            {"role": "user",   "content": p["user"]},
        ]
        sys.stdout.write(f"  call {i:>2}/{args.calls} persona={p['persona_id']:<14} round={p['round']:>1} ... ")
        sys.stdout.flush()
        call_t0 = time.perf_counter()
        first_tok_t: Optional[float] = None
        last_chunk: Optional[Dict[str, Any]] = None
        out_text_parts: List[str] = []
        n_output_tokens = 0

        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=args.max_new_tokens,
            temperature=0.85,
            top_p=0.9,
            stream=True,
        )
        for chunk in stream:
            if first_tok_t is None:
                first_tok_t = time.perf_counter()
            try:
                delta = chunk["choices"][0]["delta"].get("content", "")
            except (KeyError, IndexError, TypeError):
                delta = ""
            if delta:
                out_text_parts.append(delta)
                n_output_tokens += 1
            last_chunk = chunk
        call_total_s = time.perf_counter() - call_t0
        ttft_s = (first_tok_t - call_t0) if first_tok_t is not None else None
        decode_s = (call_total_s - (ttft_s or 0)) if ttft_s is not None else None
        decode_tps = (n_output_tokens / decode_s) if (decode_s and decode_s > 0) else None

        # The final llama-cpp-python chunk carries the timing dict.
        timings = (last_chunk or {}).get("timings") or {}
        prompt_eval_tps = timings.get("prompt_per_second")
        eval_tps = timings.get("predicted_per_second")
        prompt_tokens = timings.get("prompt_n")

        out_text = "".join(out_text_parts)
        has_think = "<think>" in out_text and "</think>" in out_text
        has_json = ('"move":' in out_text) and ('"taunt":' in out_text)

        sys.stdout.write(
            f"ttft={ttft_s*1000:>6.0f}ms  "
            f"out_toks={n_output_tokens:>3}  "
            f"decode={decode_tps:>5.1f}t/s  "
            f"total={call_total_s:>5.2f}s  "
            f"fmt={'OK' if (has_think and has_json) else 'no'}\n"
        )
        results.append({
            "i": i, "persona_id": p["persona_id"], "round": p["round"],
            "prompt_tokens": prompt_tokens,
            "output_tokens": n_output_tokens,
            "ttft_s": ttft_s,
            "call_total_s": call_total_s,
            "decode_tps_wall":  decode_tps,
            "prompt_eval_tps_llama_cpp": prompt_eval_tps,
            "decode_tps_llama_cpp":      eval_tps,
            "has_think_block": has_think,
            "has_move_taunt_json": has_json,
            "output_text_preview": out_text[:240],
        })

    # Final VRAM sample after the run completes; then stop the sampler.
    vram = sampler.stop()

    # --- aggregates
    finite = lambda xs: [x for x in xs if x is not None]
    ttft_vals       = finite([r["ttft_s"] for r in results])
    total_vals      = finite([r["call_total_s"] for r in results])
    decode_vals     = finite([r["decode_tps_wall"] for r in results])
    prompt_eval_vals= finite([r["prompt_eval_tps_llama_cpp"] for r in results])

    def stats(xs):
        if not xs: return {"n": 0}
        xs_sorted = sorted(xs)
        p50 = statistics.median(xs)
        p90 = xs_sorted[max(0, int(0.9 * len(xs_sorted)) - 1)]
        return {"n": len(xs), "p50": p50, "p90": p90,
                "mean": sum(xs)/len(xs), "min": min(xs), "max": max(xs)}

    summary = {
        "model": args.model,
        "file": args.file,
        "n_gpu_layers": args.n_gpu_layers,
        "n_ctx": args.n_ctx,
        "max_new_tokens": args.max_new_tokens,
        "cuda_available": cuda_available,
        "load_s": load_s,
        "calls": args.calls,
        "ttft_s":            stats(ttft_vals),
        "call_total_s":      stats(total_vals),
        "decode_tps_wall":   stats(decode_vals),
        "prompt_eval_tps":   stats(prompt_eval_vals),
        "vram_mb_during_run": vram,
        "format_pass_rate": sum(
            1 for r in results
            if r["has_think_block"] and r["has_move_taunt_json"]
        ) / max(1, len(results)),
    }
    return {"summary": summary, "calls": results}


# --- output ---------------------------------------------------------------

def print_summary(s: Dict[str, Any]) -> None:
    print()
    print("=" * 70)
    print(f"BENCH: {s['model']} / {s['file']}")
    print("-" * 70)
    print(f"  n_gpu_layers     {s['n_gpu_layers']}    n_ctx {s['n_ctx']}    "
          f"max_new {s['max_new_tokens']}    cuda {s['cuda_available']}")
    print(f"  cold load        {s['load_s']:.2f}s")
    def fmt(stat_d, unit="", scale=1):
        if not stat_d.get("n"):
            return "n/a"
        return (f"p50={stat_d['p50']*scale:.2f}{unit}  "
                f"p90={stat_d['p90']*scale:.2f}{unit}  "
                f"mean={stat_d['mean']*scale:.2f}{unit}")
    print(f"  TTFT             {fmt(s['ttft_s'], 'ms', scale=1000)}")
    print(f"  call total       {fmt(s['call_total_s'], 's')}")
    print(f"  decode (wall)    {fmt(s['decode_tps_wall'], ' t/s')}")
    print(f"  prompt eval      {fmt(s['prompt_eval_tps'], ' t/s')}  (from llama.cpp timings)")
    vram = s["vram_mb_during_run"]
    if vram and vram.get("peak_mb") is not None:
        print(f"  peak VRAM        {vram['peak_mb']} MiB   (mean {vram['mean_mb']} MiB, "
              f"{vram['n_samples']} samples)")
    else:
        print(f"  peak VRAM        n/a (nvidia-smi missing or no samples)")
    print(f"  format pass      {s['format_pass_rate']*100:.0f}%  "
          f"(both <think> and {{move,taunt}} present)")
    print("=" * 70)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark a GGUF via llama-cpp-python (TTFT, tok/s, VRAM).")
    p.add_argument("--model", default=DEFAULT_MODEL_REPO,
                   help=f"HF repo id of the GGUF (default {DEFAULT_MODEL_REPO})")
    p.add_argument("--file", default=DEFAULT_MODEL_FILE,
                   help=f"Filename in the repo (default {DEFAULT_MODEL_FILE})")
    p.add_argument("--n-gpu-layers", type=int, default=99,
                   help="Number of transformer layers to offload to GPU (default 99 = all). 0 = pure CPU.")
    p.add_argument("--n-threads", type=int, default=None,
                   help="CPU threads for llama.cpp (default: llama.cpp picks based on cores)")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--calls", type=int, default=6,
                   help="How many inference calls to run (default 6 = alternating short/long prompts)")
    p.add_argument("--out-dir", type=Path, default=Path("../data/bench_gguf"))
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    slug = (args.model + "_" + args.file).replace("/", "_").replace(".", "_").lower()
    out_path = args.out_dir / f"{slug}_{stamp}.json"

    result = run_bench(args)
    if "error" in result:
        return 3

    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print_summary(result["summary"])
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
