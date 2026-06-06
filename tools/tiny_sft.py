"""
tiny_sft.py — the Day 3 learnability gate.

Train the candidate base model on a small slice of self-play data (~150
rows × 1-2 epochs) and emit a checkpoint that the format gate can re-grade.

Question this script answers: "given a few hundred examples, can this
model learn to emit <think>...</think>{move,taunt}?" — NOT "does it
already speak that format" (the zero-shot probe answers that). A
zero-shot failure is not a swap trigger; a learnability failure is.

Usage (from mind-of-tashi/, with a venv that has torch+cuda+trl+datasets):
    python -m tools.tiny_sft                              # defaults
    python -m tools.tiny_sft --rows 300 --epochs 2        # bigger run
    python -m tools.tiny_sft --base kshitijthakkar/tracegenix-mini-sft-clean-3ep

Then:
    python -m tools.format_gate --model ../models/loggenix_tiny_sft --n 20 --device cuda

Output checkpoint: ../models/<slug>_tiny_sft/ (relative to mind-of-tashi/).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


DEFAULT_BASE = "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1"
DEFAULT_SFT_GLOB = "../data/sft/sft_*.jsonl"
DEFAULT_OUT_DIR = "../models"


def iter_sft_rows(paths: List[Path]) -> Iterator[Dict[str, Any]]:
    """Yield rows that have a valid messages=[system,user,assistant] triple.

    Skips per-file `_kind=header` metadata rows produced by prep_sft.py.
    """
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = row.get("messages")
                if not msgs or not isinstance(msgs, list):
                    continue
                roles = [m.get("role") for m in msgs]
                if roles != ["system", "user", "assistant"]:
                    continue
                # Sanity: assistant must contain a <think> marker — otherwise
                # this row teaches the wrong format and would poison the gate.
                if "<think>" not in msgs[-1].get("content", ""):
                    continue
                yield {"messages": msgs}


def slugify(repo_id: str) -> str:
    return repo_id.split("/")[-1].replace(".", "_").lower() + "_tiny_sft"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Day 3 learnability gate: tiny SFT for the format check.")
    p.add_argument("--base", default=DEFAULT_BASE, help=f"base model repo id (default {DEFAULT_BASE})")
    p.add_argument("--sft-glob", default=DEFAULT_SFT_GLOB, help="glob for SFT JSONL files relative to cwd")
    p.add_argument("--rows", type=int, default=150, help="how many rows to train on (default 150)")
    p.add_argument("--epochs", type=float, default=2.0, help="epochs (default 2.0)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bs", type=int, default=1, help="per-device batch size")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="trade speed for VRAM (useful on 6GB cards)")
    args = p.parse_args(argv)

    # Lazy imports so the script can `--help` without heavy deps.
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"ERROR: missing dep ({e}). Need: torch, transformers, trl, datasets.", file=sys.stderr)
        return 3

    # 1. Collect rows.
    glob_root = Path(".").resolve()
    glob_pattern = args.sft_glob
    files = sorted(glob_root.glob(glob_pattern))
    if not files:
        print(f"ERROR: no SFT files matched {glob_pattern} from {glob_root}", file=sys.stderr)
        return 4
    print(f"[tiny_sft] candidate SFT files: {len(files)}", file=sys.stderr)
    for f in files:
        print(f"           - {f.name} ({f.stat().st_size//1024} KB)", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    for row in iter_sft_rows(files):
        rows.append(row)
        if len(rows) >= args.rows:
            break

    if len(rows) < args.rows:
        print(
            f"WARNING: only {len(rows)} valid rows found (asked for {args.rows}); "
            f"proceeding with what we have.",
            file=sys.stderr,
        )
    print(f"[tiny_sft] using {len(rows)} rows", file=sys.stderr)

    dataset = Dataset.from_list(rows)

    # 2. Load model + tokenizer.
    print(f"[tiny_sft] loading tokenizer: {args.base}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[tiny_sft] loading model: {args.base} (bf16, cuda)", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # 3. Configure SFTTrainer. Conversational format = native in trl>=0.10.
    out_dir = Path(args.out_dir).resolve() / slugify(args.base)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[tiny_sft] output dir: {out_dir}", file=sys.stderr)

    cfg = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        max_length=args.max_seq_len,
        completion_only_loss=True,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tok,
    )

    print(f"[tiny_sft] starting training: {args.epochs} epochs, bs={args.bs}, "
          f"grad_accum={args.grad_accum}, lr={args.lr}", file=sys.stderr)
    trainer.train()

    # 4. Save final model (TRL also saved per-epoch — this overwrites with the final).
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[tiny_sft] saved checkpoint to {out_dir}", file=sys.stderr)
    print(f"\nNext: re-run the format gate against this checkpoint:")
    print(f"    python -m tools.format_gate --model {out_dir} --device cuda --print-failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
