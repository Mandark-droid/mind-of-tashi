"""
push_live_to_hub.py — upload sealed live-gameplay matches to a PRIVATE HF Dataset.

Companion to push_to_hub.py. Differences:

- Reads from data/live/*.jsonl (the live game's per-match files).
- Only uploads files that contain exactly one `_kind=session_complete` row —
  half-finished matches (player closed the tab mid-game) stay local until
  next time `live_traces.end_session` seals them, or forever if abandoned.
- Each row is already a multi-turn SFT example: the model sees the FULL
  conversation across all rounds, so it learns long-range patterns rather
  than relying on the prompt's history-block summary.
- Target repo: kshitijthakkar/mind-of-tashi-live-traces (private) by default.

USAGE (from mind-of-tashi/):
  python -m tools.push_live_to_hub                    # default repo, private
  python -m tools.push_live_to_hub --dry-run          # list sealed files only
  python -m tools.push_live_to_hub --repo my/repo
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # mind-of-tashi/
REPO_ROOT = ROOT.parent     # mind-of-tashi-scaffold/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


DATASET_CARD = """\
---
license: cc-by-sa-4.0
language:
  - en
  - hi
  - sa
tags:
  - self-play
  - reasoning
  - game
  - bilingual
  - sft
  - live
  - multi-turn
size_categories:
  - n<1K
pretty_name: Mind of Tashi — live gameplay traces
configs:
  - config_name: default
    default: true
    data_files:
      - split: train
        path:
          - "*.jsonl"
          - "**/*.jsonl"
---

# Mind of Tashi — live gameplay traces (private)

Every row in this dataset is **one complete match** between a real human
player and the Mind of Tashi AI. Each row is a single multi-turn `messages`
array — system prompt once at the start, then one `(user, assistant)` pair
per round — so the model learns to read the full match history and spot
the challenger's pattern over many turns, not just from a summarised
history block in the prompt.

This is the **long-context complement** to the synthetic self-play SFT
data in `kshitijthakkar/mind-of-tashi-traces` (which has ~4k-token,
single-turn rows). Mixing both at SFT time gives:

- the synthetic short-turn rows: high persona diversity, lots of moves
- these live multi-turn rows: real pattern reading across whole matches

**This repo is private.** Real players' moves are recorded; while no
usernames or personal identifiers are stored in the SFT-target portion,
the metadata may include the leaderboard display name. Treat accordingly.

## Schema

```jsonl
{
  "messages": [
    {"role": "system",    "content": "<persona system prompt>"},
    {"role": "user",      "content": "<round 1 state prompt>"},
    {"role": "assistant", "content": "<round 1 ai response, <think>...</think> + JSON>"},
    {"role": "user",      "content": "<round 2 state prompt>"},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "_meta": {
    "persona": "norbu",
    "total_turns": 14,
    "ai_moves":     ["GUARD","GUARD","STRIKE",...],
    "player_moves": ["STRIKE","STRIKE","FOCUS",...],
    "outcome": "ladder_clear",
    "won": true,
    "total_seconds": 187.4,
    "username": "<leaderboard name or null>",
    "source": "live"
  }
}
```
"""


def collect_sealed(live_dir: Path) -> List[Path]:
    """Return jsonl files under live_dir that have a session_complete row."""
    out: List[Path] = []
    for p in sorted(live_dir.rglob("*.jsonl")):
        if p.name.startswith("."):
            continue
        if p.stat().st_size == 0:
            continue
        # Cheap check: read up to the first 2 lines (session_complete should
        # be the only row, but be tolerant). Don't load multi-MB files into
        # memory.
        sealed = False
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    break
                if obj.get("_kind") == "session_complete":
                    sealed = True
                    break
                if i > 50:  # not sealed in any reasonable position
                    break
        if sealed:
            out.append(p)
    return out


def stage_sealed_only(files: List[Path], live_dir: Path, staging: Path) -> List[Path]:
    """Copy each sealed file to staging, keeping only the session_complete row."""
    staged: List[Path] = []
    for src in files:
        rel = src.relative_to(live_dir)
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if obj.get("_kind") == "session_complete":
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
        if kept == 0:
            dst.unlink()
            continue
        staged.append(dst)
    return staged


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=str,
                        default=os.environ.get(
                            "LIVE_TRACES_REPO",
                            "kshitijthakkar/mind-of-tashi-live-traces",
                        ),
                        help="target HF Dataset repo (default from "
                             "LIVE_TRACES_REPO env or kshitijthakkar/mind-of-tashi-live-traces)")
    parser.add_argument("--live-dir", type=str,
                        default=str(REPO_ROOT / "data" / "live"),
                        help="local live-traces directory (default ../data/live/)")
    parser.add_argument("--public", action="store_true",
                        help="create repo as PUBLIC (default: private)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list sealed files; push nothing")
    parser.add_argument("--commit-message", type=str, default=None,
                        help="commit message on the Hub side")
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN missing — cannot push to Hub")

    live_dir = Path(args.live_dir)
    if not live_dir.exists():
        print(f"[push_live] live dir not found: {live_dir} — nothing to push")
        return

    sealed = collect_sealed(live_dir)
    if not sealed:
        print(f"[push_live] no sealed matches under {live_dir} — nothing to push")
        return

    total_kb = sum(p.stat().st_size for p in sealed) / 1024
    print(f"[push_live] target repo: {args.repo} (private={not args.public})")
    print(f"[push_live] live dir:    {live_dir}")
    print(f"[push_live] will upload {len(sealed)} sealed match files "
          f"({total_kb / 1024:.1f} MB):")
    for p in sealed:
        print(f"             {p.relative_to(live_dir)}  ({p.stat().st_size / 1024:.1f} KB)")

    if args.dry_run:
        print("[push_live] --dry-run: nothing uploaded")
        return

    from huggingface_hub import HfApi, create_repo

    private = not args.public
    create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=private,
        token=token,
        exist_ok=True,
    )

    # Stage cleaned copies (drop everything except session_complete rows)
    # to a tempdir; rewrite the dataset card per push.
    with tempfile.TemporaryDirectory(prefix="mist-live-stage-") as tmpdir:
        staging = Path(tmpdir)
        (staging / "README.md").write_text(DATASET_CARD, encoding="utf-8")
        staged_files = stage_sealed_only(sealed, live_dir, staging)
        if not staged_files:
            print("[push_live] no staged files after filtering — nothing to push")
            return

        api = HfApi(token=token)
        commit = api.upload_folder(
            folder_path=str(staging),
            path_in_repo="",
            repo_id=args.repo,
            repo_type="dataset",
            # IMPORTANT: include BOTH "*.jsonl" (root level) AND
            # "**/*.jsonl" (nested). The pattern "**/*.jsonl" requires
            # a parent directory, so it silently ignored every file
            # in the prior pushes because we stage them at the root
            # of the staging dir. Result: the Hub repo had only the
            # dataset card with no actual data for a full day.
            allow_patterns=["*.jsonl", "**/*.jsonl", "README.md"],
            commit_message=(args.commit_message or "live traces sync"),
        )
    url = getattr(commit, "commit_url", None) or str(commit)
    print(f"[push_live] pushed → {url}")


if __name__ == "__main__":
    main()
