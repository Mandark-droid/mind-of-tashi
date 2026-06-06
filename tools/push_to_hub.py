"""
push_to_hub.py — sync the local data/ directory to a PRIVATE HF Dataset repo.

The collected self-play harvest + filtered SFT corpus are too valuable to
live only on one machine. This script pushes both into one private dataset
repo so:
  - the 10-day cron has an off-machine backup
  - the SFT trainer in-window can pull data with `datasets.load_dataset(repo_id)`
  - we can share the dataset with collaborators by adding them to the repo

The repo is PRIVATE by default. Re-running the script is idempotent — files
are uploaded with overwrite semantics, no duplicates.

USAGE (from mind-of-tashi/):
  python -m tools.push_to_hub                              # default kshitijthakkar/mind-of-tashi-traces
  python -m tools.push_to_hub --repo my-user/my-dataset
  python -m tools.push_to_hub --dry-run                    # list files, don't upload
  python -m tools.push_to_hub --public                     # only if you mean it
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
ROOT = HERE.parent      # mind-of-tashi/
REPO_ROOT = ROOT.parent  # mind-of-tashi-scaffold/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force UTF-8 stdout/stderr on Windows cp1252 consoles (mirrors prep_sft).
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
size_categories:
  - n<10K
pretty_name: The Mind of Tashi — self-play traces
configs:
  - config_name: sft
    default: true
    data_files:
      - split: train
        path: "sft/*.jsonl"
  - config_name: sft_multiturn
    data_files:
      - split: train
        path: "sft_multi/*.jsonl"
  - config_name: selfplay
    data_files:
      - split: train
        path: "selfplay/*.jsonl"
  - config_name: live
    data_files:
      - split: train
        path: "live/*.jsonl"
  - config_name: baselines
    data_files:
      - split: train
        path: "baselines/*.jsonl"
---

# The Mind of Tashi — self-play traces (private)

Self-play data harvested for SFT of a small reasoning model that plays
**The Mind of Tashi** — a simultaneous-commit ritual fighting game where
the opponent's `<think>` block is the centrepiece of the player
experience. (Repo slug `mind-of-tashi-traces` predates the 2026-05-27
rename and is kept stable to avoid breaking the harvest cron; the
public copy at hackathon launch will live under `mind-of-tashi-traces`.)

**This repo is private.** The dataset is part of an unfinished hackathon
submission and is not for public consumption yet.

## Layout

```
selfplay/
  selfplay_<UTC>.jsonl    raw harvest output, one row per side per turn
                          (flattened schema for the Hub viewer; nested
                          fields like state/teacher_meta dropped, provider
                          + model promoted to scalar columns)
sft/
  sft_<date>.jsonl        single-turn SFT corpus, TRL conversational
                          (messages = [system, user, assistant], _meta)
sft_multi/
  sft_<date>.jsonl        multi-turn SFT corpus, one example per match
                          (messages = [system, user_1, asst_1, user_2,
                          asst_2, ...]) -- same shape as the live/ config
live/
  live-<id>.jsonl         real-player matches captured during gameplay;
                          ONE multi-turn row per match (system + N user/
                          assistant pairs); long-context complement to
                          the single-turn synthetic SFT
baselines/
  <model>_pre-sft_*.jsonl pre-fine-tune evaluation runs against the
                          locked SFT student (tracegenix-mini); preserved
                          as the "before" picture for the post-SFT and
                          post-GRPO comparison in the submission essay
```

## Configs

The Hub viewer exposes five configs (`sft` is default):

| Config | Files | Shape | One example = |
|---|---|---|---|
| `sft` (default) | `sft/*.jsonl` | single-turn TRL conversational | one AI turn (history block summarises prior rounds) |
| `sft_multiturn` | `sft_multi/*.jsonl` | multi-turn TRL conversational | one whole match (model sees full conversation across rounds) |
| `selfplay` | `selfplay/*.jsonl` | flattened raw harvest, provider-tagged | one turn per side per round |
| `live` | `live/*.jsonl` | multi-turn conversational (one per real-player match) | one sealed real match |
| `baselines` | `baselines/*.jsonl` | pre-fine-tune eval rows | the "before" picture for the SFT student |

The single-turn `sft` and multi-turn `sft_multiturn` configs are
generated from the same selfplay harvest by `tools/prep_sft.py
--shape {single,multi}`. They're complementary: single-turn rows
densely teach format + register, multi-turn rows teach long-range
pattern reading. At SFT time, train on either or
`datasets.concatenate_datasets([sft, sft_multiturn])` to combine.

## Role contract (the asymmetry)

The live game is asymmetric: the human player commits a **move only**;
the AI opponent emits `<think>...</think>` + JSON move + taunt. Self-play
mirrors that:

- **player** rows (`role: "player"`) — move-only stand-ins for the user.
  `<think>` / `taunt` / `raw_completion` are stripped at harvest time.
  `is_sft_target: false`.
- **opponent** rows (`role: "opponent"`) — the side we're training. Full
  `<think>` + taunt + raw_completion preserved. `is_sft_target: true`,
  TRL `messages: [system, user, assistant]` triple included.

## Bilingual register

`<think>` is targeted as **English + IAST Hindi/Sanskrit code-switched**
(*prahār, rakṣā, prāṇa, dṛṣṭi, abhyāsa*, …). The SFT-prep step records
`_meta.bilingual_hits` per row (count of normalised lexicon matches in
the thinking). Downstream training stratifies on this score; English-only
rows are kept but down-weighted.

## Provenance

Generated by `tools/selfplay.py` rotating across a pool of free-tier
inference providers (Gemini family, Mistral, Sarvam, OpenRouter free
models) with per-spec daily quota tracking. Each row carries
`teacher_meta.provider` + `teacher_meta.model` for full traceability.
"""


def collect_jsonl(data_dir: Path) -> List[Path]:
    """List every *.jsonl under data_dir, excluding hidden/state files."""
    return [
        p for p in sorted(data_dir.rglob("*.jsonl"))
        if not p.name.startswith(".") and p.name != "cron.log"
    ]


HEADER_KINDS = {"header", "sft_header"}


def _live_file_is_sealed(path: Path) -> bool:
    """Return True if the live/*.jsonl file contains a `_kind=session_complete`
    row. Unsealed live files are per-turn durability scratch and have a
    different schema than sealed (consolidated multi-turn) files, which
    confuses the Hub viewer's parquet converter for the `live` config."""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("_kind") == "session_complete":
                    return True
    except OSError:
        return False
    return False


def _is_unrepairable_truncation(obj: dict) -> bool:
    """Detect rows where the provider truncated mid-<think> without emitting
    a closing tag AND without ever reaching the JSON line. These are
    parse_reply-fallback candidates at game time (no move available) and
    prep_sft drops them anyway, but they otherwise litter the Hub viewer
    with malformed-format examples. Repairable rows (have a trailing
    `{`-anchored JSON) are NOT touched here — those are healed in place
    by repair_unbalanced_think() upstream."""
    for src in (obj.get("raw_completion"), (obj.get("messages") or [{}])[-1].get("content") if obj.get("messages") else None):
        if not src or not isinstance(src, str):
            continue
        if "<think>" in src and "</think>" not in src and src.rfind("{") < 0:
            return True
    return False


def _flatten_selfplay_row(obj: dict) -> dict:
    """Project a selfplay row to a flat, viewer-friendly schema.

    The local row schema is rich (nested `state.history` of variable shape,
    provider-specific `teacher_meta`, optional `messages`) — great for
    analytics but it breaks Hub parquet conversion because nested types
    can't be reconciled across heterogeneous rows.

    We strip the heavy nested fields and flatten provider info to two
    scalar columns. The full row is preserved locally; this projection is
    Hub-only. Schema after flattening (one column type per name):

      match_id, turn, side, role, is_sft_target, persona, opponent_persona,
      legal_moves[list<str>], move, think (nullable), taunt (nullable),
      raw_completion (nullable), turn_reward, outcome_reward (nullable),
      match_length, final_hp_player, final_hp_opponent,
      provider, model (nullable)
    """
    tm = obj.get("teacher_meta") or {}
    provider = (
        tm.get("pool_spec")
        or tm.get("provider")
        or tm.get("backend")
        or "unknown"
    )
    flat = {
        "match_id": obj.get("match_id"),
        "turn": obj.get("turn"),
        "side": obj.get("side"),
        "role": obj.get("role"),
        "is_sft_target": bool(obj.get("is_sft_target", False)),
        "persona": obj.get("persona"),
        "opponent_persona": obj.get("opponent_persona"),
        "legal_moves": obj.get("legal_moves") or [],
        "move": obj.get("move"),
        "think": obj.get("think"),
        "taunt": obj.get("taunt"),
        "raw_completion": obj.get("raw_completion"),
        "turn_reward": obj.get("turn_reward"),
        "outcome_reward": obj.get("outcome_reward"),
        "match_length": obj.get("match_length"),
        "final_hp_player": obj.get("final_hp_player"),
        "final_hp_opponent": obj.get("final_hp_opponent"),
        "provider": provider,
        "model": tm.get("model"),
    }
    return flat


def stage_clean_jsonl(data_dir: Path, staging: Path) -> List[Path]:
    """Mirror data_dir to staging:
      - skip empty files
      - drop `_kind` header rows
      - selfplay rows: flatten to viewer-friendly schema (see _flatten_selfplay_row)
      - sft rows: unchanged (the existing schema is already flat-ish)

    Returns the list of staged jsonl files (under `staging`).
    """
    staged: List[Path] = []
    for src in collect_jsonl(data_dir):
        rel = src.relative_to(data_dir)
        if src.stat().st_size == 0:
            print(f"[push_to_hub] skip empty file {rel}")
            continue

        # Unsealed live/*.jsonl (per-turn durability scratch, no
        # session_complete row) have a different schema from sealed
        # files, breaking the Hub viewer's parquet schema inference for
        # the `live` config. Skip them — push_live_to_hub.py applies the
        # same rule against the dedicated live-traces repo.
        if "live" in rel.parts and rel.name.startswith("live-"):
            if not _live_file_is_sealed(src):
                print(f"[push_to_hub] skip unsealed live file {rel}")
                continue

        is_selfplay = "selfplay" in rel.parts

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
                if isinstance(obj, dict) and obj.get("_kind") in HEADER_KINDS:
                    continue
                # Drop rows where the model truncated mid-<think> without
                # ever emitting JSON. prep_sft already drops them from SFT
                # files via the raw_not_terminated filter; applying the
                # same rule at push time across BOTH configs (sft + selfplay)
                # is belt-and-suspenders so the Hub dataset is guaranteed
                # to never expose an unrepairable row.
                if isinstance(obj, dict) and _is_unrepairable_truncation(obj):
                    continue
                if is_selfplay and isinstance(obj, dict):
                    obj = _flatten_selfplay_row(obj)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
        if kept == 0:
            print(f"[push_to_hub] skip {rel}: header-only / no data rows")
            dst.unlink()
            continue
        staged.append(dst)

    for f in data_dir.iterdir():
        if f.is_file() and not f.name.endswith(".jsonl") and not f.name.startswith("."):
            shutil.copy2(f, staging / f.name)
    return staged


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=str, default="kshitijthakkar/mind-of-tashi-traces",
                        help="target HF Dataset repo (default kshitijthakkar/mind-of-tashi-traces)")
    parser.add_argument("--data-dir", type=str,
                        default=str(REPO_ROOT / "data"),
                        help="local data directory to mirror (default ../data/)")
    parser.add_argument("--public", action="store_true",
                        help="create the repo as PUBLIC (default: private). Be deliberate.")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be uploaded, push nothing")
    parser.add_argument("--commit-message", type=str, default=None,
                        help="commit message on the Hub side (default: 'daily harvest sync')")
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN missing in environment (.env) — cannot push")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    files = collect_jsonl(data_dir)
    if not files:
        print(f"[push_to_hub] no jsonl files under {data_dir} — nothing to push")
        return

    total_bytes = sum(p.stat().st_size for p in files)
    print(f"[push_to_hub] target repo: {args.repo} (private={not args.public})")
    print(f"[push_to_hub] data dir:    {data_dir}")
    print(f"[push_to_hub] will upload {len(files)} jsonl files "
          f"({total_bytes / 1024 / 1024:.1f} MB):")
    for p in files:
        rel = p.relative_to(data_dir)
        print(f"               {rel}  ({p.stat().st_size / 1024:.1f} KB)")

    if args.dry_run:
        print("[push_to_hub] --dry-run: nothing uploaded")
        return

    # Lazy import so --dry-run / --help don't need huggingface_hub installed.
    from huggingface_hub import HfApi, create_repo

    private = not args.public
    create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=private,
        token=token,
        exist_ok=True,
    )

    # Always rewrite the dataset card on push so card changes (e.g. new
    # configs block) reach the Hub. The local one is a working copy.
    card_path = data_dir / "README.md"
    card_path.write_text(DATASET_CARD, encoding="utf-8")

    # Stage a clean copy: strip _kind header rows + skip empty files. The
    # Hub dataset viewer rejects schema-heterogeneous rows in one config.
    with tempfile.TemporaryDirectory(prefix="mist-hub-stage-") as tmpdir:
        staging = Path(tmpdir)
        staged_files = stage_clean_jsonl(data_dir, staging)
        if not staged_files:
            print("[push_to_hub] nothing to upload after staging")
            return
        total_kb = sum(p.stat().st_size for p in staged_files) / 1024
        print(f"[push_to_hub] staged {len(staged_files)} cleaned jsonl files "
              f"({total_kb / 1024:.1f} MB) for upload")

        # Compute delete_patterns so the Hub stays in sync with local
        # staging. Anything in the live/ folder that is NOT in our staged
        # set (i.e. unsealed live files) gets removed from the Hub —
        # otherwise a previous push leaves the unsealed file on the Hub
        # and breaks the `live` config's schema. upload_folder's
        # delete_patterns runs against the remote tree before upload.
        staged_live = {p.name for p in staged_files if "live" in p.parts}
        local_live = {p.name for p in (data_dir / "live").glob("*.jsonl")} if (data_dir / "live").exists() else set()
        unsealed_live_names = sorted(local_live - staged_live)
        delete_patterns = [f"live/{n}" for n in unsealed_live_names]
        if delete_patterns:
            print(f"[push_to_hub] will delete {len(delete_patterns)} unsealed "
                  f"live file(s) from the Hub:")
            for p in delete_patterns:
                print(f"               {p}")

        api = HfApi(token=token)
        commit = api.upload_folder(
            folder_path=str(staging),
            path_in_repo="",
            repo_id=args.repo,
            repo_type="dataset",
            allow_patterns=["**/*.jsonl", "README.md"],
            delete_patterns=delete_patterns or None,
            commit_message=(args.commit_message or "daily harvest sync"),
        )
    url = getattr(commit, "commit_url", None) or str(commit)
    print(f"[push_to_hub] pushed → {url}")


if __name__ == "__main__":
    main()
