"""
leaderboard.py — durable, public, append-only run log on a Hugging Face Dataset.

Why a Dataset (not Space local FS):
  Spaces have ephemeral filesystems; persistent storage costs extra. A Dataset
  repo is the natural "shared spreadsheet" on the Hub — durable, public,
  versioned, and queryable from anywhere.

Why one file per run (not one big JSONL):
  Concurrent player writes are perfectly safe when each write is a uniquely
  named file. A single shared JSONL would race on read-modify-write and
  silently drop submissions. The dataset reader just lists `runs/*.json` and
  parses them; the leaderboard logic dedupes per user to their best clear.

Ranking (in `top_runs`):
  Only `won` runs count. For each user, keep the single best attempt — fewest
  total turns, ties broken by lowest total seconds. Then sort that one-row-per
  -user list the same way. Storage keeps every attempt for analytics; the
  board just collapses duplicates.

Config (env):
  LEADERBOARD_REPO   "<owner>/<dataset-name>"   required to write or read
  HF_TOKEN           write-scoped HF token       required to write
  LEADERBOARD_DISABLE "1" forces the no-op path  useful for local dev
"""

from __future__ import annotations
import io
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

LEADERBOARD_REPO = os.environ.get("LEADERBOARD_REPO", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
DISABLED = os.environ.get("LEADERBOARD_DISABLE", "0") == "1"

_RUNS_PREFIX = "runs/"

# Lazy-imported so the game still boots if huggingface_hub is absent locally.
_api = None
_api_err: Optional[str] = None


def _get_api():
    global _api, _api_err
    if _api is not None or _api_err is not None:
        return _api
    if DISABLED or not LEADERBOARD_REPO:
        _api_err = "leaderboard disabled or LEADERBOARD_REPO unset"
        return None
    try:
        from huggingface_hub import HfApi  # noqa: WPS433
        _api = HfApi(token=HF_TOKEN)
        return _api
    except Exception as exc:  # huggingface_hub missing
        _api_err = f"huggingface_hub unavailable: {exc}"
        return None


def status() -> Dict[str, Any]:
    """Surface readiness to the UI so it can show 'leaderboard offline' politely."""
    api = _get_api()
    return {
        "ready": api is not None,
        "repo": LEADERBOARD_REPO or None,
        "can_write": bool(api and HF_TOKEN),
        "error": _api_err,
    }


def submit_run(
    username: str,
    source: str,
    total_turns: int,
    total_seconds: float,
    per_level: List[Dict[str, Any]],
    won: bool,
    backend: str,
) -> Dict[str, Any]:
    """
    Push a single completed run as its own JSON file under runs/<ts>-<uuid>.json.

    `source` is 'hf-oauth' or 'guest' so the leaderboard can mark verified
    entries differently from typed-name guests. `per_level` is the list of
    {opponent_id, turns, seconds, won} entries that compose the run.
    """
    api = _get_api()
    if api is None or not HF_TOKEN:
        return {"ok": False, "reason": _api_err or "no HF_TOKEN; run not persisted"}

    run_id = uuid.uuid4().hex[:12]
    ts = int(time.time())
    row = {
        "run_id": run_id,
        "ts": ts,
        "username": (username or "anon").strip()[:40],
        "source": source,                     # 'hf-oauth' | 'guest'
        "total_turns": int(total_turns),
        "total_seconds": round(float(total_seconds), 2),
        "per_level": per_level,
        "won": bool(won),
        "backend": backend,
    }
    path = f"{_RUNS_PREFIX}{ts}-{run_id}.json"
    payload = io.BytesIO(json.dumps(row, ensure_ascii=False).encode("utf-8"))
    try:
        api.upload_file(
            path_or_fileobj=payload,
            path_in_repo=path,
            repo_id=LEADERBOARD_REPO,
            repo_type="dataset",
            commit_message=f"run {run_id} ({row['username']})",
        )
        return {"ok": True, "run_id": run_id, "path": path}
    except Exception as exc:
        return {"ok": False, "reason": f"upload failed: {exc}"}


def top_runs(limit: int = 20, max_fetch: int = 500) -> Dict[str, Any]:
    """
    List the dataset's run files, parse the most recent `max_fetch`, dedupe per
    user to their best win, sort, and return up to `limit` rows. Bounded fetch
    keeps the request snappy as the dataset grows.
    """
    api = _get_api()
    if api is None:
        return {"ok": False, "reason": _api_err, "rows": []}

    try:
        files = api.list_repo_files(repo_id=LEADERBOARD_REPO, repo_type="dataset")
    except Exception as exc:
        return {"ok": False, "reason": f"list failed: {exc}", "rows": []}

    run_paths = sorted(
        [f for f in files if f.startswith(_RUNS_PREFIX) and f.endswith(".json")],
        reverse=True,                          # newest first; ts is in the name
    )[:max_fetch]

    rows: List[Dict[str, Any]] = []
    from huggingface_hub import hf_hub_download  # noqa: WPS433
    for p in run_paths:
        try:
            local = hf_hub_download(
                repo_id=LEADERBOARD_REPO,
                repo_type="dataset",
                filename=p,
                token=HF_TOKEN,
            )
            with open(local, "r", encoding="utf-8") as fh:
                rows.append(json.load(fh))
        except Exception:
            continue  # one bad file shouldn't sink the whole board

    # one-row-per-user: best win wins ties go to fewer turns, then less time
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if not r.get("won"):
            continue
        key = (r.get("source", "guest"), (r.get("username") or "").lower())
        cur = best.get(key)
        if cur is None or _better(r, cur):
            best[key] = r

    ranked = sorted(
        best.values(),
        key=lambda r: (r.get("total_turns", 9999), r.get("total_seconds", 1e9)),
    )[:limit]

    return {"ok": True, "rows": ranked, "total_seen": len(rows)}


def _better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True if `a` is a better summit clear than `b`."""
    return (a.get("total_turns", 9999), a.get("total_seconds", 1e9)) \
        <  (b.get("total_turns", 9999), b.get("total_seconds", 1e9))
