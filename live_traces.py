"""
live_traces.py — capture AI turns from REAL gameplay and ship them as SFT data.

Mirrors the self-play harness's TRL-conversational schema, but the rows come
from actual human players climbing the ladder. The opponent's full
<think>+JSON output (one assistant turn per round) is captured alongside the
state that produced it, exactly as a SFTTrainer would see it.

Key differences from the synthetic self-play harvest:

  * we DO NOT generate the player side — the human's move is what they
    clicked, and only the AI's turn becomes a training target.
  * one JSONL file per match (mirrors leaderboard.py's per-run file pattern)
    so concurrent sessions never race-overwrite. Each file holds the full
    session: one row per AI turn + a closing match-end row.
  * pushed to a SEPARATE public HF Dataset (set LIVE_TRACES_REPO, e.g.
    `build-small-hackathon/mind-of-tashi-live-traces`) so we can stratify
    training on synthetic vs real-player provenance later if we want.

Lifecycle:

  capture_turn(match_id, opp, state, ai_raw, ai_parsed, player_move)
       called from app.ai_turn after each AI response. Appends to
       data/live/<match_id>.jsonl. Fast (one append, no network).

  end_session(match_id, outcome, won, total_turns, total_seconds)
       called from app.submit_run when the player ends the run. Seals the
       file into one multi-turn SFT row, then (when LIVE_TRACES_REPO +
       HF_TOKEN are configured) pushes that sealed file to the Hub dataset
       IMMEDIATELY in a background thread. On a deployed Space there is no
       cron and container storage is ephemeral — pushing at seal time is the
       only way real-player traces survive a restart. The local daily cron
       (tools/push_live_to_hub.py) remains for dev boxes; both writers use
       the same per-match filename at the repo root, so they are idempotent
       with each other.

If LIVE_TRACES_REPO or HF_TOKEN are missing, status().ready is False and
all calls are silent no-ops — the live game keeps working unaffected.
"""

from __future__ import annotations
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import prompts
from engine import MOVES
from opponents import Opponent


LIVE_TRACES_REPO = os.environ.get("LIVE_TRACES_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LIVE_TRACES_DISABLE = os.environ.get("LIVE_TRACES_DISABLE", "0") == "1"

# Local capture directory — gets uploaded daily by the cron.
HERE = Path(__file__).resolve().parent
LOCAL_DIR = Path(os.environ.get(
    "LIVE_TRACES_LOCAL_DIR",
    str(HERE.parent / "data" / "live"),
))

# Per-process file lock — the live game's Gradio queue is single-process by
# default, but we don't rely on that. POSIX appends are atomic for small
# writes anyway; this lock just keeps in-process turns from interleaving.
_LOCK = threading.RLock()

_api = None
_repo_ready = False


def _upload_sealed(path: Path, match_id: str) -> None:
    """Best-effort push of ONE sealed match file to the Hub dataset.

    Runs on a daemon thread from end_session — never blocks or fails the
    request. Per-match filenames mean concurrent sessions can't collide
    (same pattern as leaderboard.py's per-run files)."""
    global _api, _repo_ready
    if not (LIVE_TRACES_REPO and HF_TOKEN) or LIVE_TRACES_DISABLE:
        return
    try:
        from huggingface_hub import HfApi, create_repo  # noqa: WPS433
        if _api is None:
            _api = HfApi(token=HF_TOKEN)
        if not _repo_ready:
            # PUBLIC: live traces are part of the Sharing-is-Caring bundle.
            # No personal identifiers in the SFT target; the metadata only
            # carries the (already public) leaderboard display name.
            create_repo(LIVE_TRACES_REPO, repo_type="dataset", private=False,
                        token=HF_TOKEN, exist_ok=True)
            _repo_ready = True
        _api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,   # repo root — same layout as the cron
            repo_id=LIVE_TRACES_REPO,
            repo_type="dataset",
            commit_message=f"live match {match_id}",
        )
        print(f"[live_traces] pushed {path.name} -> {LIVE_TRACES_REPO}")
    except Exception as exc:
        print(f"[live_traces] hub push failed for {path.name}: {exc}")


def status() -> Dict[str, Any]:
    """For /live_traces_status diagnostics in the live game."""
    return {
        "ready": bool(LIVE_TRACES_REPO and HF_TOKEN and not LIVE_TRACES_DISABLE),
        "repo": LIVE_TRACES_REPO,
        "local_dir": str(LOCAL_DIR),
        "disabled": LIVE_TRACES_DISABLE,
    }


def new_match_id() -> str:
    """Generate a fresh match id. Called by app.py when state.round == 1 and
    no match_id is in the request — preserves session continuity."""
    return f"live-{uuid.uuid4().hex[:12]}"


def _path_for(match_id: str) -> Path:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    # match_id is server-controlled (we generate it ourselves on round 1)
    # but defensively scrub anything that could escape the dir.
    safe = "".join(c for c in match_id if c.isalnum() or c in "-_")[:64]
    if not safe:
        safe = new_match_id()
    return LOCAL_DIR / f"{safe}.jsonl"


def capture_turn(
    match_id: str,
    opp: Opponent,
    state: Dict[str, Any],
    ai_raw: str,
    ai_parsed: Dict[str, Any],
    player_move: Optional[str] = None,
    username: Optional[str] = None,
    source: Optional[str] = None,
    backend: Optional[str] = None,
) -> None:
    """Append one AI turn to the per-match JSONL file.

    `state` is the pre-resolve state as seen by the AI (i.e. what was passed
    into `Reasoner.choose_with_raw`). `ai_raw` is the model's full text blob,
    `ai_parsed` is the parsed dict (reasoning/move/taunt) — we save both.

    `player_move` is the human's committed move that round; it never enters
    the assistant target (which would break the blind-commit contract),
    but it's recorded in metadata for later analysis.
    """
    if LIVE_TRACES_DISABLE:
        return
    if not LIVE_TRACES_REPO:
        # local-only capture is still valuable (e.g. for local dev) — don't
        # skip just because the Hub repo isn't configured. The cron uploader
        # will silently skip uploading if the repo isn't set.
        pass

    legal = [m for m in MOVES if state["ai_prana"] >= MOVES[m]["cost"]]
    messages = [
        {"role": "system",    "content": prompts.build_system(opp)},
        {"role": "user",      "content": prompts.build_user(opp, state, legal)},
        {"role": "assistant", "content": ai_raw},
    ]
    row = {
        "_kind": "turn",
        "match_id": match_id,
        "turn": state.get("round"),
        "is_sft_target": True,
        "persona": opp.id,
        "move": ai_parsed.get("move"),
        "think": ai_parsed.get("reasoning"),
        "taunt": ai_parsed.get("taunt"),
        "raw_completion": ai_raw,
        "messages": messages,
        "_meta": {
            "player_move": player_move,
            "ai_hp_before": state.get("ai_hp"),
            "player_hp_before": state.get("player_hp"),
            "ai_prana_before": state.get("ai_prana"),
            "player_prana_before": state.get("player_prana"),
            "captured_at": time.time(),
            "username": username,
            "source": source,
            "backend": backend,
        },
    }
    path = _path_for(match_id)
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def end_session(
    match_id: str,
    outcome: str,
    won: bool,
    total_turns: Optional[int] = None,
    total_seconds: Optional[float] = None,
    username: Optional[str] = None,
) -> None:
    """Consolidate per-turn captures into ONE multi-turn SFT row and seal.

    During play we append per-turn rows for durability (server restart can't
    lose them). On end-of-match we rewrite the file as a SINGLE
    `_kind=session_complete` row whose `messages` field stretches across all
    turns of the match — system once, then alternating user/assistant for
    every round. That's the SFT target for long-context training: the model
    learns to read the full match history and spot the challenger's pattern
    over time, not just from the truncated history-block summary.

    The cron uploader only ships sealed files (those containing exactly one
    `session_complete` row), so half-finished matches stay local.
    """
    if LIVE_TRACES_DISABLE:
        return
    path = _path_for(match_id)
    if not path.exists():
        return  # no turns captured — nothing to seal

    # Read per-turn rows (skip anything that isn't a turn we already wrote).
    turn_rows: List[dict] = []
    with _LOCK:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if obj.get("_kind") == "turn":
                    turn_rows.append(obj)
        if not turn_rows:
            return

        # Build the multi-turn conversation. All turns share the same system
        # prompt (persona-derived); we emit it once.
        first_msgs = turn_rows[0].get("messages") or []
        if not first_msgs or first_msgs[0].get("role") != "system":
            return  # malformed; skip rather than ship garbage
        system_msg = first_msgs[0]
        messages: List[Dict[str, Any]] = [system_msg]
        for tr in turn_rows:
            tm = tr.get("messages") or []
            # tm is [system, user, assistant] — append the user + assistant only
            if len(tm) >= 3:
                messages.append(tm[1])
                messages.append(tm[2])

        persona = turn_rows[0].get("persona")
        ai_moves = [tr.get("move") for tr in turn_rows]
        player_moves = [
            (tr.get("_meta") or {}).get("player_move") for tr in turn_rows
        ]
        consolidated = {
            "_kind": "session_complete",
            "match_id": match_id,
            "is_sft_target": True,
            "messages": messages,
            "_meta": {
                "persona": persona,
                "total_turns": total_turns or len(turn_rows),
                "ai_moves": ai_moves,
                "player_moves": player_moves,
                "outcome": outcome,
                "won": bool(won),
                "total_seconds": total_seconds,
                "username": username,
                "source": "live",
                "sealed_at": time.time(),
            },
        }
        # Rewrite the file with just the consolidated row — per-turn rows
        # were durability scratch; the SFT-shape is what we keep.
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(consolidated, ensure_ascii=False) + "\n")

    # Ship it now — on a Space the container disk is ephemeral and there is
    # no cron, so seal time is the only reliable push point.
    threading.Thread(
        target=_upload_sealed, args=(path, match_id), daemon=True,
    ).start()
