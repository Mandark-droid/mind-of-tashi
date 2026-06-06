"""
selfplay.py — drive two Teacher instances against each other and harvest
<think>+move+taunt transcripts as JSONL.

ROLE ASYMMETRY (matches the live game)
--------------------------------------
The live game is asymmetric: the human commits a move only; the AI emits
<think>...</think> + move + taunt and the player reads the mind-scroll. Self-play
must respect that asymmetry — otherwise we'd train the model on a world where
the "user" also produces narrated reasoning, and inference-time history would
look nothing like training-time history.

Therefore in every match:

  * side "player"    — a Teacher whose only job is to generate plausible MOVES
                       in the user's place. May internally be a reasoning model
                       for stronger play, but its <think>/raw_completion/taunt
                       are stripped from the saved rows and NEVER become SFT
                       targets. Defaults to `mock` because the mock heuristic
                       gives more human-like aggressive play than two reasoning
                       models locking into a defensive stalemate.
  * side "opponent"  — the Teacher we're harvesting from. Full <think> + taunt
                       + raw_completion preserved. Tagged is_sft_target=True.

Blind-commit is still preserved: each side sees only its own `history`, never
the other's pending move. The two move generations are dispatched via
`asyncio.gather` so an API opponent and a fast player can fire concurrently.

USAGE (from mind-of-tashi/):
  # mock player vs Gemini opponent (the recommended default)
  python -m tools.selfplay --matches 8 \\
      --player-persona tashi --opponent-persona norbu \\
      --opponent-teacher gemini:gemini-2.0-flash

  # 5x5 sweep, mock vs Gemini, ~25 matches total
  python -m tools.selfplay --sweep --matches 1 \\
      --opponent-teacher gemini:gemini-2.0-flash

  # mixed providers; player side also reasoning (think still stripped from rows)
  python -m tools.selfplay --matches 5 \\
      --player-teacher mistral:mistral-small-latest \\
      --opponent-teacher gemini:gemini-2.0-flash

  # local Ollama on both sides, two different models
  python -m tools.selfplay --matches 5 \\
      --player-teacher ollama:llama3.1:8b \\
      --opponent-teacher ollama:qwen3:14b

  # mock/mock smoke (no network)
  python -m tools.selfplay --matches 2 --force-mock

OUTPUT
------
  data/selfplay/selfplay_<utc_iso>.jsonl
    First line is a "_kind=header" record (run config + role contract).
    Each subsequent line is one row per side per turn (schema below).

Row schema (per side per turn):
  {
    "match_id", "turn",
    "role": "player" | "opponent",
    "is_sft_target": bool,                       # true only for opponent rows
    "persona", "opponent_persona",
    "state": {"round","ai_hp","ai_prana","player_hp","player_prana","history"},
    "legal_moves": [...],
    "move": "STRIKE",
    "think": "..." | null,                       # null on player rows
    "taunt": "..." | null,                       # null on player rows
    "raw_completion": "<think>...</think>{...}" | null,  # null on player rows
    "messages": [                                # TRL "conversational" format;
      {"role": "system", "content": "..."},     # only present on opponent rows.
      {"role": "user",   "content": "..."},     # Directly consumable by TRL
      {"role": "assistant", "content": "..."}   # SFTTrainer — no prep needed.
    ],
    "teacher_meta": { provider, model, latency_ms, ... },
    "turn_reward": int,                          # HP inflicted on opponent this turn
    "outcome_reward": +1 | -1 | 0,               # filled at match end
    "match_length", "final_hp_player", "final_hp_opponent"
  }

Downstream SFT loading filters `is_sft_target == True` and feeds `messages`
straight into SFTTrainer.
"""

from __future__ import annotations
import argparse
import asyncio
import itertools
import json
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow `python -m tools.selfplay` from mind-of-tashi/.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # mind-of-tashi/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env (HF_TOKEN, GEMINI_API_KEY, OPENROUTER_API_KEY, MISTRAL_API_KEY,
# SARVAM_API_KEY) BEFORE importing teachers — they read env at construction.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from engine import MOVES, Fighter, RoundResult, apply, resolve  # noqa: E402
from opponents import BY_ID, LADDER, Opponent  # noqa: E402
from teachers import (  # noqa: E402
    PoolExhausted, QuotaState, Teacher, make_pool, make_teacher,
)
from teachers.base import build_messages  # noqa: E402


def _outcome_phrase(res: RoundResult, p_name: str, o_name: str) -> str:
    """One-line summary of a resolved round, used to render `history.outcome`
    inside each side's next prompt."""
    if res.b_dmg_taken and res.a_dmg_taken:
        return f"trade: {p_name} -{res.a_dmg_taken}, {o_name} -{res.b_dmg_taken}"
    if res.b_dmg_taken:
        return f"{o_name} -{res.b_dmg_taken}"
    if res.a_dmg_taken:
        return f"{p_name} -{res.a_dmg_taken}"
    return "no blood"


def _legal(prana: int) -> List[str]:
    return [m for m in MOVES if prana >= MOVES[m]["cost"]]


async def run_match(
    player_teacher: Teacher,
    opponent_teacher: Teacher,
    player_opp: Opponent,
    opponent_opp: Opponent,
    match_id: str,
    max_rounds: int = 30,
) -> List[Dict]:
    fp = Fighter(player_opp.name)
    fo = Fighter(opponent_opp.name)
    hist_p: List[Dict] = []  # player's perspective: ai_move=player's own move
    hist_o: List[Dict] = []  # opponent's perspective: ai_move=opponent's own move
    rows: List[Dict] = []
    p_rows: List[Dict] = []
    o_rows: List[Dict] = []
    rnd = 0

    for rnd in range(1, max_rounds + 1):
        state_p = {
            "round": rnd,
            "ai_hp": fp.hp, "ai_prana": fp.prana,
            "player_hp": fo.hp, "player_prana": fo.prana,
            "history": list(hist_p),
        }
        state_o = {
            "round": rnd,
            "ai_hp": fo.hp, "ai_prana": fo.prana,
            "player_hp": fp.hp, "player_prana": fp.prana,
            "history": list(hist_o),
        }
        legal_p = _legal(fp.prana)
        legal_o = _legal(fo.prana)

        # Blind commit, in parallel. Neither call sees the other's pending move.
        result_p, result_o = await asyncio.gather(
            player_teacher.choose(player_opp, state_p),
            opponent_teacher.choose(opponent_opp, state_o),
        )

        hp_p_before, hp_o_before = fp.hp, fo.hp
        res = resolve(fp, fo, result_p.parsed["move"], result_o.parsed["move"])
        apply(fp, fo, res)

        p_turn_reward = max(0, hp_o_before - fo.hp)
        o_turn_reward = max(0, hp_p_before - fp.hp)
        outcome_phrase = _outcome_phrase(res, player_opp.name, opponent_opp.name)

        # Player row: think/taunt/raw_completion DELIBERATELY stripped.
        # The live game has no player-side narrated reasoning; training data must
        # match that. is_sft_target=False so the SFT-prep filter never sees these.
        # `messages` is also nulled (not omitted) so player + opponent rows share
        # one schema — required for the Hub dataset viewer's parquet conversion.
        row_p = {
            "match_id": match_id, "turn": rnd,
            "role": "player",
            "is_sft_target": False,
            "persona": player_opp.id, "opponent_persona": opponent_opp.id,
            "state": state_p, "legal_moves": legal_p,
            "move": result_p.parsed["move"],
            "think": None,
            "taunt": None,
            "raw_completion": None,
            "messages": None,
            "teacher_meta": {
                # keep provider+latency for analytics; drop nothing else useful
                k: v for k, v in result_p.meta.items()
                if k in ("provider", "backend", "model", "latency_ms", "retries", "fallback")
            },
            "turn_reward": p_turn_reward,
            "outcome_reward": None,
        }
        # Opponent row: full SFT target. Built in TRL "conversational" format
        # (`messages` = [system, user, assistant]) so a row is directly trainable
        # by SFTTrainer / TRL DataCollatorForCompletionOnlyLM without any prep
        # step. The system+user pair is exactly what the live game sends to
        # llm.Reasoner; the assistant content is the model's raw <think>...{json}
        # blob (NOT the parsed dict — the student must learn to produce both
        # the think and the JSON line together).
        opp_messages = build_messages(opponent_opp, state_o)
        opp_messages.append({"role": "assistant", "content": result_o.raw})

        row_o = {
            "match_id": match_id, "turn": rnd,
            "role": "opponent",
            "is_sft_target": True,
            "persona": opponent_opp.id, "opponent_persona": player_opp.id,
            "state": state_o, "legal_moves": legal_o,
            "move": result_o.parsed["move"],
            "think": result_o.parsed["reasoning"],
            "taunt": result_o.parsed["taunt"],
            "raw_completion": result_o.raw,
            "messages": opp_messages,
            "teacher_meta": result_o.meta,
            "turn_reward": o_turn_reward,
            "outcome_reward": None,
        }
        rows.extend([row_p, row_o])
        p_rows.append(row_p)
        o_rows.append(row_o)

        # Histories: each side's view is from their own POV (ai=self, player=other).
        hist_p.append({
            "round": rnd,
            "player_move": result_o.parsed["move"],
            "ai_move": result_p.parsed["move"],
            "outcome": outcome_phrase,
        })
        hist_o.append({
            "round": rnd,
            "player_move": result_p.parsed["move"],
            "ai_move": result_o.parsed["move"],
            "outcome": outcome_phrase,
        })

        if fp.hp <= 0 or fo.hp <= 0:
            break

    if fp.hp <= 0 and fo.hp <= 0:
        p_out, o_out = 0, 0
    elif fo.hp <= 0:
        p_out, o_out = +1, -1
    elif fp.hp <= 0:
        p_out, o_out = -1, +1
    else:
        p_out, o_out = 0, 0  # round cap = draw

    for r in p_rows:
        r["outcome_reward"] = p_out
    for r in o_rows:
        r["outcome_reward"] = o_out
    for r in rows:
        r["match_length"] = rnd
        r["final_hp_player"] = fp.hp
        r["final_hp_opponent"] = fo.hp

    return rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-play harness for The Mind of Tashi")
    p.add_argument("--matches", type=int, default=5,
                   help="matches per matchup (default 5)")
    p.add_argument("--player-persona", type=str, default="tashi",
                   help="persona id for the player side (default tashi)")
    p.add_argument("--opponent-persona", type=str, default="norbu",
                   help="persona id for the opponent side (default norbu)")
    p.add_argument("--sweep", action="store_true",
                   help="ignore --*-persona; run --matches per matchup across "
                        "all 5x5 persona pairs (mirrors included)")

    p.add_argument("--player-teacher", type=str, default=None,
                   help="teacher spec for the player side (default: mock — "
                        "human-like aggressive play, no API quota cost)")
    p.add_argument("--opponent-teacher", type=str, default=None,
                   help="teacher spec for the opponent side (default: mock; "
                        "this is the side whose <think> becomes SFT data)")
    p.add_argument("--teacher", type=str, default=None,
                   help="convenience: set BOTH sides to this spec. Overridden "
                        "by --player-teacher / --opponent-teacher.")
    p.add_argument("--opponent-pool", type=str, default=None,
                   help="comma-separated specs to rotate across, respecting "
                        "per-spec daily quota. Overrides --opponent-teacher. "
                        "Example: --opponent-pool "
                        "gemini:gemini-2.5-flash,gemini:gemini-2.5-flash-lite,"
                        "gemini:gemini-3.0-flash,mistral:mistral-large-latest,"
                        "openrouter:meta-llama/llama-3.3-70b-instruct:free")
    p.add_argument("--opponent-pool-file", type=str, default=None,
                   help="path to a text file with one teacher spec per line "
                        "(# comments ok). Useful for cron — maintain the "
                        "pool list out of band. Overrides --opponent-pool.")
    p.add_argument("--player-pool", type=str, default=None,
                   help="comma-separated specs for the player side (rare — "
                        "the player is usually --player-teacher mock).")
    p.add_argument("--player-pool-file", type=str, default=None,
                   help="text-file equivalent of --player-pool.")
    p.add_argument("--quota-file", type=str, default="data/selfplay/.quota.json",
                   help="path to the daily-quota state file (default "
                        "data/selfplay/.quota.json)")
    p.add_argument("--show-quota", action="store_true",
                   help="print the current quota state and exit without harvesting")

    p.add_argument("--max-rounds", type=int, default=30,
                   help="safety cap on rounds per match (default 30)")
    p.add_argument("--max-matches-per-run", type=int, default=None,
                   help="hard ceiling on matches in one invocation — useful "
                        "for cron to bound wall time per run")
    p.add_argument("--seed", type=int, default=None,
                   help="seed Python's random for mock-backend reproducibility")
    p.add_argument("--output-dir", type=str, default="data/selfplay",
                   help="output directory (relative to cwd; default data/selfplay)")
    p.add_argument("--force-mock", action="store_true",
                   help="shortcut: both sides use the mock heuristic teacher")
    return p.parse_args(argv)


def _resolve_specs(args: argparse.Namespace) -> Tuple[str, str]:
    if args.force_mock:
        return "mock", "mock"
    default = args.teacher or "mock"
    return (args.player_teacher or default, args.opponent_teacher or default)


def _load_pool_file(path: str) -> List[str]:
    """Read pool specs from a text file: one per line, '#' comments allowed."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"pool file not found: {path}")
    specs = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            specs.append(line)
    if not specs:
        raise SystemExit(f"pool file is empty (or only comments): {path}")
    return specs


def _build_side(
    spec_single: str,
    pool_csv: Optional[str],
    pool_file: Optional[str],
    quota_path: Path,
) -> Tuple[Teacher, str]:
    """Construct the Teacher for one side. Returns (teacher, descriptive_label).

    Precedence: pool_file > pool_csv > spec_single.
    """
    if pool_file:
        specs = _load_pool_file(pool_file)
        teacher = make_pool(specs, quota_path=quota_path)
        return teacher, f"pool-file({Path(pool_file).name}, {len(specs)} specs)"
    if pool_csv:
        specs = [s.strip() for s in pool_csv.split(",") if s.strip()]
        if not specs:
            raise SystemExit("pool spec must be a comma-separated list of teachers")
        teacher = make_pool(specs, quota_path=quota_path)
        return teacher, f"pool({len(specs)} specs)"
    return make_teacher(spec_single), spec_single


async def _amain(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)

    quota_path = Path(args.quota_file)

    # --show-quota: print and exit before constructing teachers / hitting any API.
    if args.show_quota:
        snap = QuotaState(quota_path).snapshot()
        print(json.dumps(snap, indent=2))
        return

    spec_p, spec_o = _resolve_specs(args)
    player_teacher, label_p = _build_side(
        spec_p, args.player_pool, args.player_pool_file, quota_path,
    )
    any_player_pool = args.player_pool or args.player_pool_file
    any_opp_pool = args.opponent_pool or args.opponent_pool_file
    if not any_opp_pool and not any_player_pool and spec_p == spec_o:
        # Share the single-teacher client when both sides match — saves a
        # session and halves account-level rate-limit pressure.
        opponent_teacher, label_o = player_teacher, label_p
        shared = True
    else:
        opponent_teacher, label_o = _build_side(
            spec_o, args.opponent_pool, args.opponent_pool_file, quota_path,
        )
        shared = False

    print(f"[selfplay] player   teacher: {label_p}")
    print(f"[selfplay] opponent teacher: {label_o}"
          f"{'  [shared client]' if shared else ''}")
    print(f"[selfplay] player <think>/taunt/raw discarded — opponent rows only become SFT targets")

    if args.sweep:
        matchups = list(itertools.product([o.id for o in LADDER], repeat=2))
    else:
        if args.player_persona not in BY_ID or args.opponent_persona not in BY_ID:
            raise SystemExit(f"unknown persona; valid ids: {list(BY_ID.keys())}")
        matchups = [(args.player_persona, args.opponent_persona)]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"selfplay_{ts}.jsonl"

    written = 0
    sft_written = 0
    matches_done = 0
    exit_reason = "ok"
    t0 = time.time()
    try:
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "_kind": "header",
                "started_at": ts,
                "player_teacher": label_p,
                "opponent_teacher": label_o,
                "player_pool": args.player_pool,
                "opponent_pool": args.opponent_pool,
                "role_contract": (
                    "player rows are move-only (think/taunt/raw_completion null, "
                    "is_sft_target false). Opponent rows are full SFT targets."
                ),
                "matches_per_matchup": args.matches,
                "max_matches_per_run": args.max_matches_per_run,
                "matchups": [{"player": pp, "opponent": oo} for (pp, oo) in matchups],
                "max_rounds": args.max_rounds,
                "seed": args.seed,
            }) + "\n")

            outer = False
            for (pid, oid) in matchups:
                if outer:
                    break
                player_opp = BY_ID[pid]
                opponent_opp = BY_ID[oid]
                for i in range(args.matches):
                    if (args.max_matches_per_run is not None
                            and matches_done >= args.max_matches_per_run):
                        exit_reason = "max_matches_per_run reached"
                        outer = True
                        break
                    match_id = f"{pid}-vs-{oid}-{i:03d}-{uuid.uuid4().hex[:6]}"
                    try:
                        rows = await run_match(
                            player_teacher, opponent_teacher,
                            player_opp, opponent_opp,
                            match_id, max_rounds=args.max_rounds,
                        )
                    except PoolExhausted as exc:
                        print(f"[selfplay] pool exhausted mid-match — {exc}")
                        print(f"[selfplay] partial match {match_id} discarded "
                              f"to keep JSONL clean (no outcome_reward)")
                        exit_reason = "pool_exhausted"
                        outer = True
                        break
                    matches_done += 1
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        if r.get("is_sft_target"):
                            sft_written += 1
                    written += len(rows)
                    last = rows[-1] if rows else {}
                    ml = last.get("match_length", "?")
                    hp = last.get("final_hp_player", "?")
                    ho = last.get("final_hp_opponent", "?")
                    p_out = next(
                        (r["outcome_reward"] for r in rows if r["role"] == "player"),
                        0,
                    )
                    print(f"  [{pid:>13} (P) vs {oid:<13} (O)] match {i+1:>3}/{args.matches}: "
                          f"turns={ml}, p_hp={hp:>3}, o_hp={ho:>3}, p_reward={p_out:+d}")
    finally:
        await player_teacher.aclose()
        if not shared:
            await opponent_teacher.aclose()

    dt = time.time() - t0
    print(f"[selfplay] {matches_done} matches, {written} rows "
          f"({sft_written} SFT targets) to {out_path} in {dt:.1f}s "
          f"[exit: {exit_reason}]")

    # Print final quota snapshot — useful for cron logs.
    snap = QuotaState(quota_path).snapshot()
    if snap["used"]:
        print(f"[selfplay] quota after run: {json.dumps(snap['used'])}")
        if snap["exhausted_until_eod"]:
            print(f"[selfplay] exhausted today: {snap['exhausted_until_eod']}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
