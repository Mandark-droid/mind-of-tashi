"""capture_conviction.py — exercise the Conviction Meter against the REAL
llama.cpp backend and dump the per-turn signal.

Two jobs:
  1. Verify the logprobs path (llm._conviction_from_completion) flows
     end-to-end on a real GGUF — i.e. `source == "llama.cpp"`, not the
     mock/estimate fallback.
  2. Capture the first real "hesitation" data (per-token conviction + the
     peak-wavered word) for the demo clip and the Field-Notes blog.

Runs a short scripted duel where the human plays a repetitive pattern, so the
model's reads swing between confident and unsure. Writes a JSONL row per turn
to data/conviction/ (gitignored).

Usage (from mind-of-tashi/):
    python -m tools.capture_conviction --opponent tashi --rounds 5
    python -m tools.capture_conviction --cpu          # force N_GPU_LAYERS=0
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# The mind-scroll is bilingual (IAST chars like prāṇa); a cp1252 Windows console
# would crash on print. Force utf-8 and never die on an un-encodable glyph.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opponent", default="tashi")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--cpu", action="store_true", help="force N_GPU_LAYERS=0")
    ap.add_argument("--seal", default=None,
                    help="seal this opponent move id every round (tests the E3 GBNF grammar)")
    args = ap.parse_args()
    seal = args.seal.upper() if args.seal else None

    if args.cpu:
        os.environ["N_GPU_LAYERS"] = "0"
    os.environ.pop("FORCE_MOCK", None)  # we want the real backend

    import engine
    import opponents
    from llm import Reasoner

    print(f"[capture] MODEL_REPO={os.environ.get('MODEL_REPO')} "
          f"MODEL_FILE={os.environ.get('MODEL_FILE')} "
          f"N_GPU_LAYERS={os.environ.get('N_GPU_LAYERS')}")
    r = Reasoner()
    print(f"[capture] backend = {r.backend}")
    if r.backend != "llama.cpp":
        print("[capture] WARNING: real backend not active — conviction will be "
              "estimated/mock, not true logprobs.")

    opp = opponents.get(args.opponent)
    # repetitive-ish player so the model alternates between strong and weak reads
    player_moves = ["STRIKE", "STRIKE", "FOCUS", "GUARD", "STRIKE", "MIST_STEP", "GRAPPLE"]

    state = {
        "opponent_id": opp.id, "round": 1,
        "player_hp": engine.MAX_HP, "ai_hp": engine.MAX_HP,
        "player_prana": engine.START_PRANA, "ai_prana": engine.START_PRANA,
        "history": [],
    }

    outdir = os.path.join(ROOT, "data", "conviction")
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outfile = os.path.join(outdir, f"conviction_{opp.id}_{ts}.jsonl")

    real_turns = 0
    with open(outfile, "w", encoding="utf-8") as fout:
        for i in range(args.rounds):
            pm = player_moves[i % len(player_moves)]
            if seal:
                state["sealed_move"] = seal  # E3: forbid this move via GBNF each round
            parsed, _raw = r.choose_with_raw(opp, state)
            if seal and parsed["move"] == seal:
                print(f"  !! SEAL BREACH: she played the sealed move {seal}")
            cv = parsed.get("conviction") or {}
            if cv.get("source") == "llama.cpp":
                real_turns += 1

            print(f"\n--- round {state['round']} | you commit {pm} ---")
            print(f"  {opp.name} commits: {parsed['move']}   \"{parsed['taunt']}\"")
            print(f"  conviction: score={cv.get('score')} "
                  f"decision={cv.get('decision_score')} think={cv.get('think_score')} "
                  f"source={cv.get('source')} ntok={len(cv.get('tokens') or [])}")
            peak = cv.get("peak")
            if peak:
                print(f"  wavered on: \"{peak.get('t')}\"  (token #{peak.get('i')})")
            print(f"  mind-scroll: {parsed['reasoning'][:220]}")

            # resolve to advance the duel state
            p = engine.Fighter("you", hp=state["player_hp"], prana=state["player_prana"])
            a = engine.Fighter(opp.name, hp=state["ai_hp"], prana=state["ai_prana"])
            res = engine.resolve(p, a, pm, parsed["move"])
            engine.apply(p, a, res)
            state["history"].append({
                "round": state["round"], "player_move": res.a_move,
                "ai_move": res.b_move, "outcome": "",
            })
            state["player_hp"], state["ai_hp"] = p.hp, a.hp
            state["player_prana"], state["ai_prana"] = p.prana, a.prana
            state["round"] += 1

            fout.write(json.dumps({
                "round": state["round"] - 1, "player_move": pm,
                "ai_move": parsed["move"], "taunt": parsed["taunt"],
                "reasoning": parsed["reasoning"], "conviction": cv,
            }, ensure_ascii=False) + "\n")

            if p.hp <= 0 or a.hp <= 0:
                print(f"\n[capture] duel ended early at round {state['round'] - 1}")
                break

    print(f"\n[capture] backend={r.backend}  real-logprob turns={real_turns}")
    print(f"[capture] wrote {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
