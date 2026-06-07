"""
smoke_test.py — run one full episode end-to-end to verify the env works.

Usage (from mind-of-tashi/):
    python -m envs.mind_of_tashi_env.smoke_test                            # local mock
    python -m envs.mind_of_tashi_env.smoke_test --backend api              # API teachers (tiered)
    python -m envs.mind_of_tashi_env.smoke_test --inject-malformed         # exercise format_penalty
    python -m envs.mind_of_tashi_env.smoke_test --student-persona the-veiled-one
                                                                       # force boss tier (api)

The student's "completion" is faked with a tiny rotation of canned moves
(STRIKE / FOCUS / GUARD) so we exercise the env loop, parsing, reward
shaping, persona-tier routing, format-validity detection, and spend
tracking — without needing a real model loaded.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Mirror tools/selfplay.py: load mind-of-tashi/.env so API keys are picked up
# when --backend api is requested. No-op if python-dotenv isn't installed
# or the file is missing.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass

from envs.mind_of_tashi_env import make_env, OpponentBackendForbidden


CANNED_THINK = (
    "Player has guarded twice. They will likely strike to break my rhythm. "
    "Prāṇa low — I will gather breath. वायु बहाल करें."
)

# Rotate through a small playbook so the episode actually finishes within
# max_rounds. Real GRPO replaces this with sampled completions.
CANNED_PLAYBOOK = ["FOCUS", "STRIKE", "GUARD", "STRIKE", "GRAPPLE", "STRIKE"]

# A completion that's *missing* the <think> block and emits unstructured prose
# instead of JSON — the env should flag format_valid=False and apply the
# format_penalty. parse_reply will still legalise it (defaults to GUARD).
MALFORMED_COMPLETION = "I think I will just guard this round, perhaps strike next."


def fake_completion(move: str) -> str:
    return (
        f"<think>{CANNED_THINK}</think>\n"
        f'{{"move":"{move}","taunt":"प्रहार आता है"}}'
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="local", choices=["local", "api"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rounds", type=int, default=12)
    p.add_argument("--student-persona", default=None,
                   help="Pin the student persona id (default: random from LADDER)")
    p.add_argument("--opponent-persona", default=None,
                   help="Pin the opponent persona id (default: random, excluding student)")
    p.add_argument("--inject-malformed", action="store_true",
                   help="On round 2, send a malformed completion to verify the format penalty fires")
    p.add_argument("--spend-log", type=Path, default=Path("../data/grpo/spend.jsonl"),
                   help="Path to append per-step spend rows (default ../data/grpo/spend.jsonl)")
    args = p.parse_args(argv)

    try:
        env = make_env(
            opponent_backend=args.backend,
            max_rounds=args.max_rounds,
            spend_log_path=args.spend_log,
        )
    except OpponentBackendForbidden as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    obs = env.reset(
        seed=args.seed,
        student_persona=args.student_persona,
        opponent_persona=args.opponent_persona,
    )
    print(
        f"--- reset ---\n"
        f"  student persona: {obs['student_persona']}\n"
        f"  opponent persona: {obs['opponent_persona']}\n"
        f"  backend: {env.backend_label}\n"
        f"  legal moves: {obs['legal_moves']}\n"
        f"  spend log: {args.spend_log}\n"
    )

    total = {"turn": 0.0, "outcome": 0.0, "lexicon": 0.0, "format_penalty": 0.0, "total": 0.0}
    step = 0
    try:
        while True:
            if args.inject_malformed and step == 1:
                completion = MALFORMED_COMPLETION
            else:
                move = CANNED_PLAYBOOK[step % len(CANNED_PLAYBOOK)]
                completion = fake_completion(move)
            obs, reward, terminated, info = env.step(completion)
            step += 1
            for k in total:
                total[k] += reward[k]
            fmt_marker = " " if info["format_valid"] else "!"
            print(
                f"step {step:>2}{fmt_marker} student={info['student_move']:<9} "
                f"opp={info['opponent_move']:<9} "
                f"r={reward['total']:+7.2f} "
                f"(turn {reward['turn']:+.1f} / out {reward['outcome']:+.1f} / "
                f"lex {reward['lexicon']:+.2f} / fmt {reward['format_penalty']:+.1f})  "
                f"hp s/o={info['hp_after']['student']}/{info['hp_after']['opponent']}  "
                f"tier={info['tier']:<4} usd=+{info['call_usd']:.4f}  "
                f"prov={info['teacher_meta'].get('provider', '?')}"
            )
            if not info["format_valid"]:
                print(f"           format-violation: {info['format_reason']}")
            if terminated:
                break
    finally:
        env.close()

    print(
        f"\n--- episode end ---\n"
        f"  steps: {step}\n"
        f"  cumulative reward: total={total['total']:+.2f} "
        f"(turn={total['turn']:+.2f}, outcome={total['outcome']:+.2f}, "
        f"lexicon={total['lexicon']:+.2f}, format_penalty={total['format_penalty']:+.2f})\n"
        f"  cumulative API spend: ${env.get_total_spend():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
