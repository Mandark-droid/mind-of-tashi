"""
prep_sft.py — turn raw self-play JSONL into a clean SFT corpus.

The harness writes broad data (32-spec pool, ~30% of which produce
underwhelming mind-scrolls). This script filters down to rows we'd
actually train on, in TRL-conversational format:

    {"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}

USAGE (from mind-of-tashi/):
  # filter all collected runs into one corpus, default thresholds
  python -m tools.prep_sft \
      --input "data/selfplay/*.jsonl" \
      --output data/sft/sft.jsonl

  # tune the filter for a specific harvest
  python -m tools.prep_sft \
      --input data/selfplay/selfplay_20260525T192242Z.jsonl \
      --output data/sft/exp1.jsonl \
      --min-think-chars 80 \
      --no-require-lexicon-hit

  # dry run — just print filter stats, write nothing
  python -m tools.prep_sft --input "data/selfplay/*.jsonl" --inspect

FILTERS (in order; each drops then increments a counter):

  is_sft_target          must be true (player rows are role-mismatched targets)
  teacher_fallback       teacher_meta.fallback == true means pool fell through
  raw_terminates         raw_completion must end with '}' (full JSON line emitted)
  think_length           think text >= --min-think-chars
  not_meta_commentary    think doesn't start with role-breaking patterns
                         ("Okay", "Let's", "Let me", "We need", "Alright", ...)
  not_silent_fallback    think doesn't start with '(silent' / '(teacher '
  not_guard_default      not the parse_reply default (GUARD + taunt == "...")
  match_long_enough      match_length >= --min-match-length

Bilingual register is RECORDED, not enforced — every kept row carries
`_meta.bilingual_hits` (count of normalized Sanskrit/Hindi lexicon matches
in the <think> body). Downstream training can stratify or weight by this
score; English-only rows are still useful when the bilingual harvest is
thin or noisy. Pass `--require-lexicon-hit` to opt back into hard-filtering.
"""

from __future__ import annotations
import argparse
import glob
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # mind-of-tashi/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows console default is cp1252 — printing IAST diacritics crashes.
# Force stdout/stderr to UTF-8 when supported (Python 3.7+).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# --- Lexicon loading --------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase + strip combining diacritics, so 'prāṇa' matches 'prana'."""
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def load_lexicon(path: Path) -> Set[str]:
    """Read assets/sanskrit_lexicon.txt and return a set of normalized terms
    we'll look for in the model's <think> output."""
    terms: Set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # rows are tab-separated: iast \t devanagari \t english \t notes
        parts = re.split(r"\t+", line)
        for p in parts[:2]:  # iast and devanagari columns
            p = p.strip()
            if p:
                terms.add(_normalize(p))
    return terms


# --- Filter predicates ------------------------------------------------------

META_COMMENTARY_PREFIXES = (
    "okay",
    "let's",
    "let me",
    "we need",
    "we have",
    "we should",
    "we will",
    "alright",
    "first,",
    "first ",
    "i need to",
    "i'll",
    "i will",
    "i'm",
    "i am ",
    "the user",
    "the system",
    "the prompt",
    "the instruction",
    "hmm,",
    "so, ",
    "looking at",
)


def is_meta_commentary(think: str) -> bool:
    """Return True if the think reads like the model breaking out of role to
    discuss the task ("Okay let's tackle this...", "We need to produce...")."""
    head = think.lstrip().lower()[:80]
    return any(head.startswith(p) for p in META_COMMENTARY_PREFIXES)


def looks_silent_fallback(think: str) -> bool:
    head = think.lstrip()
    return head.startswith("(silent") or head.startswith("(teacher ")


def looks_guard_default(move: str, taunt: Optional[str]) -> bool:
    """parse_reply's last-ditch default emits move=GUARD with taunt='...'."""
    return move == "GUARD" and (taunt is None or taunt.strip() in ("...", ""))


def lexicon_hit_count(think: str, lexicon: Set[str]) -> int:
    norm = _normalize(think)
    return sum(1 for term in lexicon if term and term in norm)


def repair_unbalanced_think(raw: str) -> str:
    """Heal rows where the provider emitted <think> but dropped the closing
    tag and ran the reasoning straight into the JSON line. Heavily seen on
    gpt-oss-120b; ~27% of the day-2 corpus was affected, with rich
    bilingual reasoning silently lost because downstream split-on-</think>
    extracted nothing. Insert </think> right before the trailing JSON.

    Idempotent: rows that already have both tags are passed through.
    """
    if not raw or "<think>" not in raw:
        return raw
    if "</think>" in raw:
        return raw
    j = raw.rfind("{")
    if j > raw.find("<think>"):
        return raw[:j] + "</think>\n" + raw[j:]
    return raw


def extract_think(raw: str) -> str:
    """Pull the <think> body out of a possibly-malformed raw completion."""
    raw = repair_unbalanced_think(raw)
    if "<think>" in raw and "</think>" in raw:
        return raw.split("<think>", 1)[1].split("</think>", 1)[0]
    return ""


# --- Main pipeline ----------------------------------------------------------

DROP_REASONS = [
    "not_sft_target",
    "teacher_fallback",
    "raw_not_terminated",
    "think_too_short",
    "meta_commentary",
    "silent_fallback",
    "guard_default",
    "no_lexicon_hit",      # only fires when --require-lexicon-hit is set
    "match_too_short",
    "missing_messages",
]


def filter_row(
    row: dict,
    lexicon: Set[str],
    require_lexicon_hit: bool,
    min_think_chars: int,
    min_match_length: int,
) -> Optional[str]:
    """Return None if the row is kept, else a string drop_reason.

    Mutates `row` to repair unbalanced <think> tags in raw_completion +
    messages[2].content + think, so kept rows are well-formed downstream.
    """
    if not row.get("is_sft_target"):
        return "not_sft_target"
    # Heal unbalanced <think> on the row before any further checks so
    # think_length / meta_commentary tests see the real reasoning, not the
    # JSON-polluted blob from a missing-close-tag provider.
    raw = row.get("raw_completion")
    if raw:
        healed = repair_unbalanced_think(raw)
        if healed != raw:
            row["raw_completion"] = healed
            row["think"] = extract_think(healed)
            msgs = row.get("messages") or []
            if msgs and msgs[-1].get("role") == "assistant":
                msgs[-1]["content"] = healed
    if (row.get("teacher_meta") or {}).get("fallback"):
        return "teacher_fallback"
    raw = (row.get("raw_completion") or "").rstrip()
    if not raw.endswith("}"):
        return "raw_not_terminated"
    think = row.get("think") or ""
    if len(think) < min_think_chars:
        return "think_too_short"
    if looks_silent_fallback(think):
        return "silent_fallback"
    if is_meta_commentary(think):
        return "meta_commentary"
    if looks_guard_default(row.get("move", ""), row.get("taunt")):
        return "guard_default"
    # Lexicon is recorded as a score (see main); only filter here if the
    # caller explicitly opted in via --require-lexicon-hit.
    if require_lexicon_hit and lexicon and lexicon_hit_count(think, lexicon) == 0:
        return "no_lexicon_hit"
    if int(row.get("match_length") or 0) < min_match_length:
        return "match_too_short"
    if not row.get("messages"):
        return "missing_messages"
    return None


def iter_input_files(input_patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in input_patterns:
        matches = [Path(p) for p in glob.glob(pat)]
        if not matches:
            print(f"[prep_sft] WARN: no files match {pat!r}", file=sys.stderr)
        paths.extend(matches)
    # de-dup, sort
    seen: Set[str] = set()
    uniq: List[Path] = []
    for p in paths:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    uniq.sort()
    return uniq


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", required=True,
                   help="glob or path to a self-play JSONL; can repeat")
    p.add_argument("--output", type=str, default=None,
                   help="output JSONL (TRL conversational). Required unless --inspect")
    p.add_argument("--inspect", action="store_true",
                   help="dry run — print filter stats and exit, write nothing")
    p.add_argument("--lexicon", type=str,
                   default="assets/sanskrit_lexicon.txt",
                   help="path to the Sanskrit/Hindi lexicon file")
    p.add_argument("--min-think-chars", type=int, default=40,
                   help="drop rows whose <think> is shorter than this (default 40)")
    p.add_argument("--min-match-length", type=int, default=3,
                   help="drop rows from matches shorter than this many rounds")
    p.add_argument("--require-lexicon-hit", action="store_true",
                   help="hard-drop rows whose <think> has zero IAST/Devanagari "
                        "lexicon matches. OFF by default — English-only rows are "
                        "still useful and the bilingual_hits score is recorded "
                        "per row for downstream stratification.")
    p.add_argument("--sample", type=int, default=5,
                   help="print this many kept-row samples at end (default 5)")
    p.add_argument("--shape", choices=("single", "multi"), default="single",
                   help="single-turn (default; one example per AI turn) OR "
                        "multi-turn (one example per match, mirrors live/ "
                        "config -- model sees full conversation across rounds)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not args.inspect and not args.output:
        raise SystemExit("--output is required unless --inspect")

    lex_path = Path(args.lexicon)
    if not lex_path.is_absolute():
        lex_path = ROOT / lex_path
    lexicon = load_lexicon(lex_path) if lex_path.exists() else set()
    if not lexicon:
        print(f"[prep_sft] WARN: empty lexicon at {lex_path}", file=sys.stderr)
    require_lex = args.require_lexicon_hit

    files = iter_input_files(args.input)
    if not files:
        raise SystemExit("no input files found")

    drop_counter: Counter[str] = Counter()
    provider_counter: Counter[str] = Counter()
    persona_counter: Counter[str] = Counter()
    bilingual_hit_dist: Counter[int] = Counter()  # histogram of lexicon hits
    kept_rows: List[dict] = []
    kept_hits: List[int] = []  # parallel to kept_rows
    total = 0
    file_rows: Counter[str] = Counter()

    malformed = 0
    for fp in files:
        for raw in fp.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                # tolerate mid-write rows (in-flight cron harvest), truncated
                # writes, or any other malformed JSON. Counted but never
                # crashes the pipeline.
                malformed += 1
                continue
            if row.get("_kind") == "header":
                continue
            total += 1
            reason = filter_row(
                row, lexicon,
                require_lexicon_hit=require_lex,
                min_think_chars=args.min_think_chars,
                min_match_length=args.min_match_length,
            )
            if reason:
                drop_counter[reason] += 1
                continue
            kept_rows.append(row)
            hits = lexicon_hit_count(row.get("think") or "", lexicon) if lexicon else 0
            kept_hits.append(hits)
            bilingual_hit_dist[hits] += 1
            file_rows[fp.name] += 1
            spec = ((row.get("teacher_meta") or {}).get("pool_spec")
                    or (row.get("teacher_meta") or {}).get("provider")
                    or "unknown")
            provider_counter[spec] += 1
            persona_counter[row.get("persona", "?")] += 1

    # --- Report -------------------------------------------------------------
    print(f"[prep_sft] scanned {len(files)} file(s), {total} rows total")
    print(f"[prep_sft] kept   {len(kept_rows)} ({(len(kept_rows)/max(1,total))*100:.1f}%)")
    print(f"[prep_sft] dropped (in order):")
    for reason in DROP_REASONS:
        n = drop_counter.get(reason, 0)
        if n:
            print(f"             {reason:24s} {n}")
    print(f"[prep_sft] kept rows by provider:")
    for spec, n in provider_counter.most_common():
        print(f"             {spec:60s} {n}")
    print(f"[prep_sft] kept rows by persona: "
          f"{dict(persona_counter.most_common())}")
    if lexicon:
        bilingual_total = sum(h * c for h, c in bilingual_hit_dist.items())
        english_only = bilingual_hit_dist.get(0, 0)
        print(f"[prep_sft] bilingual_hits histogram (English-only rows kept "
              f"{english_only}/{len(kept_rows)}):")
        for hits, n in sorted(bilingual_hit_dist.items()):
            print(f"             hits={hits:2d}  rows={n}")
        if kept_rows:
            print(f"[prep_sft] mean bilingual hits per kept row: "
                  f"{bilingual_total / len(kept_rows):.2f}")

    # --- Write --------------------------------------------------------------
    if args.inspect:
        if args.sample and kept_rows:
            print(f"[prep_sft] sample of {min(args.sample, len(kept_rows))} kept rows:")
            for r in kept_rows[: args.sample]:
                print(f"  [{r.get('persona')}] move={r['move']!s:10s} think={r['think'][:140]!r}")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.shape == "multi":
        # Aggregate kept turns into one multi-turn example per match —
        # same shape as the live/ config. The model sees the FULL
        # conversation across rounds rather than a summary history block,
        # so long-context pattern reading falls out of the data shape.
        examples = _aggregate_multiturn(kept_rows, kept_hits, args.min_match_length)
        header = {
            "_kind": "sft_header",
            "generated_at": ts,
            "shape": "multi",
            "input_files": [str(p) for p in files],
            "total_rows_seen": total,
            "kept_turn_rows": len(kept_rows),
            "kept_match_examples": len(examples),
            "drop_counts": dict(drop_counter),
            "filter_config": {
                "min_think_chars": args.min_think_chars,
                "min_match_length": args.min_match_length,
                "require_lexicon_hit": require_lex,
                "lexicon_path": str(lex_path),
                "lexicon_size": len(lexicon),
            },
            "providers_kept_in_turn_rows": dict(provider_counter),
            "personas_kept_in_turn_rows": dict(persona_counter),
        }
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[prep_sft] [shape=multi] wrote {len(examples)} multi-turn "
              f"match examples (from {len(kept_rows)} turn rows) to {out_path}")
        return

    # --- shape == "single" ---------------------------------------------------
    with out_path.open("w", encoding="utf-8") as f:
        # 1-line header recording the prep config (judges + future-you).
        f.write(json.dumps({
            "_kind": "sft_header",
            "generated_at": ts,
            "shape": "single",
            "input_files": [str(p) for p in files],
            "total_rows_seen": total,
            "kept": len(kept_rows),
            "drop_counts": dict(drop_counter),
            "filter_config": {
                "min_think_chars": args.min_think_chars,
                "min_match_length": args.min_match_length,
                "require_lexicon_hit": require_lex,
                "lexicon_path": str(lex_path),
                "lexicon_size": len(lexicon),
            },
            "providers_kept": dict(provider_counter),
            "personas_kept": dict(persona_counter),
            "bilingual_hits_histogram": dict(bilingual_hit_dist),
        }) + "\n")
        # 2..N: one TRL example per kept row.
        for r, hits in zip(kept_rows, kept_hits):
            example = {
                "messages": r["messages"],
                # carry a small metadata blob so we can do post-hoc audits
                # and stratified sampling (bilingual_hits = bilingual register
                # density for this row; downstream can up-weight high-hit rows).
                "_meta": {
                    "match_id": r["match_id"],
                    "turn": r["turn"],
                    "persona": r["persona"],
                    "opponent_persona": r["opponent_persona"],
                    "move": r["move"],
                    "turn_reward": r.get("turn_reward"),
                    "outcome_reward": r.get("outcome_reward"),
                    "provider": ((r.get("teacher_meta") or {}).get("pool_spec")
                                 or (r.get("teacher_meta") or {}).get("provider")),
                    "bilingual_hits": hits,
                },
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"[prep_sft] [shape=single] wrote {len(kept_rows)} examples to {out_path}")


def _aggregate_multiturn(
    kept_rows: List[Dict],
    kept_hits: List[int],
    min_match_length: int,
) -> List[Dict]:
    """Group kept opponent-side turns by match_id and emit ONE multi-turn
    example per match. Mirrors the live/ config's session_complete shape so
    the synthetic and live corpora are mechanically interchangeable for
    SFTTrainer."""
    from collections import defaultdict
    by_match: Dict[str, List[Tuple[Dict, int]]] = defaultdict(list)
    for r, h in zip(kept_rows, kept_hits):
        by_match[r["match_id"]].append((r, h))

    examples = []
    for match_id, items in by_match.items():
        items.sort(key=lambda x: x[0].get("turn", 0))
        # Need at least one valid messages triple to seed the system prompt
        first_msgs = items[0][0].get("messages") or []
        if not first_msgs or first_msgs[0].get("role") != "system":
            continue
        # Build [system, then alternating user/asst for each kept turn]
        msgs: List[Dict[str, Any]] = [first_msgs[0]]
        ai_moves = []
        bil_total = 0
        for r, h in items:
            tm = r.get("messages") or []
            if len(tm) < 3:
                continue
            msgs.append(tm[1])  # user (state prompt for that round)
            msgs.append(tm[2])  # assistant (think + json)
            ai_moves.append(r.get("move"))
            bil_total += h
        # Only keep matches with at least min_match_length kept turns
        kept_turn_count = (len(msgs) - 1) // 2
        if kept_turn_count < min_match_length:
            continue
        first_row = items[0][0]
        last_row = items[-1][0]
        example = {
            "messages": msgs,
            "_meta": {
                "match_id": match_id,
                "persona": first_row.get("persona"),
                "opponent_persona": first_row.get("opponent_persona"),
                "total_turns_kept": kept_turn_count,
                "ai_moves": ai_moves,
                "outcome_reward": last_row.get("outcome_reward"),
                "match_length": last_row.get("match_length"),
                "final_hp_player": last_row.get("final_hp_player"),
                "final_hp_opponent": last_row.get("final_hp_opponent"),
                "provider": ((first_row.get("teacher_meta") or {}).get("pool_spec")
                             or (first_row.get("teacher_meta") or {}).get("provider")),
                "bilingual_hits_total": bil_total,
                "bilingual_hits_per_turn": (bil_total / kept_turn_count) if kept_turn_count else 0,
                "source": "selfplay_multiturn",
            },
        }
        examples.append(example)
    return examples


if __name__ == "__main__":
    main()
