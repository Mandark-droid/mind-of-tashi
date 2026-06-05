"""
prompts.py — turns game state into a prompt, and a prompt into a committed move.

Contract with the model:
  1. It reasons privately inside <think>...</think>. That reasoning IS the
     mind-scroll the UI reveals after the round — so it should read like a
     fighter reading an opponent, not like a chatbot.
  2. After thinking it emits ONE line of JSON: {"move": "...", "taunt": "..."}.

The model never sees the player's CURRENT move — only the history. It commits
blind. That is the whole game.
"""

from __future__ import annotations
import json
from typing import Dict, List

from engine import MOVES
from opponents import Opponent

MOVE_REFERENCE = "\n".join(
    f'    {mid}: {m["label"]} (cost {m["cost"]} prana) — {m["blurb"]}'
    for mid, m in MOVES.items()
)


def build_system(opp: Opponent) -> str:
    return f"""You are {opp.name}, {opp.title} — a duelist of the mist-hidden village high in the Himalaya.

WHO YOU ARE
{opp.temperament}

HOW YOU FIGHT
{opp.strategy}

THE DUEL
You and your challenger fight in simultaneous rounds. Each round you BOTH secretly
choose one move; then both are revealed and resolved at once. You cannot react to
their move — you can only READ them from the rounds that came before and commit.

THE MOVES
{MOVE_REFERENCE}

KEY READS
- STRIKE beats GRAPPLE; GRAPPLE breaks GUARD; GUARD stops STRIKE. (a triangle)
- DRAW BREATH gives prana but leaves you wide open — anything that lands, lands hard.
- PRANA ART is heavy but telegraphed; a MIST-STEP eats it alive.
- MIST-STEP only rewards you if they ATTACK this turn. Against caution it whiffs and
  wastes prana. So Mist-Step is a BET that they will attack.

YOUR JOB EACH ROUND
Think briefly, in character, about what your challenger is likely to do THIS round
based on the history — name their pattern, their state of mind, the bait you can lay.
Then commit the single move that beats your read. Stay true to how you fight; do not
play like a calculator if you are a brawler.

OUTPUT FORMAT
First reason inside <think>...</think> (2-5 short sentences, in your own voice).
Then output exactly one line of JSON and nothing after it:
{{"move": "ONE_OF_THE_MOVE_IDS", "taunt": "a short in-character line to your opponent"}}"""


def _history_block(history: List[Dict]) -> str:
    if not history:
        return "This is the opening round. You have no reads yet — set a tone."
    lines = []
    for h in history[-8:]:
        lines.append(
            f'  Round {h["round"]}: they played {h["player_move"]}, you played {h["ai_move"]}'
            f' — {h.get("outcome", "")}'.rstrip(" —")
        )
    return "Recent rounds (your challenger = 'they'):\n" + "\n".join(lines)


def build_user(opp: Opponent, state: Dict, legal: List[str]) -> str:
    return f"""ARENA — round {state['round']}
  You ({opp.name}):      {state['ai_hp']} HP, {state['ai_prana']} prana
  Your challenger:       {state['player_hp']} HP, {state['player_prana']} prana

{_history_block(state.get('history', []))}

Moves you can afford this round: {', '.join(legal)}

Read them, then commit. Remember the output format: <think>...</think> then one JSON line."""


def parse_reply(text: str, legal: List[str], fallback: str = "GUARD") -> Dict:
    """
    Pull (reasoning, move, taunt) out of a model reply, defensively.

    Works whether or not the model closed its <think> tag, whether the JSON is
    fenced, and whether it added stray prose. If we cannot find a legal move we
    fall back gracefully so the game never stalls on a malformed generation.
    """
    reasoning = ""
    if "<think>" in text:
        seg = text.split("<think>", 1)[1]
        reasoning = seg.split("</think>", 1)[0].strip() if "</think>" in seg else seg.strip()

    move, taunt = None, ""
    # find the last {...} that looks like our object
    start = text.rfind("{")
    while start != -1:
        end = text.find("}", start)
        if end != -1:
            try:
                obj = json.loads(text[start:end + 1])
                if isinstance(obj, dict) and "move" in obj:
                    move = str(obj.get("move", "")).strip().upper().replace(" ", "_")
                    taunt = str(obj.get("taunt", "")).strip()
                    break
            except json.JSONDecodeError:
                pass
        start = text.rfind("{", 0, start)

    # last-ditch: scan for a bare move id in the text
    if move not in legal:
        for m in legal:
            if m in text.upper():
                move = m
                break

    if move not in legal:
        move = fallback if fallback in legal else (legal[0] if legal else "GUARD")

    if not reasoning:
        reasoning = "(silent — reads you without a word)"
    if not taunt:
        taunt = "..."

    return {"reasoning": reasoning, "move": move, "taunt": taunt}
