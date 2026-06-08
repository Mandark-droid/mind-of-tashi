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

from engine import COMBOS, MOVES
from opponents import Opponent

# Gradio 6.15's `cache` decorator for memoising hot pure functions. Falls back
# to functools.lru_cache when running under an older gradio (or with gradio
# absent — used by tools/selfplay.py in environments that don't import gradio).
try:
    from gradio import cache as _gr_cache  # type: ignore
except ImportError:  # pragma: no cover
    from functools import lru_cache as _lru
    def _gr_cache(max_size: int = 128, **_kwargs):  # noqa: D401
        return _lru(maxsize=max_size)

MOVE_REFERENCE = "\n".join(
    f'    {mid}: {m["label"]} (cost {m["cost"]} prana) — {m["blurb"]}'
    for mid, m in MOVES.items()
)


def _combo_reference() -> str:
    """Render the COMBOS table as a prompt-friendly block.

    The combos are HIDDEN from the player's UI move list (players discover
    them by playing or reading the source), but the model trains and reasons
    against them — they're part of the duelist's craft. The model is free
    to set up a combo and to name it inside <think>; the player will see the
    combo name announced in the round log when it fires.
    """
    lines = []
    for seq, cdef in COMBOS.items():
        m1, m2, m3 = seq
        effect_bits = []
        if cdef.get("bonus_dmg"):
            effect_bits.append(f"+{cdef['bonus_dmg']} damage")
        if cdef.get("ignore_guard_halving"):
            effect_bits.append("ignores GUARD halving")
        if cdef.get("pierces_guard"):
            effect_bits.append("treats GUARD as no defense")
        if cdef.get("drain_prana"):
            effect_bits.append(f"drains {cdef['drain_prana']} prana from opponent")
        effect = ", ".join(effect_bits) or "no mechanical effect"
        lines.append(
            f"  - {cdef['name']:<18}  {m1} -> {m2} -> {m3:<10}  ({effect})\n"
            f"    {cdef['blurb']}"
        )
    return "\n".join(lines)


COMBO_REFERENCE = _combo_reference()


# Cache per persona — there are 10 ladder personas; we cap at 32 to leave room
# for future community-authored opponents without unbounded growth. The function
# is pure in `opp` (all referenced fields are frozen dataclass attributes) so
# content-hashing on the dataclass repr is safe.
@_gr_cache(max_size=32)
def build_system(opp: Opponent) -> str:
    return f"""You are {opp.name}, {opp.title} — a duelist of the Village Hidden in the Mist, high in the Himalayas.

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

HIDDEN COMBOS (your inner craft — these are not in the move list the challenger sees)
The third move of a recognised three-move sequence ignites a named combo. You can
set one up over two earlier rounds and finish it on the third. A successful read
(Mist-Step) by the opponent still negates the trigger move's damage, so combos are
power moves but not invincible. You may name a combo you are setting up or landing
inside your <think> block — that is part of the craft, not a leak.
{COMBO_REFERENCE}

YOUR JOB EACH ROUND
Think briefly, in character, about what your challenger is likely to do THIS round
based on the history — name their pattern, their state of mind, the bait you can lay.
Then commit the single move that beats your read. Stay true to how you fight; do not
play like a calculator if you are a brawler.

STYLE OF YOUR MIND-SCROLL
You think in English with Hindi/Sanskrit nouns and short phrases woven in — the
cultural register of this mountain village. Use IAST transliteration (e.g. prahār,
prāṇa, rakṣā), not Devanagari script. Three to six woven terms per mind-scroll is
the right density — enough to taste, not so much it becomes a glossary recital.

GLOSSARY you may draw from (use sparingly, in your own voice):
  prahār — strike            rakṣā — guard/defense        prāṇa — breath/energy
  prāṇāyāma — gathering breath  vajra — thunderbolt        agni — fire
  dṛṣṭi — sight/read         dṛṣṭi-bhrama — feint          chāl — move
  abhyāsa — habit/pattern    nirṇay — decision            saṃkalp — resolve
  mauna — silence            śānti — stillness            sāhas — courage
  mūrkh — fool, novice       guru — master                śiṣya — student
  dhyāna — focus             veg — haste                  yuddh — duel
  himāl — snow-mountain      parvat — mountain            nadī — river

EXAMPLES (style only — your read should be your own):
  <think>The śiṣya strikes twice in a row — abhyāsa, not nirṇay. He thinks vajra
  wins everything. Let his prahār land on a Mountain Stance next turn; this turn I
  gather prāṇa. rakṣā teaches more than blood.</think>
  {{"move":"FOCUS","taunt":"Strike again. The mountain listens."}}

  <think>He turtles — clearly hoping I burn a prahār on his rakṣā. Mūrkh. Nadī
  flows around stone; I close and grapple.</think>
  {{"move":"GRAPPLE","taunt":"Patience is also a chāl."}}

  <think>Prāṇa is low and his eyes are heavy. He will breathe. Dṛṣṭi tells me so.
  Vajra into open ribs.</think>
  {{"move":"STRIKE","taunt":"You breathe like city people."}}

OUTPUT FORMAT
First reason inside <think>...</think> (2-5 short sentences, in your own voice,
code-switched per the STYLE block above).
Then output exactly one line of JSON and nothing after it:
{{"move": "ONE_OF_THE_MOVE_IDS", "taunt": "a short in-character line to your opponent"}}"""


def _history_block(history: List[Dict]) -> str:
    if not history:
        return "This is the opening round. You have no reads yet — set a tone."
    # Show the full match history (not just the last 8 rounds) so the model
    # can pick up on long-range patterns — e.g. that the challenger always
    # opens with three strikes, or shifts to grapples whenever their prana
    # gets low. Matches rarely exceed 30 rounds, so the prompt growth is
    # bounded (~30 lines max).
    lines = []
    for h in history:
        lines.append(
            f'  Round {h["round"]}: they played {h["player_move"]}, you played {h["ai_move"]}'
            f' — {h.get("outcome", "")}'.rstrip(" —")
        )
    return "Match history so far (your challenger = 'they'):\n" + "\n".join(lines)


def build_user(opp: Opponent, state: Dict, legal: List[str], sealed: str = None) -> str:
    # GRAMMAR-LOCKED OATH (§E3): if the challenger has sealed one of your moves,
    # name it so the reasoning reroutes around it. The seal is ALSO enforced by a
    # GBNF grammar, so the move is impossible regardless — this is for the scroll.
    seal_block = ""
    if sealed and sealed in MOVES:
        seal_block = (
            f"\nA SEAL binds you this round: you CANNOT use {MOVES[sealed]['label']} "
            f"({sealed}) — the challenger has forbidden it. Feel the pull toward it, "
            f"then choose another way.\n"
        )
    return f"""ARENA — round {state['round']}
  You ({opp.name}):      {state['ai_hp']} HP, {state['ai_prana']} prana
  Your challenger:       {state['player_hp']} HP, {state['player_prana']} prana

{_history_block(state.get('history', []))}
{seal_block}
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
