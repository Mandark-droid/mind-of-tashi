"""
engine.py — the combat core of the duel.

A turn is SIMULTANEOUS: both fighters secretly commit a move, then both are
revealed and resolved at once. There is no "react" — only "read". That single
design choice is what makes the reasoning model load-bearing: the AI must
*predict* your move, not respond to it.

The six moves form a rock-paper-scissors core (STRIKE > GRAPPLE > GUARD > STRIKE)
wrapped in a prana (breath/energy) economy, with two high-skill moves on top:

    ART        — spend prana for a heavy ranged technique. Strong, but telegraphed.
    MIST_STEP  — the read. Spend prana to vanish; if the opponent ATTACKED you this
                 turn you fully dodge AND counter. If they didn't, you whiff and
                 burned the prana for nothing. A pure prediction.

Everything here is plain Python with no dependencies so it can be unit-tested
and reasoned about in isolation from the model and the web layer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- tunables -------------------------------------------------------------
MAX_HP = 100
MAX_PRANA = 6
START_PRANA = 1

ART_COST = 3
MIST_COST = 2

GUARD_PRANA_GAIN = 1
FOCUS_PRANA_GAIN = 2

# base attack power
STRIKE_DMG = 12
GRAPPLE_DMG = 9          # vs anything except a guard
GRAPPLE_VS_GUARD = 20    # guard-break: grapples punish turtling hard
ART_DMG = 22
MIST_COUNTER = 14        # counter damage when a Mist-Step reads an attack

FOCUS_EXPOSURE = 1.6     # damage multiplier taken while gathering breath
ART_VS_GUARD = 0.5       # guard halves an incoming technique

# --- move catalogue -------------------------------------------------------
# Himalayan reskin of the classic fighting-game verbs.
MOVES = {
    "STRIKE":    {"label": "Vajra Strike", "cost": 0,         "kind": "attack",  "glyph": "\u26a1",
                  "blurb": "A thunderbolt blow. Reliable. Beats Grapple, stopped by Guard."},
    "GUARD":     {"label": "Mountain Stance", "cost": 0,      "kind": "defend",  "glyph": "\u26f0",
                  "blurb": "Immovable. Blocks strikes, softens techniques, +1 prana. Broken by Grapple."},
    "GRAPPLE":   {"label": "River Throw", "cost": 0,          "kind": "attack",  "glyph": "\U0001f30a",
                  "blurb": "Flows around a guard and breaks it. Loses to a clean Strike."},
    "FOCUS":     {"label": "Draw Breath", "cost": 0,          "kind": "gather",  "glyph": "\U0001f343",
                  "blurb": "Gather +2 prana — but you are wide open. Anything that lands, lands hard."},
    "ART":       {"label": "Prana Art", "cost": ART_COST,     "kind": "attack",  "glyph": "\U0001f525",
                  "blurb": "Spend 3 prana for a heavy ranged technique. Devastating, but read-able."},
    "MIST_STEP": {"label": "Mist-Step", "cost": MIST_COST,    "kind": "read",    "glyph": "\U0001f32b",
                  "blurb": "Spend 2 prana to vanish. Dodge AND counter an attack — whiff against caution."},
}

ATTACKS = {m for m, v in MOVES.items() if v["kind"] == "attack"}
ALL_MOVES = list(MOVES.keys())


# --- combo table ----------------------------------------------------------
# Named 3-move sequences that grant a bonus effect on the THIRD move. Symmetric
# (either fighter can trigger them). The pattern is checked against the
# attacker's `move_history[-2:] + [current_move]` — so only the trigger round
# applies the bonus, and the combo "consumes" the build-up in the sense that
# the same combo can't re-fire next round without a fresh setup.
#
# Each combo definition:
#   name              — narrative label rendered in mind-scrolls + UI taunts
#   trigger_move      — must equal current move for the combo to fire
#   bonus_dmg         — additive damage stacked on top of the base attack
#   ignore_guard_halving — ART variant: restore full damage even vs GUARD
#   pierces_guard     — STRIKE variant: treat defender's GUARD as no defense
#   drain_prana       — side-effect: subtract N prana from the opponent on hit
#
# Effects are nullified if the attack itself is dodged (e.g. by Mist-Step).
# That's intentional — a read counter dominates a telegraphed combo.

MAX_HISTORY = 3  # how many recent moves to keep on each fighter (combo window)

COMBOS: Dict[Tuple[str, str, str], Dict[str, Any]] = {
    ("FOCUS", "GRAPPLE", "STRIKE"): {
        "name": "Dragon Current",
        "trigger_move": "STRIKE",
        "bonus_dmg": 12,
        "tag": "dragon_current",
        "blurb": "Gathered breath, flowing throw, thunderbolt strike — a current that "
                 "pulls the opponent in and then breaks them.",
    },
    ("GUARD", "GUARD", "ART"): {
        "name": "Mountain's Verdict",
        "trigger_move": "ART",
        "ignore_guard_halving": True,
        "tag": "mountains_verdict",
        "blurb": "Two stances of stillness, then judgment falls. The mountain "
                 "renders verdict — and even the firmest guard does not soften it.",
    },
    ("MIST_STEP", "MIST_STEP", "STRIKE"): {
        "name": "Mist Veil",
        "trigger_move": "STRIKE",
        "bonus_dmg": 8,
        "pierces_guard": True,
        "tag": "mist_veil",
        "blurb": "Twice the opponent grasps at mist; the third moment the mist itself "
                 "strikes — the guard is meaningless when the attacker was never there.",
    },
    ("STRIKE", "STRIKE", "STRIKE"): {
        "name": "Vajra Storm",
        "trigger_move": "STRIKE",
        "bonus_dmg": 10,
        "drain_prana": 1,
        "tag": "vajra_storm",
        "blurb": "Three thunderbolts in a row — the opponent's breath is knocked from "
                 "their chest along with their hit points.",
    },
}


def _detect_combo(
    move_history: List[str], current_move: str
) -> Optional[Dict[str, Any]]:
    """Return the combo definition that fires this round, or None.

    Looks at the LAST TWO entries in `move_history` plus `current_move`, so a
    fighter must have played at least two prior rounds to trigger anything.
    """
    if len(move_history) < 2:
        return None
    seq = (move_history[-2], move_history[-1], current_move)
    cdef = COMBOS.get(seq)
    if cdef is None:
        return None
    # The combo's trigger_move must match the current move (a defensive
    # legalisation might have replaced an intended trigger — in that case the
    # combo simply doesn't fire).
    if cdef.get("trigger_move") != current_move:
        return None
    return cdef


def _apply_combo_damage(
    base_dmg: int, attacker_move: str, defender_move: str, cdef: Dict[str, Any]
) -> int:
    """Transform `base_dmg` according to the combo's damage modifiers.

    Called after the base `_damage(...)` calculation but before any
    Mist-Step counter logic, so a successful read still nullifies the combo.
    """
    dmg = base_dmg
    # Mountain's Verdict — ART ignores GUARD halving.
    if cdef.get("ignore_guard_halving") and attacker_move == "ART" and defender_move == "GUARD":
        dmg = ART_DMG  # restore full pre-halving damage
    # Mist Veil — STRIKE treats GUARD as no defense.
    if cdef.get("pierces_guard") and attacker_move == "STRIKE" and defender_move == "GUARD":
        dmg = STRIKE_DMG  # restore base STRIKE damage (otherwise 0 vs guard)
    # Flat bonus.
    dmg += int(cdef.get("bonus_dmg", 0))
    return dmg


@dataclass
class Fighter:
    name: str
    hp: int = MAX_HP
    prana: int = START_PRANA
    # Rolling buffer of the last MAX_HISTORY moves this fighter played. Used
    # by combo detection in resolve(). Maintained by apply() after a round
    # resolves — that way move_history reflects only confirmed (post-legalise)
    # moves the fighter actually committed to.
    move_history: List[str] = field(default_factory=list)

    def can_afford(self, move: str) -> bool:
        return self.prana >= MOVES[move]["cost"]

    def clamp(self) -> None:
        self.hp = max(0, min(MAX_HP, self.hp))
        self.prana = max(0, min(MAX_PRANA, self.prana))

    def push_history(self, move: str) -> None:
        """Append `move` and truncate to MAX_HISTORY entries."""
        self.move_history.append(move)
        if len(self.move_history) > MAX_HISTORY:
            self.move_history = self.move_history[-MAX_HISTORY:]


@dataclass
class RoundResult:
    a_move: str
    b_move: str
    a_dmg_taken: int = 0          # damage A took
    b_dmg_taken: int = 0          # damage B took
    a_prana_delta: int = 0
    b_prana_delta: int = 0
    a_combo: Optional[str] = None  # narrative combo name fired by A this round (or None)
    b_combo: Optional[str] = None  # narrative combo name fired by B this round (or None)
    log: List[str] = field(default_factory=list)


def _legalize(fighter: Fighter, move: str) -> str:
    """Server-side guard: an unaffordable move collapses into Draw Breath."""
    if move not in MOVES:
        return "FOCUS"
    if not fighter.can_afford(move):
        return "FOCUS"
    return move


def resolve(a: Fighter, b: Fighter, a_move: str, b_move: str) -> RoundResult:
    """
    Resolve one simultaneous exchange. `a` is the perspective fighter (the player
    in the web layer) but the function is fully symmetric.

    Returns a RoundResult; the caller is responsible for applying it to the
    Fighter objects (so the resolution stays a pure-ish function and is easy
    to test / preview).
    """
    a_move = _legalize(a, a_move)
    b_move = _legalize(b, b_move)
    res = RoundResult(a_move=a_move, b_move=b_move)

    # 1) pay costs up front (both commit)
    res.a_prana_delta -= MOVES[a_move]["cost"]
    res.b_prana_delta -= MOVES[b_move]["cost"]

    # 2) prana gains from defensive/gather stances
    if a_move == "GUARD":
        res.a_prana_delta += GUARD_PRANA_GAIN
    if b_move == "GUARD":
        res.b_prana_delta += GUARD_PRANA_GAIN
    if a_move == "FOCUS":
        res.a_prana_delta += FOCUS_PRANA_GAIN
    if b_move == "FOCUS":
        res.b_prana_delta += FOCUS_PRANA_GAIN

    # 3) damage. Compute what each fighter does TO the other.
    a_to_b = _damage(a_move, b_move)
    b_to_a = _damage(b_move, a_move)

    # 3b) Combo detection. Check each fighter's last-two-moves + this round's
    # legalised move against the COMBOS table. A combo modifies the damage
    # output of the trigger move (and possibly drains opponent prana). Effects
    # are nullified by a successful Mist-Step in step 4 below — a clean read
    # dominates a telegraphed combo.
    a_combo = _detect_combo(a.move_history, a_move)
    b_combo = _detect_combo(b.move_history, b_move)
    if a_combo is not None:
        a_to_b = _apply_combo_damage(a_to_b, a_move, b_move, a_combo)
        if a_combo.get("drain_prana"):
            res.b_prana_delta -= int(a_combo["drain_prana"])
        res.a_combo = a_combo["name"]
        res.log.append(f"{a.name} threads a hidden combo: {a_combo['name']}.")
    if b_combo is not None:
        b_to_a = _apply_combo_damage(b_to_a, b_move, a_move, b_combo)
        if b_combo.get("drain_prana"):
            res.a_prana_delta -= int(b_combo["drain_prana"])
        res.b_combo = b_combo["name"]
        res.log.append(f"{b.name} threads a hidden combo: {b_combo['name']}.")

    # 4) Mist-Step counters (resolved after raw damage so a successful read
    #    both negates the incoming hit and adds a counter).
    if a_move == "MIST_STEP":
        if b_move in ATTACKS:
            b_to_a = 0                       # fully dodged
            a_to_b = max(a_to_b, MIST_COUNTER)
            res.log.append(f"{a.name} reads the attack and Mist-Steps — a counter lands.")
        else:
            res.log.append(f"{a.name} Mist-Steps into empty air. Prana wasted.")
    if b_move == "MIST_STEP":
        if a_move in ATTACKS:
            a_to_b = 0
            b_to_a = max(b_to_a, MIST_COUNTER)
            res.log.append(f"{b.name} reads the attack and Mist-Steps — a counter lands.")
        else:
            res.log.append(f"{b.name} Mist-Steps into empty air. Prana wasted.")

    # 5) trade narration for the common cases
    if a_move in ATTACKS and b_move in ATTACKS and a_to_b and b_to_a:
        res.log.append("Both commit — neither flinches. Blood is traded.")
    elif a_move == "GUARD" and b_move == "GRAPPLE":
        res.log.append(f"{b.name} breaks the Mountain Stance wide open.")
    elif b_move == "GUARD" and a_move == "GRAPPLE":
        res.log.append(f"{a.name} breaks the Mountain Stance wide open.")
    elif a_move == "FOCUS" and b_to_a:
        res.log.append(f"{a.name} is caught gathering breath — it costs dearly.")
    elif b_move == "FOCUS" and a_to_b:
        res.log.append(f"{b.name} is caught gathering breath — it costs dearly.")

    res.b_dmg_taken = a_to_b
    res.a_dmg_taken = b_to_a
    return res


def _damage(attacker_move: str, defender_move: str) -> int:
    """Raw damage `attacker_move` deals into `defender_move` (before Mist-Step)."""
    if attacker_move not in ATTACKS:
        return 0

    exposed = defender_move == "FOCUS"

    if attacker_move == "STRIKE":
        if defender_move == "GUARD":
            return 0                          # blocked
        if defender_move == "GRAPPLE":
            base = STRIKE_DMG                 # strike beats grapple cleanly
        else:
            base = STRIKE_DMG
        return _expose(base, exposed)

    if attacker_move == "GRAPPLE":
        if defender_move == "GUARD":
            return GRAPPLE_VS_GUARD           # guard-break
        if defender_move == "STRIKE":
            return 0                          # the grab is stuffed by the strike
        return _expose(GRAPPLE_DMG, exposed)

    if attacker_move == "ART":
        if defender_move == "GUARD":
            return _expose(int(ART_DMG * ART_VS_GUARD), exposed)
        return _expose(ART_DMG, exposed)

    return 0


def _expose(base: int, exposed: bool) -> int:
    return int(round(base * FOCUS_EXPOSURE)) if exposed else base


def apply(a: Fighter, b: Fighter, res: RoundResult) -> None:
    """Mutate fighters with a resolved round."""
    a.hp -= res.a_dmg_taken
    b.hp -= res.b_dmg_taken
    a.prana += res.a_prana_delta
    b.prana += res.b_prana_delta
    a.clamp()
    b.clamp()
    # Record the move each fighter actually committed to (the legalised one)
    # so the next round's combo detection has accurate history.
    a.push_history(res.a_move)
    b.push_history(res.b_move)


if __name__ == "__main__":
    # quick coherence dump: every move pair from a fresh, prana-rich state
    print("MOVE INTERACTION TABLE (attacker rows -> defender cols), dmg dealt to defender\n")
    header = "          " + "".join(f"{m[:6]:>9}" for m in ALL_MOVES)
    print(header)
    for am in ALL_MOVES:
        row = f"{am[:9]:<10}"
        for dm in ALL_MOVES:
            a = Fighter("A", prana=6)
            b = Fighter("B", prana=6)
            r = resolve(a, b, am, dm)
            row += f"{r.b_dmg_taken:>9}"
        print(row)

    # combo demonstration — fire each one and show the bonus on the trigger round
    print("\nHIDDEN COMBO TRIGGERS (vs FOCUS so the opponent is fully exposed)\n")
    for seq, cdef in COMBOS.items():
        a, b = Fighter("A", prana=6), Fighter("B", prana=6)
        defender_moves = ["FOCUS", "FOCUS", "FOCUS"]
        # If the combo's third move is ART and defender's GUARD is the
        # interesting case, demo against GUARD instead so the effect is visible.
        if cdef.get("ignore_guard_halving") or cdef.get("pierces_guard"):
            defender_moves[-1] = "GUARD"
        for i, (am, bm) in enumerate(zip(seq, defender_moves)):
            r = resolve(a, b, am, bm)
            apply(a, b, r)
            if i == 2:
                bonus_total = sum([
                    cdef.get("bonus_dmg", 0),
                    int(cdef.get("ignore_guard_halving", False)) * int(ART_DMG * (1 - ART_VS_GUARD)),
                    int(cdef.get("pierces_guard", False)) * STRIKE_DMG,
                ])
                drain = f", drains {cdef['drain_prana']} opp prana" if cdef.get("drain_prana") else ""
                print(
                    f"  {cdef['name']:<18}  "
                    f"{seq[0]:>9} -> {seq[1]:>9} -> {seq[2]:>9} (vs {bm})  "
                    f"-> dmg={r.b_dmg_taken} (combo:{r.a_combo}, bonus ~+{bonus_total}{drain})"
                )
