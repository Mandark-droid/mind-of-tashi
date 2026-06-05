"""
app.py — the web layer, built on gradio.Server (Gradio 6).

gradio.Server extends FastAPI, so we get to ship a fully custom frontend
(static/index.html) from `@app.get("/")` while still exposing our game logic as
an `@app.api()` endpoint that rides Gradio's queue + concurrency control and is
callable from the browser via the Gradio JS client. That combination — our own
UI, Gradio's backend — is the Off-Brand badge.

The single endpoint, `ai_turn`, is where the magic and the fairness live: the
model is asked to commit a move BEFORE it is ever told what the player chose.
It only sees the history. It is a prediction, not a reaction.
"""

from __future__ import annotations
import json
import os

from fastapi.responses import HTMLResponse
from gradio import Server

import engine
import opponents
from llm import Reasoner

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "static", "index.html")

app = Server()
reasoner = Reasoner()  # loaded once; reused across turns
print(f"[app] opponent backend: {reasoner.backend}")


# --- metadata injected into the page so the UI renders from one source ----
def _meta() -> dict:
    return {
        "backend": reasoner.backend,
        "max_hp": engine.MAX_HP,
        "max_prana": engine.MAX_PRANA,
        "start_prana": engine.START_PRANA,
        "moves": [
            {"id": mid, "label": m["label"], "cost": m["cost"],
             "glyph": m["glyph"], "blurb": m["blurb"], "kind": m["kind"]}
            for mid, m in engine.MOVES.items()
        ],
        "ladder": [
            {"id": o.id, "name": o.name, "title": o.title, "bio": o.bio,
             "accent": o.accent, "glyph": o.glyph, "difficulty": o.difficulty}
            for o in opponents.LADDER
        ],
    }


@app.api(name="ai_turn")
def ai_turn(state_json: str) -> str:
    """One simultaneous exchange. Input/output are JSON strings for safe marshalling."""
    state = json.loads(state_json)
    opp = opponents.get(state["opponent_id"])

    player = engine.Fighter("you", hp=state["player_hp"], prana=state["player_prana"])
    ai = engine.Fighter(opp.name, hp=state["ai_hp"], prana=state["ai_prana"])

    player_move = str(state.get("player_move", "FOCUS")).upper()

    # The model commits BLIND — choose() only ever reads `history`, never player_move.
    decision = reasoner.choose(opp, state)
    ai_move = decision["move"]

    result = engine.resolve(player, ai, player_move, ai_move)
    engine.apply(player, ai, result)

    # compact outcome line for the running history / future reads
    outcome = _outcome_phrase(result, player_move, ai_move)
    history = state.get("history", [])
    history.append({
        "round": state["round"],
        "player_move": result.a_move,   # legalized actual move
        "ai_move": result.b_move,
        "outcome": outcome,
    })

    status = "continue"
    next_opponent = None
    if player.hp <= 0 and ai.hp <= 0:
        status = "double_ko"
    elif player.hp <= 0:
        status = "defeat"
    elif ai.hp <= 0:
        nxt = opponents.next_after(opp.id)
        if nxt is None:
            status = "ladder_clear"
        else:
            status = "advance"
            next_opponent = nxt.id

    return json.dumps({
        "ai_move": result.b_move,
        "player_move": result.a_move,
        "reasoning": decision["reasoning"],
        "taunt": decision["taunt"],
        "log": result.log,
        "player_dmg": result.a_dmg_taken,
        "ai_dmg": result.b_dmg_taken,
        "player_hp": player.hp,
        "ai_hp": ai.hp,
        "player_prana": player.prana,
        "ai_prana": ai.prana,
        "status": status,
        "next_opponent": next_opponent,
        "history": history,
    })


def _outcome_phrase(result: engine.RoundResult, pm: str, am: str) -> str:
    if result.a_dmg_taken and result.b_dmg_taken:
        return "traded blows"
    if result.b_dmg_taken:
        return f"you landed {am} for {result.b_dmg_taken}"
    if result.a_dmg_taken:
        return f"they punished you for {result.a_dmg_taken}"
    return "a quiet round"


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    with open(INDEX, "r", encoding="utf-8") as fh:
        html = fh.read()
    return html.replace("__GAME_META__", json.dumps(_meta()))


if __name__ == "__main__":
    app.launch(show_error=True)
