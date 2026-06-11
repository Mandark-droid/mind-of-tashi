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

# Load .env BEFORE importing modules that read env at import time
# (live_traces and leaderboard both capture their config at module load).
# On a real Space, env vars come from Spaces Secrets/Variables and python-dotenv
# is a no-op; locally it makes the live game pick up HF_TOKEN, LEADERBOARD_REPO,
# LIVE_TRACES_REPO etc. without needing them exported in the parent shell.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# Initialize OTEL instrumentation before importing LLM/HTTP-using modules.
# The live game generates training-data rows via live_traces; tracing the
# AI turns captures the same cost/eval/GPU signal as the synthetic harness.
# No-op when GENAI_OTEL_DISABLE=1 or library is missing.
from otel_bootstrap import init_otel  # noqa: E402
init_otel(service_name="mind-of-tashi-live")

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from gradio import Server
# Gradio 6.15's `cache` decorator: content-hashed function memoisation that
# also bypasses the queue on cache hits. Optional import — if the deployed
# Gradio version doesn't have it (pre-6.15), we fall back to functools.lru_cache
# so the runtime still benefits from memoisation, just without the queue-bypass.
try:
    from gradio import cache as gr_cache  # type: ignore
    _GRADIO_CACHE_AVAILABLE = True
except ImportError:  # pragma: no cover  — pre-6.15 fallback
    from functools import lru_cache as _lru
    _GRADIO_CACHE_AVAILABLE = False
    def gr_cache(max_size: int = 128, **_kwargs):  # noqa: D401
        return _lru(maxsize=max_size)

import engine
import leaderboard
import live_traces
import opponents
import selfplay_live
from llm import Reasoner

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "static", "index.html")
ASSETS = os.path.join(HERE, "static", "assets")

# Read the index template ONCE at module load. Pre-6.15 we did this on every
# `/` request — a 50+ KB file read per page view. The file is regenerated only
# at deploy time, so caching it indefinitely is safe.
with open(INDEX, "r", encoding="utf-8") as _fh:
    _INDEX_TEMPLATE = _fh.read()

# HF Spaces sets SPACE_ID automatically. We use it as the single switch for
# "are we on a real Space?" — only then do we trust the OAuth session. Locally,
# /whoami always reports signed_in=False and the UI falls back to a typed name.
ON_SPACE = bool(os.environ.get("SPACE_ID"))

# GRAMMAR-LOCKED OATH (IDEAS.md §E3): prana the player spends to seal one of the
# opponent's moves for a round. Env-tunable so the cost can be balanced.
OATH_COST = int(os.environ.get("OATH_COST", "2"))

app = Server()
# Mount the HF OAuth routes (/login/huggingface, /login/callback, /logout) and
# the session middleware that /whoami reads. On a Gradio-SDK Space our custom
# gradio.Server isn't auto-wired for OAuth, so attach it explicitly — otherwise
# "Sign in with HF" 404s. attach_oauth uses real routes on a Space and mocked
# ones locally; guarded so a failure never blocks boot.
try:
    from gradio.oauth import attach_oauth
    attach_oauth(app)
except Exception as _oauth_exc:  # noqa: BLE001
    print(f"[app] OAuth not attached: {_oauth_exc}")
# Serve the arena backdrops (mp4 / webp / png) referenced by the frontend.
# Without this mount, /assets/village.mp4 etc. 404 and the page falls back
# to the poster image only (or a black sky if posters are missing too).
app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
reasoner = Reasoner()  # loaded once; reused across turns
print(f"[app] opponent backend: {reasoner.backend}")

# Warm the default self-play challenger's GGUF in the background at boot so
# the first "Watch" click doesn't stall on a multi-hundred-MB download.
if selfplay_live.SELFPLAY_MODE:
    print(f"[app] selfplay prewarm: {selfplay_live.prewarm()}")


# --- metadata injected into the page so the UI renders from one source ----
# @gr_cache singleton: _meta() takes no arguments, returns the same payload on
# every call within an app boot (the move catalogue is module-constant, the
# ladder is module-constant, and selfplay.SELFPLAY_MODE is evaluated once at
# import). Caching with max_size=1 lets us serve `/` without rebuilding the
# dict per request; gradio 6.15 also short-circuits the queue on cache hits.
@gr_cache(max_size=1)
def _meta() -> dict:
    return {
        "backend": reasoner.backend,
        "max_hp": engine.MAX_HP,
        "max_prana": engine.MAX_PRANA,
        "start_prana": engine.START_PRANA,
        "oath_cost": OATH_COST,  # §E3: prana to seal an opponent move
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
        # When SELFPLAY_MODE=1 this exposes the two teacher specs so the UI
        # can show a "Watch self-play" button and auto-drive both sides.
        "selfplay": selfplay_live.status(),
    }


@app.api(name="ai_turn")
async def ai_turn(state_json: str) -> str:
    """One simultaneous exchange. Input/output are JSON strings for safe marshalling.

    Two opponent backends:
      - normal: llm.Reasoner (transformers / llama.cpp / mock) — what every
        human-driven match uses, SELFPLAY_MODE or not.
      - self-play watch-mode (SELFPLAY_MODE=1 AND the round is flagged
        state["selfplay"]=true by the UI): selfplay_live's opponent_teacher —
        unless that spec is "house", in which case the live Reasoner stays in
        (full Conviction Meter / composure / Oath path) and only the player
        side is a teacher model.
    """
    state = json.loads(state_json)
    opp = opponents.get(state["opponent_id"])

    # Session continuity for live-trace capture: the front-end echoes back
    # whatever match_id we minted on round 1. First contact (round == 1 or
    # missing) mints a fresh one. End-of-session flush happens in /submit_run.
    match_id = state.get("match_id")
    if not match_id or state.get("round") == 1:
        match_id = live_traces.new_match_id()

    # GRAMMAR-LOCKED OATH (§E3): the player may spend prana to SEAL one of the
    # opponent's moves this round. Validate + charge server-side; the seal is
    # enforced in llm.choose_with_raw (legal filter + GBNF). This is a declared
    # resource action, NOT the player's pending move — blind-commit is intact.
    sealed_move = None
    oath_req = str(state.get("oath") or "").upper().replace(" ", "_")
    if oath_req in engine.MOVES:
        ai_legal = [m for m in engine.MOVES if state["ai_prana"] >= engine.MOVES[m]["cost"]]
        if (oath_req in ai_legal and state["player_prana"] >= OATH_COST
                and len([m for m in ai_legal if m != oath_req]) >= 1):
            sealed_move = oath_req
            state["player_prana"] = state["player_prana"] - OATH_COST  # pay before resolve
    state["sealed_move"] = sealed_move

    player = engine.Fighter("you", hp=state["player_hp"], prana=state["player_prana"])
    ai = engine.Fighter(opp.name, hp=state["ai_hp"], prana=state["ai_prana"])

    player_move = str(state.get("player_move", "FOCUS")).upper()

    # The opponent commits BLIND — it sees `history` only, never player_move.
    # Watch-mode matches flag themselves via state["selfplay"]; human matches
    # always take the live Reasoner, so SELFPLAY_MODE=1 can stay on for a
    # deployed Space without touching real players' games.
    is_selfplay = bool(selfplay_live.SELFPLAY_MODE and state.get("selfplay"))
    if is_selfplay and not selfplay_live.OPPONENT_IS_HOUSE:
        result_obj = await selfplay_live.opponent_choose(state, opp)
        decision, raw = result_obj.parsed, result_obj.raw
    else:
        decision, raw = reasoner.choose_with_raw(opp, state)
    ai_move = decision["move"]

    result = engine.resolve(player, ai, player_move, ai_move)
    engine.apply(player, ai, result)

    # Capture the AI turn into the per-match JSONL (best-effort; never fails
    # the request). The match-end summary is written in /submit_run. We
    # explicitly DO NOT capture self-play matches — those rows would mix
    # two-AI play into the real-player dataset.
    if not is_selfplay:
        try:
            live_traces.capture_turn(
                match_id=match_id, opp=opp, state=state,
                ai_raw=raw, ai_parsed=decision,
                player_move=player_move, backend=reasoner.backend,
            )
        except Exception as exc:  # never let capture failures break the game
            print(f"[live_traces] capture_turn failed: {exc}")

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
        # Conviction Meter (IDEAS.md §E1): per-token confidence read off the
        # llama.cpp logprobs. None in self-play mode / if the backend omits it.
        "conviction": decision.get("conviction"),
        # Composure (IDEAS.md §E2): 0-100, how rattled she is. Falls as the
        # player lands reads/counters; drives her rising temperature.
        "composure": decision.get("composure"),
        # Oath (IDEAS.md §E3): which move the player sealed this round (+ cost),
        # echoed back so the UI can confirm it and log the binding. None if no
        # valid oath was cast.
        "oath": ({"sealed": sealed_move, "cost": OATH_COST} if sealed_move else None),
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
        # The front-end persists this in its run state and echoes it back on
        # the next /ai_turn call (or to /submit_run on game end) so every
        # turn of a match shares one live-traces file.
        "match_id": match_id,
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
    # _INDEX_TEMPLATE is read once at module load; _meta() is memoised.
    # The whole route handler becomes a O(template_size) string substitution.
    return _INDEX_TEMPLATE.replace("__GAME_META__", json.dumps(_meta()))


# --- leaderboard wiring ---------------------------------------------------
# Identity policy: on a Space with `hf_oauth: true` we read the verified
# username from the request session and IGNORE whatever the client sends, so
# nobody can submit a run as someone else. Locally, /whoami signals "guest" and
# the UI collects a typed name — no verification, no spoofing prevention.
def _oauth_username(request: Request) -> str | None:
    if not ON_SPACE:
        return None
    try:
        info = (request.session.get("oauth_info") or {}).get("userinfo") or {}
        return info.get("preferred_username") or info.get("name")
    except Exception:
        return None


@app.get("/whoami")
async def whoami(request: Request):
    username = _oauth_username(request)
    if username:
        return {"signed_in": True, "username": username, "source": "hf-oauth", "on_space": True}
    return {
        "signed_in": False,
        "username": None,
        "source": "guest",
        "on_space": ON_SPACE,
        # HF Spaces with hf_oauth: true auto-mounts this login route.
        "login_url": "/login/huggingface" if ON_SPACE else None,
    }


@app.post("/submit_run")
async def submit_run(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "reason": "bad json"}, status_code=400)

    verified = _oauth_username(request)
    if verified:
        username, source = verified, "hf-oauth"
    else:
        username, source = str(body.get("username") or "anon")[:40], "guest"

    res = leaderboard.submit_run(
        username=username,
        source=source,
        total_turns=int(body.get("total_turns", 0)),
        total_seconds=float(body.get("total_seconds", 0)),
        per_level=body.get("per_level") or [],
        won=bool(body.get("won", False)),
        backend=reasoner.backend,
    )

    # Seal the live-traces file for the just-finished match. Best-effort —
    # cron uploads any sealed (match_end-marked) files to the live dataset.
    match_id = body.get("match_id")
    if match_id:
        try:
            live_traces.end_session(
                match_id=str(match_id),
                outcome=str(body.get("outcome") or ("ladder_clear" if body.get("won") else "defeat")),
                won=bool(body.get("won", False)),
                total_turns=int(body.get("total_turns", 0)) or None,
                total_seconds=float(body.get("total_seconds", 0)) or None,
                username=username,
            )
        except Exception as exc:
            print(f"[live_traces] end_session failed: {exc}")

    return res


@app.get("/live_traces_status")
async def live_traces_status():
    return live_traces.status()


@app.get("/prewarm_challenger")
async def prewarm_challenger(id: str = ""):
    """Fire-and-forget: download a self-play challenger's weights into the HF
    cache. The UI calls this when the picker changes so the model is warm
    before the watcher commits to a duel."""
    if not selfplay_live.SELFPLAY_MODE:
        return {"ok": False, "reason": "selfplay off"}
    return {"ok": True, "status": selfplay_live.prewarm(id or None)}


@app.api(name="player_turn")
async def player_turn(state_json: str) -> str:
    """Self-play only: ask the PLAYER teacher for its blind-commit move.

    The front-end calls this in self-play mode BEFORE /ai_turn each round —
    it gets the player teacher's chosen move (and reasoning), then submits
    that move through the normal /ai_turn path so the resolution + opponent
    side go through the same code as a human-driven game.

    Returns: {"move", "reasoning", "taunt"} or {"error": "..."}.
    """
    if not selfplay_live.SELFPLAY_MODE:
        return json.dumps({"error": "self-play disabled (set SELFPLAY_MODE=1)"})
    try:
        state = json.loads(state_json)
        opp = opponents.get(state["opponent_id"])
        result = await selfplay_live.player_choose(
            state, opp, challenger=state.get("challenger"))
        return json.dumps({
            "move": result.parsed["move"],
            "reasoning": result.parsed["reasoning"],
            "taunt": result.parsed["taunt"],
            # surfaced in the watch-mode log so a silent llama.cpp->mock
            # fallback (e.g. wheel missing) is visible, not mysterious.
            # meta["error"] covers base.Teacher's retry-exhausted GUARD
            # fallback (generation-time failures, not load-time ones).
            "backend": (result.meta.get("reasoner_backend")
                        or result.meta.get("backend")
                        or ("error" if result.meta.get("fallback") else None)),
            "model": result.meta.get("model"),
            "load_error": (result.meta.get("reasoner_error")
                           or result.meta.get("error")),
        })
    except Exception as exc:
        print(f"[selfplay] player_turn failed: {exc}")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@app.get("/top_runs")
async def top_runs():
    return leaderboard.top_runs(limit=20)


@app.get("/leaderboard_status")
async def leaderboard_status():
    return {**leaderboard.status(), "on_space": ON_SPACE}


if __name__ == "__main__":
    app.launch(show_error=True)
