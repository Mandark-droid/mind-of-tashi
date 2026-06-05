"""
llm.py — the opponent's mind.

Two backends behind one interface:

  * REAL: a small reasoning model run through llama.cpp (llama-cpp-python),
    pulling a GGUF straight from the Hugging Face Hub. This is the path that
    earns the Off-the-Grid (no cloud API) and Llama Champion badges. Swapping in
    a fine-tuned GGUF later is a one-line change (set MODEL_REPO / MODEL_FILE).

  * MOCK: a persona-aware heuristic that reads your recent pattern and answers
    in character. No weights, no download — so the whole loop is playable the
    instant you unzip this, and CI / local dev never needs a GPU.

The web layer does not know or care which backend is live.
"""

from __future__ import annotations
import os
import random
from collections import Counter
from typing import Dict, List

import prompts
from engine import MOVES, ATTACKS
from opponents import Opponent

# ----- configuration (override via Space "Variables") ---------------------
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen3-4B-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "Qwen3-4B-Q4_K_M.gguf")
N_CTX = int(os.environ.get("MODEL_N_CTX", "4096"))
N_THREADS = int(os.environ.get("MODEL_N_THREADS", str(os.cpu_count() or 4)))
FORCE_MOCK = os.environ.get("FORCE_MOCK", "0") == "1"


class Reasoner:
    def __init__(self) -> None:
        self.llm = None
        self.backend = "mock"
        if FORCE_MOCK:
            return
        try:
            from llama_cpp import Llama  # noqa: WPS433
            self.llm = Llama.from_pretrained(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", "0")),  # 0 = CPU-only, the reliable path on Spaces
                verbose=False,
            )
            self.backend = "llama.cpp"
        except Exception as exc:  # llama_cpp missing, or model not fetched yet
            print(f"[llm] falling back to mock opponent: {exc}")
            self.llm = None
            self.backend = "mock"

    # --------------------------------------------------------------------- #
    def choose(self, opp: Opponent, state: Dict) -> Dict:
        legal = [m for m in MOVES if state["ai_prana"] >= MOVES[m]["cost"]]
        if self.llm is None:
            return _mock_choose(opp, state, legal)
        return self._llm_choose(opp, state, legal)

    def _llm_choose(self, opp: Opponent, state: Dict, legal: List[str]) -> Dict:
        messages = [
            {"role": "system", "content": prompts.build_system(opp)},
            {"role": "user", "content": prompts.build_user(opp, state, legal)},
        ]
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=opp.think_tokens + 80,
            temperature=0.7 + 0.05 * (5 - opp.difficulty),  # brawlers run hotter
            top_p=0.9,
        )
        text = out["choices"][0]["message"]["content"]
        return prompts.parse_reply(text, legal)


# ------------------------------------------------------------------------- #
# Mock opponent — good enough to be fun, transparent enough to learn from.
# ------------------------------------------------------------------------- #
def _predict_player(history: List[Dict]) -> str:
    if not history:
        return "STRIKE"  # most people open aggressive
    recent = [h["player_move"] for h in history[-3:]]
    counts = Counter(recent)
    # modal recent move, tie broken toward the most recent
    best = max(counts, key=lambda m: (counts[m], recent[::-1].index(m) * -1))
    return best


# what beats a predicted player move, in priority order
_COUNTER = {
    "STRIKE":    ["MIST_STEP", "GUARD"],
    "GRAPPLE":   ["STRIKE", "MIST_STEP"],
    "GUARD":     ["GRAPPLE", "FOCUS"],
    "FOCUS":     ["ART", "STRIKE", "GRAPPLE"],
    "ART":       ["MIST_STEP", "GUARD"],
    "MIST_STEP": ["GUARD", "FOCUS"],   # refuse to attack into a read
}


def _persona_bias(opp: Opponent, predicted: str, state: Dict, legal: List[str]) -> str:
    prana = state["ai_prana"]

    if opp.id == "tashi":  # brawler: ignores reads, swings
        pool = (["STRIKE"] * 5) + (["ART"] if prana >= MOVES["ART"]["cost"] else []) + ["GRAPPLE"]
        return random.choice(pool)

    if opp.id == "norbu":  # patient counter-puncher
        last = state["history"][-1]["player_move"] if state["history"] else None
        if last == "FOCUS" and "STRIKE" in legal:
            return "STRIKE"  # punish their greed
        if prana < 2:
            return random.choice(["GUARD", "FOCUS"])
        return _first_legal(_COUNTER[predicted], legal, default="GUARD")

    if opp.id == "pema":  # read-merchant, loves the bet
        if random.random() < 0.25 and "GUARD" in legal:
            return "GUARD"  # feint to bait an attack
        return _first_legal(_COUNTER[predicted], legal, default="MIST_STEP")

    if opp.id == "drogpa":  # prana tyrant
        if prana >= MOVES["ART"]["cost"] and predicted != "MIST_STEP":
            return "ART"
        if prana < MOVES["ART"]["cost"]:
            return random.choice(["GUARD", "FOCUS"])
        return _first_legal(_COUNTER[predicted], legal, default="GUARD")

    # the-mountain: near-perfect counter, occasional layer-2 mixup
    if random.random() < 0.2:
        # assume the player expects the counter and goes one deeper
        return _first_legal(_COUNTER.get(predicted, []), legal, default="FOCUS")
    return _first_legal(_COUNTER[predicted], legal, default="MIST_STEP")


def _first_legal(prefs: List[str], legal: List[str], default: str) -> str:
    for p in prefs:
        if p in legal:
            return p
    return default if default in legal else (legal[0] if legal else "GUARD")


_REASONING = {
    "tashi": "They'll {pred_verb}? Doesn't matter. I hit harder and I hit now.",
    "norbu": "They have shown {pred}. Haste is a debt; let them pay it. I answer with {move_label}.",
    "pema": "Oh, I know you. You're going to {pred_verb} — you always do. So I'll {move_label} and watch you fall for it.",
    "drogpa": "The storm is not ready... / or it is. {move_label}. Stone does not hurry.",
    "the-mountain": "Your pattern leans toward {pred}. You expect me to counter it — so I weigh whether you have already changed. {move_label}.",
}

_TAUNT = {
    "tashi": ["Too slow!", "Is that all the leaf has?", "Sit down."],
    "norbu": ["Again.", "You are louder than you are skilled.", "Breathe. It is your last lesson."],
    "pema": ["Predictable little thing.", "I wrote this ending already.", "Dance for me."],
    "drogpa": ["...", "Avalanche.", "You should not have come up the mountain."],
    "the-mountain": ["I have already seen this duel.", "Notice yet?", "Come. The summit waits."],
}

_PRED_VERB = {
    "STRIKE": "swing", "GRAPPLE": "lunge in", "GUARD": "turtle up",
    "FOCUS": "stop to breathe", "ART": "throw a technique", "MIST_STEP": "try to read me",
}


def _mock_choose(opp: Opponent, state: Dict, legal: List[str]) -> Dict:
    predicted = _predict_player(state.get("history", []))
    move = _persona_bias(opp, predicted, state, legal)
    move = move if move in legal else _first_legal([move], legal, default="GUARD")

    reasoning = _REASONING[opp.id].format(
        pred=predicted.replace("_", " ").title(),
        pred_verb=_PRED_VERB.get(predicted, "move"),
        move_label=MOVES[move]["label"],
    )
    taunt = random.choice(_TAUNT[opp.id])
    return {"reasoning": reasoning, "move": move, "taunt": taunt}
