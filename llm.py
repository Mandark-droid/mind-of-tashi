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
import json
import math
import os
import random
from collections import Counter
from typing import Dict, Iterator, List, Tuple

import prompts
from engine import MOVES, ATTACKS, MAX_HP
from opponents import Opponent

# ----- configuration (override via Space "Variables") ---------------------
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen3-4B-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "Qwen3-4B-Q4_K_M.gguf")
N_CTX = int(os.environ.get("MODEL_N_CTX", "4096"))
N_THREADS = int(os.environ.get("MODEL_N_THREADS", str(os.cpu_count() or 4)))
FORCE_MOCK = os.environ.get("FORCE_MOCK", "0") == "1"
# llama.cpp only exposes per-token logprobs (the Conviction Meter, §E1) when the
# context is built with logits_all=True. It costs extra compute/RAM (logits for
# every token, not just the sampled one), so it's switchable for constrained
# Spaces — but ON by default because the meter is a headline feature. When off,
# the meter degrades to the labelled "estimate" path.
LOGITS_ALL = os.environ.get("LOGITS_ALL", "1") == "1"
# How many top alternatives to pull per token for the entropy-based conviction.
CONVICTION_TOPK = int(os.environ.get("CONVICTION_TOPK", "5"))
# CRACK HER COMPOSURE (IDEAS.md §E2): as the player lands reads/counters, the
# opponent's sampling temperature rises — her reasoning genuinely frays (and the
# Conviction Meter drops with it, since hotter sampling = more entropy). Gain =
# extra temperature at full tilt; MAX caps it so she degrades, not pure gibberish.
COMPOSURE_TEMP_GAIN = float(os.environ.get("COMPOSURE_TEMP_GAIN", "0.8"))
COMPOSURE_TEMP_MAX = float(os.environ.get("COMPOSURE_TEMP_MAX", "1.6"))


class Reasoner:
    def __init__(self) -> None:
        self.llm = None
        self.backend = "mock"
        if FORCE_MOCK:
            return
        try:
            from llama_cpp import Llama  # noqa: WPS433
            common = dict(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", "0")),  # 0 = CPU-only, the reliable path on Spaces
                verbose=False,
            )
            if LOGITS_ALL:
                try:
                    self.llm = Llama.from_pretrained(logits_all=True, **common)
                except TypeError as exc:  # logits_all removed in some builds
                    print(f"[llm] logits_all unsupported ({exc}); Conviction Meter "
                          "degrades to estimate")
                    self.llm = Llama.from_pretrained(**common)
            else:
                self.llm = Llama.from_pretrained(**common)
            self.backend = "llama.cpp"
        except Exception as exc:  # llama_cpp missing, or model not fetched yet
            print(f"[llm] falling back to mock opponent: {exc}")
            self.llm = None
            self.backend = "mock"

    # --------------------------------------------------------------------- #
    def choose(self, opp: Opponent, state: Dict) -> Dict:
        parsed, _raw = self.choose_with_raw(opp, state)
        return parsed

    def choose_with_raw(self, opp: Opponent, state: Dict) -> Tuple[Dict, str]:
        """Like choose() but also returns the raw model completion text.

        Self-play data collection wants the full <think>...</think>{json} string
        (that's the SFT target). For the mock backend we synthesise an
        equivalent raw string so downstream code can be agnostic.
        """
        legal = [m for m in MOVES if state["ai_prana"] >= MOVES[m]["cost"]]
        # CRACK HER COMPOSURE (§E2): how rattled she is, from past outcomes only
        # (blind-commit safe). Drives temperature on the real path and a wild-
        # swing chance on the mock path; surfaced to the UI as `composure`.
        tilt = _composure_tilt(state)
        composure = int(round(100 * (1 - tilt)))
        if self.llm is None:
            parsed = _mock_choose(opp, state, legal)
            if legal and tilt > 0 and random.random() < tilt * 0.5:
                parsed["move"] = random.choice(legal)  # composure cracks: a wild swing
            parsed["conviction"] = _degrade_conviction(_mock_conviction(opp, parsed), tilt)
            parsed["composure"] = composure
            return parsed, _synthesize_raw(parsed)
        messages = [
            {"role": "system", "content": prompts.build_system(opp)},
            {"role": "user", "content": prompts.build_user(opp, state, legal)},
        ]
        out = self._complete(messages, opp, tilt)
        raw = out["choices"][0]["message"]["content"]
        parsed = prompts.parse_reply(raw, legal)
        # CONVICTION METER (IDEAS.md §E1): read the model's per-token confidence
        # straight off the llama.cpp logprobs — the signal a cloud API can't
        # expose — and surface it to the UI as a readable "tell". (Her rising
        # temperature from §E2 naturally pushes this down as you crack her.)
        conv = _conviction_from_completion(out, parsed.get("move"))
        if conv is None:  # logprobs unavailable on this build — estimate instead
            conv = _degrade_conviction(_mock_conviction(opp, parsed), tilt)
            conv["source"] = "estimated"
        parsed["conviction"] = conv
        parsed["composure"] = composure
        return parsed, raw

    def _complete(self, messages: List[Dict], opp: Opponent, tilt: float = 0.0):
        """create_chat_completion, asking for logprobs; degrade if unsupported.

        ``tilt`` (0..1, §E2) adds to the sampling temperature so a rattled
        opponent reasons worse. Older llama-cpp-python builds reject the
        OpenAI-style ``logprobs`` / ``top_logprobs`` kwargs — in that case we
        retry without them and the Conviction Meter falls back to an estimate.
        """
        temperature = min(
            COMPOSURE_TEMP_MAX,
            0.7 + 0.05 * (5 - opp.difficulty)  # brawlers run hotter
            + max(0.0, min(1.0, tilt)) * COMPOSURE_TEMP_GAIN,  # +heat as she frays
        )
        base = dict(
            messages=messages,
            max_tokens=opp.think_tokens + 80,
            temperature=temperature,
            top_p=0.9,
        )
        # Ask for the top-K logprob distribution per token (K via CONVICTION_TOPK)
        # so conviction can be measured as distribution entropy, not just the
        # emitted token's probability. Try whichever dialect this llama-cpp build
        # speaks: OpenAI-style (bool + top_logprobs) first, then the integer
        # form, then bare. Any failure degrades the meter to an estimate.
        k = CONVICTION_TOPK
        for kwargs in ({"logprobs": True, "top_logprobs": k}, {"logprobs": k}, {}):
            try:
                return self.llm.create_chat_completion(**base, **kwargs)
            except (TypeError, ValueError) as exc:
                print(f"[llm] logprobs variant {kwargs or '{}'} rejected ({exc}); trying next")
        return self.llm.create_chat_completion(**base)


def _synthesize_raw(parsed: Dict) -> str:
    """Format a mock-backend choice as if it had come from the model.

    Lets the self-play harness treat both backends uniformly when writing the
    `raw_completion` field of its JSONL output.
    """
    obj = {"move": parsed["move"], "taunt": parsed["taunt"]}
    return f"<think>{parsed['reasoning']}</think>\n{json.dumps(obj, ensure_ascii=False)}"


# ------------------------------------------------------------------------- #
# Conviction Meter (IDEAS.md §E1 / ROADMAP §2.5) — turn the model's raw
# token-level confidence into a readable game signal. Everything here is a
# pure read on the existing llama.cpp completion: no extra inference, no
# network, blind-commit untouched. The shape returned to the UI:
#   {score, decision_score, think_score, source, peak:{i,t}|None,
#    tokens:[{t, c}]}   where c = per-token conviction 0..100
# ------------------------------------------------------------------------- #
def _iter_token_dists(out: Dict) -> Iterator[Tuple[str, float, List[Tuple[str, float]]]]:
    """Yield (token_text, token_logprob, top_alternatives) per generated token.

    ``top_alternatives`` is a list of (token, logprob) for the top-K candidates
    at that step (possibly empty). Handles both the OpenAI-style chat shape
    (``choices[0].logprobs.content = [{token, logprob, top_logprobs:[...]}]``)
    and the older llama.cpp completion shape
    (``{tokens, token_logprobs, top_logprobs:[{tok: lp}, ...]}``).
    """
    try:
        choice = out["choices"][0]
    except (KeyError, IndexError, TypeError):
        return
    lp = choice.get("logprobs") if isinstance(choice, dict) else None
    if not isinstance(lp, dict):
        return
    content = lp.get("content")
    if content:
        for item in content:
            val = item.get("logprob")
            if val is None:
                continue
            top = [
                (str(t.get("token", "")), float(t["logprob"]))
                for t in (item.get("top_logprobs") or [])
                if t.get("logprob") is not None
            ]
            yield str(item.get("token", "")), float(val), top
        return
    toks, tlps, tops = lp.get("tokens"), lp.get("token_logprobs"), lp.get("top_logprobs")
    if toks and tlps and len(toks) == len(tlps):
        for i, (tok, val) in enumerate(zip(toks, tlps)):
            if val is None:
                continue
            top: List[Tuple[str, float]] = []
            if tops and i < len(tops) and isinstance(tops[i], dict):
                top = [(str(kk), float(vv)) for kk, vv in tops[i].items() if vv is not None]
            yield str(tok), float(val), top


def _token_conviction(token_lp: float, top: List[Tuple[str, float]]) -> int:
    """Per-token conviction 0..100.

    Primary signal = how *concentrated* the model's belief was over the top-K
    alternatives: 1 - normalised Shannon entropy (peaked distribution → high
    conviction, torn distribution → low). Falls back to the emitted token's
    own probability when no distribution is available (e.g. top_logprobs=1 or
    a build that omits it).
    """
    probs = [math.exp(lp) for _, lp in top if lp is not None]
    probs = [p for p in probs if p > 0.0]
    if len(probs) >= 2:
        s = sum(probs)
        if s > 0.0:
            probs = [p / s for p in probs]
            h = -sum(p * math.log2(p) for p in probs if p > 0.0)
            hmax = math.log2(len(probs))
            conf = 1.0 - (h / hmax if hmax > 0.0 else 0.0)
            return int(round(max(0.0, min(1.0, conf)) * 100))
    p0 = math.exp(token_lp)
    return int(round(max(0.0, min(1.0, p0)) * 100))


def _conviction_from_completion(out: Dict, move_id: str | None) -> Dict | None:
    """Compute the conviction signal from a completion's logprobs, or None.

    ``score`` (the headline gauge) is the model's confidence in the tokens of
    the move it actually committed — i.e. *how exploitable was this commit*.
    ``tokens`` carries the per-token confidence of the <think> block so the UI
    can tint the mind-scroll, and ``peak`` marks the single most-hesitant word.
    """
    pairs = list(_iter_token_dists(out))
    if not pairs:
        return None

    spans = []  # [start, end, token_text, conviction_pct]
    text = ""
    for tok, lp, top in pairs:
        start = len(text)
        text += tok
        spans.append([start, len(text), tok, _token_conviction(lp, top)])

    ti, tc = text.find("<think>"), text.find("</think>")
    think_lo = (ti + len("<think>")) if ti != -1 else 0
    think_hi = tc if tc != -1 else len(text)
    post_lo = (tc + len("</think>")) if tc != -1 else 0

    think_tokens = [
        {"t": tok, "c": c}
        for s, e, tok, c in spans
        if e > think_lo and s < think_hi
    ]

    # decision tokens = the tokens spelling the committed move id, after </think>
    decision_convs: List[int] = []
    if move_id:
        mpos = text.find(move_id, post_lo)
        if mpos == -1:
            mpos = text.upper().find(move_id.upper(), post_lo)
        if mpos != -1:
            mend = mpos + len(move_id)
            decision_convs = [c for s, e, tok, c in spans if e > mpos and s < mend]
    if not decision_convs:  # couldn't pin the move id — use the whole JSON line
        decision_convs = [c for s, e, tok, c in spans if s >= post_lo] or [c for *_, c in spans]

    think_convs = [tt["c"] for tt in think_tokens] or [c for *_, c in spans]

    # peak hesitation = lowest-confidence *wordy* think token. Prefer tokens with
    # >=2 letters so the "she wavered" flash lands on a real word, not a stray
    # sub-word fragment ("v", "i"); fall back to any alpha token if none qualify.
    peak = None
    _alpha = lambda s: sum(ch.isalpha() for ch in s)
    wordy = [(i, tt) for i, tt in enumerate(think_tokens) if _alpha(tt["t"]) >= 2]
    if not wordy:
        wordy = [(i, tt) for i, tt in enumerate(think_tokens) if _alpha(tt["t"]) >= 1]
    if wordy:
        i, tt = min(wordy, key=lambda it: it[1]["c"])
        peak = {"i": i, "t": tt["t"].strip()}

    return {
        "score": int(round(sum(decision_convs) / len(decision_convs))),
        "decision_score": int(round(sum(decision_convs) / len(decision_convs))),
        "think_score": int(round(sum(think_convs) / len(think_convs))),
        "source": "llama.cpp",
        "tokens": think_tokens,
        "peak": peak,
    }


# Per-persona baseline conviction for the mock / estimate path, so the meter
# is demoable without a GGUF. tenzin (storm-voice, deliberately unpredictable)
# reads as low-conviction; the summit bosses read as ice-cold sure.
_MOCK_CONVICTION_BASE = {
    "tashi": 82, "lhamo": 70, "norbu": 74, "yeshi": 68, "pema": 78,
    "karma": 72, "drogpa": 70, "tenzin": 48, "the-mountain": 88,
    "the-veiled-one": 90,
}
_HEDGE_WORDS = {"not", "or", "yet", "wait", "no", "maybe", "perhaps", "but", "...", "—"}


def _mock_conviction(opp: Opponent, parsed: Dict) -> Dict:
    """Synthesize a believable conviction signal when real logprobs are absent.

    Marked ``source != "llama.cpp"`` so the UI can flag it as an estimate — we
    do not pass synthetic confidence off as the real thing.
    """
    base = _MOCK_CONVICTION_BASE.get(opp.id, 66)
    words = (parsed.get("reasoning") or "").split()
    tokens = [
        {"t": (" " if i else "") + w, "c": max(20, min(98, base + random.randint(-8, 8)))}
        for i, w in enumerate(words)
    ]
    peak = None
    if tokens:
        hedge_idx = next(
            (i for i, w in enumerate(words) if w.strip(".,;:!?").lower() in _HEDGE_WORDS),
            None,
        )
        di = hedge_idx if hedge_idx is not None else len(tokens) // 2
        tokens[di]["c"] = max(12, base - 45)  # carve one clear dip for the flash
        peak = {"i": di, "t": words[di].strip()}
    return {
        "score": base, "decision_score": base, "think_score": base,
        "source": "mock", "tokens": tokens, "peak": peak,
    }


# ------------------------------------------------------------------------- #
# Crack Her Composure (IDEAS.md §E2) — derive how rattled the opponent is from
# PAST outcomes only (never the pending player move), so the blind-commit
# contract holds. The tilt raises her temperature (real path) and degrades the
# conviction signal she shows.
# ------------------------------------------------------------------------- #
def _composure_tilt(state: Dict) -> float:
    """0.0 = composed, 1.0 = fully rattled.

    Weighs cumulative HP lost (she bleeds, she frays) with how often the player
    has recently *landed* on her (successful reads/counters crack composure
    faster than slow attrition).
    """
    ai_hp = state.get("ai_hp", MAX_HP)
    hp_loss = ((MAX_HP - ai_hp) / MAX_HP) if MAX_HP else 0.0
    hp_loss = max(0.0, min(1.0, hp_loss))
    hist = (state.get("history") or [])[-4:]
    hits = sum(1 for h in hist if str(h.get("outcome", "")).startswith("you landed"))
    recent = (hits / len(hist)) if hist else 0.0
    return max(0.0, min(1.0, 0.6 * hp_loss + 0.4 * recent))


def _degrade_conviction(conv: Dict | None, tilt: float) -> Dict | None:
    """Scale a conviction dict down by tilt (mock/estimate path only).

    On the real backend the rising temperature already lowers conviction via
    genuine entropy; this keeps the mock/estimate path consistent with that.
    """
    if not conv or tilt <= 0:
        return conv
    f = max(0.0, 1.0 - 0.5 * tilt)
    for key in ("score", "decision_score", "think_score"):
        if isinstance(conv.get(key), (int, float)):
            conv[key] = int(round(conv[key] * f))
    for tk in conv.get("tokens") or []:
        if isinstance(tk.get("c"), (int, float)):
            tk["c"] = int(round(tk["c"] * f))
    return conv


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

    if opp.id == "lhamo":  # bridge-walker: tempo-reader, picks the moment
        last = state["history"][-1]["player_move"] if state["history"] else None
        if last in ("FOCUS", "ART") and "STRIKE" in legal:
            return "STRIKE"  # post-commit window
        if prana < 2:
            return random.choice(["GUARD", "FOCUS"])
        if prana >= MOVES["ART"]["cost"] and predicted == "FOCUS" and "ART" in legal:
            return "ART"
        return _first_legal(["GUARD", "FOCUS"], legal, default="GUARD")

    if opp.id == "norbu":  # patient counter-puncher
        last = state["history"][-1]["player_move"] if state["history"] else None
        if last == "FOCUS" and "STRIKE" in legal:
            return "STRIKE"  # punish their greed
        if prana < 2:
            return random.choice(["GUARD", "FOCUS"])
        return _first_legal(_COUNTER[predicted], legal, default="GUARD")

    if opp.id == "yeshi":  # mirror-ice: reflects last player move back as a counter
        last = state["history"][-1]["player_move"] if state["history"] else None
        if last is None:
            return _first_legal(["MIST_STEP", "GUARD"], legal, default="GUARD")
        return _first_legal(_COUNTER.get(last, []), legal, default="GUARD")

    if opp.id == "pema":  # read-merchant, loves the bet
        if random.random() < 0.25 and "GUARD" in legal:
            return "GUARD"  # feint to bait an attack
        return _first_legal(_COUNTER[predicted], legal, default="MIST_STEP")

    if opp.id == "karma":  # glacier-heart: bank prana, then unleash ART
        if prana < 4:
            return random.choice(["FOCUS", "GUARD", "FOCUS"])  # bank, bank, bank
        if "ART" in legal and predicted != "MIST_STEP":
            return "ART"
        return _first_legal(_COUNTER[predicted], legal, default="GUARD")

    if opp.id == "drogpa":  # prana tyrant
        if prana >= MOVES["ART"]["cost"] and predicted != "MIST_STEP":
            return "ART"
        if prana < MOVES["ART"]["cost"]:
            return random.choice(["GUARD", "FOCUS"])
        return _first_legal(_COUNTER[predicted], legal, default="GUARD")

    if opp.id == "tenzin":  # storm-voice: deliberately unpredictable
        # uniform-ish over legal, slight bias toward high-variance plays
        weighted = list(legal)
        for hv in ("ART", "MIST_STEP"):
            if hv in legal:
                weighted.append(hv)  # small bias
        return random.choice(weighted)

    if opp.id == "the-veiled-one":  # transcendent: two layers deep, balanced
        # 30% layer-2 mixup, otherwise hard counter; balanced move use
        if random.random() < 0.3:
            return _first_legal(_COUNTER.get(predicted, []), legal, default="FOCUS")
        return _first_legal(_COUNTER[predicted], legal, default="MIST_STEP")

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
    "lhamo": "The wind on the ropes... they will {pred_verb}. Not yet. Now. {move_label}.",
    "norbu": "They have shown {pred}. Haste is a debt; let them pay it. I answer with {move_label}.",
    "yeshi": "They showed {pred}. The lake returns what it is shown — {move_label}, sharpened.",
    "pema": "Oh, I know you. You're going to {pred_verb} — you always do. So I'll {move_label} and watch you fall for it.",
    "karma": "The ice has not finished forming. {move_label}. A glacier does not hurry to break.",
    "drogpa": "The storm is not ready... / or it is. {move_label}. Stone does not hurry.",
    "tenzin": "The wind turns. They expect {pred}? Let it be {move_label} instead. Ha!",
    "the-mountain": "Your pattern leans toward {pred}. You expect me to counter it — so I weigh whether you have already changed. {move_label}.",
    "the-veiled-one": "I have already seen you choose {pred}. I have already chosen {move_label}. The duel is a memory.",
}

_TAUNT = {
    "tashi": ["Too slow!", "Is that all the leaf has?", "Sit down."],
    "lhamo": ["The moment was now.", "You stepped too early.", "The bridge holds. You do not."],
    "norbu": ["Again.", "You are louder than you are skilled.", "Breathe. It is your last lesson."],
    "yeshi": ["The lake returns it.", "I am only what you brought.", "Look at yourself."],
    "pema": ["Predictable little thing.", "I wrote this ending already.", "Dance for me."],
    "karma": ["Slowly.", "The ice has formed.", "Centuries learn this. You have minutes."],
    "drogpa": ["...", "Avalanche.", "You should not have come up the mountain."],
    "tenzin": ["Ha!", "The wind turns.", "Listen for the next one. You won't hear it."],
    "the-mountain": ["I have already seen this duel.", "Notice yet?", "Come. The summit waits."],
    "the-veiled-one": ["It is already done.", "You arrived where I waited.", "The veil parts only one way."],
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
