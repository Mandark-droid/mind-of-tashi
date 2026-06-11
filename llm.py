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
from typing import Dict, Iterator, List, Optional, Tuple

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

# Backend selector:
#   ""/"llamacpp"  -> llama.cpp GGUF (Off-the-Grid + Llama Champion). DEFAULT,
#                     so a local clone runs the model on llama.cpp out of the box.
#   "transformers" -> PyTorch + ZeroGPU: the deployed Space sets BACKEND=transformers
#                     to run the SFT safetensors on a dynamically-allocated GPU
#                     via @spaces.GPU (free, scales to many players). Still
#                     Off-the-Grid (local GPU, no cloud API).
#   "mock"         -> heuristic, no weights.
BACKEND = os.environ.get("BACKEND", "").strip().lower()
# The transformers backend pulls the safetensors checkpoint, NOT the GGUF.
TF_MODEL_REPO = os.environ.get("TF_MODEL_REPO", "build-small-hackathon/mind-of-tashi-micro-sft")
GPU_DURATION = int(os.environ.get("GPU_DURATION", "60"))


# --- transformers / ZeroGPU backend --------------------------------------- #
# @spaces.GPU requests a GPU for the call's duration (ZeroGPU) and is a no-op
# off-ZeroGPU, so the same code path runs locally on a normal GPU/CPU. If the
# `spaces` package is absent (pure llama.cpp/mock installs) fall back to a no-op.
try:
    import spaces as _spaces  # type: ignore
    _GPU = _spaces.GPU
except Exception:  # spaces not installed
    def _GPU(*dargs, **dkw):
        if len(dargs) == 1 and callable(dargs[0]) and not dkw:
            return dargs[0]                        # bare @_GPU
        def _deco(fn):
            return fn
        return _deco                              # @_GPU(duration=...)

# repo -> (tokenizer, model). The house opponent loads TF_MODEL_REPO; the
# self-play challenger roster can load additional repos on the same path.
_TF_CACHE: Dict[str, tuple] = {}


def _load_transformers(repo: Optional[str] = None,
                       device: Optional[str] = None) -> None:
    """Load a safetensors checkpoint. The house model loads to cuda at boot
    (ZeroGPU's CUDA-emulation intercepts startup-time `.to('cuda')`); models
    loaded at REQUEST time (self-play challengers) MUST pass device='cpu' —
    ZeroGPU refuses cuda init outside @spaces.GPU after boot. _tf_generate
    hops cpu-resident models onto the GPU inside the decorated call instead.
    transformers is the model's *native* runtime, so the mind-scrolls are
    faithful (the norm_topk_prob issue was llama.cpp-only)."""
    repo = repo or TF_MODEL_REPO
    if repo in _TF_CACHE:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Default "cuda" on ZeroGPU (boot-time emulation). Override TF_DEVICE=cpu
    # for a GPU-less local clone (slow but functional).
    device = device or os.environ.get("TF_DEVICE", "cuda")
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    _TF_CACHE[repo] = (tok, model)


@_GPU(duration=GPU_DURATION)
def _tf_generate(messages: List[Dict], max_new_tokens: int,
                 temperature: float, top_p: float,
                 repo: Optional[str] = None) -> Dict:
    """Generate one completion on the (ZeroGPU) GPU and return it in the OpenAI
    chat shape — message content + per-token top-K logprobs — so the Conviction
    Meter / parse path are reused unchanged across both backends.

    CPU-resident models (request-time challenger loads) are moved to the GPU
    here — inside @spaces.GPU, where real CUDA exists — and moved back after,
    since ZeroGPU reclaims the device between calls."""
    import torch
    tok, model = _TF_CACHE[repo or TF_MODEL_REPO]
    moved = False
    if model.device.type == "cpu" and torch.cuda.is_available():
        model.to("cuda")
        moved = True
    try:
        # apply_chat_template returns a BatchEncoding (dict) in transformers 5.x;
        # splat it into generate so input_ids + attention_mask both go through.
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        do_sample = bool(temperature and temperature > 0.0)
        pad_id = tok.pad_token_id
        if pad_id is None:  # some models define eos as a LIST — take the first
            eos = tok.eos_token_id
            pad_id = eos[0] if isinstance(eos, (list, tuple)) else eos
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=(temperature if do_sample else None),
                top_p=(top_p if do_sample else None),
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=pad_id,
            )
        gen_ids = gen.sequences[0][prompt_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        content: List[Dict] = []
        for i, step_logits in enumerate(gen.scores):
            if i >= len(gen_ids):
                break
            lp = torch.log_softmax(step_logits[0].float(), dim=-1)
            tid = int(gen_ids[i].item())
            k = min(CONVICTION_TOPK, lp.shape[-1])
            topk = torch.topk(lp, k=k)
            top = [{"token": tok.decode([int(idx)]), "logprob": float(v)}
                   for v, idx in zip(topk.values.tolist(), topk.indices.tolist())]
            content.append({"token": tok.decode([tid]),
                            "logprob": float(lp[tid].item()), "top_logprobs": top})
        return {"choices": [{"message": {"content": text},
                             "logprobs": {"content": content}}]}
    finally:
        if moved:
            model.to("cpu")
            torch.cuda.empty_cache()


class Reasoner:
    def __init__(self, repo: Optional[str] = None, filename: Optional[str] = None,
                 backend: Optional[str] = None, tf_repo: Optional[str] = None,
                 tf_device: Optional[str] = None) -> None:
        """repo/filename override MODEL_REPO/MODEL_FILE (llama.cpp path);
        tf_repo/tf_device override the transformers path (challengers pass
        tf_device='cpu' — request-time cuda loads are forbidden on ZeroGPU);
        backend overrides the BACKEND env. All exist for the self-play
        challenger roster, which loads a *different* model per challenger
        while the Space's house opponent keeps its own backend untouched."""
        self.llm = None
        self.backend = "mock"
        self.load_error: Optional[str] = None  # why we fell back, if we did
        self._tf_repo = tf_repo or TF_MODEL_REPO
        if FORCE_MOCK:
            return
        backend = (backend or BACKEND or "llamacpp")  # default = llama.cpp (Llama Champion) for local clones
        if backend == "mock":
            return
        if backend == "transformers":
            try:
                _load_transformers(self._tf_repo, device=tf_device)
                self.backend = "transformers"
                print(f"[llm] backend: transformers (ZeroGPU-ready) [{self._tf_repo}]")
            except Exception as exc:
                print(f"[llm] transformers backend failed ({exc}); falling back to mock")
                self.load_error = f"transformers: {exc}"
                self.backend = "mock"
            return
        # --- llama.cpp (default; Off-the-Grid + Llama Champion) ---
        try:
            from llama_cpp import Llama  # noqa: WPS433
            common = dict(
                repo_id=repo or MODEL_REPO,
                filename=filename or MODEL_FILE,
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
            self.load_error = f"llama.cpp: {exc}"
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
        # GRAMMAR-LOCKED OATH (§E3): the player may seal one of her moves this
        # round (validated + paid for in app.ai_turn). Drop it from her legal set
        # so the mock can't pick it and parse_reply can't fall back to it; on the
        # real path it's also enforced by a GBNF grammar below. Blind-commit safe
        # — a sealed move is public, not the player's pending choice.
        sealed = (state.get("sealed_move") or "") or None
        if sealed and sealed in legal and len(legal) > 1:
            legal = [m for m in legal if m != sealed]
        else:
            sealed = None
        # CRACK HER COMPOSURE (§E2): how rattled she is, from past outcomes only
        # (blind-commit safe). Drives temperature on the real path and a wild-
        # swing chance on the mock path; surfaced to the UI as `composure`.
        tilt = _composure_tilt(state)
        composure = int(round(100 * (1 - tilt)))
        if self.backend == "mock":
            parsed = _mock_choose(opp, state, legal)
            if legal and tilt > 0 and random.random() < tilt * 0.5:
                parsed["move"] = random.choice(legal)  # composure cracks: a wild swing
            parsed["conviction"] = _degrade_conviction(_mock_conviction(opp, parsed), tilt)
            parsed["composure"] = composure
            return parsed, _synthesize_raw(parsed)
        messages = [
            {"role": "system", "content": prompts.build_system(opp)},
            {"role": "user", "content": prompts.build_user(opp, state, legal, sealed)},
        ]
        # When an oath is active, force the move token via GBNF so the sealed move
        # is literally undecodable — the mind-scroll reroutes around the hole.
        grammar_src = _move_grammar(legal) if sealed else None
        out = self._complete(messages, opp, tilt, grammar_src)
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

    def _complete(self, messages: List[Dict], opp: Opponent, tilt: float = 0.0,
                  grammar_src: str | None = None):
        """create_chat_completion with logprobs (+ optional GBNF grammar).

        ``tilt`` (0..1, §E2) adds to the sampling temperature so a rattled
        opponent reasons worse. ``grammar_src`` (§E3) is a GBNF string that
        forces the move token — passed best-effort: if grammar load or a
        grammar-constrained generation fails, we retry without it (the legal
        filter still enforces the seal). Logprobs are requested in whichever
        dialect this llama-cpp build speaks; any failure degrades the meter.
        """
        temperature = min(
            COMPOSURE_TEMP_MAX,
            0.7 + 0.05 * (5 - opp.difficulty)  # brawlers run hotter
            + max(0.0, min(1.0, tilt)) * COMPOSURE_TEMP_GAIN,  # +heat as she frays
        )
        # transformers/ZeroGPU path: generate on GPU, return the OpenAI shape.
        # The Oath's hard GBNF mask is llama.cpp-only; on this backend the seal
        # is still enforced by the legal-move filter (sealed move dropped from
        # `legal` upstream, and parse_reply can't fall back to it).
        if self.backend == "transformers":
            return _tf_generate(messages, opp.think_tokens + 80, temperature, 0.9,
                                repo=self._tf_repo)
        base = dict(
            messages=messages,
            max_tokens=opp.think_tokens + 80,
            temperature=temperature,
            top_p=0.9,
        )

        def _try(extra: Dict):
            # Try the top-K logprob dialects in turn (OpenAI bool+top_logprobs,
            # then int, then bare); return the first that the build accepts.
            last = None
            for lp in ({"logprobs": True, "top_logprobs": CONVICTION_TOPK},
                       {"logprobs": CONVICTION_TOPK}, {}):
                try:
                    return self.llm.create_chat_completion(**base, **extra, **lp)
                except (TypeError, ValueError) as exc:
                    last = exc
            raise last if last else RuntimeError("create_chat_completion failed")

        grammar = _load_grammar(grammar_src)
        if grammar is not None:
            try:
                return _try({"grammar": grammar})
            except Exception as exc:  # grammar kwarg/gen unsupported — drop it
                print(f"[llm] GBNF grammar path failed ({exc}); retrying without")
        return _try({})


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
# Grammar-Locked Oath (IDEAS.md §E3) — when the player seals a move, build a
# GBNF grammar that lets the model think freely (<think>…</think>) but forces
# the committed move token to one of the still-allowed moves. The sealed move
# becomes literally undecodable; the mind-scroll reroutes around the hole.
# ------------------------------------------------------------------------- #
def _move_grammar(allowed: List[str]) -> str:
    """A GBNF grammar permitting a free-form think block then a JSON object whose
    ``move`` is one of ``allowed``. (``[^<]*`` for the think body assumes the
    reasoning has no stray ``<`` — true for our bilingual scrolls.)"""
    alts = " | ".join('"\\"' + m + '\\""' for m in allowed) or '"\\"FOCUS\\""'
    return (
        'root ::= think ws obj\n'
        'think ::= "<think>" tbody "</think>"\n'
        'tbody ::= [^<]*\n'
        'obj ::= "{" ws "\\"move\\"" ws ":" ws move ws "," ws "\\"taunt\\"" ws ":" ws str ws "}"\n'
        'move ::= ' + alts + '\n'
        'str ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""\n'
        'ws ::= [ \\t\\n]*\n'
    )


def _load_grammar(src: str | None):
    """Compile a GBNF string to a LlamaGrammar, or None (caller proceeds
    without the hard constraint; the legal filter still enforces the seal)."""
    if not src:
        return None
    try:
        from llama_cpp import LlamaGrammar  # noqa: WPS433
        return LlamaGrammar.from_string(src, verbose=False)
    except Exception as exc:
        print(f"[llm] grammar load failed ({exc}); proceeding without GBNF")
        return None


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
