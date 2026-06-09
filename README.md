---
title: The Mind of Tashi
emoji: 🌫️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.15.0
app_file: app.py
pinned: true
license: mit
hf_oauth: true
models:
  - build-small-hackathon/mind-of-tashi-micro-sft
  - build-small-hackathon/mind-of-tashi-micro-sft-gguf
datasets:
  - build-small-hackathon/mind-of-tashi-selfplay
---

# The Mind of Tashi — a duel of foresight

> Built for the **Build Small Hackathon · Track Two: An Adventure in Thousand
> Token Wood**.

You climb a gauntlet of ninja-monks in the Village Hidden in the Mist, high in
the Himalayas. Every round, you and your opponent **secretly commit one move at
the same time** — there is no reacting, only *reading*. The opponent is a small
reasoning model, and the thing on centre stage is its **mind**: after each
commit, the scroll reveals how it just read you ("you've drawn breath twice
unpunished — you're greedy, so I strike"). The whole game is out-thinking
something that narrates how it's out-thinking you.

## Why the AI is load-bearing

Strip the model and this is rock-paper-scissors. The model's *prediction of
your next move* is the entire mechanic — and it commits **blind**, seeing only
the history, never your current move. The signature move, **Mist-Step**, only
rewards you if your opponent attacks this turn, so it is a pure bet on a read.
That recursion ("I think you'll strike, so I Mist-Step / I think you think
that, so I draw breath") is what a reasoning model is uniquely good at.

And because the opponent is a **small local model run through llama.cpp**, the
game can read its *uncertainty*: per-token entropy drives a **Conviction
Meter**, your reads raise its sampling temperature ("crack her composure"), and
a prāṇa-spent **Oath** drops a move from its decode grammar so it literally
cannot choose it. A cloud API hides the logits; a giant model can't stream them
on a CPU Space — this is what the small-model class uniquely makes possible.

## The model — David vs Goliath

The opponent's mind is a custom MoE (~0.4B total / **~200M active per token**),
SFT'd (and GRPO-trained) to read an opponent and commit blind in an English +
Hindi/Sanskrit (IAST) code-switched register — 10–100× smaller (active) than
the frontier APIs it learned from. It ships as a Q4_K_M GGUF and runs in-Space
via llama.cpp: **no cloud API at runtime.**

## The six-artifact bundle

This Space is one of six linked artifacts:

1. **Game / Space** — you are here.
2. **Self-play dataset** — [`mind-of-tashi-selfplay`](https://huggingface.co/datasets/build-small-hackathon/mind-of-tashi-selfplay)
3. **SFT model + GGUF** — [`mind-of-tashi-micro-sft`](https://huggingface.co/build-small-hackathon/mind-of-tashi-micro-sft) · [`-sft-gguf`](https://huggingface.co/build-small-hackathon/mind-of-tashi-micro-sft-gguf)
4. **OpenEnv gym** — [`mind-of-tashi-env`](https://huggingface.co/spaces/build-small-hackathon/mind-of-tashi-env)
5. **GRPO model + GGUF** — [`mind-of-tashi-micro-grpo`](https://huggingface.co/build-small-hackathon/mind-of-tashi-micro-grpo) · [`-grpo-gguf`](https://huggingface.co/build-small-hackathon/mind-of-tashi-micro-grpo-gguf)
6. **Deployed Space** — this one, with the fine-tuned GGUF wired in.

## The moves

| Move | Cost | Beats / loses |
|---|---|---|
| **Vajra Strike** | free | beats Grapple · blocked by Guard |
| **Mountain Stance** (guard) | free, +1 prāṇa | blocks Strike, softens Art · broken by Grapple |
| **River Throw** (grapple) | free | breaks Guard · loses to Strike |
| **Draw Breath** (focus) | free, +2 prāṇa | gathers prāṇa but fully exposed |
| **Prāṇa Art** | 3 prāṇa | heavy ranged hit · eaten by Mist-Step |
| **Mist-Step** | 2 prāṇa | dodges + counters an attack · whiffs vs caution |

## Run it

```bash
pip install -r requirements.txt
python app.py            # serves the custom frontend at http://localhost:7860
```

It runs immediately on the **mock opponent** (a persona-aware heuristic) — no
model download needed. To play against the real reasoning model, set the env
vars below and the app loads the GGUF through llama.cpp on next boot.

## Swap in the real (or fine-tuned) model

The web layer is model-agnostic. Configure via Space **Variables** / env:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_REPO` | `build-small-hackathon/mind-of-tashi-micro-sft-gguf` | any GGUF repo on the Hub (swap to `-grpo-gguf` after A/B) |
| `MODEL_FILE` | `mind-of-tashi-micro-sft-Q4_K_M.gguf` | the quant file |
| `N_GPU_LAYERS` | `0` | keep 0 (CPU) on Spaces — see note |
| `FORCE_MOCK` | `0` | set `1` to force the heuristic opponent |
| `LEADERBOARD_REPO` | unset | `build-small-hackathon/mind-of-tashi-runs` to enable the board |
| `HF_TOKEN` | unset | write-scoped, in Space **Secrets**, for leaderboard writes |

**Note on hardware:** llama.cpp + **ZeroGPU** is unreliable. Run llama.cpp
**CPU-only** on a CPU-upgrade Space, or use a **dedicated GPU** Space (not
ZeroGPU). Turn-based play makes a few seconds of "reading you…" feel like
drama, not lag.

## Bonus badges targeted

- **Off the Grid** — model runs in-Space via llama.cpp, no cloud API.
- **Llama Champion** — opponent runs through the llama.cpp runtime.
- **Off-Brand** — fully custom frontend on `gradio.Server` (Gradio 6).
- **Well-Tuned** — a fine-tuned custom-MoE GGUF dropped in via `MODEL_REPO`.

## Layout

```
app.py          gradio.Server: serves the UI + the blind-commit ai_turn API
engine.py       combat core: moves, costs, the resolution matrix (no deps, testable)
opponents.py    the ten personas (temperament + strategy + think budget)
prompts.py      builds the read-prompt; parses <think> + JSON move out of replies
llm.py          llama.cpp reasoning path + persona-aware mock fallback
static/index.html   the custom Himalayan arena (self-contained HTML/CSS/JS)
```

Test the engine in isolation: `python engine.py` prints the full
move-interaction table.
