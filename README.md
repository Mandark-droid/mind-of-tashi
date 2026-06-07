---
title: The Mind of Tashi
emoji: 🌫️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.15.0
app_file: app.py
pinned: false
license: mit
hf_oauth: true
hf_oauth_scopes:
  - openid
  - profile
---

# The Mind of Tashi — a duel of foresight

> Working title: **The Mind of Tashi** (renamed from "The Mind of Tashi" — folder slugs kept for cron stability).
> Built for the **Build Small Hackathon · Track Two: An Adventure in Thousand Token Wood**.

You climb a gauntlet of ninja-monks in the Village Hidden in the Mist, high in the Himalayas. Every round, you
and your opponent **secretly commit one move at the same time** — there is no reacting,
only *reading*. The opponent is a small reasoning model, and the thing on centre stage is
its **mind**: after each commit, the scroll reveals how it just read you ("you've drawn
breath twice unpunished — you're greedy, so I strike"). The whole game is out-thinking
something that narrates how it's out-thinking you.

## Why the AI is load-bearing

Strip the model and this is rock-paper-scissors. The model's *prediction of your next move*
is the entire mechanic — and it commits **blind**, seeing only the history, never your
current move. The signature move, **Mist-Step**, only rewards you if your opponent attacks
this turn, so it is a pure bet on a read. That recursion ("I think you'll strike, so I
Mist-Step / I think you think that, so I draw breath") is what a reasoning model is uniquely
good at, and what a scripted bot is bad at.

## The moves

| Move | Cost | Beats / loses |
|---|---|---|
| **Vajra Strike** | free | beats Grapple · blocked by Guard |
| **Mountain Stance** (guard) | free, +1 prana | blocks Strike, softens Art · broken by Grapple |
| **River Throw** (grapple) | free | breaks Guard · loses to Strike |
| **Draw Breath** (focus) | free, +2 prana | gathers prana but fully exposed |
| **Prana Art** | 3 prana | heavy ranged hit · eaten by Mist-Step |
| **Mist-Step** | 2 prana | dodges + counters an attack · whiffs vs caution |

## The ladder

Tashi (Avalanche Fist) → Norbu (Patient Stone) → Pema (Snow-Fox) → Drogpa (Cliff-Bear) →
The Mountain (Mist-Master). Same engine and prompt for all five — only the **persona** and
the **thinking budget** change, so a harder opponent is literally the same model allowed to
think longer with a different soul.

## Run it

```bash
pip install -r requirements.txt
python app.py            # serves the custom frontend at http://localhost:7860
```

It runs immediately on the **mock opponent** (a persona-aware heuristic) — no model download
needed. To play against the real reasoning model, set the env vars below and the app loads a
GGUF through llama.cpp on next boot.

## Swap in the real (or fine-tuned) model

The web layer is model-agnostic. Configure via Space **Variables** / env:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_REPO` | `Qwen/Qwen3-4B-GGUF` | any GGUF repo on the Hub |
| `MODEL_FILE` | `Qwen3-4B-Q4_K_M.gguf` | the quant file |
| `N_GPU_LAYERS` | `0` | keep 0 (CPU) on Spaces — see note |
| `FORCE_MOCK` | `0` | set `1` to force the heuristic opponent |

**Note on hardware:** llama.cpp + **ZeroGPU** is unreliable (CUDA init vs dynamic
allocation — the hackathon org's own demo space hit this). Run llama.cpp **CPU-only** on a
CPU-upgrade Space, or use a **dedicated GPU** Space (not ZeroGPU). Turn-based play makes a
few seconds of "reading you…" feel like drama, not lag.

## Bonus badges targeted

- **Off the Grid** — model runs in-Space via llama.cpp, no cloud API.
- **Llama Champion** — opponent runs through the llama.cpp runtime.
- **Off-Brand** — fully custom frontend on `gradio.Server` (Gradio 6), not default Gradio.
- **Field Notes** — write-up of the design + what we learned (todo).
- **Sharing is Caring** — publish the opponents' reasoning traces to the Hub (todo).
- **Well-Tuned** *(stretch)* — fine-tune a small model to role-play the five personas and
  reliably emit move-JSON; drop the GGUF in via `MODEL_REPO`. This is the move that pushes
  past a good submission to a winning one.

## Layout

```
app.py          gradio.Server: serves the UI + the blind-commit ai_turn API
engine.py       combat core: moves, costs, the resolution matrix (no deps, testable)
opponents.py    the five personas (temperament + strategy + think budget)
prompts.py      builds the read-prompt; parses <think> + JSON move out of replies
llm.py          llama.cpp reasoning path + persona-aware mock fallback
static/index.html   the custom Himalayan arena (self-contained HTML/CSS/JS)
```

Test the engine in isolation: `python engine.py` prints the full move-interaction table.
