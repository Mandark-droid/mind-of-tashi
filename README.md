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
tags:
  - build-small-hackathon
  - thousand-token-wood
  - off-the-grid
  - llama-champion
  - off-brand
  - well-tuned
  - tiny-titan
  - sharing-is-caring
  - field-notes
  - zerogpu
  - modal
  - openbmb
  - minicpm
  - nemotron
  - reasoning-game
  - bilingual
models:
  - build-small-hackathon/mind-of-tashi-micro-sft
  - build-small-hackathon/mind-of-tashi-micro-sft-gguf
  - build-small-hackathon/mind-of-tashi-micro-grpo
  - build-small-hackathon/mind-of-tashi-micro-grpo-gguf
  - openbmb/MiniCPM5-1B-GGUF
  - unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF
datasets:
  - build-small-hackathon/mind-of-tashi-selfplay
---

# The Mind of Tashi — a duel of foresight

> Built for the **Build Small Hackathon · Track Two: An Adventure in Thousand
> Token Wood**.

**▶️ [Play the Space](https://huggingface.co/spaces/build-small-hackathon/mind-of-tashi) ·
🎬 [Watch the demo](https://www.linkedin.com/posts/kshitij-thakkar-2061b924_buildsmallhackathon-thousandtokenwood-smallmodels-activity-7470170147283017729-UfCR) (on LinkedIn) ·
📦 [The bundle](https://huggingface.co/collections/build-small-hackathon/the-mind-of-tashi-6a27107214f1265b159ade35) ·
💻 [Code](https://github.com/Mandark-droid/mind-of-tashi)**

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
| `SELFPLAY_MODE` | `0` | `1` shows **Watch self-play** (Tashi vs Tashi) on the intro card |
| `SELFPLAY_PLAYER_TEACHER` | roster default | override the player side; on a Space only local specs (`llamacpp:…`, `mock`) are accepted |
| `SELFPLAY_OPPONENT_TEACHER` | `house` (Space) | `house` keeps the deployed mind as the defender |

**Two runtimes, one model.** The deployed Space runs the student through
**transformers on Hugging Face ZeroGPU** — a GPU is allocated on demand per move
(`@spaces.GPU`), so it's fast and scales to many concurrent players, and it's
still *off the grid* (local GPU, no cloud API). For a self-hosted **llama.cpp**
runtime, clone the repo and `docker build` (or `pip install llama-cpp-python`
and set `BACKEND=llamacpp`) — the same GGUF student then runs on CPU, no GPU
needed. `BACKEND=transformers` selects the ZeroGPU path.

## Tashi vs Tashi — watch the minds duel

With `SELFPLAY_MODE=1` the intro card grows a **Watch self-play** button and a
challenger picker: choose who climbs the mountain, take your hands off, and
watch both mind-scrolls reason against each other through the same UI. The
**challenger roster** is all local GGUFs run in-process via llama.cpp (CPU —
no ZeroGPU minutes, no cloud API):

| Challenger | Why it's here |
|---|---|
| **Tashi micro GRPO** (0.4B MoE, ours) | the RL-trained student — our best read |
| **Tashi micro SFT** (0.4B MoE, ours) | the pre-GRPO checkpoint, for A/B-ing the training story |
| **MiniCPM5 1B** (OpenBMB) | sponsor-class small — can it out-read a model half its size? |
| **Nemotron 3 Nano 4B** (NVIDIA) | the big sibling — 10× the active params, same blind commit |

The defender is always the **house mind** — the Space's own deployed opponent
(ZeroGPU transformers), with the Conviction Meter, composure cracking, and
grammar-locked Oath all live. Watch-mode matches never touch the leaderboard
or the live-traces dataset, and human games are completely unaffected by the
flag. On a Space, cloud/API teacher specs are refused outright — self-play is
Off-the-Grid by construction.

## Bonus badges targeted

- **Off the Grid** — the opponent runs locally (ZeroGPU transformers in-Space, or
  llama.cpp on a clone); no cloud API at request time.
- **Llama Champion** — the GGUF student runs through the llama.cpp runtime: the
  Docker / local path, **and in the deployed Space itself** (the self-play
  challenger roster is llama.cpp end-to-end).
- **Off-Brand** — fully custom frontend on `gradio.Server` (Gradio 6).
- **Well-Tuned** — a fine-tuned custom-MoE student (SFT + GRPO), shipped as both
  safetensors and GGUF.
- **Tiny Titan** — ~0.4B total / **~200M active** per token: by far the smallest
  fine-tuned mind in the wood, and you can watch it duel 1B–4B challengers in
  self-play.
- **Sharing is Caring** — everything is published openly on the Hub: the
  [self-play dataset](https://huggingface.co/datasets/build-small-hackathon/mind-of-tashi-selfplay),
  the [live gameplay traces](https://huggingface.co/datasets/build-small-hackathon/mind-of-tashi-live-traces)
  (real matches, sealed + pushed straight from this Space),
  the [leaderboard runs](https://huggingface.co/datasets/build-small-hackathon/mind-of-tashi-runs),
  the SFT + GRPO models + GGUFs, and the OpenEnv gym — all cross-linked in one
  [collection](https://huggingface.co/collections/build-small-hackathon/the-mind-of-tashi-6a27107214f1265b159ade35).
- **Field Notes** — a written build log: the bilingual `<think>` distillation,
  the qwen3moe `norm_topk_prob` GGUF bug, the David-vs-Goliath GRPO framing, and
  the ZeroGPU port.

## Built with — thank you 🙏

A **Build Small Hackathon (Track Two)** submission, made possible by:

- **[Hugging Face](https://huggingface.co/)** — the backbone. The Hub hosts the
  [dataset](https://huggingface.co/datasets/build-small-hackathon/mind-of-tashi-selfplay),
  the [SFT](https://huggingface.co/build-small-hackathon/mind-of-tashi-micro-sft) +
  [GRPO](https://huggingface.co/build-small-hackathon/mind-of-tashi-micro-grpo)
  models, the GGUFs, the [OpenEnv gym](https://huggingface.co/spaces/build-small-hackathon/mind-of-tashi-env),
  and this Space. **ZeroGPU** gives the deployed opponent a free,
  dynamically-allocated GPU; **Gradio** powers the custom frontend; and HF
  **OAuth** drives the verified leaderboard. 🤗
- **[Modal](https://modal.com/)** — serverless GPU compute for training: the SFT
  and GRPO runs of the ~200M-active student executed on **Modal L4**.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the local inference
  runtime (the *Llama Champion* path): clone + `docker build` and the same GGUF
  student runs entirely on CPU, no cloud.
- **[NVIDIA](https://www.nvidia.com/)** — the silicon under all of it (ZeroGPU's
  RTX Pro 6000 Blackwell at runtime, L4 for training on Modal, a local RTX 3060
  for the dev loop) — and **Nemotron 3 Nano 4B** fights in the self-play
  challenger roster.
- **[OpenBMB](https://www.openbmb.cn/)** — **MiniCPM5-1B** is a self-play
  challenger: the sponsor-class 1B versus our 200M-active student, live in the
  arena.

## Watch & follow

- 🎬 **Demo video** (posted on LinkedIn): [watch the duel](https://www.linkedin.com/posts/kshitij-thakkar-2061b924_buildsmallhackathon-thousandtokenwood-smallmodels-activity-7470170147283017729-UfCR)
- 💼 **LinkedIn** announcement: [the post](https://www.linkedin.com/posts/kshitij-thakkar-2061b924_buildsmallhackathon-thousandtokenwood-smallmodels-activity-7470170147283017729-UfCR)
- 🐦 **X / Twitter** announcement: [the thread](https://x.com/Mandark12921244/status/2064405594044547331)

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
