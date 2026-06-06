# mind_of_tashi_env — GRPO env skeleton

A multi-turn gym wrapper around the blind-commit duel engine. Built for
`trl.GRPOTrainer` in `environments` mode (see ROADMAP §4 slot C).

## Quick start

From `mind-of-tashi/`:

```bash
# fast local smoke (no network, uses the mock opponent)
python -m envs.mind_of_tashi_env.smoke_test

# with the real API teacher pool (Gemini / OpenRouter / Mistral)
#   requires GEMINI_API_KEY, OPENROUTER_API_KEY, etc. — same env vars as
#   tools/selfplay.py
python -m envs.mind_of_tashi_env.smoke_test --backend api
```

## API

```python
from envs.mind_of_tashi_env import make_env

env = make_env(opponent_backend="api")   # "local" | "api"
obs = env.reset(seed=0)

# obs:
#   messages         — TRL conversational [system, user] for the student
#   state            — engine state from the student's POV
#   legal_moves      — list[str]
#   student_persona  — id of the persona the student is playing AS
#   opponent_persona — id of the persona the opponent is playing AS

# GRPO produces a completion; we hand the raw text back to the env.
completion = "<think>...</think>\n{\"move\":\"STRIKE\",\"taunt\":\"...\"}"

obs, reward, terminated, info = env.step(completion)

# reward dict:
#   turn     — dense per-step HP-delta signal
#   outcome  — sparse terminal +10 / 0 / -10
#   lexicon  — anti-anglicisation bonus (0..0.5)
#   total    — scalar the trainer optimises (sum of above)
```

## Reward shaping (matches ROADMAP §4 C2)

| Component | Formula | Default coef |
|---|---|---|
| `turn`    | Δ(opponent_hp) − Δ(student_hp) per round | 1.0 |
| `outcome` | +1 win / 0 draw / −1 loss at terminal step | 10.0 |
| `lexicon` | sanskrit_token_count / think_len | 0.5 |

Coefficients are tunable via `MindOfTashiEnv(reward_coefficients=...)`.

## Off-the-Grid badge boundary

`make_env(opponent_backend="api")` raises `OpponentBackendForbidden` when
`SPACE_ID` is set in env. Hugging Face Spaces always set `SPACE_ID`, so
the API backend can never run on a deployed Space. Training rigs are
exempt — `SPACE_ID` is only set inside HF infra.

If this env is ever packaged as its own Docker Space, that Space must
default to (or hard-code) `OPPONENT_BACKEND=local`. The runtime then
either uses the in-process mock or wires up llama.cpp via `Reasoner`
locally — no `requests`, no cloud key.

## What's NOT here yet

This is the **Day 2 skeleton** (per the compressed ROADMAP). To-be-added
in subsequent days:

- **Day 5 (06-01):** OpenEnv 0.2.x adapter layer so the same env class
  satisfies the OpenEnv `step` / `reset` HTTP contract for a Docker
  Space deployment.
- **Day 6 (06-02):** `train/grpo.py` driving this env via TRL with a
  spend-cap on API rollouts.
- **Day 6 (06-02):** opponent persona → teacher tier routing (boss
  persona → Gemini 2.5 Pro; mid → Flash; low → free OpenRouter), to
  keep API spend predictable.
