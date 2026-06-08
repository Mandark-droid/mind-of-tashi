---
license: other
base_model: kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1
tags:
  - gguf
  - llama-cpp
  - qwen3moe
  - reasoning
  - game
  - bilingual
  - sft
language:
  - en
  - hi
  - sa
pipeline_tag: text-generation
---

# The Mind of Tashi — micro student (SFT v3, GGUF)

The opponent's mind for **The Mind of Tashi**, a simultaneous-commit ritual
fighting game where the model's `<think>` block *is* the game (surfaced to the
player as the "mind-scroll"). This is the fine-tuned **GGUF** the playable
Hugging Face Space loads through **llama.cpp** — the *Off the Grid* + *Llama
Champion* contract (no cloud API at runtime).

**David vs Goliath.** The student is the author's own custom MoE —
`loggenix-moe-0.4B-0.2A` (~**0.4B total / ~200M active per token**) — SFT'd to
read an opponent and commit blind, in an English + Hindi/Sanskrit (IAST)
code-switched register. It is 10–100× smaller (active) than the API teachers it
learned from.

## Files

| File | Size | Use |
|---|---|---|
| `mind-of-tashi-micro-sft-v3-Q4_K_M.gguf` | ~256 MB | **deployed** — small + fast |
| `mind-of-tashi-micro-sft-v3-f16.gguf` | ~786 MB | zero-loss reference |

## Training

- **SFT** (TRL `SFTTrainer`, Modal L4, bf16, completion-only loss, seq 4096,
  3 epochs) on `kshitijthakkar/mind-of-tashi-traces` (configs `sft` +
  `sft_multiturn` + `live`) — self-play traces vs a frontier-API teacher pool.
- Recipe: `mind-of-tashi/train/sft.py` (self-contained, reproducible).

### ⚠️ `norm_topk_prob` — required for llama.cpp

The base ships `norm_topk_prob=false` (raw top-k expert routing), but
llama.cpp's `qwen3moe` graph **hardcodes `norm_w=true`** and ignores the GGUF
`expert_weights_norm` key. A checkpoint trained with `false` therefore produces
**garbage on every llama.cpp runtime**. v3 is trained with **`norm_topk_prob=true`**
so the weights match llama.cpp's renormalised routing — *this* is what makes the
GGUF coherent. (v1/v2 predate the fix.)

## Eval

- **Format gate via llama.cpp** (the real deploy path): f16 **18/20**, Q4_K_M
  **20/20** valid `<think> + {move,taunt}`; ~19–20/20 bilingual across 5 personas.
- **Throughput** (Q4_K_M, RTX 3060 Laptop, `n_gpu_layers=99`, n_ctx 4096):
  cold load **3.2 s**, **TTFT ~97 ms** (p50), **decode ~450 t/s** (p50, 528 p90),
  **~0.30 s per full move**, peak VRAM < ~0.8 GB single-instance, format pass 100%.
  (CPU-only deploy is slower but fine for a 0.4B.)

## Usage

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="kshitijthakkar/mind-of-tashi-micro-sft-v3-gguf",
    filename="mind-of-tashi-micro-sft-v3-Q4_K_M.gguf",
    n_ctx=4096, n_gpu_layers=0,          # 0 = CPU (Space); 99 = full GPU offload
    logits_all=True,                      # needed for the in-game Conviction Meter (token logprobs)
)
out = llm.create_chat_completion(messages=[
    {"role": "system", "content": "<persona system prompt>"},
    {"role": "user",   "content": "<arena state + match history>"},
])
# -> "<think>...IAST-coded reasoning...</think>\n{\"move\": \"...\", \"taunt\": \"...\"}"
```

The model emits a `<think>…</think>` block then one JSON line
`{"move": ..., "taunt": ...}`; the host parses defensively and falls back to a
legal move if generation is malformed.

*Private pre-window artifact (Build Small Hackathon). Slug `mind-of-tashi`/`micro`
is internal; public-facing artifacts ship as `mind-of-tashi-*`.*
