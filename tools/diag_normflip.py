"""diag_normflip.py — test whether the SFT model is robust to flipping
norm_topk_prob from false (its trained value) to true (what llama.cpp
hardcodes for qwen3moe).

If the norm=true greedy output is still clean, we can ship by simply
flipping the config flag before GGUF conversion — no retrain. If it
degrades, we must retrain SFT with norm_topk_prob=true.

Writes both norm=false (baseline) and norm=true outputs to
data/diag_tokenizer/ for side-by-side comparison.

Run with a venv that has torch+transformers (e.g. the nemotron venv):
    <python> -m tools.diag_normflip
"""
from __future__ import annotations
import sys
from pathlib import Path

OUT = Path("../data/diag_tokenizer")
OUT.mkdir(parents=True, exist_ok=True)
# Load from the already-downloaded local snapshot to avoid a slow/flaky
# re-download from the Hub (the safetensors was pulled during GGUF conversion).
_LOCAL = Path("../../traceverse-rl/artifacts/mind-of-tashi-micro-sft-v2")
MID = str(_LOCAL) if _LOCAL.exists() else "kshitijthakkar/mind-of-tashi-micro-sft-v2"


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from opponents import LADDER
    from prompts import build_system, build_user
    from teachers.base import legal_moves

    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    opp = LADDER[0]
    state = {"round": 1, "ai_hp": 100, "ai_prana": 1,
             "player_hp": 100, "player_prana": 1, "history": []}
    legal = legal_moves(state["ai_prana"])
    msgs = [{"role": "system", "content": build_system(opp)},
            {"role": "user", "content": build_user(opp, state, legal)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt").to("cuda")

    def gen():
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=90, do_sample=False,
                               repetition_penalty=1.0, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=False)

    def set_norm(val: bool):
        model.config.norm_topk_prob = val
        n = 0
        for m in model.modules():
            if hasattr(m, "norm_topk_prob"):
                m.norm_topk_prob = val
                n += 1
        return n

    orig = model.config.norm_topk_prob
    print(f"[normflip] config norm_topk_prob (as trained) = {orig}", flush=True)

    # Baseline: as trained (false)
    nset = set_norm(False)
    txt_false = gen()
    (OUT / "greedy_hf_normFALSE.txt").write_text(txt_false, encoding="utf-8")
    print(f"[normflip] norm=FALSE done (set on {nset} modules)", flush=True)

    # Flipped: true (matches llama.cpp)
    nset = set_norm(True)
    txt_true = gen()
    (OUT / "greedy_hf_normTRUE.txt").write_text(txt_true, encoding="utf-8")
    print(f"[normflip] norm=TRUE done (set on {nset} modules)", flush=True)

    print("\n================ norm=FALSE (as trained) ================", flush=True)
    print(txt_false[:400], flush=True)
    print("\n================ norm=TRUE  (llama.cpp behaviour) =======", flush=True)
    print(txt_true[:400], flush=True)


if __name__ == "__main__":
    run()
