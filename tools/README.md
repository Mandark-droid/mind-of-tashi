# `tools/` — self-play data harvest

Pre-window playbook for the Mind of Tashi SFT dataset. The harness drives
two `Teacher` instances against the engine, asymmetrically: the **player**
side is a stand-in for the human user (move only, no `<think>` saved),
the **opponent** side is the AI we'll fine-tune (full `<think>` + taunt
preserved as the SFT target). Output is JSONL in TRL conversational
format — directly trainable by `SFTTrainer` with no prep step.

This README is the harvest operator's manual. The why is in
`../../ROADMAP.md`; the per-row schema is in the `tools/selfplay.py`
module docstring.

---

## One-time setup

From the repo root:

```bash
# 1. Install deps (adds aiohttp, python-dotenv, google-genai to the live-game deps).
pip install -r mind-of-tashi/requirements.txt

# 2. Copy the env template and fill in the API keys you have.
cp mind-of-tashi/.env.example mind-of-tashi/.env
$EDITOR mind-of-tashi/.env   # populate GEMINI_API_KEY, OPENROUTER_API_KEY, MISTRAL_API_KEY, SARVAM_API_KEY

# 3. From inside mind-of-tashi/, auto-fetch the OpenRouter free-tier list
#    and write the full rotation pool (Gemini + Mistral + Sarvam + OpenRouter).
cd mind-of-tashi
python -m tools.list_openrouter_free --write pools/all-free.txt
```

`pools/all-free.txt` is **gitignored** (it changes per fetch). Re-run
`list_openrouter_free` whenever you start a fresh harvest day so the pool
reflects what's currently free on your account.

---

## A single harvest run

All commands assume cwd = `mind-of-tashi/`.

```bash
# Mock-only smoke: zero API quota, validates wiring after any code change.
python -m tools.selfplay --matches 2 --force-mock --seed 42

# Real harvest, single matchup, pool-rotated opponent.
python -m tools.selfplay \
    --matches 5 \
    --player-persona tashi --opponent-persona norbu \
    --opponent-pool-file pools/all-free.txt \
    --max-rounds 30

# Real harvest, full 5×5 persona sweep, bounded by max-matches-per-run.
python -m tools.selfplay \
    --sweep --matches 4 \
    --opponent-pool-file pools/all-free.txt \
    --max-rounds 30 \
    --max-matches-per-run 30
```

Defaults to remember:
- `--player-teacher mock` — the player is the simulated human. Mock keeps
  it aggressive (human-like) and costs no API quota. Override only if you
  want richer "user" behaviour for ablations.
- `--opponent-teacher` is ignored when `--opponent-pool*` is set.
- `--max-rounds 30` is a safety net; most matches end in 6–12 rounds.
- `--max-matches-per-run` caps total matches per invocation — cron-critical.

Each invocation writes a new file: `data/selfplay/selfplay_<UTC_ISO>.jsonl`.
Resumable: never overwrites. The quota counter (`data/selfplay/.quota.json`)
persists across runs and resets at UTC midnight.

---

## Push the harvest to a private HF Dataset

```bash
# from mind-of-tashi/, after a harvest
python -m tools.push_to_hub                                    # default kshitijthakkar/mind-of-tashi-traces (private)
python -m tools.push_to_hub --dry-run                          # list files, push nothing
python -m tools.push_to_hub --repo my-user/my-dataset
python -m tools.push_to_hub --commit-message "Daily harvest 20260527"
```

The script mirrors the local `data/` tree to a **PRIVATE** Hugging Face
Dataset (creates it on first push, exists_ok afterwards). Pattern allowed:
`**/*.jsonl` + the dataset card README. Patterns ignored: `.quota.json`,
`.tmp*`, `cron.log`. `HF_TOKEN` must be in `.env`.

The daily cron does this automatically as step 4 of the wrapper.

---

## Daily cron / Task Scheduler

The intent: ~30 matches/day for 10 days → ~1500–3000 SFT rows after
quality filtering. Each run is one to three hours wall time depending
on which pool members get exercised and how rate-limited the providers
are.

### Linux / macOS cron

```cron
# UTC midnight + 5 minutes: refresh the OpenRouter list, then harvest.
5 0 * * * cd /path/to/mind-of-tashi-scaffold/mind-of-tashi \
  && /path/to/venv/bin/python -m tools.list_openrouter_free --write pools/all-free.txt \
  && /path/to/venv/bin/python -m tools.selfplay --sweep --matches 8 \
       --opponent-pool-file pools/all-free.txt \
       --max-rounds 30 --max-matches-per-run 30 \
       >> data/selfplay/cron.log 2>&1
```

### Windows Task Scheduler — two crons

Two daily harvests run on separate schedules so the API quota cron and
the local-Ollama cron have independent failure modes and don't compete
for system resources:

| Task | Time (IST) | Source | Wrapper |
|---|---|---|---|
| `MindOfTashiHarvest` | 12:00 | API pool (Gemini / Mistral / Sarvam / OpenRouter free) | `harvest_daily.ps1` |
| `MindOfTashiHarvestOllama` | 18:00 | Local Ollama (`qwen3:1.7b` + `qwen3.5:4b`) | `harvest_daily_ollama.ps1` |

Both wrappers share the same five-step pipeline (refresh pool / selfplay
sweep / prep_sft *single* / prep_sft *multi* / push to Hub) and both write
to `data/selfplay/`, so prep_sft sees the cumulative output from both.

Install both for the full 10-day pre-hackathon harvest:

```powershell
# API-pool cron (12:00 IST daily, 10 days)
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File mind-of-tashi\tools\install_cron.ps1

# Ollama cron (18:00 IST daily, 10 days)
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File mind-of-tashi\tools\install_cron_ollama.ps1
```

Inspect / tail / uninstall:

```powershell
# inspect
schtasks /query /tn MindOfTashiHarvest        /v /fo list
schtasks /query /tn MindOfTashiHarvestOllama  /v /fo list

# tail the shared log (both crons append here, lines prefixed `[ollama]`
# for the Ollama wrapper so you can tell them apart at a glance)
Get-Content -Wait data\selfplay\cron.log

# remove one or both
.\mind-of-tashi\tools\uninstall_cron.ps1
.\mind-of-tashi\tools\uninstall_cron_ollama.ps1
```

Tune the harvest size by editing the relevant `tools.selfplay` line in
the wrapper directly. Re-run the install script to push the change to
the scheduler (it's idempotent).

---

## Inspecting the harvest

Quota state (no API calls):
```bash
python -m tools.selfplay --show-quota
```

Sample output:
```json
{
  "date_utc": "2026-05-26",
  "used":     {"gemini:gemini-2.5-flash": 17, "openrouter:meta-llama/llama-3.3-70b-instruct:free": 23, ...},
  "exhausted_until_eod": ["gemini:gemini-2.5-pro"],
  "remaining": {"gemini:gemini-2.5-flash": 33, ...}
}
```

Quick row count across all harvested files:
```bash
# row totals (Linux/macOS)
wc -l data/selfplay/*.jsonl

# SFT-target count
grep -h '"is_sft_target": true' data/selfplay/*.jsonl | wc -l

# distribution of teacher providers in the harvest
python -c "import json,glob; from collections import Counter; \
  c=Counter(); \
  [c.update([json.loads(l).get('teacher_meta',{}).get('pool_spec') \
    or json.loads(l).get('teacher_meta',{}).get('backend') \
    for l in open(f,'r',encoding='utf-8') if '\"is_sft_target\": true' in l]) \
    for f in glob.glob('data/selfplay/*.jsonl')]; \
  print(c.most_common())"
```

Spot-check a row (Python is safer than `cat` because of UTF-8):
```bash
python -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1],'r',encoding='utf-8')]; \
  opp=[x for x in r[1:] if x.get('is_sft_target')]; \
  import pprint; pprint.pp({k:opp[5][k] for k in ('move','think','taunt','turn_reward')})" \
  data/selfplay/selfplay_<timestamp>.jsonl
```

---

## What "good" output looks like

A healthy opponent row has:

- `move` that is one of the six engine ids and **not** a `GUARD` fallback
  from a malformed completion (rare cases excepted).
- `think` of ~2–5 sentences, in-character for the persona, with 3–6
  IAST Hindi/Sanskrit terms woven in (*prahār*, *rakṣā*, *prāṇa*, *dṛṣṭi*,
  *abhyāsa*, …). English-only rows survive harvest but should be filtered
  out at SFT-prep time.
- `taunt` of one short in-character line (not "...").
- `messages` of `[system, user, assistant]` with the system + user
  identical to what the live game sends and the assistant being the full
  raw `<think>...</think>\n{json}` blob.
- `outcome_reward` ∈ {+1, −1, 0} populated at match end.

A bad row (drop at SFT-prep):

- `move == "GUARD"` and `taunt == "..."` — typically a parser fallback
  on a malformed model completion.
- `think` shorter than ~10 words — usually truncation or empty thinking.
- `think` is English-only with zero lexicon hits — bilingual drift.
- `teacher_meta.fallback == true` — the pool fell through entirely.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `module 'aiohttp' not found` | Deps not installed | `pip install -r requirements.txt` |
| All matches end in GUARD/GUARD draws | Token cap too low for the chosen API teacher | Already fixed for Gemini (thinking_budget=0) and OpenAI-compat (max_tokens × 3). If reappears on a new provider, raise the cap in that teacher's file. |
| `PoolExhausted` very early in the run | One of the pool's models is down or your account doesn't have access | The harness exits clean. Re-run `tools.list_openrouter_free` to refresh, or comment dead specs out of the pool file. |
| Latency 10+ s/call on Gemini | Hidden thinking still enabled | The Gemini teacher sets `thinking_budget=0`. If your installed `google-genai` is old, upgrade: `pip install -U google-genai`. |
| Empty `pools/all-free.txt` | `OPENROUTER_API_KEY` missing or empty | Populate it in `.env` and re-run the fetcher. Fall back to `pools/all-free.example.txt` for now. |
| Race / data loss between cron and a manual run | Two harness instances writing the same quota file | Don't run two harvesters concurrently. Quota state is asyncio-safe but not multi-process. |
| Output file has only the header line | Process killed before any match completed | Check the cron log; usually a transient provider 5xx hit early. The pool will fail over on the next call once it knows the spec is dead. |

---

## What this does NOT do

- **No SFT prep.** A separate `tools/prep_sft.py` (next workstream) will
  consume these JSONL files, filter by `is_sft_target`, drop GUARD fallbacks
  and English-only drift, deduplicate near-identical thinks, and emit the
  final TRL-ready dataset.
- **No GRPO / OpenEnv gym.** That comes after SFT lands. See `../../ROADMAP.md`
  workstreams B → C.
- **No model evaluation.** A `tools/eval_ladder.py` will probably grow once
  we have a trained checkpoint.

---

## Files

```
tools/
  selfplay.py             the harness — async, asyncio.gather blind commits
  list_openrouter_free.py auto-fetch the OpenRouter free-tier rotation pool
  prep_sft.py             filter raw harvest -> TRL-ready SFT JSONL
  push_to_hub.py          sync data/ to a PRIVATE HF Dataset repo
  harvest_daily.ps1       Windows wrapper: refresh -> harvest -> prep -> push
  install_cron.ps1        register MindOfTashiHarvest in Task Scheduler (10 days)
  uninstall_cron.ps1      remove MindOfTashiHarvest
  README.md               this file

teachers/
  base.py                 Teacher ABC, retry/backoff, prompt building
  mock.py                 heuristic from llm._mock_choose
  llamacpp.py             wraps llm.Reasoner for self-play use
  ollama.py               async HTTP, 127.0.0.1:11434
  gemini.py               google-genai async; thinking_budget=0
  openai_compat.py        OpenRouter / Mistral / Sarvam (shared shape)
  pool.py                 TeacherPool — rotates across specs, quota-aware
  quota.py                file-backed daily counter (UTC reset, atomic write)

pools/
  all-free.example.txt    static template, committed
  all-free.txt            user-maintained (gitignored), auto-generated

data/selfplay/            JSONL harvest output (jsonl tracked, .quota* + log not)
data/sft/                 cumulative filtered SFT corpora (tracked)
data/README.md            dataset card auto-written by push_to_hub.py
```
