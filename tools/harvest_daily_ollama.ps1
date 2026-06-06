# harvest_daily_ollama.ps1 -- daily Mind of Tashi self-play harvest, OLLAMA edition.
#
# Sibling to harvest_daily.ps1 (the API-pool variant). Both crons share the
# same data/selfplay/ output directory, the same prep_sft pipeline, and the
# same Hub repos -- only step 2's teacher specs differ.
#
# Why two separate crons:
#   - Ollama runs entirely on the local 6 GB GPU; the API cron burns the
#     daily free-tier quotas across ~32 Gemini/OpenRouter/Mistral/Sarvam
#     specs. Independent failure modes, so we want them on independent
#     schedules.
#   - Splitting them by 6 hours (API at 12:00 IST, Ollama at 18:00 IST)
#     means today's data has landed on Hub by the time the user looks at
#     the dataset viewer in the evening, AND it spreads the wall-time
#     load across the day rather than running an 80-min job back-to-back.
#
# Combined target across 10 days: ~10k SFT rows
#   (each cron contributes ~400 single-turn + ~30 multi-turn per day).
#
# Pipeline (each step logs to data/selfplay/cron.log; failures don't abort):
#   1. (skipped) refresh OpenRouter pool -- not used here; Ollama is local.
#   2. run a 5x5 persona sweep, qwen3:1.7b vs qwen3.5:4b, max 30 rounds.
#   3. filter cumulative selfplay output into BOTH single-turn and
#      multi-turn SFT JSONLs.
#   4. push the data/ directory to the private HF Dataset.
#   5. push sealed live-gameplay matches to the live HF Dataset.
#
# ASCII-only, same as harvest_daily.ps1 -- PowerShell 5.1 reads .ps1 using
# the active code page (cp1252 here) unless the file has a UTF-8 BOM.

$ErrorActionPreference = "Continue"

# --- force UTF-8 for the entire pipeline -------------------------------------
# See harvest_daily.ps1 for the rationale. Without these three lines, every
# Python stdout line in cron.log appears letter-spaced (UTF-8 bytes decoded
# under cp1252 then re-encoded as UTF-8).
$env:PYTHONIOENCODING       = 'utf-8'
[Console]::OutputEncoding   = [System.Text.UTF8Encoding]::new()
$OutputEncoding             = [System.Text.UTF8Encoding]::new()

# --- locate the project --------------------------------------------------- #
$scriptDir = $PSScriptRoot                           # mind-of-tashi\tools
$projDir   = Split-Path -Parent $scriptDir           # mind-of-tashi
$repoRoot  = Split-Path -Parent $projDir             # mind-of-tashi-scaffold
$python    = Join-Path $repoRoot 'venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Error "venv python not found at $python -- aborting"
    exit 2
}

Set-Location $projDir

# --- prepare logging ------------------------------------------------------ #
$logDir = Join-Path $repoRoot 'data\selfplay'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir 'cron.log'

function Log {
    $msg = ($args -join ' ')
    $ts = Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'
    "[$ts] [ollama] $msg" | Add-Content -Path $logFile -Encoding utf8
}

Log "===== harvest start ====="
Log "python: $python"
Log "projDir: $projDir"

# --- step 2: self-play harvest, Ollama teachers --------------------------- #
# Player: qwen3:1.7b (~1.4 GB) -- fast moves, no API quota cost. Its
# reasoning is discarded at row-write time per role asymmetry.
# Opponent: qwen3.5:4b (~3.4 GB) -- richer reads, ~3s/turn on the user's
# RTX 3060. Its full <think>+JSON is the SFT target.
Log "[step 2] self-play sweep, ollama qwen3:1.7b vs qwen3.5:4b, max 30 rounds, cap 50 matches"
& $python -m tools.selfplay --sweep --matches 1 `
    --player-teacher ollama:qwen3:1.7b `
    --opponent-teacher ollama:qwen3.5:4b `
    --output-dir "..\data\selfplay" `
    --quota-file "..\data\selfplay\.quota.json" `
    --max-rounds 30 --max-matches-per-run 50 2>&1 | Add-Content -Path $logFile -Encoding utf8
Log "[step 2] exit code $LASTEXITCODE"

# --- step 3: prep cumulative SFT corpus, BOTH shapes ---------------------- #
$dayStamp = Get-Date -Format 'yyyyMMdd'
$sftSingleOut = "..\data\sft\sft_$dayStamp.jsonl"
$sftMultiOut  = "..\data\sft_multi\sft_$dayStamp.jsonl"

Log "[step 3a] prep_sft single-turn -> $sftSingleOut"
& $python -m tools.prep_sft `
    --input "..\data\selfplay\*.jsonl" `
    --output $sftSingleOut `
    --shape single 2>&1 | Add-Content -Path $logFile -Encoding utf8
Log "[step 3a] exit code $LASTEXITCODE"

Log "[step 3b] prep_sft multi-turn -> $sftMultiOut"
& $python -m tools.prep_sft `
    --input "..\data\selfplay\*.jsonl" `
    --output $sftMultiOut `
    --shape multi 2>&1 | Add-Content -Path $logFile -Encoding utf8
Log "[step 3b] exit code $LASTEXITCODE"

# --- step 4: push self-play + SFT to private HF Dataset ------------------- #
Log "[step 4] push synthetic traces to private HF Dataset"
& $python -m tools.push_to_hub `
    --commit-message "Daily Ollama harvest $dayStamp" 2>&1 | Add-Content -Path $logFile -Encoding utf8
Log "[step 4] exit code $LASTEXITCODE"

# --- step 5: push sealed live-gameplay matches to live HF Dataset --------- #
Log "[step 5] push live-gameplay matches to live HF Dataset"
& $python -m tools.push_live_to_hub `
    --commit-message "Live matches $dayStamp (Ollama cron tick)" 2>&1 | Add-Content -Path $logFile -Encoding utf8
Log "[step 5] exit code $LASTEXITCODE"

Log "===== harvest done ====="
