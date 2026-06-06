# harvest_daily.ps1 -- daily Mind of Tashi self-play harvest.
#
# Pipeline (each step appends to data/selfplay/cron.log and continues on error):
#   1. refresh the OpenRouter free-tier rotation pool
#   2. run a 5x5 persona sweep, 1 match per matchup, max 30 rounds, cap 25
#   3. filter the cumulative selfplay output into a TRL-ready SFT JSONL
#   4. push the data/ directory to the private HF Dataset repo
#   5. push sealed live-gameplay matches to the live HF Dataset
#
# Registered as a Windows Task Scheduler entry that fires daily at 12:00
# local for 10 days. See tools/README.md for install/uninstall commands.
#
# ASCII-only on purpose: PowerShell 5.1 reads .ps1 files with the active
# code page (cp1252 on most Windows installs) unless they have a UTF-8
# BOM. Em-dashes, curly quotes, and other non-ASCII would otherwise be
# misread as multi-byte sequences and cascade parser errors throughout
# the file.

$ErrorActionPreference = "Continue"  # individual step failures shouldn't kill the run

# --- force UTF-8 for the entire pipeline -------------------------------------
# Without this, native command stdout (Python's UTF-8 bytes) gets decoded by
# PowerShell using [Console]::OutputEncoding (which defaults to the system
# code page, cp437/cp1252 on most Windows installs). The mis-decoded string
# then gets re-encoded by Add-Content -Encoding utf8, leaving every Python
# line in the log as letter-spaced mojibake. The fix is to make every layer
# agree on UTF-8:
#   1. $env:PYTHONIOENCODING tells Python to emit UTF-8 even when redirected
#      (Python 3 usually does this for pipes already, but explicit is safer).
#   2. [Console]::OutputEncoding tells PowerShell to read native-command
#      stdout bytes as UTF-8 (this is the line that actually fixes it).
#   3. $OutputEncoding tells PowerShell what to use when writing to native
#      commands' stdin (not strictly needed here, but keeps the pipeline
#      symmetric).
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
    # Use $args (PowerShell's automatic variable for unbound positional
    # args) rather than typed params. Two reasons:
    #   1. PowerShell 5.1's positional-arg parser misreads commas inside
    #      parenthesised string segments as argument separators when a
    #      function is called bare ('Log <str>'). Joining $args sidesteps.
    #   2. The [Parameter(...)] decorator syntax silently fails to register
    #      the function in some 5.1 builds, leaving 'Log' undefined.
    $msg = ($args -join ' ')
    $ts = Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'
    "[$ts] $msg" | Add-Content -Path $logFile -Encoding utf8
}

Log "===== harvest start ====="
Log "python: $python"
Log "projDir: $projDir"

# --- step 1: refresh OpenRouter free pool --------------------------------- #
Log "[step 1] refresh OpenRouter free pool"
& $python -m tools.list_openrouter_free --write pools\all-free.txt 2>&1 | Add-Content -Path $logFile -Encoding utf8
Log "[step 1] exit code $LASTEXITCODE"

# --- step 2: self-play harvest -------------------------------------------- #
# IMPORTANT: cwd is $projDir (mind-of-tashi), so --output-dir and --quota-file
# need to point UP one level to mind-of-tashi-scaffold/data/selfplay/. Without
# these flags the harness writes to mind-of-tashi/data/selfplay/ instead,
# and step 3's prep_sft can't find the new rows.
Log "[step 2] self-play sweep, 1 match per matchup, max 30 rounds, cap 50 matches total"
& $python -m tools.selfplay --sweep --matches 1 `
    --opponent-pool-file pools\all-free.txt `
    --output-dir "..\data\selfplay" `
    --quota-file "..\data\selfplay\.quota.json" `
    --max-rounds 30 --max-matches-per-run 50 2>&1 | Add-Content -Path $logFile -Encoding utf8
Log "[step 2] exit code $LASTEXITCODE"

# --- step 3: prep cumulative SFT corpus, BOTH shapes ---------------------- #
# Each day's SFT files are cumulative snapshots -- one for single-turn
# (one example per AI turn, default TRL shape) and one for multi-turn
# (one example per match, matches the live/ config schema so SFTTrainer
# can mix them). Same selfplay/ input, two shape-aware aggregations.
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
    --commit-message "Daily harvest $dayStamp" 2>&1 | Add-Content -Path $logFile -Encoding utf8
Log "[step 4] exit code $LASTEXITCODE"

# --- step 5: push sealed live-gameplay matches to live HF Dataset --------- #
# Skips when LIVE_TRACES_REPO is unset OR when no sealed matches exist
# the script logs a one-liner and exits 0 in both cases.
Log "[step 5] push live-gameplay matches to live HF Dataset"
& $python -m tools.push_live_to_hub `
    --commit-message "Live matches $dayStamp" 2>&1 | Add-Content -Path $logFile -Encoding utf8
Log "[step 5] exit code $LASTEXITCODE"

Log "===== harvest done ====="
