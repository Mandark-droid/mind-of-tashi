# install_cron_ollama.ps1 -- register the daily Ollama-side harvest as a
# Windows Scheduled Task, sibling to the API-pool MindOfTashiHarvest task.
# Idempotent: re-running replaces the existing entry.
#
# Run once:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File install_cron_ollama.ps1
#
# Inspect:
#   schtasks /query /tn MindOfTashiHarvestOllama /v /fo list
# Remove:
#   .\uninstall_cron_ollama.ps1

$ErrorActionPreference = "Stop"

$scriptDir     = $PSScriptRoot                                          # mind-of-tashi\tools
$harvestScript = Join-Path $scriptDir 'harvest_daily_ollama.ps1'

if (-not (Test-Path $harvestScript)) {
    throw "harvest_daily_ollama.ps1 not found at $harvestScript"
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$harvestScript`""

# Fire at 18:00 local -- 6 hours after the API-pool cron at 12:00. If 18:00
# has already passed today, first run is tomorrow.
$startDt = (Get-Date).Date.AddHours(18)
if ($startDt -lt (Get-Date)) {
    $startDt = $startDt.AddDays(1)
}

# 10 daily runs total. End-boundary 30 min after the 10th 18:00 trigger.
$endDt = $startDt.AddDays(9).AddMinutes(30)

$trigger = New-ScheduledTaskTrigger -Daily -At $startDt
$trigger.EndBoundary = $endDt.ToString('s')

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName 'MindOfTashiHarvestOllama' `
    -Description 'Mind of Tashi daily Ollama self-play harvest (10-day pre-hackathon, sibling to MindOfTashiHarvest)' `
    -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null

Write-Host "Registered MindOfTashiHarvestOllama:"
Write-Host ("  first run: {0}" -f $startDt.ToString('yyyy-MM-dd HH:mm zzz'))
Write-Host ("  last run:  {0}" -f $startDt.AddDays(9).ToString('yyyy-MM-dd HH:mm zzz'))
Write-Host ("  ends at:   {0}" -f $endDt.ToString('yyyy-MM-dd HH:mm zzz'))
Write-Host ""
Write-Host "schtasks /query summary:"
schtasks /query /tn MindOfTashiHarvestOllama /fo list
