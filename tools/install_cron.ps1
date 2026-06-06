# install_cron.ps1 — register the daily Mind of Tashi harvest as a Windows
# Scheduled Task. Idempotent: re-running replaces the existing entry.
#
# Run once:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File install_cron.ps1
#
# Inspect:
#   schtasks /query /tn MindOfTashiHarvest /v /fo list
# Remove:
#   .\uninstall_cron.ps1

$ErrorActionPreference = "Stop"

$scriptDir     = $PSScriptRoot                                   # mind-of-tashi\tools
$harvestScript = Join-Path $scriptDir 'harvest_daily.ps1'

if (-not (Test-Path $harvestScript)) {
    throw "harvest_daily.ps1 not found at $harvestScript"
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$harvestScript`""

# Fire at 12:00 local. If 12:00 has already passed today, first run is
# tomorrow — never miss a slot by triggering immediately on register.
$startDt = (Get-Date).Date.AddHours(12)
if ($startDt -lt (Get-Date)) {
    $startDt = $startDt.AddDays(1)
}

# 10 daily runs total. End-boundary is 30 minutes after the 10th 12:00
# trigger so the last run has time to spin up before the trigger expires.
$endDt = $startDt.AddDays(9).AddMinutes(30)

$trigger = New-ScheduledTaskTrigger -Daily -At $startDt
$trigger.EndBoundary = $endDt.ToString('s')

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName 'MindOfTashiHarvest' `
    -Description 'Mind of Tashi daily self-play harvest (10-day pre-hackathon)' `
    -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null

Write-Host "Registered MindOfTashiHarvest:"
Write-Host ("  first run: {0}" -f $startDt.ToString('yyyy-MM-dd HH:mm zzz'))
Write-Host ("  last run:  {0}" -f $startDt.AddDays(9).ToString('yyyy-MM-dd HH:mm zzz'))
Write-Host ("  ends at:   {0}" -f $endDt.ToString('yyyy-MM-dd HH:mm zzz'))
Write-Host ""
Write-Host "schtasks /query summary:"
schtasks /query /tn MindOfTashiHarvest /fo list
