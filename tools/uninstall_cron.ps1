# uninstall_cron.ps1 — remove the MindOfTashiHarvest scheduled task.
$ErrorActionPreference = "Stop"

$task = Get-ScheduledTask -TaskName 'MindOfTashiHarvest' -ErrorAction SilentlyContinue
if ($task) {
    Unregister-ScheduledTask -TaskName 'MindOfTashiHarvest' -Confirm:$false
    Write-Host "Removed MindOfTashiHarvest scheduled task."
} else {
    Write-Host "MindOfTashiHarvest is not registered. Nothing to do."
}
