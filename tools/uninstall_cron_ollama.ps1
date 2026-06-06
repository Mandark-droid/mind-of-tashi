# uninstall_cron_ollama.ps1 -- remove the MindOfTashiHarvestOllama scheduled task.
$ErrorActionPreference = "Stop"

$task = Get-ScheduledTask -TaskName 'MindOfTashiHarvestOllama' -ErrorAction SilentlyContinue
if ($task) {
    Unregister-ScheduledTask -TaskName 'MindOfTashiHarvestOllama' -Confirm:$false
    Write-Host "Removed MindOfTashiHarvestOllama scheduled task."
} else {
    Write-Host "MindOfTashiHarvestOllama is not registered. Nothing to do."
}
