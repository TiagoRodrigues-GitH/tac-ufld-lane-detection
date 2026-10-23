# Full OpenLane, end to end and unattended: download -> labels -> training container.
# Run by the scheduled task "TAC-UFLD OpenLane full pipeline" (now and at every logon); safe to rerun at any time:
#   1. scripts/download_openlane.py --annotations (resumable; retried up to 5 times, 10 min apart)
#   2. when it succeeds: results/openlane_full_7models/.download_done
#   3. waits for Docker Desktop, then builds the image and starts the service openlane-full
#      (docker-compose.openlane.yml), which resumes by itself after a shutdown and writes DONE.txt at the end.
# Log: results/openlane_full_pipeline.log
#   Register:  powershell -ExecutionPolicy Bypass -File scripts\docker\openlane_full_pipeline.ps1 -Register
#   Remove:    Unregister-ScheduledTask -TaskName "TAC-UFLD OpenLane full pipeline" -Confirm:$false
param([switch]$Register)

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($Register) {
    $script = $MyInvocation.MyCommand.Path
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`"" -WorkingDirectory $repo
    $now = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(30)
    $logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 3) -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName "TAC-UFLD OpenLane full pipeline" -Action $action -Trigger $now, $logon `
        -Settings $settings -Description "Downloads the full OpenLane, then starts the openlane-full training container" `
        -Force | Out-Null
    Write-Output "registered: TAC-UFLD OpenLane full pipeline"
    exit 0
}

$results = Join-Path $repo "results"
$run = Join-Path $results "openlane_full_7models"
$log = Join-Path $results "openlane_full_pipeline.log"
$marker = Join-Path $run ".download_done"
$python = Join-Path $repo "..\.venv\Scripts\python.exe"
$openlane = Join-Path $repo "..\..\datasets\OpenLane"
New-Item -ItemType Directory -Force $run | Out-Null
function Log($m) { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) -Encoding utf8 }

$mutex = New-Object System.Threading.Mutex($false, "Global\TacUfldOpenLaneFullPipeline")
if (-not $mutex.WaitOne(0)) { exit 0 }   # another copy is already running

try {
    if (-not (Test-Path $marker)) {
        for ($i = 1; $i -le 5; $i++) {
            Log "download attempt $i (images + annotations)"
            $env:PYTHONUTF8 = "1"
            & $python (Join-Path $repo "scripts\download_openlane.py") $openlane --annotations --workers 2 *>> $log
            if ($LASTEXITCODE -eq 0) { Set-Content $marker (Get-Date -Format s); Log "download complete"; break }
            Log "download attempt $i failed (exit $LASTEXITCODE); retrying in 10 min"
            Start-Sleep -Seconds 600
        }
        if (-not (Test-Path $marker)) { Log "download did not complete after 5 attempts; rerun the task"; exit 1 }
    }
    if (Test-Path (Join-Path $run "DONE.txt")) { Log "training already finished"; exit 0 }
    for ($i = 0; $i -lt 60; $i++) {   # Docker Desktop needs a minute or two after logon
        docker info *> $null
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep -Seconds 10
    }
    $state = docker inspect -f "{{.State.Running}}" tac-ufld-openlane_full_7models 2>$null
    if ($state -eq "true") { Log "training container already running"; exit 0 }
    Log "building the image and starting the training container (openlane-full)"
    Set-Location $repo
    docker compose -f docker-compose.openlane.yml up -d --build openlane-full *>> $log
    Log "docker compose exit $LASTEXITCODE"
} finally {
    $mutex.ReleaseMutex()
}
