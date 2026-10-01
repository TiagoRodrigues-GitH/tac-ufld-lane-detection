# Overnight OpenLane pilot: training + evaluation + report, then the robustness evaluation.
# Runs detached; progress in results/openlane_pilot/run.log, console output in
# results/openlane_pilot_{run,robustness}.{out,err}.log. Start it with:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\run_openlane_pilot.ps1' -WindowStyle Minimized
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path (Split-Path -Parent $repo) ".venv\Scripts\python.exe"
if (-not $env:OPENLANE_ROOT) {
  $env:OPENLANE_ROOT = Join-Path (Split-Path -Parent (Split-Path -Parent $repo)) "datasets\OpenLane"
}
$env:PYTHONUTF8 = "1"
$results = Join-Path $repo "results"
New-Item -ItemType Directory -Force $results | Out-Null
$status = Join-Path $results "openlane_pilot_status.log"

function Step([string]$name, [string[]]$arguments) {
  "[{0}] {1} started (OPENLANE_ROOT={2})" -f (Get-Date -Format s), $name, $env:OPENLANE_ROOT | Out-File $status -Append -Encoding utf8
  $p = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -NoNewWindow -Wait -PassThru `
      -RedirectStandardOutput (Join-Path $results "openlane_pilot_$name.out.log") `
      -RedirectStandardError (Join-Path $results "openlane_pilot_$name.err.log")
  "[{0}] {1} exit code {2}" -f (Get-Date -Format s), $name, $p.ExitCode | Out-File $status -Append -Encoding utf8
}

Step "run" @("-m", "tac_ufld", "run", "--config", "configs/openlane_pilot.yaml")
Step "robustness" @("-m", "tac_ufld", "robustness", "--runs", "results/openlane_pilot")
