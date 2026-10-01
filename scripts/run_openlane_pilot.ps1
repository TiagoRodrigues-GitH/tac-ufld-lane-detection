# Overnight OpenLane pilot: training + evaluation + report, then the robustness evaluation.
# Runs detached; progress in results/openlane_pilot/run.log, console output in
# results/openlane_pilot_console.log. Start it with:
#   Start-Process powershell -ArgumentList '-NoProfile','-File','scripts\run_openlane_pilot.ps1' -WindowStyle Minimized
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path (Split-Path -Parent $repo) ".venv\Scripts\python.exe"
if (-not $env:OPENLANE_ROOT) {
  $env:OPENLANE_ROOT = Join-Path (Split-Path -Parent (Split-Path -Parent $repo)) "datasets\OpenLane"
}
$log = Join-Path $repo "results\openlane_pilot_console.log"
New-Item -ItemType Directory -Force (Join-Path $repo "results") | Out-Null
"[{0}] OPENLANE_ROOT={1}" -f (Get-Date -Format s), $env:OPENLANE_ROOT | Out-File $log -Encoding utf8
"[{0}] run" -f (Get-Date -Format s) | Out-File $log -Append -Encoding utf8
& $python -m tac_ufld run --config configs/openlane_pilot.yaml *>> $log
"[{0}] run exit code {1}; robustness" -f (Get-Date -Format s), $LASTEXITCODE | Out-File $log -Append -Encoding utf8
& $python -m tac_ufld robustness --runs results/openlane_pilot *>> $log
"[{0}] robustness exit code {1}; done" -f (Get-Date -Format s), $LASTEXITCODE | Out-File $log -Append -Encoding utf8
