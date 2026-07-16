$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\admin\miniconda3\envs\pacman\python.exe"
Set-Location $root

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_phase1_k8_v2_screen.ps1
if ($LASTEXITCODE -ne 0) {
    throw "K=8 v2 screen stage failed with exit code $LASTEXITCODE"
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_phase1_k8_long20k.ps1
if ($LASTEXITCODE -ne 0) {
    throw "K=8 v2 long20k stage failed with exit code $LASTEXITCODE"
}

& $python scripts\run_phase1_long_supervisor.py
if ($LASTEXITCODE -ne 0) {
    throw "K=8 v2 long-training supervisor failed with exit code $LASTEXITCODE"
}
