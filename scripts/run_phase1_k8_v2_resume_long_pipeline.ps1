$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\admin\miniconda3\envs\pacman\python.exe"
$artifacts = Join-Path $root "artifacts\phase1"
Set-Location $root

$supervisor = Start-Process `
    -FilePath $python `
    -ArgumentList "scripts\run_phase1_long_supervisor.py" `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $artifacts "phase1_v2_supervisor.stdout.log") `
    -RedirectStandardError (Join-Path $artifacts "phase1_v2_supervisor.stderr.log") `
    -PassThru

try {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File scripts\run_phase1_k8_long20k.ps1
    if ($LASTEXITCODE -ne 0) {
        throw "K=8 v2 long20k stage failed with exit code $LASTEXITCODE"
    }

    Wait-Process -Id $supervisor.Id
    $supervisor.Refresh()
    if ($supervisor.ExitCode -ne 0) {
        throw "K=8 v2 supervisor failed with exit code $($supervisor.ExitCode)"
    }
}
finally {
    if (-not $supervisor.HasExited) {
        Stop-Process -Id $supervisor.Id -ErrorAction SilentlyContinue
    }
}
