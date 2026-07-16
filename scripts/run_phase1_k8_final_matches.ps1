param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [int]$SeedStart = 4000,
    [int]$SeedCount = 50,
    [string]$OutputDir = "artifacts\phase1\evaluations\k8_final_acceptance"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\admin\miniconda3\envs\pacman\python.exe"
Set-Location $root

if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "Final K=8 checkpoint does not exist: $Checkpoint"
}

& $python scripts\evaluate_match.py `
    --agent "K8-final=checkpoint:$Checkpoint" `
    --agent "K1=checkpoint:artifacts/phase1/temporal_k1_v2_screen1000.pt" `
    --agent "K4=checkpoint:artifacts/phase1/temporal_k4_v2_screen1000.pt" `
    --agent "K16=checkpoint:artifacts/phase1/temporal_k16_v2_screen1000.pt" `
    --pair "K8-final:K1" `
    --pair "K8-final:K4" `
    --pair "K8-final:K16" `
    --seed-start $SeedStart `
    --seed-count $SeedCount `
    --turns 800 `
    --device cpu `
    --output-dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    throw "Final K=8 match acceptance failed with exit code $LASTEXITCODE"
}
