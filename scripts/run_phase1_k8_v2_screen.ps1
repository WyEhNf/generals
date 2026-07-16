$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\admin\miniconda3\envs\pacman\python.exe"
Set-Location $root

& $python scripts\train_bc.py `
    --input data\processed\raw_replays_v1_shuffled `
    --split train `
    --limit-replays 1200 `
    --steps 1000 `
    --batch-size 2 `
    --sequence-length 64 `
    --burn-in 16 `
    --model-config configs\models\temporal_fixed_k8_v2.json `
    --backbone-lr 1e-5 `
    --temporal-lr 1e-4 `
    --head-lr 5e-5 `
    --gradient-accumulation 4 `
    --shuffle-buffer-size 256 `
    --replay-shuffle-buffer-size 512 `
    --shuffle-shards `
    --warmup-steps 250 `
    --min-lr-ratio 1.0 `
    --enemy-king-loss-weight 0.02 `
    --checkpoint-every-steps 250 `
    --val-limit-replays 200 `
    --val-every-steps 250 `
    --val-batches 200 `
    --max-val-predicted-pass-rate 0.60 `
    --max-val-move-false-pass-rate 0.50 `
    --device cuda `
    --amp `
    --init-checkpoint checkpoints\bc\model.pt `
    --output artifacts\phase1\temporal_k8_v2_screen1000.pt `
    --metrics-output artifacts\phase1\temporal_k8_v2_screen1000.metrics.jsonl

if ($LASTEXITCODE -ne 0) {
    throw "K=8 v2 screen failed with exit code $LASTEXITCODE"
}
