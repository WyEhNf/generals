#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.model import BCPolicyNet
from generals_bot.training.temporal import TemporalModelConfig, TemporalPolicyNet
from generals_bot.training.trainer import (
    TemporalTrainConfig,
    _make_temporal_loaders,
    evaluate_temporal_bc,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a temporal BC checkpoint without updating it."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--limit-replays", type=int, default=200)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "temporal_policy_v1":
        raise ValueError("checkpoint is not a temporal_policy_v1 model")

    stored = dict(checkpoint["train_config"])
    config = TemporalTrainConfig(**stored)
    config = replace(
        config,
        split=args.split,
        limit_replays=args.limit_replays,
        batch_size=args.batch_size,
        shuffle_buffer_size=0,
        val_split=args.split,
        val_limit_replays=args.limit_replays,
        val_every_steps=1,
        val_batches=args.batches,
        device=args.device,
        use_amp=args.amp,
    )

    encoder = ObservationEncoder(history_length=config.history_length)
    temporal_config = TemporalModelConfig(**checkpoint["temporal_config"])
    spatial = BCPolicyNet(
        encoder.num_channels,
        base_channels=int(checkpoint["base_channels"]),
        temporal_dim=temporal_config.temporal_dim,
    ).to(device)
    model = TemporalPolicyNet(
        encoder.num_channels, spatial, temporal_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    _, val_loader = _make_temporal_loaders(config, encoder)
    if val_loader is None:
        raise RuntimeError("validation loader was not created")
    metrics = evaluate_temporal_bc(model, val_loader, config, device)
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "limit_replays": args.limit_replays,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "seed": config.seed,
        "sequence_length": config.sequence_length,
        "burn_in": config.burn_in,
        "chunk_size": config.chunk_size,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
