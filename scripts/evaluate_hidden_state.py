#!/usr/bin/env python3
"""Evaluate what temporal checkpoints retain about a hidden enemy General."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.model import BCPolicyNet
from generals_bot.training.temporal import TemporalModelConfig, TemporalPolicyNet
from generals_bot.training.trainer import (
    TemporalTrainConfig, _make_temporal_loaders, _move_batch,
    _temporal_sequence_forward,
)

BUCKETS = ((1, 8), (9, 16), (17, 32), (33, 64))


def unseen_bucket(streak: int) -> str:
    if streak <= 0:
        return "visible"
    for low, high in BUCKETS:
        if low <= streak <= high:
            return f"unseen_{low}_{high}"
    return "unseen_65_plus"


class Metrics:
    def __init__(self) -> None:
        self.rows = defaultdict(lambda: defaultdict(float))

    def add(self, group: str, *, hit1: bool, hit5: bool, hit10: bool,
            nll: float, distance: float, confidence: float,
            target_probability: float, unreachable: bool) -> None:
        r = self.rows[group]
        r["count"] += 1
        r["top1"] += hit1
        r["top5"] += hit5
        r["top10"] += hit10
        r["nll"] += nll
        r["mean_distance"] += distance
        r["confidence"] += confidence
        r["target_probability"] += target_probability
        r["confidence_error"] += abs(confidence - float(hit1))
        r["unreachable_rate"] += unreachable

    def summary(self) -> dict:
        out = {}
        for group, raw in self.rows.items():
            n = int(raw["count"])
            out[group] = {"count": n}
            for key, value in raw.items():
                if key != "count":
                    out[group][key] = value / n
        return out


def load_model(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = TemporalTrainConfig(**checkpoint["train_config"])
    temporal = TemporalModelConfig(**checkpoint["temporal_config"])
    encoder = ObservationEncoder(history_length=config.history_length)
    spatial = BCPolicyNet(
        encoder.num_channels,
        base_channels=int(checkpoint["base_channels"]),
        temporal_dim=temporal.temporal_dim,
    ).to(device)
    model = TemporalPolicyNet(encoder.num_channels, spatial, temporal).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, config, encoder, model


@torch.no_grad()
def evaluate(path: str, args) -> dict:
    device = torch.device(args.device)
    checkpoint, stored, encoder, model = load_model(path, device)
    config = replace(
        stored, split=args.split, limit_replays=args.limit_replays,
        batch_size=args.batch_size, shuffle_buffer_size=0,
        val_split=args.split, val_limit_replays=args.limit_replays,
        val_every_steps=1, val_batches=args.batches, device=args.device,
        use_amp=args.amp,
    )
    _, loader = _make_temporal_loaders(config, encoder)
    metrics = Metrics()
    evaluated = 0
    visible_channel = encoder.channel_names.index("visible")
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            batch = _move_batch(batch, device)
            with torch.amp.autocast("cuda", enabled=args.amp):
                output = _temporal_sequence_forward(model, batch)
            logits = output["enemy_king_logits"].flatten(1).float()
            valid = batch["enemy_king_valid_mask"]
            labels = batch["enemy_king_label"]
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
            probs = torch.softmax(logits, 1)
            top = probs.topk(min(10, probs.shape[1]), 1).indices
            pred = top[:, 0]
            distances = batch["enemy_king_distance"].gather(1, pred[:, None]).squeeze(1)
            target_p = probs.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12)
            confidence = probs.max(1).values
            lengths = batch["seq_lengths"].tolist()
            offset = 0
            for length in lengths:
                unseen = 0
                for local in range(length):
                    i = offset + local
                    width = batch["x"].shape[-1]
                    tile = int(labels[i])
                    row, col = divmod(tile, width)
                    is_visible = bool(batch["x"][i, visible_channel, row, col] > 0.5)
                    unseen = 0 if is_visible else unseen + 1
                    if bool(batch["burn_in_mask"][i]):
                        continue
                    group = unseen_bucket(unseen)
                    raw_distance = distances[i]
                    distance_cap = batch["x"].shape[-2] + batch["x"].shape[-1] - 2
                    values = dict(
                        hit1=bool(pred[i] == labels[i]),
                        hit5=bool((top[i, :5] == labels[i]).any()),
                        hit10=bool((top[i] == labels[i]).any()),
                        nll=float(-target_p[i].log()),
                        distance=float(raw_distance.clamp_max(distance_cap)),
                        confidence=float(confidence[i]),
                        target_probability=float(target_p[i]),
                        unreachable=not bool(torch.isfinite(raw_distance)),
                    )
                    metrics.add("all", **values)
                    metrics.add(group, **values)
                    evaluated += 1
                offset += length
    return {
        "checkpoint": path,
        "chunk_size": checkpoint["temporal_config"]["chunk_size"],
        "split": args.split,
        "batches": args.batches,
        "evaluated_steps": evaluated,
        "metrics": metrics.summary(),
        "limitations": [
            "The enemy-General head had zero auxiliary loss in these checkpoints.",
            "Unseen streak resets at each sampled sequence boundary.",
            "Dynamic hidden armies are not labelled by the current temporal batch schema.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit-replays", type=int, default=500)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    result = {"models": [evaluate(path, args) for path in args.checkpoint]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
