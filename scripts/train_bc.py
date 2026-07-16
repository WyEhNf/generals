#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from generals_bot.training.dataset import REPLAY_FILTER_RULES
    from generals_bot.training.temporal import TemporalModelConfig
    from generals_bot.training.trainer import (
        TemporalTrainConfig,
        TrainConfig,
        train_bc,
        train_temporal_bc,
    )
except RuntimeError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1)


DEFAULT_INPUT = "data/processed/raw_replays_v1/replays.jsonl.gz"
DEFAULT_OUTPUT = "artifacts/bc_policy_smoke.pt"


def main() -> int:
    """Run behavior cloning training from CLI arguments."""
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args, parser=parser)
    metrics = train_temporal_bc(config) if isinstance(config, TemporalTrainConfig) else train_bc(config)
    print(json.dumps(metrics, indent=2))
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the shared spatial/temporal behavior-cloning CLI parser."""
    parser = argparse.ArgumentParser(description="Train a behavior cloning policy.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--limit-replays", type=_none_or_int, default=100)
    parser.add_argument(
        "--replay-version",
        type=int,
        action="append",
        default=None,
        help="Only train/evaluate on this replay version. May be passed more than once.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--history-length", type=int, default=1)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.0)
    parser.add_argument("--distance-loss-weight", type=float, default=0.0)
    parser.add_argument("--distance-loss-cap", type=float, default=10.0)
    parser.add_argument("--distance-split-penalty", type=float, default=0.5)
    parser.add_argument("--actionness-loss-weight", type=float, default=0.0)
    parser.add_argument("--actionness-pass-weight", type=float, default=0.05)
    parser.add_argument("--enemy-king-loss-weight", type=float, default=0.0)
    parser.add_argument("--enemy-king-distance-loss-weight", type=float, default=1.0)
    parser.add_argument("--enemy-king-distance-cap", type=float, default=30.0)
    parser.add_argument("--freeze-non-king-head", action="store_true")
    parser.add_argument("--late-pass-turn", type=int, default=30)
    parser.add_argument("--late-pass-weight", type=float, default=0.2)
    parser.add_argument("--replay-filter-rule", default="none", choices=REPLAY_FILTER_RULES)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--tensorboard-log-dir", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--skip-train-batches", type=int, default=0)
    parser.add_argument("--override-resume-lr", action="store_true")
    parser.add_argument("--shuffle-buffer-size", type=int, default=8192)
    parser.add_argument("--shuffle-shards", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--val-limit-replays", type=_none_or_int, default=100)
    parser.add_argument("--val-every-steps", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    temporal = parser.add_argument_group("temporal behavior cloning")
    temporal.add_argument(
        "--model-config",
        default=None,
        help="Temporal model JSON. Supplying this flag enables temporal BC training.",
    )
    temporal.add_argument("--sequence-length", type=int, default=64)
    temporal.add_argument("--burn-in", type=int, default=16)
    temporal.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional explicit K; must match chunk_size in --model-config.",
    )
    temporal.add_argument("--backbone-lr", type=float, default=1e-5)
    temporal.add_argument("--temporal-lr", type=float, default=1e-4)
    temporal.add_argument("--head-lr", type=float, default=5e-5)
    temporal.add_argument("--gradient-accumulation", type=int, default=1)
    temporal.add_argument("--replay-shuffle-buffer-size", type=int, default=2048)
    temporal.add_argument("--max-val-predicted-pass-rate", type=float, default=1.0)
    temporal.add_argument("--max-val-move-false-pass-rate", type=float, default=1.0)
    temporal.add_argument("--amp", action="store_true")
    return parser


def config_from_args(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> TrainConfig:
    """Convert parsed CLI arguments into a spatial or temporal train config."""
    common = dict(
        input_path=args.input,
        output_path=args.output,
        split=args.split,
        limit_replays=args.limit_replays,
        replay_versions=_replay_versions(args.replay_version),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        history_length=args.history_length,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
        distance_loss_weight=args.distance_loss_weight,
        distance_loss_cap=args.distance_loss_cap,
        distance_split_penalty=args.distance_split_penalty,
        actionness_loss_weight=args.actionness_loss_weight,
        actionness_pass_weight=args.actionness_pass_weight,
        enemy_king_loss_weight=args.enemy_king_loss_weight,
        enemy_king_distance_loss_weight=args.enemy_king_distance_loss_weight,
        enemy_king_distance_cap=args.enemy_king_distance_cap,
        freeze_non_king_head=args.freeze_non_king_head,
        late_pass_turn=args.late_pass_turn,
        late_pass_weight=args.late_pass_weight,
        replay_filter_rule=args.replay_filter_rule,
        checkpoint_every_steps=args.checkpoint_every_steps,
        device=args.device,
        metrics_output_path=args.metrics_output,
        tensorboard_log_dir=args.tensorboard_log_dir,
        init_checkpoint=args.init_checkpoint,
        resume_from=args.resume_from,
        skip_train_batches=args.skip_train_batches,
        override_resume_lr=args.override_resume_lr,
        shuffle_buffer_size=args.shuffle_buffer_size,
        shuffle_shards=args.shuffle_shards,
        seed=args.seed,
        val_split=args.val_split,
        val_limit_replays=args.val_limit_replays,
        val_every_steps=args.val_every_steps,
        val_batches=args.val_batches,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    if args.model_config is None:
        return TrainConfig(**common)

    try:
        temporal_model = TemporalModelConfig.from_json(args.model_config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if parser is not None:
            parser.error(f"invalid --model-config: {exc}")
        raise
    chunk_size = temporal_model.chunk_size
    if args.chunk_size is not None and args.chunk_size != chunk_size:
        message = (
            f"--chunk-size={args.chunk_size} does not match "
            f"model config chunk_size={chunk_size}"
        )
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    if args.base_channels != temporal_model.base_channels:
        message = (
            f"--base-channels={args.base_channels} does not match "
            f"model config base_channels={temporal_model.base_channels}"
        )
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    return TemporalTrainConfig(
        **common,
        model_config_path=args.model_config,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
        chunk_size=chunk_size,
        backbone_lr=args.backbone_lr,
        temporal_lr=args.temporal_lr,
        head_lr=args.head_lr,
        gradient_accumulation=args.gradient_accumulation,
        use_amp=args.amp,
        replay_shuffle_buffer_size=args.replay_shuffle_buffer_size,
        max_val_predicted_pass_rate=args.max_val_predicted_pass_rate,
        max_val_move_false_pass_rate=args.max_val_move_false_pass_rate,
    )


def _none_or_int(value: str) -> int | None:
    """Parse an integer CLI value, accepting 'none' for no limit."""
    if value.lower() == "none":
        return None
    return int(value)


def _replay_versions(values: list[int] | None) -> tuple[int, ...] | None:
    """Normalize optional repeated replay version flags."""
    if values is None:
        return None
    return tuple(dict.fromkeys(values))


if __name__ == "__main__":
    raise SystemExit(main())
