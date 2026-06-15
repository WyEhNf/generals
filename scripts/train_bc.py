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
    from generals_bot.training.trainer import TrainConfig, train_bc
except RuntimeError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1)


DEFAULT_INPUT = "data/processed/raw_replays_v1/replays.jsonl.gz"
DEFAULT_OUTPUT = "artifacts/bc_policy_smoke.pt"


def main() -> int:
    """Run behavior cloning training from CLI arguments."""
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
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = TrainConfig(
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
    )
    metrics = train_bc(config)
    print(json.dumps(metrics, indent=2))
    print(args.output)
    return 0


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
