#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generals_bot.training.ppo import DEFAULT_INIT_CHECKPOINT, PPOConfig, train_ppo


def main() -> int:
    """Parse CLI arguments and run terminal-reward self-play PPO."""
    parser = argparse.ArgumentParser(description="Train a self-play PPO policy.")
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--output", default="artifacts/ppo_terminal_v1.pt")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--tensorboard-log-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--enemy-king-loss-weight", type=float, default=0.0)
    parser.add_argument("--enemy-king-distance-loss-weight", type=float, default=1.0)
    parser.add_argument("--enemy-king-distance-cap", type=float, default=30.0)
    parser.add_argument("--map-size", type=int, default=18)
    parser.add_argument("--max-turns", type=int, default=800)
    parser.add_argument("--checkpoint-every-updates", type=int, default=0)
    parser.add_argument("--keep-init-value-head", action="store_true")
    parser.add_argument("--opponent-pool-config", default=None)
    parser.add_argument("--reward-config", default=None)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--num-envs-scope",
        choices=("global", "per-rank"),
        default="global",
    )
    parser.add_argument("--ddp-backend", default="nccl")
    parser.add_argument("--torch-num-threads", type=int, default=None)
    args = parser.parse_args()

    summary = train_ppo(
        PPOConfig(
            init_checkpoint=args.init_checkpoint,
            output_path=args.output,
            resume_from=args.resume_from,
            metrics_output_path=args.metrics_output,
            tensorboard_log_dir=args.tensorboard_log_dir,
            device=args.device,
            seed=args.seed,
            updates=args.updates,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            target_kl=args.target_kl,
            grad_clip=args.grad_clip,
            enemy_king_loss_weight=args.enemy_king_loss_weight,
            enemy_king_distance_loss_weight=args.enemy_king_distance_loss_weight,
            enemy_king_distance_cap=args.enemy_king_distance_cap,
            map_size=args.map_size,
            max_turns=args.max_turns,
            checkpoint_every_updates=args.checkpoint_every_updates,
            reset_value_head=not args.keep_init_value_head,
            opponent_pool_config=args.opponent_pool_config,
            reward_config_path=args.reward_config,
            distributed=args.distributed,
            num_envs_scope=args.num_envs_scope,
            ddp_backend=args.ddp_backend,
            torch_num_threads=args.torch_num_threads,
        )
    )
    if int(summary.get("rank", 0)) == 0:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
