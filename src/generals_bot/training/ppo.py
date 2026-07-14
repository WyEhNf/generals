from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "torch and tqdm are required for PPO training. Install with: "
        ".venv/bin/python -m pip install -r requirements-train.txt"
    ) from exc

from generals_bot.sim import GameConfig, GeneralsEnv, MapGenerationConfig, generate_1v1_map
from generals_bot.sim.types import Action, Observation, PassAction
from generals_bot.training.action_codec import label_to_action, valid_action_mask
from generals_bot.training.batched_policy import BatchedBCPolicyRunner
from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.king import build_enemy_king_targets
from generals_bot.training.model import BCPolicyNet, POLICY_MODE, load_model_state_compatible
from generals_bot.training.losses import enemy_king_prediction_loss
from generals_bot.training.rewards import (
    CityAttackOutcomeEvent,
    CityCaptureLifecycleEvent,
    REWARD_COMPONENTS,
    RewardBreakdown,
    RewardCalculator,
    RewardState,
    load_reward_config,
)


DEFAULT_INIT_CHECKPOINT = (
    "checkpoints/bc_3epoch/model.pt"
)


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for self-play PPO."""

    init_checkpoint: str = DEFAULT_INIT_CHECKPOINT
    output_path: str = "artifacts/ppo_terminal_v1.pt"
    resume_from: str | None = None
    metrics_output_path: str | None = None
    tensorboard_log_dir: str | None = None
    device: str = "auto"
    seed: int = 0
    updates: int = 1
    num_envs: int = 32
    rollout_steps: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 1024
    lr: float = 1e-5
    weight_decay: float = 0.0
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    target_kl: float = 0.02
    grad_clip: float = 1.0
    enemy_king_loss_weight: float = 0.0
    enemy_king_distance_loss_weight: float = 1.0
    enemy_king_distance_cap: float = 30.0
    map_size: int = 18
    max_turns: int = 800
    checkpoint_every_updates: int = 0
    reset_value_head: bool = True
    opponent_pool_config: str | None = None
    reward_config_path: str | None = None
    distributed: bool = False
    num_envs_scope: str = "global"
    ddp_backend: str = "nccl"
    torch_num_threads: int | None = None


@dataclass(frozen=True)
class DistributedContext:
    """Runtime distributed-training metadata for one PPO process."""

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device | None = None

    @property
    def is_rank0(self) -> bool:
        """Return whether this process is responsible for writes and logs."""
        return self.rank == 0


@dataclass(frozen=True)
class OpponentSpec:
    """One frozen opponent checkpoint entry from a PPO pool config."""

    id: str
    path: Path
    weight: float


@dataclass(frozen=True)
class OpponentPool:
    """Resolved frozen-opponent pool used by PPO rollout collection."""

    config_path: Path
    self_play_weight: float
    opponent_device: str
    opponent_pass_bias: float
    opponents: tuple[OpponentSpec, ...]


@dataclass(frozen=True)
class Matchup:
    """Per-environment learner/opponent assignment."""

    opponent_id: str | None
    learner_players: tuple[int, ...]

    @property
    def is_self_play(self) -> bool:
        """Return whether both sides are controlled by the learner."""
        return self.opponent_id is None


@dataclass
class SampledActions:
    """Actions and policy statistics sampled from the current model."""

    actions: list[Action | PassAction]
    labels: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    entropy: torch.Tensor


@dataclass
class RolloutBatch:
    """Flat PPO rollout batch collected from multiple self-play environments."""

    x: torch.Tensor
    valid_action_mask: torch.Tensor
    action_label: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    enemy_king_label: torch.Tensor
    enemy_king_distance: torch.Tensor
    enemy_king_valid_mask: torch.Tensor
    rewards: torch.Tensor
    done: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    sample_weight: torch.Tensor
    stream_ids: list[tuple[int, int]]
    metadata: list[dict[str, int]]

    @property
    def size(self) -> int:
        """Return the number of player-turn samples in the rollout."""
        return int(self.action_label.numel())

    @property
    def real_size(self) -> int:
        """Return the number of non-padding samples in the rollout."""
        return int(round(float(self.sample_weight.sum().item())))


@dataclass
class RolloutResult:
    """A collected PPO rollout and aggregate simulator statistics."""

    batch: RolloutBatch
    stats: dict[str, float]


def train_ppo(config: PPOConfig) -> dict[str, Any]:
    """Train a PPO policy from a BC or PPO checkpoint."""
    _validate_config(config)
    dist_context = _setup_distributed(config)
    device = dist_context.device or _resolve_device(config.device)
    _seed_everything(config.seed)
    metric_log_path = _resolve_metrics_output_path(config)
    if dist_context.is_rank0:
        metric_log_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(config.resume_from or config.init_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, encoder, base_channels, missing_optional = _load_model_from_checkpoint(checkpoint, device)
    if config.resume_from is None and config.reset_value_head:
        _reset_value_head(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    update_model = _wrap_model_for_distributed(model, dist_context)

    global_update = 0
    rollout_steps_seen = 0
    if config.resume_from is not None:
        if "optimizer_state" in checkpoint and not missing_optional:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _move_optimizer_state(optimizer, device)
        global_update = int(checkpoint.get("global_update", 0))
        rollout_steps_seen = int(checkpoint.get("rollout_steps_seen", 0))

    _seed_everything(config.seed + dist_context.rank * 1_000_000)
    local_envs, env_id_offset = _rank_env_assignment(
        config.num_envs,
        world_size=dist_context.world_size,
        rank=dist_context.rank,
        scope=config.num_envs_scope,
    )
    collector = SelfPlayRolloutCollector(
        model=model,
        encoder=encoder,
        config=config,
        device=device,
        num_envs=local_envs,
        seed=config.seed + dist_context.rank * 1_000_000,
        env_id_offset=env_id_offset,
    )
    tensorboard_writer = (
        _create_tensorboard_writer(config.tensorboard_log_dir)
        if dist_context.is_rank0
        else None
    )
    last_metrics: dict[str, float] = {}
    metric_log_mode = "a" if config.resume_from is not None and metric_log_path.exists() else "w"

    try:
        metric_log_context = (
            metric_log_path.open(metric_log_mode, encoding="utf-8")
            if dist_context.is_rank0
            else nullcontext(None)
        )
        with metric_log_context as metric_log:
            progress = tqdm(
                range(global_update, config.updates),
                desc="train_ppo",
                leave=False,
                disable=not dist_context.is_rank0,
            )
            for _ in progress:
                update_start = time.perf_counter()
                collect_start = time.perf_counter()
                rollout = collector.collect()
                collect_seconds = time.perf_counter() - collect_start
                aligned_batch, padded_samples = _align_rollout_batch_for_distributed(
                    rollout.batch,
                    dist_context,
                )
                rollout.stats["distributed_world_size"] = float(dist_context.world_size)
                rollout.stats["distributed_rank_envs"] = float(local_envs)
                rollout.stats["distributed_used_samples"] = float(aligned_batch.real_size)
                rollout.stats["distributed_batch_samples"] = float(aligned_batch.size)
                rollout.stats["distributed_padded_samples"] = float(padded_samples)
                rollout.stats["distributed_dropped_samples"] = 0.0
                rollout.stats["time_collect_sec"] = float(collect_seconds)
                update_start_only = time.perf_counter()
                update_metrics = ppo_update(
                    model=update_model,
                    optimizer=optimizer,
                    batch=aligned_batch,
                    config=config,
                    device=device,
                    dist_context=dist_context,
                )
                update_seconds = time.perf_counter() - update_start_only
                local_metrics = {**rollout.stats, **update_metrics}
                local_metrics["time_update_sec"] = float(update_seconds)
                local_metrics["time_total_update_sec"] = float(time.perf_counter() - update_start)
                last_metrics = _aggregate_distributed_metrics(local_metrics, dist_context)
                global_used_samples = int(
                    last_metrics.get("distributed_used_samples", aligned_batch.size)
                )
                if last_metrics.get("time_total_update_sec", 0.0) > 0:
                    last_metrics["samples_per_sec"] = (
                        float(global_used_samples) / last_metrics["time_total_update_sec"]
                    )
                global_update += 1
                rollout_steps_seen += global_used_samples
                if dist_context.is_rank0:
                    if metric_log is None:  # pragma: no cover
                        raise RuntimeError("rank0 metric log was not opened")
                    _write_metric_row(
                        metric_log,
                        update=global_update,
                        rollout_steps_seen=rollout_steps_seen,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        metrics=last_metrics,
                    )
                    _write_tensorboard_scalars(
                        tensorboard_writer,
                        step=global_update,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        metrics=last_metrics,
                    )
                if dist_context.is_rank0 and _should_save_periodic_checkpoint(config, global_update):
                    _save_ppo_checkpoint(
                        config,
                        encoder,
                        model,
                        optimizer,
                        metrics=last_metrics,
                        metric_log_path=metric_log_path,
                        base_channels=base_channels,
                        global_update=global_update,
                        rollout_steps_seen=rollout_steps_seen,
                        output_path=_periodic_checkpoint_path(config, global_update),
                        dist_context=dist_context,
                        rank_envs=local_envs,
                    )
                progress.set_postfix({
                    key: round(value, 4)
                    for key, value in last_metrics.items()
                    if key in {"policy_loss", "value_loss", "entropy", "approx_kl"}
                })
    except Exception:
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        _cleanup_distributed(dist_context)
        raise

    if tensorboard_writer is not None:
        tensorboard_writer.close()
    _barrier_if_distributed(dist_context)
    if dist_context.is_rank0:
        _save_ppo_checkpoint(
            config,
            encoder,
            model,
            optimizer,
            metrics=last_metrics,
            metric_log_path=metric_log_path,
            base_channels=base_channels,
            global_update=global_update,
            rollout_steps_seen=rollout_steps_seen,
            dist_context=dist_context,
            rank_envs=local_envs,
        )
    summary = {
        **last_metrics,
        "checkpoint_path": config.output_path,
        "metrics_output_path": str(metric_log_path),
        "global_update": global_update,
        "rollout_steps_seen": rollout_steps_seen,
        "rank": dist_context.rank,
        "world_size": dist_context.world_size,
    }
    _cleanup_distributed(dist_context)
    return summary


class SelfPlayRolloutCollector:
    """Collect fixed-size PPO rollouts from local self-play environments."""

    def __init__(
        self,
        *,
        model: BCPolicyNet,
        encoder: ObservationEncoder,
        config: PPOConfig,
        device: torch.device,
        num_envs: int | None = None,
        seed: int | None = None,
        env_id_offset: int = 0,
    ) -> None:
        """Create environments and policy-history state for rollout collection."""
        self.model = model
        self.encoder = encoder
        self.config = config
        self.device = device
        self.num_envs = config.num_envs if num_envs is None else int(num_envs)
        self.env_id_offset = int(env_id_offset)
        self.opponent_pool = (
            load_opponent_pool_config(
                config.opponent_pool_config,
                training_device=str(device),
            )
            if config.opponent_pool_config is not None
            else None
        )
        self.opponent_runners = self._load_opponent_runners()
        self.reward_calculator = RewardCalculator(
            load_reward_config(config.reward_config_path)
        )
        self.histories: dict[tuple[int, int], deque[Observation]] = {}
        self.envs: list[GeneralsEnv] = []
        self.observations: list[dict[int, Observation]] = []
        self.matchups: list[Matchup] = []
        self.episode_returns = [{0: 0.0, 1: 0.0} for _ in range(self.num_envs)]
        self.episode_lengths = [0 for _ in range(self.num_envs)]
        self.episode_city_captures = [{0: 0, 1: 0} for _ in range(self.num_envs)]
        self.pending_city_events: list[list[CityCaptureLifecycleEvent]] = [
            []
            for _ in range(self.num_envs)
        ]
        self.pending_city_attack_events: list[list[CityAttackOutcomeEvent]] = [
            []
            for _ in range(self.num_envs)
        ]
        self.global_stream_step: dict[tuple[int, int], int] = {}
        self.next_map_seed = config.seed if seed is None else int(seed)
        for env_id in range(self.num_envs):
            self._reset_env(env_id)

    def collect(self) -> RolloutResult:
        """Collect one rollout and compute terminal-reward GAE targets."""
        was_training = self.model.training
        self.model.eval()
        x_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        log_prob_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        king_label_rows: list[torch.Tensor] = []
        king_distance_rows: list[torch.Tensor] = []
        king_valid_mask_rows: list[torch.Tensor] = []
        reward_rows: list[float] = []
        done_rows: list[bool] = []
        stream_ids: list[tuple[int, int]] = []
        metadata: list[dict[str, int]] = []
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        player0_wins = 0
        player1_wins = 0
        draws = 0
        invalid_actions = 0
        submitted_actions = 0
        self_play_episodes = 0
        pool_episodes = 0
        learner_pool_wins = 0
        learner_pool_losses = 0
        learner_pool_draws = 0
        reward_component_sums = {component: 0.0 for component in REWARD_COMPONENTS}
        reward_sample_count = 0
        reward_adjustments_by_index: dict[int, RewardBreakdown] = {}
        opponent_stats = {
            opponent_id: {"episodes": 0, "learner_wins": 0, "learner_losses": 0, "learner_draws": 0}
            for opponent_id in self.opponent_runners
        }

        def add_reward_adjustment(
            *,
            env_id: int,
            event: CityAttackOutcomeEvent,
            breakdown: RewardBreakdown,
        ) -> None:
            """Apply a retroactive reward adjustment to its target rollout row."""
            if event.reward_index is None:
                return
            if event.reward_index < len(reward_rows):
                reward_rows[event.reward_index] += breakdown.total
            else:
                reward_adjustments_by_index[event.reward_index] = (
                    reward_adjustments_by_index.get(event.reward_index, RewardBreakdown())
                    + breakdown
                )
            self.episode_returns[env_id][event.player_id] += float(breakdown.total)
            for component, value in breakdown.as_metrics().items():
                reward_component_sums[component] += float(value)

        with torch.no_grad():
            for _ in range(self.config.rollout_steps):
                learner_items = self._learner_items(mutate_history=True)
                x_cpu, valid_mask_cpu, keys, observations = self._encode_items(learner_items)
                king_targets = self._enemy_king_targets(keys, observations)
                sampled = sample_policy_actions(
                    self.model,
                    x_cpu.to(self.device, dtype=torch.float32),
                    valid_mask_cpu.to(self.device),
                    observations=observations,
                )

                x_rows.extend(row.to(torch.float16).cpu() for row in x_cpu)
                mask_rows.extend(row.cpu() for row in valid_mask_cpu)
                label_rows.extend(row.cpu() for row in sampled.labels)
                log_prob_rows.extend(row.cpu() for row in sampled.log_probs)
                value_rows.extend(row.cpu() for row in sampled.values)
                king_label_rows.extend(row.cpu() for row in king_targets["label"])
                king_distance_rows.extend(row.to(torch.float16).cpu() for row in king_targets["distance"])
                king_valid_mask_rows.extend(row.cpu() for row in king_targets["valid_mask"])
                stream_ids.extend(keys)
                metadata.extend(
                    {
                        "env_id": self.env_id_offset + env_id,
                        "player_id": player,
                        "turn": observation.turn,
                        "global_stream_step": self.global_stream_step.get(
                            (env_id, player), 0
                        ),
                    }
                    for (env_id, player), observation in zip(keys, observations, strict=True)
                )
                for key in keys:
                    self.global_stream_step[key] = (
                        self.global_stream_step.get(key, 0) + 1
                    )

                actions_by_env: dict[int, dict[int, Action | PassAction]] = {}
                for (env_id, player), action in zip(keys, sampled.actions, strict=True):
                    actions_by_env.setdefault(env_id, {})[player] = action
                for (env_id, player), action in self._opponent_actions().items():
                    actions_by_env.setdefault(env_id, {})[player] = action
                reward_indices_by_stream = {
                    key: len(reward_rows) + index
                    for index, key in enumerate(keys)
                }

                rewards_by_stream: dict[tuple[int, int], float] = {}
                done_by_env: dict[int, bool] = {}
                for env_id, actions in actions_by_env.items():
                    matchup = self.matchups[env_id]
                    env = self.envs[env_id]
                    before_reward_state = RewardState.from_env(env)
                    reward_indices_by_player = {
                        player: reward_indices_by_stream[(env_id, player)]
                        for player in matchup.learner_players
                        if (env_id, player) in reward_indices_by_stream
                    }
                    if self.reward_calculator.config.city_attack_outcome.attribution == "retroactive":
                        for event in self.pending_city_attack_events[env_id]:
                            if event.reward_index is None:
                                event.reward_index = reward_indices_by_player.get(
                                    event.player_id
                                )
                    _, _, done, info = env.step(actions)
                    after_reward_state = RewardState.from_env(env)
                    city_capture_counts = self.reward_calculator.city_capture_counts(
                        info["executed_actions"],
                        before_reward_state.city_tiles,
                    )
                    for player, count in city_capture_counts.items():
                        self.episode_city_captures[env_id][player] += count
                    self.pending_city_events[env_id].extend(
                        self.reward_calculator.city_capture_lifecycle_events(
                            info["executed_actions"],
                            before_reward_state.city_tiles,
                            capture_turn=before_reward_state.turn,
                        )
                    )
                    self.pending_city_attack_events[env_id].extend(
                        self.reward_calculator.city_attack_outcome_events(
                            info["executed_actions"],
                            before_reward_state.city_tiles,
                            attack_turn=before_reward_state.turn,
                            reward_indices_by_player=reward_indices_by_player,
                        )
                    )
                    reward_breakdowns = self.reward_calculator.rewards(
                        before=before_reward_state,
                        after=after_reward_state,
                        executed_actions=info["executed_actions"],
                        winner=info["winner"],
                        done=bool(done),
                        episode_city_captures=self.episode_city_captures[env_id],
                    )
                    lifecycle_breakdowns, pending_events = (
                        self.reward_calculator.city_capture_lifecycle_rewards(
                            self.pending_city_events[env_id],
                            after_reward_state,
                        )
                    )
                    self.pending_city_events[env_id] = pending_events
                    if self.reward_calculator.config.city_attack_outcome.attribution == "resolution":
                        attack_outcome_rewards, pending_attack_events = (
                            self.reward_calculator.city_attack_outcome_rewards_for_step(
                                self.pending_city_attack_events[env_id],
                                after_reward_state,
                            )
                        )
                        self.pending_city_attack_events[env_id] = pending_attack_events
                        for player in (0, 1):
                            reward_breakdowns[player] += attack_outcome_rewards.get(
                                player,
                                RewardBreakdown(),
                            )
                    else:
                        resolved_attack_events, pending_attack_events = (
                            self.reward_calculator.city_attack_outcome_rewards(
                                self.pending_city_attack_events[env_id],
                                after_reward_state,
                            )
                        )
                        self.pending_city_attack_events[env_id] = pending_attack_events
                        for event, breakdown in resolved_attack_events:
                            add_reward_adjustment(
                                env_id=env_id,
                                event=event,
                                breakdown=breakdown,
                            )
                    reward_breakdowns = {
                        player: (
                            reward_breakdowns[player]
                            + lifecycle_breakdowns[player]
                        )
                        for player in (0, 1)
                    }
                    self.observations[env_id] = env.observations()
                    submitted_actions += sum(
                        len(actions)
                        for actions in info["submitted_actions"].values()
                    )
                    invalid_actions += sum(
                        1
                        for executed in info["executed_actions"]
                        if not executed.valid
                    )
                    for player in matchup.learner_players:
                        breakdown = reward_breakdowns[player]
                        reward = breakdown.total
                        self.episode_returns[env_id][player] += float(reward)
                        rewards_by_stream[(env_id, player)] = float(reward)
                        for component, value in breakdown.as_metrics().items():
                            reward_component_sums[component] += float(value)
                        reward_sample_count += 1
                    self.episode_lengths[env_id] += 1
                    done_by_env[env_id] = bool(done)
                    if done:
                        completed_returns.extend(
                            self.episode_returns[env_id][player]
                            for player in matchup.learner_players
                        )
                        completed_lengths.append(self.episode_lengths[env_id])
                        winner = info["winner"]
                        if winner == 0:
                            player0_wins += 1
                        elif winner == 1:
                            player1_wins += 1
                        else:
                            draws += 1
                        if matchup.is_self_play:
                            self_play_episodes += 1
                        else:
                            pool_episodes += 1
                            learner_player = matchup.learner_players[0]
                            learner_reward = terminal_reward(winner, learner_player)
                            opponent_id = str(matchup.opponent_id)
                            opponent_stats[opponent_id]["episodes"] += 1
                            if learner_reward > 0:
                                learner_pool_wins += 1
                                opponent_stats[opponent_id]["learner_wins"] += 1
                            elif learner_reward < 0:
                                learner_pool_losses += 1
                                opponent_stats[opponent_id]["learner_losses"] += 1
                            else:
                                learner_pool_draws += 1
                                opponent_stats[opponent_id]["learner_draws"] += 1
                        self._reset_env(env_id)

                for env_id, player in keys:
                    reward_index = reward_indices_by_stream[(env_id, player)]
                    adjustment = reward_adjustments_by_index.pop(
                        reward_index,
                        RewardBreakdown(),
                    )
                    reward_rows.append(
                        rewards_by_stream.get((env_id, player), 0.0)
                        + adjustment.total
                    )
                    done_rows.append(done_by_env.get(env_id, False))

            last_values = self._bootstrap_values()
            if self.reward_calculator.config.city_attack_outcome.attribution == "retroactive":
                for events in self.pending_city_attack_events:
                    for event in events:
                        event.reward_index = None

        rewards = torch.tensor(reward_rows, dtype=torch.float32)
        old_values = torch.stack(value_rows).to(torch.float32)
        done = torch.tensor(done_rows, dtype=torch.bool)
        advantages, returns = compute_gae(
            rewards=rewards,
            values=old_values,
            done=done,
            stream_ids=stream_ids,
            last_values_by_stream=last_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        self.model.train(was_training)
        batch = RolloutBatch(
            x=torch.stack(x_rows),
            valid_action_mask=torch.stack(mask_rows).to(torch.bool),
            action_label=torch.stack(label_rows).to(torch.long),
            old_log_prob=torch.stack(log_prob_rows).to(torch.float32),
            old_value=old_values,
            enemy_king_label=torch.stack(king_label_rows).to(torch.long),
            enemy_king_distance=torch.stack(king_distance_rows).to(torch.float32),
            enemy_king_valid_mask=torch.stack(king_valid_mask_rows).to(torch.bool),
            rewards=rewards,
            done=done,
            returns=returns,
            advantages=advantages,
            sample_weight=torch.ones_like(rewards, dtype=torch.float32),
            stream_ids=stream_ids,
            metadata=metadata,
        )
        stats = {
            "completed_episodes": float(len(completed_lengths)),
            "reward_attribution_mode": self.reward_calculator.config.city_attack_outcome.attribution,
            "mean_episode_return": _mean(completed_returns),
            "mean_episode_length": _mean(completed_lengths),
            "win_rate_player0": _rate(player0_wins, len(completed_lengths)),
            "win_rate_player1": _rate(player1_wins, len(completed_lengths)),
            "draw_rate": _rate(draws, len(completed_lengths)),
            "invalid_actions": float(invalid_actions),
            "submitted_actions": float(submitted_actions),
            "rollout_samples": float(batch.size),
            "reward_sample_count": float(reward_sample_count),
            "self_play_episodes": float(self_play_episodes),
            "pool_episodes": float(pool_episodes),
            "learner_pool_win_rate": _rate(learner_pool_wins, pool_episodes),
            "learner_pool_loss_rate": _rate(learner_pool_losses, pool_episodes),
            "learner_pool_draw_rate": _rate(learner_pool_draws, pool_episodes),
        }
        for component in REWARD_COMPONENTS:
            stats[f"reward_{component}_mean"] = _safe_divide(
                reward_component_sums[component],
                reward_sample_count,
            )
        for opponent_id, counters in opponent_stats.items():
            episodes = counters["episodes"]
            stats[f"opponent_{opponent_id}_episodes"] = float(episodes)
            stats[f"opponent_{opponent_id}_learner_win_rate"] = _rate(
                counters["learner_wins"],
                episodes,
            )
            stats[f"opponent_{opponent_id}_learner_loss_rate"] = _rate(
                counters["learner_losses"],
                episodes,
            )
            stats[f"opponent_{opponent_id}_learner_draw_rate"] = _rate(
                counters["learner_draws"],
                episodes,
            )
        return RolloutResult(batch=batch, stats=stats)

    def _learner_items(self, *, mutate_history: bool) -> list[tuple[tuple[int, int], Observation]]:
        """Return current learner observations, optionally appending history."""
        items = []
        for env_id, env in enumerate(self.envs):
            for player in env.alive_players:
                if not self._is_learner(env_id, player):
                    continue
                observation = self.observations[env_id][player]
                key = (env_id, player)
                if mutate_history:
                    self.histories.setdefault(
                        key,
                        deque(maxlen=self.encoder.history_length),
                    ).append(observation)
                items.append((key, observation))
        return items

    def _opponent_actions(self) -> dict[tuple[int, int], Action | PassAction]:
        """Return actions from frozen pool opponents for current rollout observations."""
        items_by_opponent: dict[str, list[tuple[tuple[int, int], Observation]]] = {}
        for env_id, env in enumerate(self.envs):
            matchup = self.matchups[env_id]
            if matchup.is_self_play:
                continue
            opponent_id = str(matchup.opponent_id)
            for player in env.alive_players:
                if self._is_learner(env_id, player):
                    continue
                observation = self.observations[env_id][player]
                items_by_opponent.setdefault(opponent_id, []).append(
                    ((env_id, player), observation)
                )

        actions: dict[tuple[int, int], Action | PassAction] = {}
        for opponent_id, items in items_by_opponent.items():
            actions.update(self.opponent_runners[opponent_id].act_many(items))
        return actions

    def _is_learner(self, env_id: int, player: int) -> bool:
        """Return whether a player stream is trained in the current matchup."""
        return player in self.matchups[env_id].learner_players

    def _encode_items(
        self,
        items: list[tuple[tuple[int, int], Observation]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[Observation]]:
        """Encode observations and valid action masks for a same-size rollout batch."""
        encoded = []
        masks = []
        keys = []
        observations = []
        for key, observation in items:
            history = self.histories.get(key, deque(maxlen=self.encoder.history_length))
            encoded.append(self.encoder.encode_history(tuple(history) or (observation,)))
            masks.append(valid_action_mask(observation))
            keys.append(key)
            observations.append(observation)
        x = torch.from_numpy(np.ascontiguousarray(np.stack(encoded, axis=0)))
        valid_mask = torch.from_numpy(np.stack(masks, axis=0)).to(torch.bool)
        return x, valid_mask, keys, observations

    def _enemy_king_targets(
        self,
        keys: list[tuple[int, int]],
        observations: list[Observation],
    ) -> dict[str, torch.Tensor]:
        """Build enemy general belief targets for current learner observations."""
        targets = build_enemy_king_targets(
            observations,
            enemy_general_tiles=[
                self.envs[env_id].config.generals[1 - player]
                for env_id, player in keys
            ],
            mountains=[
                self.envs[env_id].config.mountains
                for env_id, _ in keys
            ],
        )
        return {
            "label": torch.from_numpy(targets.labels),
            "distance": torch.from_numpy(targets.distances),
            "valid_mask": torch.from_numpy(targets.valid_mask),
        }

    def _bootstrap_values(self) -> dict[tuple[int, int], float]:
        """Estimate V(s_next) for streams that remain alive after the rollout."""
        items = []
        encoded = []
        keys = []
        for env_id, env in enumerate(self.envs):
            for player in env.alive_players:
                if not self._is_learner(env_id, player):
                    continue
                observation = self.observations[env_id][player]
                key = (env_id, player)
                history = list(self.histories.get(key, ()))
                history.append(observation)
                encoded.append(self.encoder.encode_history(tuple(history)))
                keys.append(key)
                items.append(observation)
        if not items:
            return {}
        x = torch.from_numpy(np.ascontiguousarray(np.stack(encoded, axis=0))).to(
            self.device,
            dtype=torch.float32,
        )
        output = self.model(x)
        values = output["value"].detach().cpu().to(torch.float32)
        return {
            key: float(value)
            for key, value in zip(keys, values, strict=True)
        }

    def _reset_env(self, env_id: int) -> None:
        """Replace one environment with a freshly generated fixed-size map."""
        game_config = self._next_game_config()
        env = GeneralsEnv(game_config)
        observations = env.reset()
        if env_id < len(self.envs):
            self.envs[env_id] = env
            self.observations[env_id] = observations
        else:
            self.envs.append(env)
            self.observations.append(observations)
        self.episode_returns[env_id] = {0: 0.0, 1: 0.0}
        self.episode_lengths[env_id] = 0
        self.episode_city_captures[env_id] = {0: 0, 1: 0}
        self.pending_city_events[env_id] = []
        self.pending_city_attack_events[env_id] = []
        matchup = self._sample_matchup()
        if env_id < len(self.matchups):
            self.matchups[env_id] = matchup
        else:
            self.matchups.append(matchup)
        self.histories.pop((env_id, 0), None)
        self.histories.pop((env_id, 1), None)
        self.global_stream_step.pop((env_id, 0), None)
        self.global_stream_step.pop((env_id, 1), None)
        for runner in self.opponent_runners.values():
            runner.reset_key((env_id, 0))
            runner.reset_key((env_id, 1))

    def _next_game_config(self) -> GameConfig:
        """Generate the next fixed-size self-play map."""
        seed = self.next_map_seed
        self.next_map_seed += 1
        map_size = self.config.map_size
        return generate_1v1_map(
            seed=seed,
            config=MapGenerationConfig(
                min_size=map_size,
                max_size=map_size,
                size_weights=None,
                min_general_distance=max(1, min(15, map_size * 2 - 4)),
                max_turns=self.config.max_turns,
            ),
            max_turns=self.config.max_turns,
        )

    def _load_opponent_runners(self) -> dict[str, BatchedBCPolicyRunner]:
        """Load frozen opponent checkpoints if an opponent pool is configured."""
        if self.opponent_pool is None:
            return {}
        return {
            spec.id: BatchedBCPolicyRunner(
                spec.path,
                device=self.opponent_pool.opponent_device,
                pass_bias=self.opponent_pool.opponent_pass_bias,
            )
            for spec in self.opponent_pool.opponents
        }

    def _sample_matchup(self) -> Matchup:
        """Sample a new matchup for one environment reset."""
        if self.opponent_pool is None:
            return Matchup(opponent_id=None, learner_players=(0, 1))
        choices: list[str | None] = []
        weights: list[float] = []
        if self.opponent_pool.self_play_weight > 0:
            choices.append(None)
            weights.append(self.opponent_pool.self_play_weight)
        for opponent in self.opponent_pool.opponents:
            if opponent.weight > 0:
                choices.append(opponent.id)
                weights.append(opponent.weight)
        opponent_id = random.choices(choices, weights=weights, k=1)[0]
        if opponent_id is None:
            return Matchup(opponent_id=None, learner_players=(0, 1))
        return Matchup(
            opponent_id=opponent_id,
            learner_players=(random.randrange(2),),
        )


def load_opponent_pool_config(
    config_path: str | Path,
    *,
    training_device: str,
) -> OpponentPool:
    """Load and validate a JSON PPO opponent-pool config."""
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("opponent pool config must contain version=1")

    self_play_weight = float(data.get("self_play_weight", 0.0))
    if self_play_weight < 0:
        raise ValueError("self_play_weight must be nonnegative")

    opponent_device = str(data.get("opponent_device", "same_as_training"))
    if opponent_device == "same_as_training":
        opponent_device = training_device
    opponent_pass_bias = float(data.get("opponent_pass_bias", 0.0))
    if opponent_pass_bias < 0:
        raise ValueError("opponent_pass_bias must be nonnegative")

    opponents_data = data.get("opponents", [])
    if not isinstance(opponents_data, list):
        raise ValueError("opponents must be a list")

    seen_ids: set[str] = set()
    opponents: list[OpponentSpec] = []
    for raw in opponents_data:
        if not isinstance(raw, dict):
            raise ValueError("each opponent must be an object")
        opponent_id = str(raw.get("id", "")).strip()
        if not opponent_id:
            raise ValueError("each opponent must have a nonempty id")
        if opponent_id in seen_ids:
            raise ValueError(f"duplicate opponent id: {opponent_id}")
        seen_ids.add(opponent_id)
        if "path" not in raw:
            raise ValueError(f"opponent {opponent_id} is missing path")
        opponent_path = Path(str(raw["path"]))
        if not opponent_path.is_absolute():
            opponent_path = path.parent / opponent_path
        opponent_path = opponent_path.resolve()
        if not opponent_path.exists():
            raise FileNotFoundError(f"opponent checkpoint not found: {opponent_path}")
        weight = float(raw.get("weight", 1.0))
        if weight < 0:
            raise ValueError(f"opponent {opponent_id} weight must be nonnegative")
        opponents.append(OpponentSpec(id=opponent_id, path=opponent_path, weight=weight))

    total_weight = self_play_weight + sum(opponent.weight for opponent in opponents)
    if total_weight <= 0:
        raise ValueError("opponent pool must have positive total sampling weight")
    return OpponentPool(
        config_path=path.resolve(),
        self_play_weight=self_play_weight,
        opponent_device=opponent_device,
        opponent_pass_bias=opponent_pass_bias,
        opponents=tuple(opponents),
    )


def sample_policy_actions(
    model: BCPolicyNet,
    x: torch.Tensor,
    valid_action_mask: torch.Tensor,
    *,
    observations: list[Observation],
) -> SampledActions:
    """Sample hierarchical PPO actions from the model under a legal-action mask."""
    output = model(x)
    move_logits = output["move_logits"].flatten(start_dim=1)
    action_logits = output["action_logits"].reshape(-1)
    move_mask = valid_action_mask[:, :-1]
    pass_label = valid_action_mask.shape[1] - 1
    has_move = move_mask.any(dim=1)

    labels = torch.full(
        (x.shape[0],),
        pass_label,
        dtype=torch.long,
        device=x.device,
    )
    move_probability = torch.sigmoid(action_logits)
    sampled_move_gate = torch.bernoulli(move_probability).to(torch.bool) & has_move
    if bool(sampled_move_gate.any()):
        masked_logits = move_logits.masked_fill(~move_mask, -1e9)
        move_probabilities = torch.softmax(masked_logits[sampled_move_gate], dim=1)
        labels[sampled_move_gate] = torch.multinomial(move_probabilities, num_samples=1).squeeze(1)

    log_probs, entropy, values = evaluate_policy_actions_from_output(
        output,
        valid_action_mask,
        labels,
    )
    actions = [
        label_to_action(
            int(label),
            player_id=observation.player_id,
            height=observation.height,
            width=observation.width,
        )
        for label, observation in zip(labels.detach().cpu(), observations, strict=True)
    ]
    return SampledActions(
        actions=actions,
        labels=labels.detach(),
        log_probs=log_probs.detach(),
        values=values.detach(),
        entropy=entropy.detach(),
    )


def evaluate_policy_actions(
    model: BCPolicyNet,
    x: torch.Tensor,
    valid_action_mask: torch.Tensor,
    action_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute log-probability, entropy, and value for PPO labels."""
    return evaluate_policy_actions_from_output(
        model(x),
        valid_action_mask,
        action_labels,
    )


def evaluate_policy_actions_from_output(
    output: dict[str, torch.Tensor],
    valid_action_mask: torch.Tensor,
    action_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute hierarchical action log-probs and entropy from model output."""
    move_logits = output["move_logits"].flatten(start_dim=1)
    action_logits = output["action_logits"].reshape(-1)
    values = output["value"].reshape(-1)
    move_mask = valid_action_mask[:, :-1]
    pass_label = valid_action_mask.shape[1] - 1
    has_move = move_mask.any(dim=1)
    is_pass = action_labels == pass_label
    is_move = ~is_pass

    masked_move_logits = move_logits.masked_fill(~move_mask, -1e9)
    move_log_probs = F.log_softmax(masked_move_logits, dim=1)
    move_probabilities = torch.softmax(masked_move_logits, dim=1)
    move_entropy = -(move_probabilities * move_log_probs).sum(dim=1)
    move_labels = action_labels.clamp(0, pass_label - 1)
    selected_move_log_probs = move_log_probs.gather(1, move_labels[:, None]).squeeze(1)

    log_prob = torch.zeros_like(action_logits)
    move_rows = has_move & is_move
    pass_rows = has_move & is_pass
    log_prob[move_rows] = F.logsigmoid(action_logits[move_rows]) + selected_move_log_probs[move_rows]
    log_prob[pass_rows] = F.logsigmoid(-action_logits[pass_rows])

    move_probability = torch.sigmoid(action_logits)
    gate_entropy = _binary_entropy_from_logits(action_logits)
    entropy = torch.zeros_like(action_logits)
    entropy[has_move] = gate_entropy[has_move] + move_probability[has_move] * move_entropy[has_move]
    return log_prob, entropy, values


def ppo_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    config: PPOConfig,
    device: torch.device,
    dist_context: DistributedContext | None = None,
) -> dict[str, float]:
    """Run PPO epochs over one rollout batch and return aggregate metrics."""
    model.train()
    if dist_context is None:
        dist_context = DistributedContext(device=device)
    advantages = _normalize_advantages(
        batch.advantages,
        sample_weight=batch.sample_weight,
        dist_context=dist_context,
    )
    sample_count = batch.size
    if sample_count == 0 or batch.real_size == 0:
        raise ValueError("cannot update PPO with an empty rollout")
    minibatch_size = _local_minibatch_size(config.minibatch_size, dist_context.world_size)
    accumulator = _MetricAccumulator()
    early_stop = False

    for _ in range(config.ppo_epochs):
        permutation = torch.randperm(sample_count)
        for start in range(0, sample_count, minibatch_size):
            indexes = permutation[start : start + minibatch_size]
            minibatch = _slice_rollout_batch(batch, advantages, indexes, device)
            output = model(minibatch["x"])
            log_probs, entropy, values = evaluate_policy_actions_from_output(
                output,
                minibatch["valid_action_mask"],
                minibatch["action_label"],
            )
            ratio = torch.exp(log_probs - minibatch["old_log_prob"])
            unclipped_policy = ratio * minibatch["advantage"]
            clipped_policy = torch.clamp(
                ratio,
                1.0 - config.clip_range,
                1.0 + config.clip_range,
            ) * minibatch["advantage"]
            sample_weight = minibatch["sample_weight"].to(dtype=torch.float32)
            policy_loss = -_weighted_tensor_mean(
                torch.min(unclipped_policy, clipped_policy),
                sample_weight,
            )
            value_pred_clipped = minibatch["old_value"] + torch.clamp(
                values - minibatch["old_value"],
                -config.clip_range,
                config.clip_range,
            )
            value_loss_unclipped = (values - minibatch["returns"]).pow(2)
            value_loss_clipped = (value_pred_clipped - minibatch["returns"]).pow(2)
            value_loss = 0.5 * _weighted_tensor_mean(
                torch.max(value_loss_unclipped, value_loss_clipped),
                sample_weight,
            )
            entropy_mean = _weighted_tensor_mean(entropy, sample_weight)
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
            king_metrics: dict[str, float] = {}
            if config.enemy_king_loss_weight > 0:
                real_rows = sample_weight > 0
                if bool(real_rows.any()):
                    king_loss, king_metrics = enemy_king_prediction_loss(
                        output["enemy_king_logits"][real_rows],
                        labels=minibatch["enemy_king_label"][real_rows],
                        valid_mask=minibatch["enemy_king_valid_mask"][real_rows],
                        distances=minibatch["enemy_king_distance"][real_rows],
                        distance_weight=config.enemy_king_distance_loss_weight,
                        distance_cap=config.enemy_king_distance_cap,
                    )
                else:
                    king_loss = output["enemy_king_logits"].sum() * 0.0
                    king_metrics = {
                        "enemy_king_loss": 0.0,
                        "enemy_king_ce_loss": 0.0,
                        "enemy_king_distance_loss": 0.0,
                        "enemy_king_top1_acc": 0.0,
                        "enemy_king_mean_distance": 0.0,
                        "enemy_king_target_valid_rate": 0.0,
                    }
                loss = loss + config.enemy_king_loss_weight * king_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            approx_kl = _weighted_tensor_mean(
                minibatch["old_log_prob"] - log_probs,
                sample_weight,
            )
            clip_fraction = _weighted_tensor_mean(
                (torch.abs(ratio - 1.0) > config.clip_range)
                .to(torch.float32),
                sample_weight,
            )
            metrics = {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy_mean.detach().cpu()),
                "approx_kl": float(approx_kl.detach().cpu()),
                "clip_fraction": float(clip_fraction.detach().cpu()),
            }
            metrics.update(king_metrics)
            accumulator.update(
                batch_size=float(sample_weight.sum().detach().cpu()),
                **metrics,
            )
            should_stop = config.target_kl > 0 and float(approx_kl.detach().cpu()) > config.target_kl
            if _distributed_any(should_stop, dist_context):
                early_stop = True
                break
        if early_stop:
            break

    summary = accumulator.summary()
    summary["advantage_mean"] = _weighted_float_mean(batch.advantages, batch.sample_weight)
    summary["advantage_std"] = _weighted_float_std(batch.advantages, batch.sample_weight)
    summary["return_mean"] = _weighted_float_mean(batch.returns, batch.sample_weight)
    summary["explained_variance"] = explained_variance(
        batch.returns,
        batch.old_value,
        sample_weight=batch.sample_weight,
    )
    summary["ppo_early_stop"] = float(1 if early_stop else 0)
    return summary


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    done: torch.Tensor,
    stream_ids: list[tuple[int, int]],
    last_values_by_stream: dict[tuple[int, int], float],
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages for interleaved player streams."""
    if not (len(rewards) == len(values) == len(done) == len(stream_ids)):
        raise ValueError("GAE inputs must have matching lengths")
    next_values = {
        stream: float(value)
        for stream, value in last_values_by_stream.items()
    }
    next_advantages: dict[tuple[int, int], float] = {}
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    for index in range(len(rewards) - 1, -1, -1):
        stream = stream_ids[index]
        terminal = bool(done[index])
        next_value = 0.0 if terminal else next_values.get(stream, 0.0)
        next_advantage = 0.0 if terminal else next_advantages.get(stream, 0.0)
        delta = float(rewards[index]) + gamma * next_value - float(values[index])
        advantage = delta + gamma * gae_lambda * next_advantage
        advantages[index] = advantage
        next_values[stream] = float(values[index])
        next_advantages[stream] = advantage
    returns = advantages + values.to(torch.float32)
    return advantages, returns


def terminal_reward(winner: int | None, player_id: int) -> float:
    """Return pure terminal win/loss/draw reward for one player."""
    if winner is None:
        return 0.0
    return 1.0 if player_id == winner else -1.0


def _terminal_rewards(winner: int | None) -> dict[int, float]:
    """Return pure terminal rewards for both players."""
    return {
        player: terminal_reward(winner, player)
        for player in (0, 1)
    }


def explained_variance(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> float:
    """Compute value-function explained variance over non-padding samples."""
    if sample_weight is None:
        variance = torch.var(targets, unbiased=False)
        if float(variance) <= 1e-8:
            return 0.0
        residual_variance = torch.var(targets - predictions, unbiased=False)
        return float((1.0 - residual_variance / variance).cpu())
    variance = _weighted_tensor_variance(targets, sample_weight)
    if float(variance) <= 1e-8:
        return 0.0
    residual_variance = _weighted_tensor_variance(targets - predictions, sample_weight)
    return float((1.0 - residual_variance / variance).cpu())


def _weighted_tensor_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return a tensor mean that ignores zero-weight padding samples."""
    weights = weights.to(device=values.device, dtype=values.dtype)
    denominator = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denominator


def _weighted_tensor_variance(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return a weighted population variance with zero for empty weights."""
    weights = weights.to(device=values.device, dtype=values.dtype)
    mean = _weighted_tensor_mean(values, weights)
    return _weighted_tensor_mean((values - mean).pow(2), weights)


def _weighted_float_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    """Return a Python float weighted mean for metric logging."""
    return float(_weighted_tensor_mean(values.to(torch.float32), weights).detach().cpu())


def _weighted_float_std(values: torch.Tensor, weights: torch.Tensor) -> float:
    """Return a Python float weighted standard deviation for metric logging."""
    variance = _weighted_tensor_variance(values.to(torch.float32), weights)
    return float(torch.sqrt(variance).detach().cpu())


def _slice_rollout_batch(
    batch: RolloutBatch,
    advantages: torch.Tensor,
    indexes: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move one minibatch slice to the training device."""
    return {
        "x": batch.x[indexes].to(device=device, dtype=torch.float32),
        "valid_action_mask": batch.valid_action_mask[indexes].to(device=device),
        "action_label": batch.action_label[indexes].to(device=device),
        "old_log_prob": batch.old_log_prob[indexes].to(device=device),
        "old_value": batch.old_value[indexes].to(device=device),
        "enemy_king_label": batch.enemy_king_label[indexes].to(device=device),
        "enemy_king_distance": batch.enemy_king_distance[indexes].to(device=device),
        "enemy_king_valid_mask": batch.enemy_king_valid_mask[indexes].to(device=device),
        "returns": batch.returns[indexes].to(device=device),
        "advantage": advantages[indexes].to(device=device),
        "sample_weight": batch.sample_weight[indexes].to(device=device),
    }


def _normalize_advantages(
    advantages: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    dist_context: DistributedContext | None = None,
) -> torch.Tensor:
    """Return standardized advantages with a safe zero-variance fallback."""
    if sample_weight is None:
        sample_weight = torch.ones_like(advantages, dtype=torch.float32)
    if dist_context is not None and dist_context.enabled:
        device = _distributed_tensor_device(dist_context)
        local = advantages.to(device=device, dtype=torch.float32)
        local_weight = sample_weight.to(device=device, dtype=torch.float32)
        stats = torch.tensor(
            [
                float((local * local_weight).sum()),
                float((local * local * local_weight).sum()),
                float(local_weight.sum()),
            ],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        count = max(float(stats[2].item()), 1.0)
        mean = float(stats[0].item() / count)
        variance = max(float(stats[1].item() / count - mean * mean), 0.0)
        std = variance ** 0.5
        if std <= 1e-8:
            return advantages - mean
        return (advantages - mean) / (std + 1e-8)

    mean = _weighted_tensor_mean(advantages.to(torch.float32), sample_weight)
    variance = _weighted_tensor_variance(advantages.to(torch.float32), sample_weight)
    std = torch.sqrt(variance)
    if float(std) <= 1e-8:
        return advantages - mean
    return (advantages - mean) / (std + 1e-8)


def _rank_env_assignment(
    num_envs: int,
    *,
    world_size: int,
    rank: int,
    scope: str,
) -> tuple[int, int]:
    """Return this rank's local env count and global env-id offset."""
    if scope == "per-rank":
        return num_envs, rank * num_envs
    if scope != "global":
        raise ValueError("num_envs_scope must be 'global' or 'per-rank'")
    if num_envs < world_size:
        raise ValueError("global num_envs must be at least world_size")
    base = num_envs // world_size
    extra = num_envs % world_size
    local_envs = base + (1 if rank < extra else 0)
    offset = base * rank + min(rank, extra)
    return local_envs, offset


def _local_minibatch_size(global_minibatch_size: int, world_size: int) -> int:
    """Return per-rank minibatch size for a global effective minibatch."""
    if global_minibatch_size % world_size != 0:
        raise ValueError("minibatch_size must be divisible by world_size")
    local = global_minibatch_size // world_size
    if local <= 0:
        raise ValueError("local minibatch size must be positive")
    return local


def _setup_distributed(config: PPOConfig) -> DistributedContext:
    """Initialize torch.distributed and return rank metadata."""
    if config.torch_num_threads is not None:
        torch.set_num_threads(config.torch_num_threads)
    if not config.distributed:
        return DistributedContext(device=_resolve_device(config.device))
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available")
    missing = [name for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK") if name not in os.environ]
    if missing:
        raise RuntimeError(
            "--distributed must be launched by torchrun; missing "
            + ", ".join(missing)
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = _resolve_device(config.device, local_rank=local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=config.ddp_backend)
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def _cleanup_distributed(dist_context: DistributedContext) -> None:
    """Destroy the process group when this run initialized distributed training."""
    if dist_context.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _barrier_if_distributed(dist_context: DistributedContext) -> None:
    """Synchronize ranks when distributed training is active."""
    if dist_context.enabled and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _wrap_model_for_distributed(
    model: BCPolicyNet,
    dist_context: DistributedContext,
) -> torch.nn.Module:
    """Wrap the model in DistributedDataParallel when requested."""
    if not dist_context.enabled:
        return model
    if dist_context.device is not None and dist_context.device.type == "cuda":
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_context.device.index],
            output_device=dist_context.device.index,
            find_unused_parameters=True,
        )
    return torch.nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=True,
    )


def _align_rollout_batch_for_distributed(
    batch: RolloutBatch,
    dist_context: DistributedContext,
) -> tuple[RolloutBatch, int]:
    """Pad local rollout batches so every DDP rank runs the same number of steps."""
    if not dist_context.enabled:
        return batch, 0
    device = _distributed_tensor_device(dist_context)
    local_count = torch.tensor(batch.size, device=device, dtype=torch.long)
    max_count = local_count.clone()
    torch.distributed.all_reduce(max_count, op=torch.distributed.ReduceOp.MAX)
    resolved_max = int(max_count.item())
    if resolved_max <= 0:
        raise ValueError("distributed PPO cannot update with an empty rank rollout")
    if batch.size == resolved_max:
        return batch, 0
    return _pad_rollout_batch(batch, resolved_max), resolved_max - batch.size


def _pad_rollout_batch(batch: RolloutBatch, count: int) -> RolloutBatch:
    """Return a rollout batch padded with zero-weight dummy samples."""
    if count < batch.size:
        raise ValueError("cannot pad rollout batch to a smaller count")
    if count == batch.size:
        return batch
    if batch.size <= 0:
        raise ValueError("cannot pad an empty rollout batch")
    padding = count - batch.size
    return RolloutBatch(
        x=_pad_tensor_with_first_row(batch.x, padding),
        valid_action_mask=_pad_tensor_with_first_row(batch.valid_action_mask, padding),
        action_label=_pad_tensor_with_first_row(batch.action_label, padding),
        old_log_prob=_pad_tensor_with_first_row(batch.old_log_prob, padding),
        old_value=_pad_tensor_with_first_row(batch.old_value, padding),
        enemy_king_label=_pad_tensor_with_first_row(batch.enemy_king_label, padding),
        enemy_king_distance=_pad_tensor_with_first_row(batch.enemy_king_distance, padding),
        enemy_king_valid_mask=_pad_tensor_with_first_row(
            batch.enemy_king_valid_mask,
            padding,
        ),
        rewards=_pad_tensor_with_first_row(batch.rewards, padding),
        done=_pad_tensor_with_first_row(batch.done, padding),
        returns=_pad_tensor_with_first_row(batch.returns, padding),
        advantages=_pad_tensor_with_first_row(batch.advantages, padding),
        sample_weight=torch.cat(
            [
                batch.sample_weight,
                torch.zeros(
                    padding,
                    dtype=batch.sample_weight.dtype,
                    device=batch.sample_weight.device,
                ),
            ],
            dim=0,
        ),
        stream_ids=batch.stream_ids + [(-1, -1)] * padding,
        metadata=batch.metadata
        + [
            {"env_id": -1, "player_id": -1, "turn": -1, "padding": 1}
            for _ in range(padding)
        ],
    )


def _pad_tensor_with_first_row(tensor: torch.Tensor, padding: int) -> torch.Tensor:
    """Append copies of the first row to preserve tensor shapes while padding."""
    if padding <= 0:
        return tensor
    pad_rows = tensor[:1].expand((padding, *tensor.shape[1:])).clone()
    return torch.cat([tensor, pad_rows], dim=0)


def _distributed_any(value: bool, dist_context: DistributedContext) -> bool:
    """Return whether any rank produced a true boolean."""
    if not dist_context.enabled:
        return value
    device = _distributed_tensor_device(dist_context)
    flag = torch.tensor(1 if value else 0, device=device, dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(int(flag.item()))


def _distributed_tensor_device(dist_context: DistributedContext) -> torch.device:
    """Return a device usable by the active distributed backend."""
    if dist_context.device is not None:
        return dist_context.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _aggregate_distributed_metrics(
    metrics: dict[str, float],
    dist_context: DistributedContext,
) -> dict[str, float]:
    """Aggregate per-rank metric dictionaries into one global dictionary."""
    if not dist_context.enabled:
        return dict(metrics)
    gathered: list[dict[str, float]] = [{} for _ in range(dist_context.world_size)]
    torch.distributed.all_gather_object(gathered, dict(metrics))
    if not dist_context.is_rank0:
        return dict(metrics)
    return _merge_metric_dicts(gathered)


def _merge_metric_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    """Merge rank-local metrics using counts for rates and means."""
    result: dict[str, float] = {}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        if key in {
            "completed_episodes",
            "invalid_actions",
            "submitted_actions",
            "rollout_samples",
            "self_play_episodes",
            "pool_episodes",
            "distributed_used_samples",
            "distributed_batch_samples",
            "distributed_padded_samples",
            "distributed_dropped_samples",
        } or (key.startswith("opponent_") and key.endswith("_episodes")):
            result[key] = float(sum(float(row.get(key, 0.0)) for row in rows))
        elif key == "distributed_world_size":
            result[key] = float(max(float(row.get(key, 0.0)) for row in rows))
        elif key == "distributed_rank_envs":
            result["distributed_total_rank_envs"] = float(
                sum(float(row.get(key, 0.0)) for row in rows)
            )
        elif key in {"ppo_early_stop"}:
            result[key] = float(max(float(row.get(key, 0.0)) for row in rows))
        elif key.startswith("time_"):
            result[key] = float(max(float(row.get(key, 0.0)) for row in rows))
        elif key.startswith("reward_") and key.endswith("_mean"):
            result[key] = _weighted_mean(rows, key, "reward_sample_count")
        elif key in {"mean_episode_return", "mean_episode_length", "win_rate_player0", "win_rate_player1", "draw_rate"}:
            result[key] = _weighted_mean(rows, key, "completed_episodes")
        elif key.startswith("learner_pool_"):
            result[key] = _weighted_mean(rows, key, "pool_episodes")
        elif key.startswith("opponent_") and "_learner_" in key:
            prefix = key.split("_learner_", 1)[0]
            result[key] = _weighted_mean(rows, key, f"{prefix}_episodes")
        elif key == "reward_sample_count":
            result[key] = float(sum(float(row.get(key, 0.0)) for row in rows))
        else:
            weight_key = (
                "distributed_used_samples"
                if any("distributed_used_samples" in row for row in rows)
                else "rollout_samples"
            )
            result[key] = _weighted_mean(rows, key, weight_key)
    return result


def _weighted_mean(rows: list[dict[str, float]], key: str, weight_key: str) -> float:
    """Return a weighted mean with an unweighted fallback."""
    weighted_sum = 0.0
    total_weight = 0.0
    values = []
    for row in rows:
        if key not in row:
            continue
        value = float(row[key])
        values.append(value)
        weight = float(row.get(weight_key, 0.0))
        if weight > 0:
            weighted_sum += value * weight
            total_weight += weight
    if total_weight > 0:
        return float(weighted_sum / total_weight)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _binary_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute Bernoulli entropy from logits without constructing distributions."""
    probabilities = torch.sigmoid(logits)
    return (
        F.binary_cross_entropy_with_logits(
            logits,
            probabilities,
            reduction="none",
        )
    )


def _load_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[BCPolicyNet, ObservationEncoder, int, list[str]]:
    """Create a BCPolicyNet and load policy weights from a checkpoint."""
    policy_mode = str(checkpoint.get("policy_mode", "joint"))
    if policy_mode != POLICY_MODE:
        raise ValueError(f"PPO requires policy_mode={POLICY_MODE}, got {policy_mode}")
    history_length = int(checkpoint.get("history_length", 1))
    encoder = ObservationEncoder(history_length=history_length)
    input_channels = int(checkpoint["input_channels"])
    if input_channels != encoder.num_channels:
        raise ValueError(
            f"checkpoint input_channels={input_channels} does not match "
            f"encoder channels={encoder.num_channels}"
        )
    base_channels = int(checkpoint.get("base_channels", 64))
    model = BCPolicyNet(input_channels, base_channels=base_channels).to(device)
    missing_optional = load_model_state_compatible(model, checkpoint["model_state"])
    return model, encoder, base_channels, missing_optional


def _reset_value_head(model: BCPolicyNet) -> None:
    """Reinitialize the untrained BC value head before fresh PPO training."""
    with torch.no_grad():
        for module in model.value_head.modules():
            reset = getattr(module, "reset_parameters", None)
            if reset is not None:
                reset()
        final_layer = model.value_head[-1]
        if getattr(final_layer, "bias", None) is not None:
            final_layer.bias.zero_()


def _save_ppo_checkpoint(
    config: PPOConfig,
    encoder: ObservationEncoder,
    model: BCPolicyNet,
    optimizer: torch.optim.Optimizer,
    *,
    metrics: dict[str, float],
    metric_log_path: Path,
    base_channels: int,
    global_update: int,
    rollout_steps_seen: int,
    output_path: str | Path | None = None,
    dist_context: DistributedContext | None = None,
    rank_envs: int | None = None,
) -> None:
    """Persist PPO checkpoint and a JSON sidecar."""
    if dist_context is None:
        dist_context = DistributedContext()
    output = Path(config.output_path if output_path is None else output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    distributed_metadata = {
        "distributed": config.distributed,
        "distributed_world_size": dist_context.world_size,
        "distributed_rank": dist_context.rank,
        "num_envs_scope": config.num_envs_scope,
        "rank_envs": rank_envs,
    }
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "input_channels": encoder.num_channels,
        "feature_names": list(encoder.channel_names),
        "history_length": encoder.history_length,
        "base_channels": base_channels,
        "policy_mode": POLICY_MODE,
        "algo": "ppo",
        "init_checkpoint": config.init_checkpoint,
        "ppo_config": asdict(config),
        "metrics": metrics,
        "metrics_output_path": str(metric_log_path),
        "tensorboard_log_dir": config.tensorboard_log_dir,
        "global_update": global_update,
        "rollout_steps_seen": rollout_steps_seen,
        **distributed_metadata,
    }
    torch.save(payload, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(output),
                "algo": "ppo",
                "policy_mode": POLICY_MODE,
                "init_checkpoint": config.init_checkpoint,
                "ppo_config": asdict(config),
                "metrics": metrics,
                "metrics_output_path": str(metric_log_path),
                "tensorboard_log_dir": config.tensorboard_log_dir,
                "global_update": global_update,
                "rollout_steps_seen": rollout_steps_seen,
                **distributed_metadata,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _resolve_device(device: str, *, local_rank: int | None = None) -> torch.device:
    """Resolve auto/cpu/cuda device strings."""
    if device == "auto":
        if local_rank is not None and torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if local_rank is not None and device == "cuda":
        return torch.device(f"cuda:{local_rank}")
    return torch.device(device)


def _resolve_metrics_output_path(config: PPOConfig) -> Path:
    """Resolve the PPO metrics JSONL path."""
    if config.metrics_output_path is not None:
        return Path(config.metrics_output_path)
    output = Path(config.output_path)
    return output.with_suffix(output.suffix + ".metrics.jsonl")


def _periodic_checkpoint_path(config: PPOConfig, update: int) -> Path:
    """Return the periodic checkpoint path for an update number."""
    output = Path(config.output_path)
    return output.with_name(f"{output.stem}_update{update:05d}{output.suffix}")


def _should_save_periodic_checkpoint(config: PPOConfig, update: int) -> bool:
    """Return whether a periodic checkpoint should be written now."""
    return config.checkpoint_every_updates > 0 and update % config.checkpoint_every_updates == 0


def _write_metric_row(
    handle: Any,
    *,
    update: int,
    rollout_steps_seen: int,
    learning_rate: float,
    metrics: dict[str, float],
) -> None:
    """Append one PPO metric row to a JSONL file."""
    row = {
        "phase": "train",
        "update": update,
        "rollout_steps_seen": rollout_steps_seen,
        "lr": learning_rate,
        **metrics,
    }
    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    handle.flush()


def _create_tensorboard_writer(log_dir: str | None) -> Any | None:
    """Create a TensorBoard writer if a log directory is configured."""
    if log_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        try:
            from tensorboardX import SummaryWriter  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TensorBoard logging requested but neither torch.utils.tensorboard "
                "nor tensorboardX is installed"
            ) from exc
    return SummaryWriter(log_dir)


def _write_tensorboard_scalars(
    writer: Any | None,
    *,
    step: int,
    learning_rate: float,
    metrics: dict[str, float],
) -> None:
    """Write PPO metrics to TensorBoard."""
    if writer is None:
        return
    writer.add_scalar("train/lr", learning_rate, step)
    for key, value in metrics.items():
        writer.add_scalar(f"train/{key}", value, step)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer tensors to the selected device after loading state."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible starts."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(config: PPOConfig) -> None:
    """Validate PPO config values that would otherwise fail deep in training."""
    if config.updates < 0:
        raise ValueError("updates must be nonnegative")
    if config.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if config.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if config.ppo_epochs <= 0:
        raise ValueError("ppo_epochs must be positive")
    if config.minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if config.map_size < 2:
        raise ValueError("map_size must be at least 2")
    if config.max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if config.enemy_king_loss_weight < 0:
        raise ValueError("enemy_king_loss_weight must be nonnegative")
    if config.enemy_king_distance_loss_weight < 0:
        raise ValueError("enemy_king_distance_loss_weight must be nonnegative")
    if config.enemy_king_distance_cap <= 0:
        raise ValueError("enemy_king_distance_cap must be positive")
    if config.num_envs_scope not in {"global", "per-rank"}:
        raise ValueError("num_envs_scope must be 'global' or 'per-rank'")
    if config.torch_num_threads is not None and config.torch_num_threads <= 0:
        raise ValueError("torch_num_threads must be positive")
    if config.distributed and not config.ddp_backend:
        raise ValueError("ddp_backend must be nonempty in distributed mode")


def _mean(values: list[float] | list[int]) -> float:
    """Return a JSON-safe mean with 0 for empty inputs."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _rate(count: int, total: int) -> float:
    """Return a JSON-safe rate with 0 for an empty denominator."""
    if total <= 0:
        return 0.0
    return float(count / total)


def _safe_divide(numerator: float, denominator: int) -> float:
    """Return a JSON-safe ratio with 0 for an empty denominator."""
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


class _MetricAccumulator:
    """Accumulate weighted scalar metrics over minibatches."""

    def __init__(self) -> None:
        """Create an empty metric accumulator."""
        self.totals: dict[str, float] = {}
        self.count = 0.0

    def update(self, *, batch_size: float, **metrics: float) -> None:
        """Add a metric row weighted by minibatch size."""
        if batch_size <= 0:
            return
        self.count += batch_size
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * batch_size

    def summary(self) -> dict[str, float]:
        """Return weighted mean metrics."""
        if self.count == 0:
            return {}
        return {
            key: value / self.count
            for key, value in self.totals.items()
        }
