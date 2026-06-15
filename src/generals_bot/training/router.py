from __future__ import annotations

import json
import random
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "torch and tqdm are required for router PPO training. Install with: "
        ".venv/bin/python -m pip install -r requirements-train.txt"
    ) from exc

from generals_bot.agents.base import Policy
from generals_bot.sim import GeneralsEnv, MapGenerationConfig, generate_1v1_map
from generals_bot.sim.types import Action, GameConfig, Observation, PassAction
from generals_bot.training.batched_policy import ActionKey, ActionLike, BatchedBCPolicyRunner
from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.model import (
    BCPolicyNet,
    ROUTER_MODEL_TYPE,
    RouterPolicyNet,
    load_model_state_compatible,
)
from generals_bot.training.ppo import (
    DistributedContext,
    Matchup,
    _MetricAccumulator,
    _barrier_if_distributed,
    _cleanup_distributed,
    _create_tensorboard_writer,
    _distributed_any,
    _distributed_tensor_device,
    _local_minibatch_size,
    _rank_env_assignment,
    _resolve_device,
    _setup_distributed,
    _weighted_float_mean,
    _weighted_float_std,
    _weighted_mean,
    _weighted_tensor_mean,
    _write_metric_row,
    _write_tensorboard_scalars,
    compute_gae,
    explained_variance,
    load_opponent_pool_config,
    terminal_reward,
)
from generals_bot.training.rewards import (
    REWARD_COMPONENTS,
    RewardCalculator,
    RewardState,
    load_reward_config,
)


DEFAULT_EXPERT_CONFIG = "configs/routers/four_expert_router_v1.json"
DEFAULT_INIT_ROUTER_FROM = "checkpoints/city_decay085_700/model.pt"


@dataclass(frozen=True)
class RouterPPOConfig:
    """Configuration for PPO training of a frozen-expert router."""

    expert_config: str = DEFAULT_EXPERT_CONFIG
    init_router_from: str = DEFAULT_INIT_ROUTER_FROM
    output_path: str = "artifacts/router/four_expert_terminal_v1.pt"
    resume_from: str | None = None
    metrics_output_path: str | None = None
    tensorboard_log_dir: str | None = None
    opponent_pool_config: str | None = None
    reward_config_path: str | None = None
    device: str = "auto"
    seed: int = 0
    updates: int = 1
    num_envs: int = 32
    rollout_steps: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 1024
    lr: float = 1e-5
    router_head_lr: float = 3e-5
    weight_decay: float = 0.0
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    target_kl: float = 0.02
    grad_clip: float = 1.0
    map_size: int = 18
    max_turns: int = 800
    checkpoint_every_updates: int = 0
    distributed: bool = False
    num_envs_scope: str = "global"
    ddp_backend: str = "nccl"
    torch_num_threads: int | None = None


@dataclass(frozen=True)
class RouterExpertSpec:
    """One frozen expert checkpoint that the router can choose."""

    id: str
    path: Path


@dataclass(frozen=True)
class RouterExpertConfig:
    """Resolved router expert configuration."""

    config_path: Path
    experts: tuple[RouterExpertSpec, ...]


@dataclass(frozen=True)
class SampledExperts:
    """Router samples and policy statistics for one learner batch."""

    labels: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    entropy: torch.Tensor
    max_probabilities: torch.Tensor


@dataclass
class RouterRolloutBatch:
    """One router PPO rollout batch."""

    x: torch.Tensor
    expert_label: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    rewards: torch.Tensor
    done: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    sample_weight: torch.Tensor
    stream_ids: list[tuple[int, int]]
    metadata: list[dict[str, int]]

    @property
    def size(self) -> int:
        """Return the padded number of samples."""
        return int(self.expert_label.numel())

    @property
    def real_size(self) -> int:
        """Return the number of non-padding samples."""
        return int(round(float(self.sample_weight.sum().item())))


@dataclass(frozen=True)
class RouterRolloutResult:
    """A collected router PPO rollout and simulator statistics."""

    batch: RouterRolloutBatch
    stats: dict[str, float]


class FrozenExpertActionRunner:
    """Run all frozen expert checkpoints for the same observation streams."""

    def __init__(
        self,
        expert_config: RouterExpertConfig,
        *,
        device: str,
    ) -> None:
        """Load all expert policies and prepare per-expert histories."""
        self.expert_ids = tuple(spec.id for spec in expert_config.experts)
        self.runners = {
            spec.id: BatchedBCPolicyRunner(spec.path, device=device)
            for spec in expert_config.experts
        }

    def reset_key(self, key: ActionKey) -> None:
        """Clear one stream in every expert runner."""
        for runner in self.runners.values():
            runner.reset_key(key)

    def reset_all(self) -> None:
        """Clear all expert histories."""
        for runner in self.runners.values():
            runner.reset_all()

    def act_many_all(
        self,
        items: Iterable[tuple[ActionKey, Observation]],
    ) -> dict[ActionKey, list[ActionLike]]:
        """Return every expert's greedy action for each item."""
        resolved_items = list(items)
        actions_by_expert = {
            expert_id: self.runners[expert_id].act_many(resolved_items)
            for expert_id in self.expert_ids
        }
        return {
            key: [actions_by_expert[expert_id][key] for expert_id in self.expert_ids]
            for key, _ in resolved_items
        }


class BatchedRouterPolicyRunner:
    """Run a trained router checkpoint over multiple observation streams."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
    ) -> None:
        """Load a router checkpoint and its frozen experts."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if str(checkpoint.get("model_type", "")) != ROUTER_MODEL_TYPE:
            raise ValueError(
                f"router checkpoint must have model_type={ROUTER_MODEL_TYPE!r}"
            )
        self.expert_ids = tuple(str(value) for value in checkpoint["expert_ids"])
        expert_config = _expert_config_from_checkpoint(checkpoint)
        self.expert_runner = FrozenExpertActionRunner(
            expert_config,
            device=str(self.device),
        )
        self.history_length = int(checkpoint.get("history_length", 1))
        self.encoder = ObservationEncoder(history_length=self.history_length)
        input_channels = int(checkpoint["input_channels"])
        if input_channels != self.encoder.num_channels:
            raise ValueError(
                f"router input_channels={input_channels} does not match "
                f"encoder channels={self.encoder.num_channels}"
            )
        self.model = RouterPolicyNet(
            input_channels,
            expert_count=len(self.expert_ids),
            base_channels=int(checkpoint.get("base_channels", 64)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.histories: dict[ActionKey, deque[Observation]] = {}

    def reset_key(self, key: ActionKey) -> None:
        """Clear one stream's router and expert histories."""
        self.histories.pop(key, None)
        self.expert_runner.reset_key(key)

    def reset_all(self) -> None:
        """Clear all stored histories."""
        self.histories.clear()
        self.expert_runner.reset_all()

    def summary(self) -> dict[str, Any]:
        """Return current runner metadata for evaluation summaries."""
        return {
            "device": str(self.device),
            "policy_mode": "router",
            "history_length": self.history_length,
            "expert_ids": list(self.expert_ids),
        }

    @torch.no_grad()
    def act_many(
        self,
        items: Iterable[tuple[ActionKey, Observation]],
    ) -> dict[ActionKey, ActionLike]:
        """Select deterministic router actions for many keyed observations."""
        resolved_items = list(items)
        if not resolved_items:
            return {}
        expert_actions = self.expert_runner.act_many_all(resolved_items)
        x, keys = self._encode_items(resolved_items)
        output = self.model(x.to(self.device, dtype=torch.float32))
        labels = output["expert_logits"].argmax(dim=1).detach().cpu().tolist()
        return {
            key: expert_actions[key][int(label)]
            for key, label in zip(keys, labels, strict=True)
        }

    def _encode_items(
        self,
        items: list[tuple[ActionKey, Observation]],
    ) -> tuple[torch.Tensor, list[ActionKey]]:
        """Encode observations while updating router histories."""
        encoded = []
        keys = []
        for key, observation in items:
            history = self.histories.setdefault(
                key,
                deque(maxlen=self.history_length),
            )
            history.append(observation)
            encoded.append(self.encoder.encode_history(tuple(history)))
            keys.append(key)
        x = torch.from_numpy(np.ascontiguousarray(np.stack(encoded, axis=0)))
        return x, keys


class RouterModelPolicy(Policy):
    """Serial simulator policy wrapper for a trained router checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
    ) -> None:
        """Load a router checkpoint for serial match execution."""
        self.runner = BatchedRouterPolicyRunner(checkpoint_path, device=device)
        self.player_id: int | None = None

    def reset(self, player_id: int, config: GameConfig) -> None:
        """Remember the player id and clear stored histories."""
        del config
        self.player_id = player_id
        self.runner.reset_all()

    def act(self, observation: Observation) -> ActionLike:
        """Return the router-selected expert action for one observation."""
        key = (observation.player_id,)
        return self.runner.act_many([(key, observation)])[key]


class RouterRolloutCollector:
    """Collect fixed-size PPO rollouts for a frozen-expert router."""

    def __init__(
        self,
        *,
        model: RouterPolicyNet,
        encoder: ObservationEncoder,
        expert_config: RouterExpertConfig,
        config: RouterPPOConfig,
        device: torch.device,
        num_envs: int | None = None,
        seed: int | None = None,
        env_id_offset: int = 0,
    ) -> None:
        """Create environments, experts, and rollout state."""
        self.model = model
        self.encoder = encoder
        self.expert_config = expert_config
        self.config = config
        self.device = device
        self.num_envs = config.num_envs if num_envs is None else int(num_envs)
        self.env_id_offset = int(env_id_offset)
        self.expert_runner = FrozenExpertActionRunner(
            expert_config,
            device=str(device),
        )
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
        self.next_map_seed = config.seed if seed is None else int(seed)
        for env_id in range(self.num_envs):
            self._reset_env(env_id)

    def collect(self) -> RouterRolloutResult:
        """Collect one rollout and compute router GAE targets."""
        was_training = self.model.training
        self.model.eval()
        x_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        log_prob_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
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
        expert_counts = {expert_id: 0 for expert_id in self.expert_runner.expert_ids}
        turn_bucket_counts = {
            bucket: {expert_id: 0 for expert_id in self.expert_runner.expert_ids}
            for bucket in _ROUTER_TURN_BUCKETS
        }
        max_probability_sum = 0.0
        max_probability_count = 0
        opponent_stats = {
            opponent_id: {"episodes": 0, "learner_wins": 0, "learner_losses": 0, "learner_draws": 0}
            for opponent_id in self.opponent_runners
        }

        with torch.no_grad():
            for _ in range(self.config.rollout_steps):
                learner_items = self._learner_items(mutate_history=True)
                x_cpu, keys, observations = self._encode_items(learner_items)
                sampled = sample_router_experts(
                    self.model,
                    x_cpu.to(self.device, dtype=torch.float32),
                )
                expert_actions = self.expert_runner.act_many_all(learner_items)

                x_rows.extend(row.to(torch.float16).cpu() for row in x_cpu)
                label_rows.extend(row.cpu() for row in sampled.labels)
                log_prob_rows.extend(row.cpu() for row in sampled.log_probs)
                value_rows.extend(row.cpu() for row in sampled.values)
                stream_ids.extend(keys)
                for (env_id, player), observation in zip(keys, observations, strict=True):
                    metadata.append({
                        "env_id": self.env_id_offset + env_id,
                        "player_id": player,
                        "turn": observation.turn,
                    })
                for label, probability, observation in zip(
                    sampled.labels.detach().cpu().tolist(),
                    sampled.max_probabilities.detach().cpu().tolist(),
                    observations,
                    strict=True,
                ):
                    expert_id = self.expert_runner.expert_ids[int(label)]
                    expert_counts[expert_id] += 1
                    turn_bucket_counts[_router_turn_bucket(observation.turn)][expert_id] += 1
                    max_probability_sum += float(probability)
                    max_probability_count += 1

                actions_by_env: dict[int, dict[int, Action | PassAction]] = {}
                for (env_id, player), label in zip(keys, sampled.labels.detach().cpu().tolist(), strict=True):
                    selected = expert_actions[(env_id, player)][int(label)]
                    actions_by_env.setdefault(env_id, {})[player] = selected
                for (env_id, player), action in self._opponent_actions().items():
                    actions_by_env.setdefault(env_id, {})[player] = action

                rewards_by_stream: dict[tuple[int, int], float] = {}
                done_by_env: dict[int, bool] = {}
                for env_id, actions in actions_by_env.items():
                    matchup = self.matchups[env_id]
                    env = self.envs[env_id]
                    before_reward_state = RewardState.from_env(env)
                    _, _, done, info = env.step(actions)
                    after_reward_state = RewardState.from_env(env)
                    reward_breakdowns = self.reward_calculator.rewards(
                        before=before_reward_state,
                        after=after_reward_state,
                        executed_actions=info["executed_actions"],
                        winner=info["winner"],
                        done=bool(done),
                    )
                    self.observations[env_id] = env.observations()
                    submitted_actions += sum(
                        len(value)
                        for value in info["submitted_actions"].values()
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
                    reward_rows.append(rewards_by_stream.get((env_id, player), 0.0))
                    done_rows.append(done_by_env.get(env_id, False))

            last_values = self._bootstrap_values()

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
        batch = RouterRolloutBatch(
            x=torch.stack(x_rows),
            expert_label=torch.stack(label_rows).to(torch.long),
            old_log_prob=torch.stack(log_prob_rows).to(torch.float32),
            old_value=old_values,
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
            "mean_episode_return": _mean(completed_returns),
            "mean_episode_length": _mean(completed_lengths),
            "win_rate_player0": _rate(player0_wins, len(completed_lengths)),
            "win_rate_player1": _rate(player1_wins, len(completed_lengths)),
            "draw_rate": _rate(draws, len(completed_lengths)),
            "invalid_actions": float(invalid_actions),
            "submitted_actions": float(submitted_actions),
            "rollout_samples": float(batch.real_size),
            "reward_sample_count": float(reward_sample_count),
            "self_play_episodes": float(self_play_episodes),
            "pool_episodes": float(pool_episodes),
            "learner_pool_win_rate": _rate(learner_pool_wins, pool_episodes),
            "learner_pool_loss_rate": _rate(learner_pool_losses, pool_episodes),
            "learner_pool_draw_rate": _rate(learner_pool_draws, pool_episodes),
            "router_max_expert_probability": _safe_divide(
                max_probability_sum,
                max_probability_count,
            ),
        }
        for component in REWARD_COMPONENTS:
            stats[f"reward_{component}_mean"] = _safe_divide(
                reward_component_sums[component],
                reward_sample_count,
            )
        _add_router_selection_metrics(stats, expert_counts, turn_bucket_counts)
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
        return RouterRolloutResult(batch=batch, stats=stats)

    def _learner_items(self, *, mutate_history: bool) -> list[tuple[tuple[int, int], Observation]]:
        """Return current learner observations, optionally appending router history."""
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
        """Return actions from frozen opponent-pool policies."""
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
        """Return whether a player stream is controlled by the router."""
        return player in self.matchups[env_id].learner_players

    def _encode_items(
        self,
        items: list[tuple[tuple[int, int], Observation]],
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[Observation]]:
        """Encode router observations for one same-size rollout batch."""
        encoded = []
        keys = []
        observations = []
        for key, observation in items:
            history = self.histories.get(key, deque(maxlen=self.encoder.history_length))
            encoded.append(self.encoder.encode_history(tuple(history) or (observation,)))
            keys.append(key)
            observations.append(observation)
        x = torch.from_numpy(np.ascontiguousarray(np.stack(encoded, axis=0)))
        return x, keys, observations

    def _bootstrap_values(self) -> dict[tuple[int, int], float]:
        """Estimate V(s_next) for streams that remain alive after the rollout."""
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
        if not keys:
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
        matchup = self._sample_matchup()
        if env_id < len(self.matchups):
            self.matchups[env_id] = matchup
        else:
            self.matchups.append(matchup)
        self.histories.pop((env_id, 0), None)
        self.histories.pop((env_id, 1), None)
        self.expert_runner.reset_key((env_id, 0))
        self.expert_runner.reset_key((env_id, 1))
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
        """Sample a new router self-play or frozen-pool matchup."""
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


def train_router_ppo(config: RouterPPOConfig) -> dict[str, Any]:
    """Train a router policy that chooses among frozen expert checkpoints."""
    _validate_router_config(config)
    expert_config = load_router_expert_config(config.expert_config)
    dist_context = _setup_distributed(config)
    device = dist_context.device or _resolve_device(config.device)
    _seed_everything(config.seed)
    metric_log_path = _resolve_metrics_output_path(config)
    if dist_context.is_rank0:
        metric_log_path.parent.mkdir(parents=True, exist_ok=True)

    model, encoder, base_channels, global_update, rollout_steps_seen, checkpoint = (
        _load_or_init_router_model(config, expert_config, device)
    )
    optimizer = _build_router_optimizer(model, config)
    if config.resume_from is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        _move_optimizer_state(optimizer, device)
    update_model = _wrap_router_model_for_distributed(model, dist_context)

    _seed_everything(config.seed + dist_context.rank * 1_000_000)
    local_envs, env_id_offset = _rank_env_assignment(
        config.num_envs,
        world_size=dist_context.world_size,
        rank=dist_context.rank,
        scope=config.num_envs_scope,
    )
    collector = RouterRolloutCollector(
        model=model,
        encoder=encoder,
        expert_config=expert_config,
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
                desc="train_router_ppo",
                leave=False,
                disable=not dist_context.is_rank0,
            )
            for _ in progress:
                update_start = time.perf_counter()
                collect_start = time.perf_counter()
                rollout = collector.collect()
                collect_seconds = time.perf_counter() - collect_start
                aligned_batch, padded_samples = _align_router_batch_for_distributed(
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
                update_metrics = router_ppo_update(
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
                last_metrics = _aggregate_router_metrics(local_metrics, dist_context)
                global_used_samples = int(
                    last_metrics.get("distributed_used_samples", aligned_batch.real_size)
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
                    _save_router_checkpoint(
                        config,
                        expert_config,
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
        _save_router_checkpoint(
            config,
            expert_config,
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


def load_router_expert_config(config_path: str | Path) -> RouterExpertConfig:
    """Load and validate a router expert configuration."""
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("router expert config must contain version=1")
    raw_experts = data.get("experts", [])
    if not isinstance(raw_experts, list) or not raw_experts:
        raise ValueError("router expert config must contain a nonempty experts list")
    experts = []
    seen_ids = set()
    for raw in raw_experts:
        if not isinstance(raw, dict):
            raise ValueError("each router expert must be an object")
        expert_id = str(raw.get("id", "")).strip()
        if not expert_id:
            raise ValueError("each router expert must have a nonempty id")
        if expert_id in seen_ids:
            raise ValueError(f"duplicate router expert id: {expert_id}")
        seen_ids.add(expert_id)
        if "path" not in raw:
            raise ValueError(f"router expert {expert_id} is missing path")
        expert_path = Path(str(raw["path"]))
        if not expert_path.is_absolute():
            expert_path = path.parent / expert_path
        expert_path = expert_path.resolve()
        if not expert_path.exists():
            raise FileNotFoundError(f"router expert checkpoint not found: {expert_path}")
        experts.append(RouterExpertSpec(id=expert_id, path=expert_path))
    return RouterExpertConfig(config_path=path, experts=tuple(experts))


def sample_router_experts(model: torch.nn.Module, x: torch.Tensor) -> SampledExperts:
    """Sample expert ids from the router policy."""
    output = model(x)
    logits = output["expert_logits"]
    probabilities = torch.softmax(logits, dim=1)
    labels = torch.multinomial(probabilities, num_samples=1).squeeze(1)
    log_probs, entropy, values, max_probabilities = evaluate_router_experts_from_output(
        output,
        labels,
    )
    return SampledExperts(
        labels=labels.detach(),
        log_probs=log_probs.detach(),
        values=values.detach(),
        entropy=entropy.detach(),
        max_probabilities=max_probabilities.detach(),
    )


def evaluate_router_experts_from_output(
    output: dict[str, torch.Tensor],
    expert_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute router log-probs, entropy, value, and max expert probability."""
    logits = output["expert_logits"]
    log_probs_by_expert = F.log_softmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1)
    selected_log_probs = log_probs_by_expert.gather(
        1,
        expert_labels[:, None],
    ).squeeze(1)
    entropy = -(probabilities * log_probs_by_expert).sum(dim=1)
    values = output["value"].reshape(-1)
    max_probabilities = probabilities.max(dim=1).values
    return selected_log_probs, entropy, values, max_probabilities


def router_ppo_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: RouterRolloutBatch,
    config: RouterPPOConfig,
    device: torch.device,
    dist_context: DistributedContext | None = None,
) -> dict[str, float]:
    """Run PPO epochs over one router rollout batch."""
    model.train()
    if dist_context is None:
        dist_context = DistributedContext(device=device)
    advantages = _normalize_router_advantages(
        batch.advantages,
        sample_weight=batch.sample_weight,
        dist_context=dist_context,
    )
    if batch.size == 0 or batch.real_size == 0:
        raise ValueError("cannot update router PPO with an empty rollout")
    minibatch_size = _local_minibatch_size(config.minibatch_size, dist_context.world_size)
    accumulator = _MetricAccumulator()
    early_stop = False

    for _ in range(config.ppo_epochs):
        permutation = torch.randperm(batch.size)
        for start in range(0, batch.size, minibatch_size):
            indexes = permutation[start : start + minibatch_size]
            minibatch = _slice_router_batch(batch, advantages, indexes, device)
            output = model(minibatch["x"])
            log_probs, entropy, values, max_probabilities = evaluate_router_experts_from_output(
                output,
                minibatch["expert_label"],
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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            approx_kl = _weighted_tensor_mean(
                minibatch["old_log_prob"] - log_probs,
                sample_weight,
            )
            clip_fraction = _weighted_tensor_mean(
                (torch.abs(ratio - 1.0) > config.clip_range).to(torch.float32),
                sample_weight,
            )
            metrics = {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy_mean.detach().cpu()),
                "approx_kl": float(approx_kl.detach().cpu()),
                "clip_fraction": float(clip_fraction.detach().cpu()),
                "router_max_expert_probability_update": float(
                    _weighted_tensor_mean(max_probabilities, sample_weight).detach().cpu()
                ),
            }
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


def _load_or_init_router_model(
    config: RouterPPOConfig,
    expert_config: RouterExpertConfig,
    device: torch.device,
) -> tuple[RouterPolicyNet, ObservationEncoder, int, int, int, dict[str, Any]]:
    """Load a router checkpoint or initialize one from a policy checkpoint."""
    if config.resume_from is not None:
        checkpoint = torch.load(config.resume_from, map_location=device)
        if str(checkpoint.get("model_type", "")) != ROUTER_MODEL_TYPE:
            raise ValueError("resume_from must point to a router checkpoint")
        expert_ids = tuple(str(value) for value in checkpoint["expert_ids"])
        expected_ids = tuple(spec.id for spec in expert_config.experts)
        if expert_ids != expected_ids:
            raise ValueError(
                f"resume experts {expert_ids} do not match config experts {expected_ids}"
            )
        history_length = int(checkpoint.get("history_length", 1))
        encoder = ObservationEncoder(history_length=history_length)
        input_channels = int(checkpoint["input_channels"])
        if input_channels != encoder.num_channels:
            raise ValueError(
                f"router input_channels={input_channels} does not match "
                f"encoder channels={encoder.num_channels}"
            )
        base_channels = int(checkpoint.get("base_channels", 64))
        model = RouterPolicyNet(
            input_channels,
            expert_count=len(expert_ids),
            base_channels=base_channels,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        return (
            model,
            encoder,
            base_channels,
            int(checkpoint.get("global_update", 0)),
            int(checkpoint.get("rollout_steps_seen", 0)),
            checkpoint,
        )

    source = torch.load(config.init_router_from, map_location=device)
    history_length = int(source.get("history_length", 1))
    encoder = ObservationEncoder(history_length=history_length)
    input_channels = int(source["input_channels"])
    if input_channels != encoder.num_channels:
        raise ValueError(
            f"source input_channels={input_channels} does not match "
            f"encoder channels={encoder.num_channels}"
        )
    base_channels = int(source.get("base_channels", 64))
    model = RouterPolicyNet(
        input_channels,
        expert_count=len(expert_config.experts),
        base_channels=base_channels,
    ).to(device)
    _warm_start_router_from_policy(model, source["model_state"])
    _zero_init_router_expert_output(model)
    return model, encoder, base_channels, 0, 0, {}


def _warm_start_router_from_policy(
    model: RouterPolicyNet,
    source_state: dict[str, torch.Tensor],
) -> None:
    """Copy matching trunk and value tensors from a policy checkpoint."""
    target_state = model.state_dict()
    copied = {}
    allowed_prefixes = ("stem.", "down1.", "down2.", "bottleneck.", "up2.", "up1.", "value_head.")
    for key, value in source_state.items():
        if not key.startswith(allowed_prefixes):
            continue
        if key in target_state and target_state[key].shape == value.shape:
            copied[key] = value
    missing = [key for key in target_state if key.startswith(allowed_prefixes) and key not in copied]
    if missing:
        raise RuntimeError(f"router warm-start missing compatible tensors: {missing}")
    target_state.update(copied)
    model.load_state_dict(target_state)


def _zero_init_router_expert_output(model: RouterPolicyNet) -> None:
    """Make the initial expert distribution uniform by zeroing final logits."""
    final_layer = model.expert_head[-1]
    with torch.no_grad():
        final_layer.weight.zero_()
        if final_layer.bias is not None:
            final_layer.bias.zero_()


def _build_router_optimizer(
    model: RouterPolicyNet,
    config: RouterPPOConfig,
) -> torch.optim.Optimizer:
    """Create optimizer parameter groups with a separate router-head lr."""
    head_param_ids = {id(parameter) for parameter in model.expert_head.parameters()}
    head_params = [parameter for parameter in model.parameters() if id(parameter) in head_param_ids]
    body_params = [parameter for parameter in model.parameters() if id(parameter) not in head_param_ids]
    return torch.optim.AdamW(
        [
            {"params": body_params, "lr": config.lr},
            {"params": head_params, "lr": config.router_head_lr},
        ],
        weight_decay=config.weight_decay,
    )


def _wrap_router_model_for_distributed(
    model: RouterPolicyNet,
    dist_context: DistributedContext,
) -> torch.nn.Module:
    """Wrap the router in DDP without unused-parameter graph traversal."""
    if not dist_context.enabled:
        return model
    if dist_context.device is not None and dist_context.device.type == "cuda":
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_context.device.index],
            output_device=dist_context.device.index,
        )
    return torch.nn.parallel.DistributedDataParallel(model)


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move optimizer tensors to the training device after loading state."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _slice_router_batch(
    batch: RouterRolloutBatch,
    advantages: torch.Tensor,
    indexes: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move one router minibatch slice to the training device."""
    return {
        "x": batch.x[indexes].to(device=device, dtype=torch.float32),
        "expert_label": batch.expert_label[indexes].to(device=device),
        "old_log_prob": batch.old_log_prob[indexes].to(device=device),
        "old_value": batch.old_value[indexes].to(device=device),
        "returns": batch.returns[indexes].to(device=device),
        "advantage": advantages[indexes].to(device=device),
        "sample_weight": batch.sample_weight[indexes].to(device=device),
    }


def _normalize_router_advantages(
    advantages: torch.Tensor,
    *,
    sample_weight: torch.Tensor,
    dist_context: DistributedContext,
) -> torch.Tensor:
    """Return standardized router advantages over non-padding samples."""
    if dist_context.enabled:
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
    variance = _weighted_tensor_mean(
        (advantages.to(torch.float32) - mean).pow(2),
        sample_weight,
    )
    std = torch.sqrt(variance)
    if float(std) <= 1e-8:
        return advantages - mean
    return (advantages - mean) / (std + 1e-8)


def _align_router_batch_for_distributed(
    batch: RouterRolloutBatch,
    dist_context: DistributedContext,
) -> tuple[RouterRolloutBatch, int]:
    """Pad local router batches so every DDP rank has the same count."""
    if not dist_context.enabled:
        return batch, 0
    device = _distributed_tensor_device(dist_context)
    local_count = torch.tensor(batch.size, device=device, dtype=torch.long)
    max_count = local_count.clone()
    torch.distributed.all_reduce(max_count, op=torch.distributed.ReduceOp.MAX)
    resolved_max = int(max_count.item())
    if resolved_max <= 0:
        raise ValueError("distributed router PPO cannot update with an empty rollout")
    if batch.size == resolved_max:
        return batch, 0
    return _pad_router_batch(batch, resolved_max), resolved_max - batch.size


def _pad_router_batch(batch: RouterRolloutBatch, count: int) -> RouterRolloutBatch:
    """Return a router batch padded with zero-weight dummy samples."""
    if count < batch.size:
        raise ValueError("cannot pad router batch to a smaller count")
    if count == batch.size:
        return batch
    if batch.size <= 0:
        raise ValueError("cannot pad an empty router batch")
    padding = count - batch.size
    return RouterRolloutBatch(
        x=_pad_tensor_with_first_row(batch.x, padding),
        expert_label=_pad_tensor_with_first_row(batch.expert_label, padding),
        old_log_prob=_pad_tensor_with_first_row(batch.old_log_prob, padding),
        old_value=_pad_tensor_with_first_row(batch.old_value, padding),
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


def _aggregate_router_metrics(
    metrics: dict[str, float],
    dist_context: DistributedContext,
) -> dict[str, float]:
    """Aggregate per-rank router metrics into one rank0 dictionary."""
    if not dist_context.enabled:
        return _finalize_router_rates(dict(metrics))
    gathered: list[dict[str, float]] = [{} for _ in range(dist_context.world_size)]
    torch.distributed.all_gather_object(gathered, dict(metrics))
    if not dist_context.is_rank0:
        return dict(metrics)
    result = _merge_router_metric_dicts(gathered)
    return _finalize_router_rates(result)


def _merge_router_metric_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    """Merge router rank-local metric dictionaries."""
    result: dict[str, float] = {}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        if key.endswith("_count") or key in {
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
            "reward_sample_count",
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
        elif key.endswith("_rate"):
            continue
        else:
            weight_key = (
                "distributed_used_samples"
                if any("distributed_used_samples" in row for row in rows)
                else "rollout_samples"
            )
            result[key] = _weighted_mean(rows, key, weight_key)
    return result


def _add_router_selection_metrics(
    stats: dict[str, float],
    expert_counts: dict[str, int],
    turn_bucket_counts: dict[str, dict[str, int]],
) -> None:
    """Add expert selection counts and rates to a metric dictionary."""
    total = sum(expert_counts.values())
    for expert_id, count in expert_counts.items():
        stats[f"router_expert_{expert_id}_count"] = float(count)
        stats[f"router_expert_{expert_id}_rate"] = _rate(count, total)
    for bucket, counts in turn_bucket_counts.items():
        bucket_total = sum(counts.values())
        for expert_id, count in counts.items():
            prefix = f"router_turn_{bucket}_expert_{expert_id}"
            stats[f"{prefix}_count"] = float(count)
            stats[f"{prefix}_rate"] = _rate(count, bucket_total)


def _finalize_router_rates(metrics: dict[str, float]) -> dict[str, float]:
    """Recompute router rates from aggregate counts."""
    expert_ids = sorted(
        key[len("router_expert_") : -len("_count")]
        for key in metrics
        if key.startswith("router_expert_") and key.endswith("_count")
    )
    total = sum(float(metrics.get(f"router_expert_{expert_id}_count", 0.0)) for expert_id in expert_ids)
    for expert_id in expert_ids:
        metrics[f"router_expert_{expert_id}_rate"] = _safe_divide(
            float(metrics.get(f"router_expert_{expert_id}_count", 0.0)),
            total,
        )
    for bucket in _ROUTER_TURN_BUCKETS:
        bucket_total = sum(
            float(metrics.get(f"router_turn_{bucket}_expert_{expert_id}_count", 0.0))
            for expert_id in expert_ids
        )
        for expert_id in expert_ids:
            prefix = f"router_turn_{bucket}_expert_{expert_id}"
            if f"{prefix}_count" in metrics:
                metrics[f"{prefix}_rate"] = _safe_divide(
                    float(metrics.get(f"{prefix}_count", 0.0)),
                    bucket_total,
                )
    return metrics


def _save_router_checkpoint(
    config: RouterPPOConfig,
    expert_config: RouterExpertConfig,
    encoder: ObservationEncoder,
    model: RouterPolicyNet,
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
    """Persist a router PPO checkpoint and JSON sidecar."""
    if dist_context is None:
        dist_context = DistributedContext()
    output = Path(config.output_path if output_path is None else output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    expert_payload = [
        {"id": spec.id, "path": _portable_path(spec.path)}
        for spec in expert_config.experts
    ]
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
        "model_type": ROUTER_MODEL_TYPE,
        "algo": "router_ppo",
        "expert_config": str(expert_config.config_path),
        "experts": expert_payload,
        "expert_ids": [spec.id for spec in expert_config.experts],
        "init_router_from": config.init_router_from,
        "router_config": asdict(config),
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
                "algo": "router_ppo",
                "model_type": ROUTER_MODEL_TYPE,
                "expert_config": str(expert_config.config_path),
                "experts": expert_payload,
                "expert_ids": [spec.id for spec in expert_config.experts],
                "init_router_from": config.init_router_from,
                "router_config": asdict(config),
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


def _expert_config_from_checkpoint(checkpoint: dict[str, Any]) -> RouterExpertConfig:
    """Resolve expert paths stored in a router checkpoint."""
    experts = tuple(
        RouterExpertSpec(id=str(raw["id"]), path=_resolve_checkpoint_path(raw["path"]))
        for raw in checkpoint["experts"]
    )
    return RouterExpertConfig(
        config_path=Path(str(checkpoint.get("expert_config", ""))),
        experts=experts,
    )


def _resolve_checkpoint_path(path_value: object) -> Path:
    """Resolve a checkpoint-stored path relative to the current workspace."""
    path = Path(str(path_value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _portable_path(path: Path) -> str:
    """Return a repo-relative path when possible, otherwise an absolute path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _resolve_metrics_output_path(config: RouterPPOConfig) -> Path:
    """Resolve the router PPO metrics JSONL path."""
    if config.metrics_output_path is not None:
        return Path(config.metrics_output_path)
    output = Path(config.output_path)
    return output.with_suffix(output.suffix + ".metrics.jsonl")


def _periodic_checkpoint_path(config: RouterPPOConfig, update: int) -> Path:
    """Return the periodic router checkpoint path for an update number."""
    output = Path(config.output_path)
    return output.with_name(f"{output.stem}_update{update:05d}{output.suffix}")


def _should_save_periodic_checkpoint(config: RouterPPOConfig, update: int) -> bool:
    """Return whether a periodic router checkpoint should be written now."""
    return config.checkpoint_every_updates > 0 and update % config.checkpoint_every_updates == 0


def _router_turn_bucket(turn: int) -> str:
    """Return the router metrics bucket for a turn number."""
    if turn < 50:
        return "0_50"
    if turn < 100:
        return "50_100"
    if turn < 300:
        return "100_300"
    return "300_plus"


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_router_config(config: RouterPPOConfig) -> None:
    """Validate router PPO configuration values."""
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
    if config.lr <= 0:
        raise ValueError("lr must be positive")
    if config.router_head_lr <= 0:
        raise ValueError("router_head_lr must be positive")
    if config.map_size < 2:
        raise ValueError("map_size must be at least 2")
    if config.max_turns <= 0:
        raise ValueError("max_turns must be positive")
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


def _rate(numerator: int | float, denominator: int | float) -> float:
    """Return a JSON-safe rate with 0 for empty denominators."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Return numerator divided by denominator, or zero when denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


_ROUTER_TURN_BUCKETS = ("0_50", "50_100", "100_300", "300_plus")
