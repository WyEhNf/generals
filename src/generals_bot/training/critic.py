from __future__ import annotations

import json
import math
import random
from collections import deque
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
        "torch and tqdm are required for critic training. Install with: "
        ".venv/bin/python -m pip install -r requirements-train.txt"
    ) from exc

from generals_bot.sim import GameConfig, GeneralsEnv, MapGenerationConfig, generate_1v1_map
from generals_bot.sim.types import Action, Observation, PassAction
from generals_bot.training.action_codec import valid_action_mask
from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.model import BCPolicyNet, POLICY_MODE, load_model_state_compatible
from generals_bot.training.ppo import (
    explained_variance,
    sample_policy_actions,
    terminal_reward,
)


DEFAULT_CRITIC_CHECKPOINT = "checkpoints/ppo_1h/model.pt"


@dataclass(frozen=True)
class CriticConfig:
    """Configuration for value-head outcome training."""

    checkpoint_path: str = DEFAULT_CRITIC_CHECKPOINT
    output_path: str = "artifacts/value_head_outcome_v1.pt"
    metrics_output_path: str | None = None
    tensorboard_log_dir: str | None = None
    device: str = "auto"
    seed: int = 0
    updates: int = 1
    num_envs: int = 16
    episodes_per_update: int = 16
    eval_episodes: int = 0
    eval_every_updates: int = 1
    sample_stride: int = 4
    train_epochs: int = 4
    minibatch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    map_size: int = 18
    max_turns: int = 800
    train_value_head_only: bool = True
    reset_value_head: bool = False
    checkpoint_every_updates: int = 0


@dataclass
class OutcomeBatch:
    """Outcome-labelled value samples from completed self-play games."""

    x: torch.Tensor
    target: torch.Tensor
    metadata: list[dict[str, int]]

    @property
    def size(self) -> int:
        """Return the number of value samples in the batch."""
        return int(self.target.numel())


@dataclass
class OutcomeCollection:
    """A value training batch plus aggregate self-play statistics."""

    batch: OutcomeBatch
    stats: dict[str, float]


def train_value_head(config: CriticConfig) -> dict[str, Any]:
    """Train a model value head on complete-game outcome targets."""
    _validate_config(config)
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    metric_log_path = _resolve_metrics_output_path(config)
    metric_log_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model, encoder, base_channels = _load_model_from_checkpoint(checkpoint, device)
    if config.reset_value_head:
        _reset_value_head(model)
    trainable_parameters = _select_trainable_parameters(model, config)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    collector = OutcomeSelfPlayCollector(
        model=model,
        encoder=encoder,
        config=config,
        device=device,
        seed_offset=0,
    )
    eval_collector = OutcomeSelfPlayCollector(
        model=model,
        encoder=encoder,
        config=config,
        device=device,
        seed_offset=1_000_000,
    )
    tensorboard_writer = _create_tensorboard_writer(config.tensorboard_log_dir)
    samples_seen = 0
    last_metrics: dict[str, float] = {}

    try:
        with metric_log_path.open("w", encoding="utf-8") as metric_log:
            progress = tqdm(range(config.updates), desc="train_value_head", leave=False)
            for update_index in progress:
                update = update_index + 1
                collection = collector.collect(config.episodes_per_update)
                samples_seen += collection.batch.size
                before = evaluate_value_batch(model, collection.batch, device, prefix="train_before")
                train_metrics = train_value_head_on_batch(
                    model=model,
                    optimizer=optimizer,
                    batch=collection.batch,
                    config=config,
                    device=device,
                )
                after = evaluate_value_batch(model, collection.batch, device, prefix="train_after")
                last_metrics = {
                    **collection.stats,
                    **before,
                    **train_metrics,
                    **after,
                }
                if _should_eval(config, update):
                    eval_collection = eval_collector.collect(config.eval_episodes)
                    last_metrics.update({
                        f"eval_collection/{key}": value
                        for key, value in eval_collection.stats.items()
                    })
                    last_metrics.update({
                        f"eval/{key}": value
                        for key, value in evaluate_value_batch(
                            model,
                            eval_collection.batch,
                            device,
                            prefix="",
                        ).items()
                    })
                _write_metric_row(
                    metric_log,
                    update=update,
                    samples_seen=samples_seen,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    metrics=last_metrics,
                )
                _write_tensorboard_scalars(
                    tensorboard_writer,
                    step=update,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    metrics=last_metrics,
                )
                if _should_save_periodic_checkpoint(config, update):
                    _save_critic_checkpoint(
                        config,
                        encoder,
                        model,
                        optimizer,
                        metrics=last_metrics,
                        metric_log_path=metric_log_path,
                        base_channels=base_channels,
                        update=update,
                        samples_seen=samples_seen,
                        output_path=_periodic_checkpoint_path(config, update),
                    )
                progress.set_postfix({
                    key: round(value, 4)
                    for key, value in last_metrics.items()
                    if key
                    in {
                        "train_after/value_loss",
                        "train_after/explained_variance",
                        "eval/value_loss",
                        "eval/explained_variance",
                    }
                })
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()

    _save_critic_checkpoint(
        config,
        encoder,
        model,
        optimizer,
        metrics=last_metrics,
        metric_log_path=metric_log_path,
        base_channels=base_channels,
        update=config.updates,
        samples_seen=samples_seen,
    )
    return {
        **last_metrics,
        "checkpoint_path": config.output_path,
        "metrics_output_path": str(metric_log_path),
        "global_update": config.updates,
        "samples_seen": samples_seen,
    }


class OutcomeSelfPlayCollector:
    """Collect complete self-play episodes and label states by final outcome."""

    def __init__(
        self,
        *,
        model: BCPolicyNet,
        encoder: ObservationEncoder,
        config: CriticConfig,
        device: torch.device,
        seed_offset: int,
    ) -> None:
        """Create fixed-size self-play environments for outcome collection."""
        self.model = model
        self.encoder = encoder
        self.config = config
        self.device = device
        self.next_map_seed = config.seed + seed_offset
        self.histories: dict[tuple[int, int], deque[Observation]] = {}
        self.envs: list[GeneralsEnv] = []
        self.observations: list[dict[int, Observation]] = []
        self.episode_lengths = [0 for _ in range(config.num_envs)]
        self.episode_buffers: list[list[tuple[torch.Tensor, int, int]]] = [
            [] for _ in range(config.num_envs)
        ]
        for env_id in range(config.num_envs):
            self._reset_env(env_id)

    def collect(self, episode_count: int) -> OutcomeCollection:
        """Collect complete episodes and return outcome-labelled value samples."""
        was_training = self.model.training
        self.model.eval()
        x_rows: list[torch.Tensor] = []
        target_rows: list[float] = []
        metadata: list[dict[str, int]] = []
        completed_lengths: list[int] = []
        player0_wins = 0
        player1_wins = 0
        draws = 0
        submitted_actions = 0
        invalid_actions = 0
        completed = 0

        with torch.no_grad():
            while completed < episode_count:
                items = self._current_items(mutate_history=True)
                x_cpu, valid_mask_cpu, keys, observations = self._encode_items(items)
                sampled = sample_policy_actions(
                    self.model,
                    x_cpu.to(self.device, dtype=torch.float32),
                    valid_mask_cpu.to(self.device),
                    observations=observations,
                )
                for row, (env_id, player), observation in zip(
                    x_cpu,
                    keys,
                    observations,
                    strict=True,
                ):
                    if observation.turn % self.config.sample_stride == 0:
                        self.episode_buffers[env_id].append(
                            (row.to(torch.float16).cpu(), player, observation.turn)
                        )

                actions_by_env: dict[int, dict[int, Action | PassAction]] = {}
                for (env_id, player), action in zip(keys, sampled.actions, strict=True):
                    actions_by_env.setdefault(env_id, {})[player] = action

                for env_id, actions in actions_by_env.items():
                    _, _, done, info = self.envs[env_id].step(actions)
                    self.observations[env_id] = self.envs[env_id].observations()
                    submitted_actions += sum(
                        len(actions)
                        for actions in info["submitted_actions"].values()
                    )
                    invalid_actions += sum(
                        1
                        for executed in info["executed_actions"]
                        if not executed.valid
                    )
                    self.episode_lengths[env_id] += 1
                    if done:
                        winner = info["winner"]
                        completed += 1
                        completed_lengths.append(self.episode_lengths[env_id])
                        if winner == 0:
                            player0_wins += 1
                        elif winner == 1:
                            player1_wins += 1
                        else:
                            draws += 1
                        for row, player, turn in self.episode_buffers[env_id]:
                            x_rows.append(row)
                            target_rows.append(terminal_reward(winner, player))
                            metadata.append({
                                "env_id": env_id,
                                "player_id": player,
                                "turn": turn,
                            })
                        self._reset_env(env_id)

        self.model.train(was_training)
        if not x_rows:
            raise ValueError("outcome collection produced no samples")
        batch = OutcomeBatch(
            x=torch.stack(x_rows),
            target=torch.tensor(target_rows, dtype=torch.float32),
            metadata=metadata,
        )
        stats = {
            "completed_episodes": float(len(completed_lengths)),
            "mean_episode_length": _mean(completed_lengths),
            "win_rate_player0": _rate(player0_wins, len(completed_lengths)),
            "win_rate_player1": _rate(player1_wins, len(completed_lengths)),
            "draw_rate": _rate(draws, len(completed_lengths)),
            "invalid_actions": float(invalid_actions),
            "submitted_actions": float(submitted_actions),
            "outcome_samples": float(batch.size),
            "target_mean": float(batch.target.mean()),
            "target_std": float(batch.target.std(unbiased=False)),
        }
        return OutcomeCollection(batch=batch, stats=stats)

    def _current_items(self, *, mutate_history: bool) -> list[tuple[tuple[int, int], Observation]]:
        """Return current alive-player observations, optionally appending history."""
        items = []
        for env_id, env in enumerate(self.envs):
            for player in env.alive_players:
                observation = self.observations[env_id][player]
                key = (env_id, player)
                if mutate_history:
                    self.histories.setdefault(
                        key,
                        deque(maxlen=self.encoder.history_length),
                    ).append(observation)
                items.append((key, observation))
        return items

    def _encode_items(
        self,
        items: list[tuple[tuple[int, int], Observation]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[Observation]]:
        """Encode observations and valid action masks for policy inference."""
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

    def _reset_env(self, env_id: int) -> None:
        """Replace one self-play environment and clear its episode buffer."""
        game_config = self._next_game_config()
        env = GeneralsEnv(game_config)
        observations = env.reset()
        if env_id < len(self.envs):
            self.envs[env_id] = env
            self.observations[env_id] = observations
        else:
            self.envs.append(env)
            self.observations.append(observations)
        self.episode_lengths[env_id] = 0
        self.episode_buffers[env_id].clear()
        self.histories.pop((env_id, 0), None)
        self.histories.pop((env_id, 1), None)

    def _next_game_config(self) -> GameConfig:
        """Generate the next fixed-size map for self-play."""
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


def train_value_head_on_batch(
    *,
    model: BCPolicyNet,
    optimizer: torch.optim.Optimizer,
    batch: OutcomeBatch,
    config: CriticConfig,
    device: torch.device,
) -> dict[str, float]:
    """Optimize the value head on one outcome-labelled batch."""
    model.train()
    accumulator = _MetricAccumulator()
    sample_count = batch.size
    for _ in range(config.train_epochs):
        permutation = torch.randperm(sample_count)
        for start in range(0, sample_count, config.minibatch_size):
            indexes = permutation[start : start + config.minibatch_size]
            x = batch.x[indexes].to(device=device, dtype=torch.float32)
            target = batch.target[indexes].to(device=device)
            value = model(x)["value"].reshape(-1)
            loss = F.mse_loss(value, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                config.grad_clip,
            )
            optimizer.step()
            accumulator.update(batch_size=int(indexes.numel()), value_loss=float(loss.detach().cpu()))
    return {
        f"train/{key}": value
        for key, value in accumulator.summary().items()
    }


def evaluate_value_batch(
    model: BCPolicyNet,
    batch: OutcomeBatch,
    device: torch.device,
    *,
    prefix: str,
    chunk_size: int = 2048,
) -> dict[str, float]:
    """Evaluate value loss and explained variance for one value batch."""
    was_training = model.training
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, batch.size, chunk_size):
            x = batch.x[start : start + chunk_size].to(device=device, dtype=torch.float32)
            predictions.append(model(x)["value"].detach().cpu().reshape(-1))
    model.train(was_training)
    prediction = torch.cat(predictions).to(torch.float32)
    target = batch.target.to(torch.float32)
    loss = F.mse_loss(prediction, target)
    metrics = {
        "value_loss": float(loss),
        "explained_variance": explained_variance(target, prediction),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std(unbiased=False)),
        "target_mean": float(target.mean()),
        "target_std": float(target.std(unbiased=False)),
    }
    if not prefix:
        return metrics
    return {
        f"{prefix}/{key}": value
        for key, value in metrics.items()
    }


def _validate_config(config: CriticConfig) -> None:
    """Validate critic training configuration values."""
    if config.updates < 1:
        raise ValueError("updates must be positive")
    if config.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if config.episodes_per_update < 1:
        raise ValueError("episodes_per_update must be positive")
    if config.eval_episodes < 0:
        raise ValueError("eval_episodes must be nonnegative")
    if config.eval_every_updates < 1:
        raise ValueError("eval_every_updates must be positive")
    if config.sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    if config.train_epochs < 1:
        raise ValueError("train_epochs must be positive")
    if config.minibatch_size < 1:
        raise ValueError("minibatch_size must be positive")
    if config.lr <= 0:
        raise ValueError("lr must be positive")
    if config.map_size < 3:
        raise ValueError("map_size must be at least 3")
    if config.max_turns < 1:
        raise ValueError("max_turns must be positive")


def _select_trainable_parameters(
    model: BCPolicyNet,
    config: CriticConfig,
) -> list[torch.nn.Parameter]:
    """Freeze policy parameters by default and return trainable parameters."""
    if not config.train_value_head_only:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return list(model.parameters())
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.value_head.parameters():
        parameter.requires_grad = True
    return list(model.value_head.parameters())


def _reset_value_head(model: BCPolicyNet) -> None:
    """Reinitialize value head weights while keeping a zero output bias."""
    with torch.no_grad():
        for module in model.value_head.modules():
            reset = getattr(module, "reset_parameters", None)
            if reset is not None:
                reset()
        final_layer = model.value_head[-1]
        if getattr(final_layer, "bias", None) is not None:
            final_layer.bias.zero_()


def _load_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[BCPolicyNet, ObservationEncoder, int]:
    """Create a BCPolicyNet and load weights from a compatible checkpoint."""
    policy_mode = str(checkpoint.get("policy_mode", "joint"))
    if policy_mode != POLICY_MODE:
        raise ValueError(f"critic training requires policy_mode={POLICY_MODE}, got {policy_mode}")
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
    load_model_state_compatible(model, checkpoint["model_state"])
    return model, encoder, base_channels


def _save_critic_checkpoint(
    config: CriticConfig,
    encoder: ObservationEncoder,
    model: BCPolicyNet,
    optimizer: torch.optim.Optimizer,
    *,
    metrics: dict[str, float],
    metric_log_path: Path,
    base_channels: int,
    update: int,
    samples_seen: int,
    output_path: str | Path | None = None,
) -> None:
    """Persist a value-head training checkpoint and JSON sidecar."""
    output = Path(config.output_path if output_path is None else output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "input_channels": encoder.num_channels,
        "feature_names": list(encoder.channel_names),
        "history_length": encoder.history_length,
        "base_channels": base_channels,
        "policy_mode": POLICY_MODE,
        "algo": "value_head_outcome",
        "source_checkpoint": config.checkpoint_path,
        "critic_config": asdict(config),
        "metrics": metrics,
        "metrics_output_path": str(metric_log_path),
        "tensorboard_log_dir": config.tensorboard_log_dir,
        "global_update": update,
        "samples_seen": samples_seen,
    }
    torch.save(payload, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(output),
                "algo": "value_head_outcome",
                "policy_mode": POLICY_MODE,
                "source_checkpoint": config.checkpoint_path,
                "critic_config": asdict(config),
                "metrics": metrics,
                "metrics_output_path": str(metric_log_path),
                "tensorboard_log_dir": config.tensorboard_log_dir,
                "global_update": update,
                "samples_seen": samples_seen,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _resolve_device(device: str) -> torch.device:
    """Resolve auto/cpu/cuda device strings."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_metrics_output_path(config: CriticConfig) -> Path:
    """Resolve the critic metrics JSONL path."""
    if config.metrics_output_path is not None:
        return Path(config.metrics_output_path)
    output = Path(config.output_path)
    return output.with_suffix(output.suffix + ".metrics.jsonl")


def _periodic_checkpoint_path(config: CriticConfig, update: int) -> Path:
    """Return the periodic checkpoint path for an update number."""
    output = Path(config.output_path)
    return output.with_name(f"{output.stem}_update{update:05d}{output.suffix}")


def _should_save_periodic_checkpoint(config: CriticConfig, update: int) -> bool:
    """Return whether a periodic checkpoint should be written now."""
    return config.checkpoint_every_updates > 0 and update % config.checkpoint_every_updates == 0


def _should_eval(config: CriticConfig, update: int) -> bool:
    """Return whether to run held-out outcome collection for this update."""
    return config.eval_episodes > 0 and update % config.eval_every_updates == 0


def _write_metric_row(
    handle: Any,
    *,
    update: int,
    samples_seen: int,
    learning_rate: float,
    metrics: dict[str, float],
) -> None:
    """Append one critic metric row to a JSONL file."""
    row = {
        "phase": "train",
        "update": update,
        "samples_seen": samples_seen,
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
    """Write critic metrics to TensorBoard."""
    if writer is None:
        return
    writer.add_scalar("train/lr", learning_rate, step)
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean(values: list[int] | list[float]) -> float:
    """Return the arithmetic mean for a possibly empty list."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _rate(count: int, total: int) -> float:
    """Return count divided by total, or 0 for empty totals."""
    if total <= 0:
        return 0.0
    return float(count / total)


class _MetricAccumulator:
    """Accumulate weighted scalar metrics."""

    def __init__(self) -> None:
        """Create an empty metric accumulator."""
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, *, batch_size: int, **metrics: float) -> None:
        """Add batch-weighted scalar metrics."""
        self.count += batch_size
        for key, value in metrics.items():
            if math.isfinite(value):
                self.totals[key] = self.totals.get(key, 0.0) + value * batch_size

    def summary(self) -> dict[str, float]:
        """Return average metrics accumulated so far."""
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.totals.items()}
