from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "torch and tqdm are required for enemy king predictor training. Install with: "
        ".venv/bin/python -m pip install -r requirements-train.txt"
    ) from exc

from generals_bot.training.dataset import (
    MixedReplayBCDataset,
    REPLAY_FILTER_RULES,
    RawReplayBCDataset,
    collate_bc_samples,
)
from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.losses import enemy_king_prediction_loss
from generals_bot.training.model import EnemyKingPredictorNet, PREDICTOR_MODEL_TYPE


@dataclass(frozen=True)
class EnemyKingPredictorTrainConfig:
    """Configuration for a standalone enemy king predictor training run."""

    input_path: str
    output_path: str
    selfplay_input_path: str | None = None
    split: str = "train"
    limit_replays: int | None = None
    selfplay_limit_replays: int | None = None
    replay_versions: tuple[int, ...] | None = None
    replay_filter_rule: str = "none"
    selfplay_replay_filter_rule: str = "none"
    raw_source_weight: float = 1.0
    selfplay_source_weight: float = 1.0
    mixed_stop_when_any_exhausted: bool = True
    steps: int = 200
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    base_channels: int = 64
    history_length: int = 1
    enemy_king_distance_loss_weight: float = 1.0
    enemy_king_distance_cap: float = 30.0
    checkpoint_every_steps: int = 0
    grad_clip: float = 1.0
    device: str = "auto"
    metrics_output_path: str | None = None
    tensorboard_log_dir: str | None = None
    resume_from: str | None = None
    override_resume_lr: bool = False
    shuffle_buffer_size: int = 8192
    shuffle_shards: bool = False
    seed: int = 0
    val_split: str = "val"
    val_limit_replays: int | None = 100
    val_every_steps: int = 500
    val_batches: int = 20


def train_enemy_king_predictor(config: EnemyKingPredictorTrainConfig) -> dict[str, Any]:
    """Train a standalone enemy king predictor and save a checkpoint."""
    _validate_config(config)
    encoder = ObservationEncoder(history_length=config.history_length)
    device = _resolve_device(config.device)
    metric_log_path = _resolve_metrics_output_path(config)
    metric_log_path.parent.mkdir(parents=True, exist_ok=True)
    loader = _make_loader(
        config,
        encoder=encoder,
        split=config.split,
        limit_replays=config.limit_replays,
        shuffle_buffer_size=config.shuffle_buffer_size,
        shuffle_shards=config.shuffle_shards,
        seed=config.seed,
    )
    checkpoint = _load_resume_checkpoint(config, device)
    base_channels = _effective_base_channels(config, checkpoint)
    model = EnemyKingPredictorNet(
        encoder.num_channels,
        base_channels=base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    global_step = 0
    samples_seen = 0
    if checkpoint is not None:
        _validate_resume_checkpoint(checkpoint, encoder)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        _move_optimizer_state(optimizer, device)
        if config.override_resume_lr:
            _override_optimizer_lr(optimizer, config.lr)
        global_step = int(checkpoint.get("global_step", 0))
        samples_seen = int(checkpoint.get("samples_seen", 0))
    tensorboard_writer = _create_tensorboard_writer(config.tensorboard_log_dir)
    metrics = _MetricAccumulator()
    last_val_metrics: dict[str, float] = {}
    last_val_step: int | None = None
    remaining_steps = max(0, config.steps - global_step)
    progress = tqdm(
        loader,
        total=remaining_steps,
        desc="train_enemy_king_predictor",
        leave=False,
    )
    try:
        with metric_log_path.open(_metric_log_mode(config, metric_log_path), encoding="utf-8") as metric_log:
            for batch in progress:
                if global_step >= config.steps:
                    break
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, step_metrics, batch_size = _batch_loss_and_metrics(model, batch, config)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

                global_step += 1
                samples_seen += batch_size
                metrics.add(step_metrics, weight=batch_size)
                _write_metric_row(
                    metric_log,
                    phase="train",
                    step=global_step,
                    samples_seen=samples_seen,
                    learning_rate=_current_lr(optimizer),
                    metrics=step_metrics,
                )
                _write_tensorboard_scalars(
                    tensorboard_writer,
                    phase="train",
                    step=global_step,
                    learning_rate=_current_lr(optimizer),
                    metrics=step_metrics,
                )
                progress.set_postfix({key: f"{value:.4f}" for key, value in step_metrics.items()})

                if _should_validate(config, global_step):
                    last_val_step = global_step
                    last_val_metrics = evaluate_enemy_king_predictor(
                        model,
                        config,
                        encoder=encoder,
                        device=device,
                    )
                    _write_metric_row(
                        metric_log,
                        phase="val",
                        step=global_step,
                        samples_seen=samples_seen,
                        learning_rate=_current_lr(optimizer),
                        metrics=last_val_metrics,
                    )
                    _write_tensorboard_scalars(
                        tensorboard_writer,
                        phase="val",
                        step=global_step,
                        learning_rate=_current_lr(optimizer),
                        metrics=last_val_metrics,
                    )

                if _should_save_periodic_checkpoint(config, global_step):
                    _save_checkpoint(
                        config,
                        encoder,
                        model,
                        optimizer,
                        metrics.averages(),
                        metric_log_path,
                        base_channels=base_channels,
                        global_step=global_step,
                        samples_seen=samples_seen,
                        last_val_metrics=last_val_metrics,
                        output_path=_periodic_checkpoint_path(config, global_step),
                    )

            if (
                _validation_enabled(config)
                and global_step > 0
                and global_step != last_val_step
            ):
                last_val_metrics = evaluate_enemy_king_predictor(
                    model,
                    config,
                    encoder=encoder,
                    device=device,
                )
                _write_metric_row(
                    metric_log,
                    phase="val",
                    step=global_step,
                    samples_seen=samples_seen,
                    learning_rate=_current_lr(optimizer),
                    metrics=last_val_metrics,
                )
                _write_tensorboard_scalars(
                    tensorboard_writer,
                    phase="val",
                    step=global_step,
                    learning_rate=_current_lr(optimizer),
                    metrics=last_val_metrics,
                )
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()

    final_metrics = metrics.averages()
    _save_checkpoint(
        config,
        encoder,
        model,
        optimizer,
        final_metrics,
        metric_log_path,
        base_channels=base_channels,
        global_step=global_step,
        samples_seen=samples_seen,
        last_val_metrics=last_val_metrics,
    )
    summary = {
        **final_metrics,
        "checkpoint_path": config.output_path,
        "metrics_output_path": str(metric_log_path),
        "global_step": global_step,
        "samples_seen": samples_seen,
        "last_val_metrics": last_val_metrics,
    }
    return summary


def evaluate_enemy_king_predictor(
    model: EnemyKingPredictorNet,
    config: EnemyKingPredictorTrainConfig,
    *,
    encoder: ObservationEncoder,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate predictor metrics over a limited validation stream."""
    if config.val_batches <= 0:
        return {}
    loader = _make_loader(
        config,
        encoder=encoder,
        split=config.val_split,
        limit_replays=config.val_limit_replays,
        shuffle_buffer_size=0,
        shuffle_shards=False,
        seed=config.seed,
    )
    was_training = model.training
    model.eval()
    metrics = _MetricAccumulator()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= config.val_batches:
                break
            batch = _move_batch(batch, device)
            _, step_metrics, batch_size = _batch_loss_and_metrics(model, batch, config)
            metrics.add(step_metrics, weight=batch_size)
    if was_training:
        model.train()
    return metrics.averages()


def _batch_loss_and_metrics(
    model: EnemyKingPredictorNet,
    batch: dict[str, Any],
    config: EnemyKingPredictorTrainConfig,
) -> tuple[torch.Tensor, dict[str, float], int]:
    """Compute standalone predictor loss and metrics for one batch."""
    output = model(batch["x"])
    loss, metrics = enemy_king_prediction_loss(
        output["enemy_king_logits"],
        labels=batch["enemy_king_label"],
        valid_mask=batch["enemy_king_valid_mask"],
        distances=batch["enemy_king_distance"],
        distance_weight=config.enemy_king_distance_loss_weight,
        distance_cap=config.enemy_king_distance_cap,
    )
    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics, int(batch["x"].shape[0])


def _save_checkpoint(
    config: EnemyKingPredictorTrainConfig,
    encoder: ObservationEncoder,
    model: EnemyKingPredictorNet,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    metric_log_path: Path,
    *,
    base_channels: int,
    global_step: int,
    samples_seen: int,
    last_val_metrics: dict[str, float],
    output_path: str | Path | None = None,
) -> None:
    """Persist predictor model, optimizer, config, and metrics."""
    output = Path(config.output_path if output_path is None else output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": PREDICTOR_MODEL_TYPE,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "input_channels": encoder.num_channels,
        "feature_names": list(encoder.channel_names),
        "history_length": encoder.history_length,
        "base_channels": base_channels,
        "train_config": asdict(config),
        "metrics": metrics,
        "metrics_output_path": str(metric_log_path),
        "tensorboard_log_dir": config.tensorboard_log_dir,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "last_val_metrics": last_val_metrics,
    }
    torch.save(payload, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "model_type": PREDICTOR_MODEL_TYPE,
                "train_config": asdict(config),
                "metrics": metrics,
                "checkpoint_path": str(output),
                "metrics_output_path": str(metric_log_path),
                "tensorboard_log_dir": config.tensorboard_log_dir,
                "global_step": global_step,
                "samples_seen": samples_seen,
                "last_val_metrics": last_val_metrics,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def _make_loader(
    config: EnemyKingPredictorTrainConfig,
    *,
    encoder: ObservationEncoder,
    split: str,
    limit_replays: int | None,
    shuffle_buffer_size: int,
    shuffle_shards: bool,
    seed: int,
) -> DataLoader:
    """Build a replay DataLoader for predictor training or validation."""
    raw_dataset = RawReplayBCDataset(
        config.input_path,
        split=split,
        limit_replays=limit_replays,
        include_pass=True,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_shards=shuffle_shards,
        seed=seed,
        replay_filter_rule=config.replay_filter_rule,
        replay_versions=config.replay_versions,
        history_length=config.history_length,
    )
    dataset = raw_dataset
    if config.selfplay_input_path is not None:
        selfplay_dataset = RawReplayBCDataset(
            config.selfplay_input_path,
            split=split,
            limit_replays=config.selfplay_limit_replays,
            include_pass=True,
            shuffle_buffer_size=shuffle_buffer_size,
            shuffle_shards=shuffle_shards,
            seed=seed + 1009,
            replay_filter_rule=config.selfplay_replay_filter_rule,
            replay_versions=None,
            history_length=config.history_length,
        )
        dataset = MixedReplayBCDataset(
            [raw_dataset, selfplay_dataset],
            weights=[
                config.raw_source_weight,
                config.selfplay_source_weight,
            ],
            seed=seed + 2003,
            stop_when_any_exhausted=config.mixed_stop_when_any_exhausted,
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=0,
        collate_fn=lambda batch: collate_bc_samples(batch, encoder=encoder),
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move predictor tensor fields to the requested device."""
    result = dict(batch)
    for key in (
        "x",
        "enemy_king_label",
        "enemy_king_distance",
        "enemy_king_valid_mask",
    ):
        result[key] = batch[key].to(device)
    return result


class _MetricAccumulator:
    """Weighted average accumulator for scalar metrics."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._weight = 0.0

    def add(self, metrics: dict[str, float], *, weight: int) -> None:
        self._weight += float(weight)
        for key, value in metrics.items():
            self._totals[key] = self._totals.get(key, 0.0) + float(value) * float(weight)

    def averages(self) -> dict[str, float]:
        if self._weight == 0:
            return {}
        return {key: value / self._weight for key, value in self._totals.items()}


def _validate_config(config: EnemyKingPredictorTrainConfig) -> None:
    """Validate standalone predictor training options."""
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.base_channels <= 0:
        raise ValueError("base_channels must be positive")
    if config.history_length <= 0:
        raise ValueError("history_length must be positive")
    if config.limit_replays is not None and config.limit_replays < 0:
        raise ValueError("limit_replays must be nonnegative")
    if config.selfplay_limit_replays is not None and config.selfplay_limit_replays < 0:
        raise ValueError("selfplay_limit_replays must be nonnegative")
    if config.raw_source_weight < 0:
        raise ValueError("raw_source_weight must be nonnegative")
    if config.selfplay_source_weight < 0:
        raise ValueError("selfplay_source_weight must be nonnegative")
    if config.selfplay_replay_filter_rule not in REPLAY_FILTER_RULES:
        raise ValueError(
            f"unknown selfplay_replay_filter_rule: {config.selfplay_replay_filter_rule}"
        )
    if config.selfplay_input_path is None and config.raw_source_weight <= 0:
        raise ValueError("raw_source_weight must be positive without a self-play source")
    if (
        config.selfplay_input_path is not None
        and config.raw_source_weight + config.selfplay_source_weight <= 0
    ):
        raise ValueError("mixed replay source weights must have positive total")
    if config.enemy_king_distance_loss_weight < 0:
        raise ValueError("enemy_king_distance_loss_weight must be nonnegative")
    if config.enemy_king_distance_cap <= 0:
        raise ValueError("enemy_king_distance_cap must be positive")
    if config.grad_clip < 0:
        raise ValueError("grad_clip must be nonnegative")


def _load_resume_checkpoint(
    config: EnemyKingPredictorTrainConfig,
    device: torch.device,
) -> dict[str, Any] | None:
    """Load a standalone predictor checkpoint for resume training."""
    if config.resume_from is None:
        return None
    return torch.load(config.resume_from, map_location=device)


def _validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    encoder: ObservationEncoder,
) -> None:
    """Validate that a checkpoint is compatible with the configured encoder."""
    model_type = str(checkpoint.get("model_type", ""))
    if model_type != PREDICTOR_MODEL_TYPE:
        raise ValueError(
            f"resume checkpoint model_type={model_type!r} does not match "
            f"{PREDICTOR_MODEL_TYPE!r}"
        )
    input_channels = int(checkpoint.get("input_channels", encoder.num_channels))
    if input_channels != encoder.num_channels:
        raise ValueError(
            f"checkpoint input_channels={input_channels} does not match "
            f"encoder channels={encoder.num_channels}"
        )
    history_length = int(checkpoint.get("history_length", 1))
    if history_length != encoder.history_length:
        raise ValueError(
            f"checkpoint history_length={history_length} does not match "
            f"configured history_length={encoder.history_length}"
        )
    if "optimizer_state" not in checkpoint:
        raise ValueError("resume checkpoint is missing optimizer_state")


def _effective_base_channels(
    config: EnemyKingPredictorTrainConfig,
    checkpoint: dict[str, Any] | None,
) -> int:
    """Return the model width for a fresh or resumed predictor run."""
    if checkpoint is None:
        return config.base_channels
    return int(checkpoint.get("base_channels", config.base_channels))


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors to the active device after loading."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _override_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Set all resumed optimizer parameter groups to a new learning rate."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _resolve_metrics_output_path(config: EnemyKingPredictorTrainConfig) -> Path:
    """Resolve per-step metrics JSONL output path."""
    if config.metrics_output_path is not None:
        return Path(config.metrics_output_path)
    output = Path(config.output_path)
    return output.with_suffix(output.suffix + ".metrics.jsonl")


def _metric_log_mode(config: EnemyKingPredictorTrainConfig, metric_log_path: Path) -> str:
    """Choose append mode only when resuming into an existing metrics file."""
    if config.resume_from is not None and metric_log_path.exists():
        return "a"
    return "w"


def _write_metric_row(
    metric_log: Any,
    *,
    phase: str,
    step: int,
    samples_seen: int,
    learning_rate: float,
    metrics: dict[str, float],
) -> None:
    """Append one metric row to a JSONL log."""
    row = {
        "phase": phase,
        "step": step,
        "samples_seen": samples_seen,
        **metrics,
        "lr": learning_rate,
    }
    metric_log.write(json.dumps(row, separators=(",", ":")) + "\n")
    metric_log.flush()


def _create_tensorboard_writer(log_dir: str | None) -> Any | None:
    """Create a TensorBoard SummaryWriter when requested."""
    if log_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tensorboard is required for --tensorboard-log-dir. "
            "Install with: .venv/bin/python -m pip install -r requirements-train.txt"
        ) from exc
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir)


def _write_tensorboard_scalars(
    writer: Any | None,
    *,
    phase: str,
    step: int,
    learning_rate: float,
    metrics: dict[str, float],
) -> None:
    """Write scalar metrics to TensorBoard when enabled."""
    if writer is None:
        return
    for name, value in metrics.items():
        writer.add_scalar(f"{phase}/{name}", value, step)
    writer.add_scalar(f"{phase}/lr", learning_rate, step)


def _should_validate(config: EnemyKingPredictorTrainConfig, global_step: int) -> bool:
    """Return whether validation should run after this step."""
    return _validation_enabled(config) and global_step % config.val_every_steps == 0


def _validation_enabled(config: EnemyKingPredictorTrainConfig) -> bool:
    """Return whether validation is configured."""
    return config.val_batches > 0 and config.val_every_steps > 0


def _should_save_periodic_checkpoint(
    config: EnemyKingPredictorTrainConfig,
    global_step: int,
) -> bool:
    """Return whether a periodic checkpoint should be written."""
    return config.checkpoint_every_steps > 0 and global_step % config.checkpoint_every_steps == 0


def _periodic_checkpoint_path(
    config: EnemyKingPredictorTrainConfig,
    global_step: int,
) -> Path:
    """Return the output path for one periodic checkpoint."""
    output = Path(config.output_path)
    return output.with_name(f"{output.stem}_step{global_step:05d}{output.suffix}")


def _resolve_device(device: str) -> torch.device:
    """Resolve auto/cpu/cuda into a torch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate of the first optimizer group."""
    return float(optimizer.param_groups[0]["lr"])
