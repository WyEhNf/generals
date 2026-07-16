from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, IterableDataset
    from tqdm import tqdm
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "torch and tqdm are required for training. Install with: "
        ".venv/bin/python -m pip install -r requirements-train.txt"
    ) from exc

from generals_bot.training.dataset import RawReplayBCDataset, collate_bc_samples
from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.model import (
    BCPolicyNet,
    POLICY_MODE,
    load_model_state_compatible,
)
from generals_bot.training.losses import enemy_king_prediction_loss
from generals_bot.training.action_codec import ACTION_PLANES


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for a behavior cloning training run."""

    input_path: str
    output_path: str
    split: str = "train"
    limit_replays: int | None = None
    replay_versions: tuple[int, ...] | None = None
    steps: int = 200
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    base_channels: int = 64
    history_length: int = 1
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.0
    distance_loss_weight: float = 0.0
    distance_loss_cap: float = 10.0
    distance_split_penalty: float = 0.5
    actionness_loss_weight: float = 0.0
    actionness_pass_weight: float = 0.05
    enemy_king_loss_weight: float = 0.0
    enemy_king_distance_loss_weight: float = 1.0
    enemy_king_distance_cap: float = 30.0
    freeze_non_king_head: bool = False
    late_pass_turn: int = 30
    late_pass_weight: float = 0.2
    replay_filter_rule: str = "none"
    checkpoint_every_steps: int = 0
    grad_clip: float = 1.0
    device: str = "auto"
    metrics_output_path: str | None = None
    tensorboard_log_dir: str | None = None
    init_checkpoint: str | None = None
    resume_from: str | None = None
    skip_train_batches: int = 0
    override_resume_lr: bool = False
    shuffle_buffer_size: int = 8192
    shuffle_shards: bool = False
    seed: int = 0
    val_split: str = "val"
    val_limit_replays: int | None = 100
    val_every_steps: int = 500
    val_batches: int = 20
    warmup_steps: int = 0
    min_lr_ratio: float = 1.0


def train_bc(config: TrainConfig) -> dict[str, Any]:
    """Train a BC policy model and save a checkpoint."""
    _validate_config(config)
    _seed_training(config.seed)
    if config.skip_train_batches < 0:
        raise ValueError("skip_train_batches must be nonnegative")
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
    checkpoint = _load_model_checkpoint(config, device)
    base_channels = _effective_base_channels(config, checkpoint)
    model = _create_model(config, encoder, checkpoint, device, base_channels)
    _apply_parameter_freeze(model, config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    global_step = 0
    samples_seen = 0
    if checkpoint is not None:
        missing_optional = load_model_state_compatible(model, checkpoint["model_state"])
        if config.resume_from is not None:
            if missing_optional:
                raise ValueError(
                    "resume checkpoint is missing newly added model parameters; "
                    "use --init-checkpoint to initialize from it with a fresh optimizer"
                )
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _move_optimizer_state(optimizer, device)
            if config.override_resume_lr:
                _override_optimizer_lr(optimizer, config.lr)
            global_step, samples_seen = _resume_progress(config, checkpoint)

    model.train()
    metrics = _MetricAccumulator()
    remaining_steps = max(0, config.steps - global_step)
    train_iter = iter(loader)
    skipped_batches = _skip_train_batches(train_iter, config.skip_train_batches)
    progress = tqdm(train_iter, total=remaining_steps, desc="train_bc", leave=False)
    tensorboard_writer = _create_tensorboard_writer(config.tensorboard_log_dir)
    last_val_metrics: dict[str, float] = {}
    last_val_step: int | None = None
    metric_log_mode = _metric_log_mode(config, metric_log_path)
    try:
        with metric_log_path.open(metric_log_mode, encoding="utf-8") as metric_log:
            if remaining_steps > 0:
                for batch in progress:
                    if global_step >= config.steps:
                        break
                    batch = _move_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, step_metrics, batch_size = _batch_loss_and_metrics(
                        model,
                        batch,
                        policy_loss_weight=config.policy_loss_weight,
                        value_loss_weight=config.value_loss_weight,
                        distance_loss_weight=config.distance_loss_weight,
                        distance_loss_cap=config.distance_loss_cap,
                        distance_split_penalty=config.distance_split_penalty,
                        actionness_loss_weight=config.actionness_loss_weight,
                        actionness_pass_weight=config.actionness_pass_weight,
                        enemy_king_loss_weight=config.enemy_king_loss_weight,
                        enemy_king_distance_loss_weight=config.enemy_king_distance_loss_weight,
                        enemy_king_distance_cap=config.enemy_king_distance_cap,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

                    global_step += 1
                    samples_seen += batch_size
                    metrics.update(batch_size=batch_size, **step_metrics)
                    learning_rate = float(optimizer.param_groups[0]["lr"])
                    _write_metric_row(
                        metric_log,
                        phase="train",
                        step=global_step,
                        samples_seen=samples_seen,
                        learning_rate=learning_rate,
                        metrics=step_metrics,
                    )
                    _write_tensorboard_scalars(
                        tensorboard_writer,
                        phase="train",
                        step=global_step,
                        learning_rate=learning_rate,
                        metrics=step_metrics,
                    )
                    if _should_validate(config, global_step):
                        last_val_metrics = evaluate_bc(config, model, encoder, device)
                        last_val_step = global_step
                        _write_metric_row(
                            metric_log,
                            phase="val",
                            step=global_step,
                            samples_seen=samples_seen,
                            learning_rate=learning_rate,
                            metrics=last_val_metrics,
                        )
                        _write_tensorboard_scalars(
                            tensorboard_writer,
                            phase="val",
                            step=global_step,
                            learning_rate=learning_rate,
                            metrics=last_val_metrics,
                        )
                    if _should_save_periodic_checkpoint(config, global_step):
                        _save_checkpoint(
                            config,
                            encoder,
                            model,
                            optimizer,
                            metrics.summary(),
                            metric_log_path,
                            base_channels=base_channels,
                            global_step=global_step,
                            samples_seen=samples_seen,
                            last_val_metrics=last_val_metrics,
                            output_path=_periodic_checkpoint_path(config, global_step),
                        )
                    progress.set_postfix(metrics.latest())
            if (
                _validation_enabled(config)
                and global_step > 0
                and global_step != last_val_step
            ):
                learning_rate = float(optimizer.param_groups[0]["lr"])
                last_val_metrics = evaluate_bc(config, model, encoder, device)
                _write_metric_row(
                    metric_log,
                    phase="val",
                    step=global_step,
                    samples_seen=samples_seen,
                    learning_rate=learning_rate,
                    metrics=last_val_metrics,
                )
                _write_tensorboard_scalars(
                    tensorboard_writer,
                    phase="val",
                    step=global_step,
                    learning_rate=learning_rate,
                    metrics=last_val_metrics,
                )
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()

    summary = metrics.summary()
    _save_checkpoint(
        config,
        encoder,
        model,
        optimizer,
        summary,
        metric_log_path,
        base_channels=base_channels,
        global_step=global_step,
        samples_seen=samples_seen,
        last_val_metrics=last_val_metrics,
    )
    return {
        **summary,
        "checkpoint_path": str(Path(config.output_path)),
        "metrics_output_path": str(metric_log_path),
        "global_step": global_step,
        "samples_seen": samples_seen,
        "skipped_train_batches": skipped_batches,
        "last_val_metrics": last_val_metrics,
    }


def _skip_train_batches(loader: Any, count: int) -> int:
    """Advance a training iterator without updating model state."""
    skipped = 0
    if count <= 0:
        return skipped
    progress = tqdm(range(count), desc="skip_train_batches", leave=False)
    for _ in progress:
        try:
            next(loader)
        except StopIteration:
            break
        skipped += 1
    return skipped


def evaluate_bc(
    config: TrainConfig,
    model: BCPolicyNet,
    encoder: ObservationEncoder,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a bounded validation stream."""
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
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > config.val_batches:
                break
            batch = _move_batch(batch, device)
            _, step_metrics, batch_size = _batch_loss_and_metrics(
                model,
                batch,
                policy_loss_weight=config.policy_loss_weight,
                value_loss_weight=config.value_loss_weight,
                distance_loss_weight=config.distance_loss_weight,
                distance_loss_cap=config.distance_loss_cap,
                distance_split_penalty=config.distance_split_penalty,
                actionness_loss_weight=config.actionness_loss_weight,
                actionness_pass_weight=config.actionness_pass_weight,
                enemy_king_loss_weight=config.enemy_king_loss_weight,
                enemy_king_distance_loss_weight=config.enemy_king_distance_loss_weight,
                enemy_king_distance_cap=config.enemy_king_distance_cap,
            )
            metrics.update(batch_size=batch_size, **step_metrics)
    model.train(was_training)
    return metrics.summary()


def _save_checkpoint(
    config: TrainConfig,
    encoder: ObservationEncoder,
    model: BCPolicyNet,
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
    """Persist model, optimizer, config, and metrics."""
    output = Path(config.output_path if output_path is None else output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "input_channels": encoder.num_channels,
            "feature_names": list(encoder.channel_names),
            "history_length": encoder.history_length,
            "base_channels": base_channels,
            "policy_mode": POLICY_MODE,
            "train_config": asdict(config),
            "metrics": metrics,
            "metrics_output_path": str(metric_log_path),
            "tensorboard_log_dir": config.tensorboard_log_dir,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "last_val_metrics": last_val_metrics,
        },
        output,
    )
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "metrics": metrics,
                "checkpoint_path": str(output),
                "policy_mode": POLICY_MODE,
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


def _resolve_metrics_output_path(config: TrainConfig) -> Path:
    """Resolve the per-step metrics JSONL output path."""
    if config.metrics_output_path is not None:
        return Path(config.metrics_output_path)
    output = Path(config.output_path)
    return output.with_suffix(output.suffix + ".metrics.jsonl")


def _metric_log_mode(config: TrainConfig, metric_log_path: Path) -> str:
    """Return append mode only when resuming into an existing metric log."""
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
    """Append one training step metric row to the JSONL log."""
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
    """Write training metrics to TensorBoard when enabled."""
    if writer is None:
        return
    for name, value in metrics.items():
        writer.add_scalar(f"{phase}/{name}", value, step)
    writer.add_scalar(f"{phase}/lr", learning_rate, step)


def _make_loader(
    config: TrainConfig,
    *,
    encoder: ObservationEncoder,
    split: str,
    limit_replays: int | None,
    shuffle_buffer_size: int,
    shuffle_shards: bool,
    seed: int,
) -> DataLoader:
    """Build a raw replay BC DataLoader for one split."""
    dataset = RawReplayBCDataset(
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
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=0,
        collate_fn=lambda batch: collate_bc_samples(
            batch,
            encoder=encoder,
            late_pass_turn=config.late_pass_turn,
            late_pass_weight=config.late_pass_weight,
        ),
    )


def _validate_config(config: TrainConfig) -> None:
    """Validate BC training options that interact across fields."""
    if config.init_checkpoint is not None and config.resume_from is not None:
        raise ValueError("init_checkpoint and resume_from are mutually exclusive")
    if config.policy_loss_weight < 0:
        raise ValueError("policy_loss_weight must be nonnegative")
    if config.enemy_king_loss_weight < 0:
        raise ValueError("enemy_king_loss_weight must be nonnegative")
    if config.enemy_king_distance_loss_weight < 0:
        raise ValueError("enemy_king_distance_loss_weight must be nonnegative")
    if config.enemy_king_distance_cap <= 0:
        raise ValueError("enemy_king_distance_cap must be positive")
    if config.freeze_non_king_head and config.enemy_king_loss_weight <= 0:
        raise ValueError("freeze_non_king_head requires enemy_king_loss_weight > 0")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if not 0 < config.min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in (0, 1]")


def _load_model_checkpoint(config: TrainConfig, device: torch.device) -> dict[str, Any] | None:
    """Load an initialization or resume checkpoint when requested."""
    checkpoint_path = config.resume_from or config.init_checkpoint
    if checkpoint_path is None:
        return None
    return torch.load(checkpoint_path, map_location=device)


def _apply_parameter_freeze(model: BCPolicyNet, config: TrainConfig) -> None:
    """Optionally freeze every parameter except the enemy king auxiliary head."""
    if not config.freeze_non_king_head:
        return
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("enemy_king_head.")


def _resume_progress(
    config: TrainConfig,
    checkpoint: dict[str, Any],
) -> tuple[int, int]:
    """Return resume progress from checkpoint fields or legacy metrics logs."""
    if "global_step" in checkpoint:
        return int(checkpoint["global_step"]), int(checkpoint.get("samples_seen", 0))
    return _infer_progress_from_metric_log(config, checkpoint)


def _infer_progress_from_metric_log(
    config: TrainConfig,
    checkpoint: dict[str, Any],
) -> tuple[int, int]:
    """Infer legacy checkpoint progress from the associated JSONL metric log."""
    metric_paths: list[Path] = []
    if checkpoint.get("metrics_output_path") is not None:
        metric_paths.append(Path(str(checkpoint["metrics_output_path"])))
    if config.resume_from is not None:
        resume_path = Path(config.resume_from)
        metric_paths.append(resume_path.with_suffix(resume_path.suffix + ".metrics.jsonl"))

    global_step = 0
    samples_seen = 0
    for metric_path in metric_paths:
        if not metric_path.exists():
            continue
        for line in metric_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            global_step = max(global_step, int(row.get("step", 0)))
            samples_seen = max(samples_seen, int(row.get("samples_seen", 0)))
    return global_step, samples_seen


def _create_model(
    config: TrainConfig,
    encoder: ObservationEncoder,
    checkpoint: dict[str, Any] | None,
    device: torch.device,
    base_channels: int,
) -> BCPolicyNet:
    """Create a model compatible with a fresh run or resume checkpoint."""
    if checkpoint is None:
        return BCPolicyNet(encoder.num_channels, base_channels=base_channels).to(device)
    input_channels = int(checkpoint.get("input_channels", encoder.num_channels))
    if input_channels != encoder.num_channels:
        raise ValueError(
            f"checkpoint input_channels={input_channels} does not match "
            f"encoder channels={encoder.num_channels}"
        )
    checkpoint_history_length = int(checkpoint.get("history_length", 1))
    if checkpoint_history_length != encoder.history_length:
        raise ValueError(
            f"checkpoint history_length={checkpoint_history_length} does not match "
            f"configured history_length={encoder.history_length}"
        )
    checkpoint_policy_mode = str(checkpoint.get("policy_mode", "joint"))
    if checkpoint_policy_mode != POLICY_MODE:
        raise ValueError(
            f"checkpoint policy_mode={checkpoint_policy_mode} does not match "
            f"configured policy_mode={POLICY_MODE}"
        )
    return BCPolicyNet(
        input_channels,
        base_channels=base_channels,
    ).to(device)


def _effective_base_channels(
    config: TrainConfig,
    checkpoint: dict[str, Any] | None,
) -> int:
    """Return the model width for a fresh or resumed run."""
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


def _batch_loss_and_metrics(
    model: BCPolicyNet,
    batch: dict[str, Any],
    *,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 0.0,
    distance_loss_weight: float = 0.0,
    distance_loss_cap: float = 10.0,
    distance_split_penalty: float = 0.5,
    actionness_loss_weight: float = 0.0,
    actionness_pass_weight: float = 0.05,
    enemy_king_loss_weight: float = 0.0,
    enemy_king_distance_loss_weight: float = 1.0,
    enemy_king_distance_cap: float = 30.0,
) -> tuple[torch.Tensor, dict[str, float], int]:
    """Compute hierarchical policy loss and scalar metrics for one batch."""
    if policy_loss_weight < 0:
        raise ValueError("policy_loss_weight must be nonnegative")
    if distance_loss_weight < 0:
        raise ValueError("distance_loss_weight must be nonnegative")
    if actionness_loss_weight < 0:
        raise ValueError("actionness_loss_weight must be nonnegative")
    if actionness_pass_weight < 0:
        raise ValueError("actionness_pass_weight must be nonnegative")
    if enemy_king_loss_weight < 0:
        raise ValueError("enemy_king_loss_weight must be nonnegative")
    if enemy_king_distance_loss_weight < 0:
        raise ValueError("enemy_king_distance_loss_weight must be nonnegative")
    if enemy_king_distance_cap <= 0:
        raise ValueError("enemy_king_distance_cap must be positive")
    output = model(batch["x"])
    move_logits = output["move_logits"].flatten(start_dim=1)
    action_logits = output["action_logits"].reshape(-1)
    valid_mask = batch["valid_action_mask"]
    move_valid_mask = valid_mask[:, :-1]
    labels = batch["policy_label"]
    pass_label = valid_mask.shape[1] - 1
    is_move = labels != pass_label
    sample_weight = batch["sample_weight"]
    action_loss = _weighted_binary_loss(
        action_logits,
        is_move.to(torch.float32),
        sample_weight,
    )
    move_loss = _move_cross_entropy_loss(
        move_logits,
        labels=labels,
        move_mask=is_move,
        valid_move_mask=move_valid_mask,
        sample_weight=sample_weight,
    )
    policy_loss = action_loss + move_loss
    loss = policy_loss_weight * policy_loss
    metrics = {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "policy_loss_weight": float(policy_loss_weight),
        "action_loss": float(action_loss.item()),
        "move_loss": float(move_loss.item()),
        "loss_weight_mean": float(sample_weight.mean().item()),
    }
    if distance_loss_weight > 0:
        distance_loss = _distance_aware_loss(
            move_logits,
            labels=labels,
            valid_move_mask=move_valid_mask,
            sample_weight=sample_weight,
            label_height=int(batch["x"].shape[-2]),
            label_width=int(batch["x"].shape[-1]),
            cap=distance_loss_cap,
            split_penalty=distance_split_penalty,
        )
        loss = loss + distance_loss_weight * distance_loss
        metrics["loss"] = float(loss.item())
        metrics["distance_loss"] = float(distance_loss.item())
    if actionness_loss_weight > 0:
        actionness_loss = _actionness_loss(
            action_logits,
            labels=labels,
            valid_move_mask=move_valid_mask,
            sample_weight=sample_weight,
            pass_weight=actionness_pass_weight,
        )
        loss = loss + actionness_loss_weight * actionness_loss
        metrics["loss"] = float(loss.item())
        metrics["actionness_loss"] = float(actionness_loss.item())
    if value_loss_weight > 0:
        value_loss = F.mse_loss(output["value"], batch["value_target"])
        loss = loss + value_loss_weight * value_loss
        metrics["loss"] = float(loss.item())
        metrics["value_loss"] = float(value_loss.item())
    if enemy_king_loss_weight > 0:
        king_loss, king_metrics = enemy_king_prediction_loss(
            output["enemy_king_logits"],
            labels=batch["enemy_king_label"],
            valid_mask=batch["enemy_king_valid_mask"],
            distances=batch["enemy_king_distance"],
            distance_weight=enemy_king_distance_loss_weight,
            distance_cap=enemy_king_distance_cap,
        )
        loss = loss + enemy_king_loss_weight * king_loss
        metrics["loss"] = float(loss.item())
        metrics.update(king_metrics)

    staged_pred, move_pred, action_pred = _hierarchical_predictions(
        action_logits,
        move_logits,
        move_valid_mask,
        pass_label=pass_label,
    )
    invalid_pred = staged_pred != pass_label
    invalid_rate = torch.zeros_like(invalid_pred, dtype=torch.float32)
    if bool(invalid_pred.any()):
        invalid_rate[invalid_pred] = (
            ~move_valid_mask.gather(1, staged_pred.clamp_max(pass_label - 1)[:, None]).squeeze(1)
        )[invalid_pred].to(torch.float32)
    metrics.update({
        "accuracy": float((staged_pred == labels).float().mean().item()),
        "masked_accuracy": float((staged_pred == labels).float().mean().item()),
        "staged_accuracy": float((staged_pred == labels).float().mean().item()),
        "action_accuracy": float((action_pred == is_move).float().mean().item()),
        "move_accuracy": _move_accuracy(move_pred, labels, is_move),
        "invalid_rate": float(invalid_rate.mean().item()),
        "pass_accuracy": _pass_accuracy(staged_pred, labels, pass_label),
    })
    return loss, metrics, int(labels.numel())


def _weighted_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted BCEWithLogits over the move/pass action gate."""
    per_sample = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)


def _move_cross_entropy_loss(
    move_logits: torch.Tensor,
    *,
    labels: torch.Tensor,
    move_mask: torch.Tensor,
    valid_move_mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted move CE only for expert move samples."""
    if not bool(move_mask.any()):
        return move_logits.new_zeros(())
    masked_logits = move_logits.masked_fill(~valid_move_mask, -1e9)
    per_sample = F.cross_entropy(
        masked_logits[move_mask],
        labels[move_mask],
        reduction="none",
    )
    weights = sample_weight[move_mask]
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def _hierarchical_predictions(
    action_logits: torch.Tensor,
    move_logits: torch.Tensor,
    valid_move_mask: torch.Tensor,
    *,
    pass_label: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return staged action labels, conditional move predictions, and gate decisions."""
    has_non_pass = valid_move_mask.any(dim=1)
    action_pred = (action_logits > 0) & has_non_pass
    move_pred = move_logits.masked_fill(~valid_move_mask, -1e9).argmax(dim=1)
    staged_pred = torch.full_like(move_pred, pass_label)
    staged_pred[action_pred] = move_pred[action_pred]
    return staged_pred, move_pred, action_pred


def _distance_aware_loss(
    move_logits: torch.Tensor,
    *,
    labels: torch.Tensor,
    valid_move_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    label_height: int,
    label_width: int,
    cap: float,
    split_penalty: float,
) -> torch.Tensor:
    """Penalize conditional move probability mass by distance from the expert move."""
    if cap <= 0:
        raise ValueError("distance_loss_cap must be positive")
    pass_label = ACTION_PLANES * label_height * label_width
    move_mask = labels != pass_label
    if not bool(move_mask.any()):
        return move_logits.new_zeros(())
    move_labels = labels[move_mask]
    distances = _move_distance_targets(
        move_labels,
        num_classes=move_logits.shape[1],
        label_height=label_height,
        label_width=label_width,
        cap=cap,
        split_penalty=split_penalty,
        device=move_logits.device,
    )
    masked_logits = move_logits[move_mask].masked_fill(~valid_move_mask[move_mask], -1e9)
    probabilities = torch.softmax(masked_logits, dim=1)
    per_sample = (probabilities * distances).sum(dim=1)
    move_weight = sample_weight[move_mask]
    return (
        (per_sample * move_weight).sum()
        / move_weight.sum().clamp_min(1e-8)
    )


def _actionness_loss(
    action_logits: torch.Tensor,
    *,
    labels: torch.Tensor,
    valid_move_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    pass_weight: float,
) -> torch.Tensor:
    """Optionally add extra weighted supervision to the explicit action gate."""
    if pass_weight < 0:
        raise ValueError("pass_weight must be nonnegative")
    pass_label = valid_move_mask.shape[1]
    has_non_pass = valid_move_mask.any(dim=1)
    if not bool(has_non_pass.any()):
        return action_logits.new_zeros(())

    targets = (labels != pass_label).to(torch.float32)
    per_sample = F.binary_cross_entropy_with_logits(
        action_logits,
        targets,
        reduction="none",
    )
    weights = sample_weight.clone()
    weights[labels == pass_label] = weights[labels == pass_label] * pass_weight
    weights = weights * has_non_pass.to(weights.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def _move_accuracy(
    move_pred: torch.Tensor,
    labels: torch.Tensor,
    move_mask: torch.Tensor,
) -> float:
    """Compute conditional move accuracy on samples whose expert label is a move."""
    if not bool(move_mask.any()):
        return 0.0
    return float((move_pred[move_mask] == labels[move_mask]).float().mean().item())


def _distance_targets(
    labels: torch.Tensor,
    *,
    num_classes: int,
    label_height: int,
    label_width: int,
    cap: float,
    split_penalty: float,
    device: torch.device,
) -> torch.Tensor:
    """Build normalized per-action distances to each sample's expert action."""
    board_cells = label_height * label_width
    pass_label = ACTION_PLANES * board_cells
    if num_classes != pass_label + 1:
        raise ValueError("num_classes does not match label board shape")
    move_distances = _move_distance_targets(
        labels.clamp_max(pass_label - 1),
        num_classes=pass_label,
        label_height=label_height,
        label_width=label_width,
        cap=cap,
        split_penalty=split_penalty,
        device=device,
    )
    pass_distances = torch.ones((labels.shape[0], 1), device=device)
    distances = torch.cat([move_distances, pass_distances], dim=1)
    distances[labels.to(device) == pass_label] = 0.0
    return distances


def _move_distance_targets(
    labels: torch.Tensor,
    *,
    num_classes: int,
    label_height: int,
    label_width: int,
    cap: float,
    split_penalty: float,
    device: torch.device,
) -> torch.Tensor:
    """Build normalized per-move distances to each sample's expert move."""
    board_cells = label_height * label_width
    expected_classes = ACTION_PLANES * board_cells
    if num_classes != expected_classes:
        raise ValueError("num_classes does not match label board shape")

    action_labels = torch.arange(expected_classes, device=device)
    candidate_planes = torch.div(action_labels, board_cells, rounding_mode="floor")
    candidate_starts = action_labels % board_cells
    candidate_rows = torch.div(candidate_starts, label_width, rounding_mode="floor")
    candidate_cols = candidate_starts % label_width
    candidate_dirs = candidate_planes % 4
    direction_rows = torch.tensor([-1, 1, 0, 0], device=device)
    direction_cols = torch.tensor([0, 0, -1, 1], device=device)
    candidate_end_rows = candidate_rows + direction_rows[candidate_dirs]
    candidate_end_cols = candidate_cols + direction_cols[candidate_dirs]
    candidate_split = candidate_planes >= 4

    labels = labels.to(device)
    expert_labels = labels.clamp_max(expected_classes - 1)
    expert_planes = torch.div(expert_labels, board_cells, rounding_mode="floor")
    expert_starts = expert_labels % board_cells
    expert_rows = torch.div(expert_starts, label_width, rounding_mode="floor")
    expert_cols = expert_starts % label_width
    expert_dirs = expert_planes % 4
    expert_end_rows = expert_rows + direction_rows[expert_dirs]
    expert_end_cols = expert_cols + direction_cols[expert_dirs]
    expert_split = expert_planes >= 4

    start_distance = (
        (candidate_rows[None, :] - expert_rows[:, None]).abs()
        + (candidate_cols[None, :] - expert_cols[:, None]).abs()
    )
    end_distance = (
        (candidate_end_rows[None, :] - expert_end_rows[:, None]).abs()
        + (candidate_end_cols[None, :] - expert_end_cols[:, None]).abs()
    )
    split_distance = (
        candidate_split[None, :] != expert_split[:, None]
    ).to(torch.float32) * split_penalty
    move_distances = (start_distance + end_distance).to(torch.float32) + split_distance
    move_distances = move_distances.clamp_max(cap) / cap
    return move_distances


def _validation_enabled(config: TrainConfig) -> bool:
    """Return whether validation metrics should be collected."""
    return config.val_every_steps > 0 and config.val_batches > 0


def _should_validate(config: TrainConfig, global_step: int) -> bool:
    """Return whether this global step should run validation."""
    return _validation_enabled(config) and global_step % config.val_every_steps == 0


def _should_save_periodic_checkpoint(config: TrainConfig, global_step: int) -> bool:
    """Return whether this step should write an intermediate checkpoint."""
    return config.checkpoint_every_steps > 0 and global_step % config.checkpoint_every_steps == 0


def _periodic_checkpoint_path(config: TrainConfig, global_step: int) -> Path:
    """Build the checkpoint path for a periodic step save."""
    output = Path(config.output_path)
    return output.with_name(f"{output.stem}_step{global_step:05d}{output.suffix}")


def _resolve_device(device: str) -> torch.device:
    """Resolve auto/cpu/cuda device strings to a torch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields in a collated batch to the target device."""
    result = dict(batch)
    for key in (
        "x",
        "policy_label",
        "valid_action_mask",
        "value_target",
        "sample_weight",
        "enemy_king_label",
        "enemy_king_distance",
        "enemy_king_valid_mask",
        "burn_in_mask",
        "chunk_boundary",
        "turn",
        "replay_version",
    ):
        if key in batch:
            result[key] = batch[key].to(device)
    return result


def _pass_accuracy(pred: torch.Tensor, labels: torch.Tensor, pass_label: int) -> float:
    """Compute accuracy restricted to samples whose label is pass."""
    pass_mask = labels == pass_label
    if not bool(pass_mask.any()):
        return 0.0
    return float((pred[pass_mask] == labels[pass_mask]).float().mean().item())


class _MetricAccumulator:
    """Accumulate weighted mean metrics over training batches."""

    def __init__(self) -> None:
        """Create an empty metric accumulator."""
        self.totals: dict[str, float] = {}
        self.count = 0
        self._latest: dict[str, float] = {}

    def update(self, *, batch_size: int, **values: float) -> None:
        """Add one batch of scalar metrics."""
        self.count += batch_size
        self._latest = values
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + value * batch_size

    def latest(self) -> dict[str, str]:
        """Return the latest metrics formatted for tqdm."""
        return {key: f"{value:.4f}" for key, value in self._latest.items()}

    def summary(self) -> dict[str, float]:
        """Return weighted mean metrics."""
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.totals.items()}


# ---------------------------------------------------------------------------
# Temporal BC Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalTrainConfig(TrainConfig):
    """Configuration for temporal behavior cloning."""

    model_config_path: str = ""
    sequence_length: int = 64
    burn_in: int = 16
    chunk_size: int = 8
    backbone_lr: float = 1e-5
    temporal_lr: float = 1e-4
    head_lr: float = 5e-5
    gradient_accumulation: int = 1
    use_amp: bool = False
    replay_shuffle_buffer_size: int = 2048
    max_val_predicted_pass_rate: float = 1.0
    max_val_move_false_pass_rate: float = 1.0


def train_temporal_bc(config: TemporalTrainConfig) -> dict[str, Any]:
    """Train a temporal BC policy model and save a checkpoint."""
    _validate_temporal_config(config)
    _seed_training(config.seed)
    encoder = ObservationEncoder(history_length=config.history_length)
    device = _resolve_device(config.device)
    metric_log_path = _resolve_metrics_output_path(config)
    metric_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load temporal model config
    from generals_bot.training.temporal import TemporalModelConfig, TemporalPolicyNet

    temporal_config = TemporalModelConfig.from_json(config.model_config_path)
    _validate_temporal_model_training_alignment(config, temporal_config)

    # Create or load model
    model, optimizer, global_step, samples_seen, base_channels, data_epoch = (
        _create_temporal_training_state(config, encoder, temporal_config, device)
    )

    # A resumed run starts from a newly shuffled epoch instead of silently
    # replaying the beginning of the same deterministic stream.
    loader, val_loader = _make_temporal_loaders(
        config,
        encoder,
        start_epoch=data_epoch,
    )

    remaining_steps = max(0, config.steps - global_step)
    train_iter = iter(loader)
    skipped = 0
    for _ in range(config.skip_train_batches):
        try:
            next(train_iter)
            skipped += 1
        except StopIteration:
            break
    progress = tqdm(
        train_iter,
        total=remaining_steps * config.gradient_accumulation,
        desc="train_temporal_bc",
        leave=False,
    )
    tensorboard_writer = _create_tensorboard_writer(config.tensorboard_log_dir)
    last_val_metrics: dict[str, float] = {}
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=4096.0)
        if config.use_amp
        else None
    )
    metrics = _MetricAccumulator()
    update_metrics = _MetricAccumulator()
    metric_log_mode = _metric_log_mode(config, metric_log_path)

    try:
        with metric_log_path.open(metric_log_mode, encoding="utf-8") as metric_log:
            if remaining_steps > 0:
                optimizer.zero_grad(set_to_none=True)
                accum_step = 0
                for batch in progress:
                    if global_step >= config.steps:
                        break
                    batch = _move_batch(batch, device)

                    with torch.amp.autocast(
                        "cuda", enabled=config.use_amp,
                    ):
                        loss, step_metrics, batch_size = _temporal_batch_loss(
                            model, batch, config,
                        )
                        loss = loss / config.gradient_accumulation

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    accum_step += 1
                    samples_seen += batch_size
                    metrics.update(batch_size=batch_size, **step_metrics)
                    update_metrics.update(batch_size=batch_size, **step_metrics)
                    did_optimizer_step = False
                    if accum_step % config.gradient_accumulation == 0:
                        _apply_temporal_lr_schedule(
                            optimizer,
                            config,
                            step=global_step + 1,
                        )
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_clip,
                        )
                        if scaler is not None:
                            scale_before = float(scaler.get_scale())
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer_step_succeeded = (
                                float(scaler.get_scale()) >= scale_before
                            )
                        else:
                            optimizer.step()
                            optimizer_step_succeeded = True
                        optimizer.zero_grad(set_to_none=True)
                        accum_step = 0
                        if optimizer_step_succeeded:
                            global_step += 1
                            did_optimizer_step = True

                    progress.set_postfix(metrics.latest())

                    if not did_optimizer_step:
                        if accum_step == 0:
                            update_metrics = _MetricAccumulator()
                        continue

                    logged_metrics = update_metrics.summary()
                    logged_metrics.update({
                        "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                        "temporal_lr": float(optimizer.param_groups[1]["lr"]),
                        "head_lr": float(optimizer.param_groups[2]["lr"]),
                        "amp_scale": (
                            float(scaler.get_scale()) if scaler is not None else 1.0
                        ),
                    })
                    update_metrics = _MetricAccumulator()
                    _write_metric_row(
                        metric_log,
                        phase="train",
                        step=global_step,
                        samples_seen=samples_seen,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        metrics=logged_metrics,
                    )
                    _write_tensorboard_scalars(
                        tensorboard_writer,
                        phase="train",
                        step=global_step,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        metrics=logged_metrics,
                    )

                    # Validation
                    if (
                        val_loader is not None
                        and global_step > 0
                        and config.val_every_steps > 0
                        and global_step % config.val_every_steps == 0
                    ):
                        last_val_metrics = evaluate_temporal_bc(
                            model, val_loader, config, device,
                        )
                        _write_metric_row(
                            metric_log,
                            phase="val",
                            step=global_step,
                            samples_seen=samples_seen,
                            learning_rate=float(optimizer.param_groups[0]["lr"]),
                            metrics=last_val_metrics,
                        )
                        _write_tensorboard_scalars(
                            tensorboard_writer,
                            phase="val",
                            step=global_step,
                            learning_rate=float(optimizer.param_groups[0]["lr"]),
                            metrics=last_val_metrics,
                        )
                        health_error = _temporal_validation_health_error(
                            config,
                            last_val_metrics,
                        )
                        if health_error is not None:
                            _save_temporal_checkpoint(
                                config,
                                encoder,
                                model,
                                optimizer,
                                metrics=metrics.summary(),
                                metric_log_path=metric_log_path,
                                base_channels=base_channels,
                                global_step=global_step,
                                samples_seen=samples_seen,
                                last_val_metrics=last_val_metrics,
                                temporal_config=temporal_config,
                                data_epoch=int(loader.dataset.current_epoch),
                                output_path=_periodic_checkpoint_path(config, global_step),
                            )
                            raise RuntimeError(health_error)

                    # Checkpoint
                    if _should_save_periodic_checkpoint(config, global_step):
                        _save_temporal_checkpoint(
                            config, encoder, model, optimizer,
                            metrics=metrics.summary(),
                            metric_log_path=metric_log_path,
                            base_channels=base_channels,
                            global_step=global_step,
                            samples_seen=samples_seen,
                            last_val_metrics=last_val_metrics,
                            temporal_config=temporal_config,
                            data_epoch=int(loader.dataset.current_epoch),
                            output_path=_periodic_checkpoint_path(config, global_step),
                        )
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()

    _save_temporal_checkpoint(
        config, encoder, model, optimizer,
        metrics=metrics.summary(),
        metric_log_path=metric_log_path,
        base_channels=base_channels,
        global_step=global_step,
        samples_seen=samples_seen,
        last_val_metrics=last_val_metrics,
        temporal_config=temporal_config,
        data_epoch=int(loader.dataset.current_epoch),
    )
    return {
        **metrics.summary(),
        "checkpoint_path": config.output_path,
        "metrics_output_path": str(metric_log_path),
        "global_step": global_step,
    }


def _create_temporal_training_state(
    config: TemporalTrainConfig,
    encoder: ObservationEncoder,
    temporal_config: Any,
    device: torch.device,
) -> tuple:
    """Create or resume a TemporalPolicyNet and optimizer."""
    from generals_bot.training.temporal import TemporalPolicyNet

    checkpoint = _load_model_checkpoint(config, device)
    base_channels = config.base_channels
    if checkpoint is not None:
        base_channels = int(checkpoint.get("base_channels", base_channels))

    spatial = BCPolicyNet(
        encoder.num_channels,
        base_channels=base_channels,
        temporal_dim=temporal_config.temporal_dim,
    ).to(device)

    checkpoint_is_temporal = (
        checkpoint is not None
        and str(checkpoint.get("model_type", "")) == "temporal_policy_v1"
    )
    if checkpoint is not None and not checkpoint_is_temporal:
        _load_temporal_spatial_init(spatial, checkpoint["model_state"])

    model = TemporalPolicyNet(encoder.num_channels, spatial, temporal_config).to(device)
    if checkpoint_is_temporal:
        model.load_state_dict(checkpoint["model_state"])
    model.train()

    # Grouped optimizer
    backbone_params = list(spatial.stem.parameters()) + \
        list(spatial.down1.parameters()) + \
        list(spatial.down2.parameters()) + \
        list(spatial.bottleneck.parameters()) + \
        list(spatial.up2.parameters()) + \
        list(spatial.up1.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = (
        list(spatial.move_head.parameters())
        + list(spatial.enemy_king_head.parameters())
        + list(spatial.pass_head.parameters())
        + list(spatial.value_head.parameters())
    )
    head_ids = {id(p) for p in head_params}
    temporal_params = [
        p for p in model.parameters()
        if id(p) not in backbone_ids and id(p) not in head_ids and p.requires_grad
    ]

    optimizer = torch.optim.AdamW([
        {"params": list(backbone_params), "lr": config.backbone_lr},
        {"params": list(temporal_params), "lr": config.temporal_lr},
        {"params": list(head_params), "lr": config.head_lr},
    ], weight_decay=config.weight_decay)

    global_step = 0
    samples_seen = 0
    data_epoch = 0
    if checkpoint is not None and config.resume_from is not None:
        if "optimizer_state" not in checkpoint:
            raise ValueError("resume checkpoint is missing optimizer_state")
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as exc:
            raise ValueError("failed to restore temporal optimizer state") from exc
        _move_optimizer_state(optimizer, device)
        if config.override_resume_lr:
            for pg, learning_rate in zip(
                optimizer.param_groups,
                (config.backbone_lr, config.temporal_lr, config.head_lr),
            ):
                pg["lr"] = learning_rate
        global_step, samples_seen = _resume_progress(config, checkpoint)
        data_epoch = int(checkpoint.get("data_epoch", 0)) + 1

    return model, optimizer, global_step, samples_seen, base_channels, data_epoch


def _load_temporal_spatial_init(
    spatial: BCPolicyNet,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load a spatial checkpoint, reducing stacked-history stem weights if needed.

    Released policies use eight 22-channel frames (176 channels), while the
    temporal model intentionally consumes one 22-channel frame.  In that case
    the newest-frame slice initializes the temporal stem and the learned
    temporal memory replaces the discarded frame stack.
    """
    migrated = dict(state_dict)
    stem_key = "stem.0.weight"
    source = migrated.get(stem_key)
    target = spatial.state_dict()[stem_key]
    if source is not None and source.shape != target.shape:
        same_convolution = (
            source.ndim == target.ndim == 4
            and source.shape[0] == target.shape[0]
            and source.shape[2:] == target.shape[2:]
        )
        can_take_latest_frame = (
            same_convolution
            and source.shape[1] > target.shape[1]
            and source.shape[1] % target.shape[1] == 0
        )
        if not can_take_latest_frame:
            raise RuntimeError(
                "cannot migrate checkpoint stem from "
                f"{tuple(source.shape)} to {tuple(target.shape)}"
            )
        migrated[stem_key] = source[:, -target.shape[1] :, :, :].clone()
    load_model_state_compatible(spatial, migrated)


def _temporal_batch_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    config: TemporalTrainConfig,
) -> tuple:
    """Compute temporal BC loss with burn-in masking."""
    output = _temporal_sequence_forward(model, batch)
    policy_label = batch["policy_label"]
    valid_mask = batch["valid_action_mask"]
    sample_weight = batch["sample_weight"].to(dtype=torch.float32)
    burn_in_mask = batch.get("burn_in_mask")
    if burn_in_mask is not None:
        sample_weight = sample_weight * (~burn_in_mask).to(torch.float32)

    move_logits = output["move_logits"].flatten(start_dim=1)
    action_logits = output["action_logits"].reshape(-1)
    values = output["value"].reshape(-1)
    move_mask = valid_mask[:, :-1]
    pass_label = valid_mask.shape[1] - 1
    has_move = move_mask.any(dim=1)
    is_pass = policy_label == pass_label
    is_move = ~is_pass

    # Gate loss
    gate_loss = _weighted_binary_loss(action_logits, is_move.to(torch.float32), sample_weight)

    # Move CE loss
    mask_value = torch.finfo(move_logits.dtype).min
    masked_move_logits = move_logits.masked_fill(~move_mask, mask_value)
    move_ce = F.cross_entropy(
        masked_move_logits[is_move],
        policy_label[is_move].clamp(0, pass_label - 1),
        reduction="none",
    )
    move_loss = _safe_weighted_mean(move_ce, sample_weight[is_move])

    gate_pred_move = action_logits > 0
    move_prediction = masked_move_logits.argmax(dim=1)
    move_correct = move_prediction == policy_label.clamp(0, pass_label - 1)
    gate_correct = gate_pred_move == is_move
    action_correct = torch.where(
        is_pass,
        ~gate_pred_move,
        gate_pred_move & move_correct,
    )
    predicted_pass = ~gate_pred_move

    policy_loss = gate_loss + move_loss
    loss = config.policy_loss_weight * policy_loss

    # Value loss
    value_loss = torch.tensor(0.0, device=policy_label.device)
    if config.value_loss_weight > 0:
        value_loss = _safe_weighted_mean(
            F.mse_loss(values, batch["value_target"], reduction="none"),
            sample_weight,
        )
        loss = loss + config.value_loss_weight * value_loss

    # Enemy king loss
    king_metrics: dict[str, float] = {}
    if config.enemy_king_loss_weight > 0 and "enemy_king_label" in batch:
        from generals_bot.training.losses import enemy_king_prediction_loss

        real = sample_weight > 0
        if bool(real.any()):
            king_loss, king_metrics = enemy_king_prediction_loss(
                output["enemy_king_logits"][real],
                labels=batch["enemy_king_label"][real],
                valid_mask=batch["enemy_king_valid_mask"][real],
                distances=batch["enemy_king_distance"][real],
                distance_weight=config.enemy_king_distance_loss_weight,
                distance_cap=config.enemy_king_distance_cap,
            )
            loss = loss + config.enemy_king_loss_weight * king_loss

    effective_batch = max(1.0, float(sample_weight.sum().item()))
    metrics = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "gate_loss": float(gate_loss.detach().cpu()),
        "move_loss": float(move_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "burn_in_ratio": float(burn_in_mask.float().mean().item()) if burn_in_mask is not None else 0.0,
        "gate_accuracy": float(
            _safe_weighted_mean(gate_correct.float(), sample_weight).detach().cpu()
        ),
        "action_accuracy": float(
            _safe_weighted_mean(action_correct.float(), sample_weight).detach().cpu()
        ),
        "move_accuracy": float(
            _safe_weighted_mean(
                move_correct[is_move].float(), sample_weight[is_move],
            ).detach().cpu()
        ),
        "predicted_pass_rate": float(
            _safe_weighted_mean(predicted_pass.float(), sample_weight).detach().cpu()
        ),
        "target_pass_rate": float(
            _safe_weighted_mean(is_pass.float(), sample_weight).detach().cpu()
        ),
        "move_false_pass_rate": float(
            _safe_weighted_mean(
                predicted_pass[is_move].float(), sample_weight[is_move],
            ).detach().cpu()
        ),
        "pass_false_move_rate": float(
            _safe_weighted_mean(
                gate_pred_move[is_pass].float(), sample_weight[is_pass],
            ).detach().cpu()
        ),
        "action_logit_mean": float(
            _safe_weighted_mean(action_logits, sample_weight).detach().cpu()
        ),
    }
    turns = batch.get("turn")
    if turns is not None:
        early = turns < config.late_pass_turn
        middle = (turns >= config.late_pass_turn) & (turns < 100)
        late = turns >= 100
        for name, mask in (("early", early), ("mid", middle), ("late", late)):
            metrics[f"predicted_pass_rate_{name}"] = float(
                _safe_weighted_mean(
                    predicted_pass[mask].float(), sample_weight[mask],
                ).detach().cpu()
            )
    replay_versions = batch.get("replay_version")
    if replay_versions is not None:
        for version_tensor in torch.unique(replay_versions):
            version = int(version_tensor.item())
            if version < 0:
                continue
            mask = replay_versions == version
            metrics[f"predicted_pass_rate_v{version}"] = float(
                _safe_weighted_mean(
                    predicted_pass[mask].float(), sample_weight[mask],
                ).detach().cpu()
            )
            metrics[f"action_accuracy_v{version}"] = float(
                _safe_weighted_mean(
                    action_correct[mask].float(), sample_weight[mask],
                ).detach().cpu()
            )
            move_mask_for_version = mask & is_move
            metrics[f"move_false_pass_rate_v{version}"] = float(
                _safe_weighted_mean(
                    predicted_pass[move_mask_for_version].float(),
                    sample_weight[move_mask_for_version],
                ).detach().cpu()
            )
    metrics.update(king_metrics)
    return loss, metrics, int(effective_batch)


def _temporal_sequence_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run a flattened sequence batch in causal tick order.

    Temporal collation stores rows in sequence-major order.  The temporal
    policy must therefore see ``B`` streams for ``T`` successive calls, not
    one independent ``B*T`` frame batch.
    """
    seq_lengths = batch.get("seq_lengths")
    if seq_lengths is None or seq_lengths.numel() == 0:
        raise ValueError("temporal batch requires non-empty seq_lengths")
    if not bool((seq_lengths == seq_lengths[0]).all()):
        raise ValueError("temporal sequence batch requires equal padded lengths")
    batch_size = int(seq_lengths.numel())
    sequence_length = int(seq_lengths[0].item())
    x = batch["x"]
    if x.shape[0] != batch_size * sequence_length:
        raise ValueError(
            f"temporal x has {x.shape[0]} rows; expected "
            f"{batch_size}*{sequence_length}"
        )
    frames = x.reshape(batch_size, sequence_length, *x.shape[1:])
    boundary = batch.get("chunk_boundary")
    if boundary is None:
        boundary = torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        )
    else:
        boundary = boundary.reshape(batch_size, sequence_length)

    by_key: dict[str, list[torch.Tensor]] = {}
    for tick in range(sequence_length):
        tick_output = model(
            frames[:, tick],
            chunk_boundary=boundary[:, tick],
            reset_memory=tick == 0,
        )
        for key, value in tick_output.items():
            by_key.setdefault(key, []).append(value)
    return {
        key: torch.stack(values, dim=1).flatten(start_dim=0, end_dim=1)
        for key, values in by_key.items()
    }


def evaluate_temporal_bc(
    model: torch.nn.Module,
    val_loader: Any,
    config: TemporalTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Run validation for temporal BC."""
    was_training = model.training
    model.eval()
    metrics = _MetricAccumulator()
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= config.val_batches:
                break
            batch = _move_batch(batch, device)
            with torch.amp.autocast("cuda", enabled=config.use_amp):
                _, step_metrics, batch_size = _temporal_batch_loss(
                    model, batch, config,
                )
            metrics.update(batch_size=batch_size, **step_metrics)
    model.train(was_training)
    return metrics.summary()


def _temporal_validation_health_error(
    config: TemporalTrainConfig,
    metrics: dict[str, float],
) -> str | None:
    """Return a reason to stop when the hierarchical gate has collapsed."""
    predicted_pass = float(metrics.get("predicted_pass_rate", 0.0))
    if predicted_pass > config.max_val_predicted_pass_rate:
        return (
            "temporal validation predicted_pass_rate exceeded limit: "
            f"{predicted_pass:.4f} > {config.max_val_predicted_pass_rate:.4f}"
        )
    false_pass = float(metrics.get("move_false_pass_rate", 0.0))
    if false_pass > config.max_val_move_false_pass_rate:
        return (
            "temporal validation move_false_pass_rate exceeded limit: "
            f"{false_pass:.4f} > {config.max_val_move_false_pass_rate:.4f}"
        )
    return None


def _make_temporal_loaders(
    config: TemporalTrainConfig,
    encoder: ObservationEncoder,
    *,
    start_epoch: int = 0,
) -> tuple:
    """Create temporal DataLoader(s)."""
    def _temporal_collate(batch):
        return temporal_collate_sequences(
            batch,
            encoder=encoder,
            late_pass_turn=config.late_pass_turn,
            late_pass_weight=config.late_pass_weight,
        )

    temporal_samples = _TemporalSequenceIterableDataset(
        config,
        shuffle_buffer_size=config.shuffle_buffer_size,
        seed=config.seed,
        start_epoch=start_epoch,
        repeat=True,
    )
    loader = DataLoader(
        temporal_samples,
        batch_size=config.batch_size,
        drop_last=True,
        collate_fn=_temporal_collate,
    )

    val_loader = None
    if config.val_every_steps > 0:
        val_config = TemporalTrainConfig(
            **{
                **config.__dict__,
                "split": config.val_split,
                "limit_replays": config.val_limit_replays,
                "shuffle_buffer_size": 0,
                "replay_shuffle_buffer_size": 0,
            },
        )
        val_samples = _TemporalSequenceIterableDataset(
            val_config,
            shuffle_buffer_size=0,
            seed=config.seed,
            start_epoch=0,
            repeat=False,
        )
        val_loader = DataLoader(
            val_samples,
            batch_size=config.batch_size,
            drop_last=False,
            collate_fn=_temporal_collate,
        )

    return loader, val_loader


class _TemporalSequenceIterableDataset(IterableDataset):
    """Stream temporal sequences through a bounded shuffle buffer."""

    def __init__(
        self,
        config: TemporalTrainConfig,
        *,
        shuffle_buffer_size: int,
        seed: int,
        start_epoch: int = 0,
        repeat: bool = False,
    ) -> None:
        super().__init__()
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be nonnegative")
        self.config = config
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.start_epoch = start_epoch
        self.repeat = repeat
        self.current_epoch = start_epoch

    def __iter__(self):
        epoch = self.start_epoch
        while True:
            self.current_epoch = epoch
            epoch_seed = self.seed + epoch
            sequences = _iter_temporal_sequences(
                self.config,
                seed=epoch_seed,
            )
            yielded = False
            shuffled = _bounded_shuffle(
                sequences,
                buffer_size=self.shuffle_buffer_size,
                seed=epoch_seed,
            )
            for sequence in _bucket_temporal_sequences(
                shuffled,
                group_size=self.config.batch_size,
            ):
                yielded = True
                yield sequence
            if not yielded:
                raise RuntimeError("temporal dataset produced no sequences")
            if not self.repeat:
                return
            epoch += 1


def _bounded_shuffle(iterable: Any, *, buffer_size: int, seed: int):
    """Yield every input once with deterministic bounded-memory shuffling."""
    if buffer_size < 0:
        raise ValueError("buffer_size must be nonnegative")
    if buffer_size <= 1:
        yield from iterable
        return
    rng = random.Random(seed)
    buffer: list[Any] = []
    for item in iterable:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            yield buffer.pop(rng.randrange(len(buffer)))
    while buffer:
        yield buffer.pop(rng.randrange(len(buffer)))


def _bucket_temporal_sequences(iterable: Any, *, group_size: int):
    """Emit consecutive same-size groups so padding cannot skew GroupNorm."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    buckets: dict[tuple[int, int], list[Any]] = {}
    for sequence in iterable:
        observation = sequence.steps[0].observation
        key = (observation.height, observation.width)
        bucket = buckets.setdefault(key, [])
        bucket.append(sequence)
        if len(bucket) == group_size:
            yield from bucket
            bucket.clear()


def _iter_temporal_sequences(
    config: TemporalTrainConfig,
    *,
    seed: int | None = None,
):
    """Yield TemporalSequence objects one replay at a time.

    Iterates replay files, extracts per-player trajectories via
    :func:`samples_from_raw_replay`, and segments them into fixed-length
    sequences with chunk-boundary markers.
    """
    from collections import defaultdict

    from generals_bot.training.sequence import (
        build_sequences_from_samples,
    )
    from generals_bot.training.dataset import (
        samples_from_raw_replay,
        replay_in_split,
    )
    from generals_bot.replay import iter_raw_replays

    replay_count = 0
    replay_seed = config.seed if seed is None else seed
    replays = iter_raw_replays(
        config.input_path,
        shard_seed=replay_seed if config.shuffle_shards else None,
    )
    replays = _bounded_shuffle(
        replays,
        buffer_size=config.replay_shuffle_buffer_size,
        seed=replay_seed,
    )
    for replay in replays:
        if config.limit_replays is not None and replay_count >= config.limit_replays:
            break
        if not replay_in_split(replay.replay_id, config.split):
            continue
        if config.replay_versions is not None and replay.version not in config.replay_versions:
            continue
        samples = samples_from_raw_replay(
            replay,
            history_length=config.history_length,
            include_pass=True,
        )
        if not _pass_replay_filter(samples, config.replay_filter_rule):
            continue
        # Group by player_id
        by_player: dict[int, list[dict]] = defaultdict(list)
        for sample in samples:
            by_player[sample.player_id].append({
                "observation": sample.observation,
                "action": sample.action,
                "value_target": sample.value_target,
                "player_id": sample.player_id,
                "turn": sample.turn,
                "replay_id": sample.replay_id,
                "enemy_general_tile": sample.enemy_general_tile,
                "mountains": sample.mountains,
                "replay_version": replay.version,
            })

        for player_id, player_samples in by_player.items():
            player_samples.sort(key=lambda s: s["turn"])
            sequences = build_sequences_from_samples(
                player_samples,
                sequence_length=config.sequence_length,
                burn_in=config.burn_in,
                chunk_size=config.chunk_size,
                stride=max(1, config.sequence_length - config.burn_in),
            )
            yield from sequences

        replay_count += 1

def _build_temporal_dataset(
    config: TemporalTrainConfig,
    encoder: ObservationEncoder | None = None,
) -> list[Any]:
    """Materialize temporal sequences for tests and small diagnostics only."""
    del encoder
    sequences = list(_iter_temporal_sequences(config))
    if not sequences:
        raise RuntimeError(
            f"No temporal sequences generated from {config.input_path} "
            f"(split={config.split}, limit={config.limit_replays})"
        )
    return sequences


def _pass_replay_filter(samples: Any, rule: str) -> bool:
    """Check whether a raw replay passes the configured filter rule."""
    if rule == "none" or not rule:
        return True
    from generals_bot.training.dataset import evaluate_replay_filter_rule

    return bool(evaluate_replay_filter_rule(samples, rule).accepted)


def temporal_collate_sequences(
    sequences: list[Any],
    *,
    encoder: ObservationEncoder,
    late_pass_turn: int,
    late_pass_weight: float,
) -> dict[str, Any]:
    """Collate TemporalSequence objects into a training batch."""
    from generals_bot.training.sequence import temporal_collate_fn

    return temporal_collate_fn(
        sequences,
        encoder=encoder,
        late_pass_turn=late_pass_turn,
        late_pass_weight=late_pass_weight,
    )


def _save_temporal_checkpoint(
    config: TemporalTrainConfig,
    encoder: ObservationEncoder,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    metrics: dict[str, float],
    metric_log_path: Path,
    base_channels: int,
    global_step: int,
    samples_seen: int,
    last_val_metrics: dict[str, float],
    temporal_config: Any,
    data_epoch: int,
    output_path: str | Path | None = None,
) -> None:
    """Persist temporal model checkpoint."""
    from dataclasses import asdict as dc_asdict

    output = Path(config.output_path if output_path is None else output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "input_channels": encoder.num_channels,
            "feature_names": list(encoder.channel_names),
            "history_length": encoder.history_length,
            "base_channels": base_channels,
            "policy_mode": POLICY_MODE,
            "model_type": "temporal_policy_v1",
            "temporal_config": {
                "base_channels": base_channels,
                "temporal_dim": temporal_config.temporal_dim,
                "chunk_token_dim": temporal_config.chunk_token_dim,
                "chunk_transformer_layers": temporal_config.chunk_transformer_layers,
                "chunk_transformer_heads": temporal_config.chunk_transformer_heads,
                "chunk_transformer_ffn_dim": temporal_config.chunk_transformer_ffn_dim,
                "max_chunk_memory": temporal_config.max_chunk_memory,
                "dropout": temporal_config.dropout,
                "chunk_size": temporal_config.chunk_size,
                "encoder_type": temporal_config.encoder_type,
                "memory_type": temporal_config.memory_type,
                "gru_layers": temporal_config.gru_layers,
                "event_context_mode": temporal_config.event_context_mode,
                "spatial_grid_size": temporal_config.spatial_grid_size,
                "spatial_transformer_layers": temporal_config.spatial_transformer_layers,
            },
            "train_config": dc_asdict(config),
            "metrics": metrics,
            "metrics_output_path": str(metric_log_path),
            "tensorboard_log_dir": config.tensorboard_log_dir,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "data_epoch": data_epoch,
            "data_resume_policy": "next_reshuffled_epoch",
            "last_val_metrics": last_val_metrics,
        },
        output,
    )


def _validate_temporal_config(config: TemporalTrainConfig) -> None:
    """Validate temporal BC training configuration."""
    _validate_config(config)
    if config.sequence_length <= config.burn_in:
        raise ValueError("sequence_length must exceed burn_in")
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if config.burn_in < 0:
        raise ValueError("burn_in must be nonnegative")
    if config.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    if config.replay_shuffle_buffer_size < 0:
        raise ValueError("replay_shuffle_buffer_size must be nonnegative")
    for name, value in (
        ("max_val_predicted_pass_rate", config.max_val_predicted_pass_rate),
        ("max_val_move_false_pass_rate", config.max_val_move_false_pass_rate),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")


def _validate_temporal_model_training_alignment(
    config: TemporalTrainConfig,
    model_config: Any,
) -> None:
    """Reject temporal settings that leave inference memory untrained."""
    if config.chunk_size != model_config.chunk_size:
        raise ValueError("training chunk_size does not match model config")
    horizon = model_config.chunk_size * model_config.max_chunk_memory
    if config.sequence_length < horizon:
        raise ValueError(
            "sequence_length must cover chunk_size * max_chunk_memory "
            f"({config.sequence_length} < {horizon})"
        )
    if config.sequence_length % config.chunk_size != 0:
        raise ValueError("sequence_length must be divisible by chunk_size")
    if config.burn_in % config.chunk_size != 0:
        raise ValueError("burn_in must be divisible by chunk_size")
    stride = config.sequence_length - config.burn_in
    if stride % config.chunk_size != 0:
        raise ValueError("sequence stride must be divisible by chunk_size")


def _seed_training(seed: int) -> None:
    """Seed model initialization, dropout, and CPU/CUDA sampling."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_temporal_lr_schedule(
    optimizer: torch.optim.Optimizer,
    config: TemporalTrainConfig,
    *,
    step: int,
) -> None:
    """Apply linear warmup followed by cosine decay to all parameter groups."""
    if config.warmup_steps == 0 and config.min_lr_ratio == 1.0:
        return
    if config.warmup_steps > 0 and step <= config.warmup_steps:
        scale = step / config.warmup_steps
    else:
        decay_steps = max(1, config.steps - config.warmup_steps)
        progress = min(
            1.0,
            max(0.0, (step - config.warmup_steps) / decay_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine
    for group, base_lr in zip(
        optimizer.param_groups,
        (config.backbone_lr, config.temporal_lr, config.head_lr),
        strict=True,
    ):
        group["lr"] = base_lr * scale


def _safe_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute weighted mean, clamping denominator to avoid div-by-zero."""
    denominator = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denominator


def _weighted_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute sample-weighted BCEWithLogitsLoss."""
    loss_per_sample = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
    )
    return _safe_weighted_mean(loss_per_sample, weights)
