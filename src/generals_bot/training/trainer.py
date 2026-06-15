from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
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


def train_bc(config: TrainConfig) -> dict[str, Any]:
    """Train a BC policy model and save a checkpoint."""
    _validate_config(config)
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
    ):
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
