from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "torch is required for training losses. Install with: "
        ".venv/bin/python -m pip install -r requirements-train.txt"
    ) from exc


def enemy_king_prediction_loss(
    logits: torch.Tensor,
    *,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    distances: torch.Tensor,
    distance_weight: float,
    distance_cap: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute CE plus game-distance auxiliary loss for enemy general belief."""
    if distance_weight < 0:
        raise ValueError("distance_weight must be nonnegative")
    if distance_cap <= 0:
        raise ValueError("distance_cap must be positive")
    flat_logits = logits.flatten(start_dim=1)
    if flat_logits.shape != valid_mask.shape:
        raise ValueError("enemy king logits and valid mask shapes do not match")
    labels = labels.to(flat_logits.device)
    valid_mask = valid_mask.to(flat_logits.device)
    distances = distances.to(flat_logits.device, dtype=torch.float32)
    mask_value = torch.finfo(flat_logits.dtype).min
    masked_logits = flat_logits.masked_fill(~valid_mask, mask_value)
    ce_loss = F.cross_entropy(masked_logits, labels)
    probabilities = torch.softmax(masked_logits, dim=1)
    normalized_distances = distances.clamp_max(distance_cap) / distance_cap
    distance_loss = (probabilities * normalized_distances).sum(dim=1).mean()
    loss = ce_loss + distance_weight * distance_loss

    predictions = masked_logits.argmax(dim=1)
    prediction_distances = distances.gather(1, predictions[:, None]).squeeze(1)
    target_is_valid = valid_mask.gather(1, labels[:, None]).squeeze(1)
    metrics = {
        "enemy_king_loss": float(loss.detach().cpu()),
        "enemy_king_ce_loss": float(ce_loss.detach().cpu()),
        "enemy_king_distance_loss": float(distance_loss.detach().cpu()),
        "enemy_king_top1_acc": float((predictions == labels).to(torch.float32).mean().detach().cpu()),
        "enemy_king_mean_distance": float(
            prediction_distances.clamp_max(distance_cap).mean().detach().cpu()
        ),
        "enemy_king_target_valid_rate": float(
            target_is_valid.to(torch.float32).mean().detach().cpu()
        ),
    }
    return loss, metrics
