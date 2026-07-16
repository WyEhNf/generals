from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "torch is required for training models. Install with: "
        ".venv/bin/python -m pip install -r requirements-train.txt"
    ) from exc


class ResidualBlock(nn.Module):
    """Small residual convolution block used throughout the policy net."""

    def __init__(self, channels: int) -> None:
        """Create a same-shape residual block."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block."""
        return self.activation(x + self.net(x))


class DownBlock(nn.Module):
    """Stride-2 downsampling block for the U-Net encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create a downsampling block."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample and refine a feature map."""
        return self.net(x)


class UpBlock(nn.Module):
    """Upsampling block that merges a skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """Create an upsampling block."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample x to the skip resolution and merge features."""
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.net(torch.cat([x, skip], dim=1))


POLICY_MODE = "hierarchical"
PREDICTOR_MODEL_TYPE = "enemy_king_predictor_v1"
ROUTER_MODEL_TYPE = "router_policy_v1"
OPTIONAL_STATE_PREFIXES = ("enemy_king_head.", "bottleneck_fuse.")


class BCPolicyNet(nn.Module):
    """U-Net style behavior cloning network with policy, value, and auxiliary heads.

    The forward pass can be split into :meth:`encode_frame` and
    :meth:`decode_policy` so that temporal models can inject belief
    state at the bottleneck before decoding.
    """

    def __init__(
        self,
        input_channels: int,
        base_channels: int = 64,
        *,
        temporal_dim: int = 0,
    ) -> None:
        """Create the policy network.

        When *temporal_dim* > 0, a 1×1 bottleneck-fusion convolution is
        created so that :meth:`decode_policy` can accept an optional belief
        tensor of shape ``(B, temporal_dim)``.
        """
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        self._base_channels = base_channels
        self._temporal_dim = temporal_dim
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c1, 3, padding=1),
            nn.GroupNorm(_groups(c1), c1),
            nn.SiLU(),
            ResidualBlock(c1),
        )
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.bottleneck = ResidualBlock(c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.move_head = nn.Conv2d(c1, 8, 1)
        self.enemy_king_head = nn.Conv2d(c1, 1, 1)
        self.pass_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, c1),
            nn.SiLU(),
            nn.Linear(c1, 1),
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, c1),
            nn.SiLU(),
            nn.Linear(c1, 1),
        )
        if temporal_dim > 0:
            self.bottleneck_fuse = nn.Conv2d(c3 + temporal_dim, c3, 1)
        else:
            self.bottleneck_fuse = nn.Identity()

    def encode_frame(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a frame into skip features and a bottleneck.

        Returns ``(s1, s2, bottleneck)`` for use with :meth:`decode_policy`.
        """
        s1 = self.stem(x)
        s2 = self.down1(s1)
        bn = self.down2(s2)
        bn = self.bottleneck(bn)
        return s1, s2, bn

    def decode_policy(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        bottleneck: torch.Tensor,
        *,
        belief: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode bottleneck and skip features into policy outputs.

        A vector belief is spatially tiled. A spatial belief map is resized
        to the bottleneck resolution before concatenation.
        """
        x = bottleneck
        if belief is not None and self._temporal_dim > 0:
            if belief.ndim == 2:
                belief_spatial = belief[:, :, None, None].expand(
                    -1, -1, x.shape[-2], x.shape[-1],
                )
            elif belief.ndim == 4:
                belief_spatial = F.interpolate(
                    belief,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                raise ValueError("belief must have shape (B,D) or (B,D,H,W)")
            x = torch.cat([x, belief_spatial], dim=1)
            x = self.bottleneck_fuse(x)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        action_logits = self.pass_head(x)
        return {
            "move_logits": self.move_head(x),
            "action_logits": action_logits,
            "pass_logits": action_logits,
            "value": self.value_head(x).squeeze(-1),
            "enemy_king_logits": self.enemy_king_head(x),
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return policy logits, value estimate, and auxiliary prediction logits."""
        s1, s2, bn = self.encode_frame(x)
        return self.decode_policy(s1, s2, bn)


class EnemyKingPredictorNet(nn.Module):
    """U-Net style predictor that only estimates the enemy general location."""

    def __init__(self, input_channels: int, base_channels: int = 64) -> None:
        """Create the predictor network."""
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c1, 3, padding=1),
            nn.GroupNorm(_groups(c1), c1),
            nn.SiLU(),
            ResidualBlock(c1),
        )
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.bottleneck = ResidualBlock(c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.enemy_king_head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return enemy general belief logits."""
        s1 = self.stem(x)
        s2 = self.down1(s1)
        x = self.down2(s2)
        x = self.bottleneck(x)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return {"enemy_king_logits": self.enemy_king_head(x)}


class RouterPolicyNet(nn.Module):
    """U-Net style value network that selects one frozen expert per turn."""

    def __init__(
        self,
        input_channels: int,
        *,
        expert_count: int,
        base_channels: int = 64,
    ) -> None:
        """Create the router network."""
        super().__init__()
        if expert_count <= 0:
            raise ValueError("expert_count must be positive")
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c1, 3, padding=1),
            nn.GroupNorm(_groups(c1), c1),
            nn.SiLU(),
            ResidualBlock(c1),
        )
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.bottleneck = ResidualBlock(c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.expert_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, c1),
            nn.SiLU(),
            nn.Linear(c1, expert_count),
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, c1),
            nn.SiLU(),
            nn.Linear(c1, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return expert logits and a player-relative value estimate."""
        s1 = self.stem(x)
        s2 = self.down1(s1)
        x = self.down2(s2)
        x = self.bottleneck(x)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return {
            "expert_logits": self.expert_head(x),
            "value": self.value_head(x).squeeze(-1),
        }


def flatten_policy_logits(move_logits: torch.Tensor, pass_logits: torch.Tensor) -> torch.Tensor:
    """Flatten spatial move logits and append pass logits."""
    return torch.cat([move_logits.flatten(start_dim=1), pass_logits], dim=1)


def load_model_state_compatible(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> list[str]:
    """Load state dicts while allowing newly added optional heads to be absent."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    bad_missing = [
        key
        for key in missing
        if not key.startswith(OPTIONAL_STATE_PREFIXES)
    ]
    if bad_missing or unexpected:
        details = []
        if bad_missing:
            details.append(f"missing keys: {bad_missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise RuntimeError("checkpoint model_state is incompatible: " + "; ".join(details))
    return missing


def _groups(channels: int) -> int:
    """Choose a valid GroupNorm group count for a channel size."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
