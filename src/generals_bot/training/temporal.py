"""Temporal policy network with within-chunk accumulation and chunk memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TemporalModelConfig:
    """Fixed-chunk temporal model hyperparameters."""

    base_channels: int = 64
    temporal_dim: int = 192
    chunk_token_dim: int = 192
    chunk_transformer_layers: int = 2
    chunk_transformer_heads: int = 4
    chunk_transformer_ffn_dim: int = 512
    max_chunk_memory: int = 32
    dropout: float = 0.0
    chunk_size: int = 8
    encoder_type: str = "global"
    memory_type: str = "transformer"
    gru_layers: int = 3
    event_context_mode: str = "feature"
    spatial_grid_size: int = 4
    spatial_transformer_layers: int = 1
    history_readout: str = "last_belief"

    def __post_init__(self) -> None:
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.temporal_dim <= 0 or self.chunk_token_dim <= 0:
            raise ValueError("temporal dimensions must be positive")
        if self.chunk_transformer_layers <= 0:
            raise ValueError("chunk_transformer_layers must be positive")
        if self.chunk_transformer_heads <= 0:
            raise ValueError("chunk_transformer_heads must be positive")
        if self.temporal_dim % self.chunk_transformer_heads != 0:
            raise ValueError("temporal_dim must be divisible by transformer heads")
        if self.chunk_transformer_ffn_dim <= 0:
            raise ValueError("chunk_transformer_ffn_dim must be positive")
        if self.max_chunk_memory <= 0 or self.chunk_size <= 0:
            raise ValueError("chunk memory and chunk size must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.encoder_type not in {"global", "ewsc"}:
            raise ValueError(f"unknown temporal encoder_type: {self.encoder_type}")
        if self.memory_type not in {"transformer", "gru"}:
            raise ValueError(f"unknown temporal memory_type: {self.memory_type}")
        if self.gru_layers <= 0:
            raise ValueError("gru_layers must be positive")
        if self.encoder_type == "ewsc" and self.memory_type != "transformer":
            raise ValueError("ewsc currently requires transformer chunk memory")
        if self.event_context_mode not in {"feature", "explicit"}:
            raise ValueError(
                f"unknown event_context_mode: {self.event_context_mode}"
            )
        if self.encoder_type != "ewsc" and self.event_context_mode != "feature":
            raise ValueError("explicit event context is only valid for ewsc")
        if self.spatial_grid_size <= 0 or self.spatial_transformer_layers <= 0:
            raise ValueError("spatial encoder dimensions must be positive")
        if self.history_readout not in {"last_belief", "cross_attention"}:
            raise ValueError(
                f"unknown temporal history_readout: {self.history_readout}"
            )
        if self.history_readout == "cross_attention" and self.encoder_type != "ewsc":
            raise ValueError("cross-attention history readout requires ewsc")

    @classmethod
    def from_json(cls, path: str | Path) -> TemporalModelConfig:
        """Load configuration from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("temporal model config must be a JSON object")
        if data.get("version") != 1:
            raise ValueError("temporal model config must contain version=1")
        return cls(
            base_channels=int(data.get("base_channels", 64)),
            temporal_dim=int(data.get("temporal_dim", 192)),
            chunk_token_dim=int(data.get("chunk_token_dim", 192)),
            chunk_transformer_layers=int(data.get("chunk_transformer_layers", 2)),
            chunk_transformer_heads=int(data.get("chunk_transformer_heads", 4)),
            chunk_transformer_ffn_dim=int(data.get("chunk_transformer_ffn_dim", 512)),
            max_chunk_memory=int(data.get("max_chunk_memory", 32)),
            dropout=float(data.get("dropout", 0.0)),
            chunk_size=int(data.get("chunk_size", 8)),
            encoder_type=str(data.get("encoder_type", "global")),
            memory_type=str(data.get("memory_type", "transformer")),
            gru_layers=int(data.get("gru_layers", 3)),
            event_context_mode=str(data.get("event_context_mode", "feature")),
            spatial_grid_size=int(data.get("spatial_grid_size", 4)),
            spatial_transformer_layers=int(data.get("spatial_transformer_layers", 1)),
            history_readout=str(data.get("history_readout", "last_belief")),
        )


# ---------------------------------------------------------------------------
# Within-Chunk Gated Accumulator
# ---------------------------------------------------------------------------


class WithinChunkAccumulator(nn.Module):
    """Gated recurrent accumulator that compresses per-frame bottleneck features.

    At each tick the accumulator updates an internal spatial state *S* via
    a learned gate:

        gate = σ(Conv([features, S]))
        S_new = gate · S + (1 − gate) · features

    The state tensor has shape ``(B, C, H, W)``.  Batch elements are
    independent — calling :meth:`finalize` with a mask only finalizes the
    selected rows, while others retain their accumulated state.
    """

    def __init__(self, channels: int = 192):
        super().__init__()
        self.gate_conv = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.channels = channels
        self._state: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Update the internal state with new features and return the new state."""
        if self._state is None or self._state.shape != features.shape:
            self._state = torch.zeros_like(features)
        gate_input = torch.cat([features, self._state], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_input))
        self._state = gate * self._state + (1.0 - gate) * features
        return self._state

    def finalize(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool the current state into chunk tokens and reset finalized rows.

        When *mask* is ``None`` all rows are finalized and the state is
        reset.  Otherwise only rows where *mask* is ``True`` are finalized;
        those rows' state is zeroed while other rows keep their
        accumulated state.

        Returns a tensor of shape ``(N, channels)`` where *N* is the
        number of finalized rows.
        """
        if self._state is None:
            raise RuntimeError("finalize called before any forward pass")
        if mask is None:
            token = F.adaptive_avg_pool2d(self._state, 1).squeeze(-1).squeeze(-1)
            self._state = None
            return token
        if not bool(mask.any()):
            return self._state.new_zeros(0, self.channels)
        selected = self._state[mask]
        token = F.adaptive_avg_pool2d(selected, 1).squeeze(-1).squeeze(-1)
        self._state[mask] = 0.0
        return token

    def reset(self) -> None:
        """Discard accumulated state."""
        self._state = None


class EventWeightedSpatialAccumulator(nn.Module):
    """Compress a chunk without discarding its coarse spatial layout.

    Feature-change magnitude supplies causal per-cell temporal attention.  The
    output combines event-weighted mean, last frame, net delta, and temporal
    maximum before pooling to a fixed spatial grid.
    """

    def __init__(
        self,
        channels: int,
        token_dim: int,
        grid_size: int,
        *,
        explicit_event_context: bool = False,
    ) -> None:
        super().__init__()
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        self.grid_size = grid_size
        self.explicit_event_context = bool(explicit_event_context)
        self.project = nn.Conv2d(channels * 4, token_dim, 1)
        if self.explicit_event_context:
            # feature, visibility, ownership, army deltas.  Softplus keeps all
            # event contributions non-negative while allowing them to learn.
            initial = torch.log(torch.expm1(torch.tensor(1.0)))
            self.event_weight_logits = nn.Parameter(initial.repeat(4))
        else:
            self.register_parameter("event_weight_logits", None)
        self._first: torch.Tensor | None = None
        self._last: torch.Tensor | None = None
        self._maximum: torch.Tensor | None = None
        self._weighted_sum: torch.Tensor | None = None
        self._weight_sum: torch.Tensor | None = None
        self._score_max: torch.Tensor | None = None
        self._active: torch.Tensor | None = None
        self._last_event_context: torch.Tensor | None = None

    def forward(
        self,
        features: torch.Tensor,
        *,
        event_context: torch.Tensor | None = None,
    ) -> None:
        if self.explicit_event_context:
            if event_context is None:
                raise ValueError("explicit EWSC requires event_context")
            if event_context.shape[1] != 5:
                raise ValueError(
                    "event_context must contain visible, three ownership, and army channels"
                )
            if event_context.shape[0] != features.shape[0] or event_context.shape[-2:] != features.shape[-2:]:
                raise ValueError("event_context must match feature batch and spatial shape")
        batch = features.shape[0]
        if self._last is None or self._last.shape != features.shape:
            self._initialize(features, event_context=event_context)
            return
        assert self._active is not None
        assert self._first is not None and self._maximum is not None
        assert self._weighted_sum is not None and self._weight_sum is not None
        assert self._score_max is not None
        active = self._active[:, None, None, None]
        feature_delta = (features - self._last).abs().mean(dim=1, keepdim=True)
        if self.explicit_event_context:
            assert event_context is not None and self._last_event_context is not None
            visibility_delta = (
                event_context[:, 0:1] - self._last_event_context[:, 0:1]
            ).abs()
            ownership_delta = (
                event_context[:, 1:4] - self._last_event_context[:, 1:4]
            ).abs().mean(dim=1, keepdim=True)
            army_delta = (
                event_context[:, 4:5] - self._last_event_context[:, 4:5]
            ).abs()
            components = torch.cat(
                [feature_delta, visibility_delta, ownership_delta, army_delta],
                dim=1,
            )
            assert self.event_weight_logits is not None
            weights = F.softplus(self.event_weight_logits)[None, :, None, None]
            score = (components * weights).sum(dim=1, keepdim=True)
        else:
            score = feature_delta
        new_max = torch.maximum(self._score_max, score)
        old_scale = torch.exp(self._score_max - new_max)
        new_scale = torch.exp(score - new_max)
        updated_sum = self._weighted_sum * old_scale + features * new_scale
        updated_weight = self._weight_sum * old_scale + new_scale
        self._first = torch.where(active, self._first, features)
        self._maximum = torch.where(active, torch.maximum(self._maximum, features), features)
        self._weighted_sum = torch.where(active, updated_sum, features)
        self._weight_sum = torch.where(active, updated_weight, torch.ones_like(score))
        self._score_max = torch.where(active, new_max, torch.zeros_like(score))
        self._last = features
        self._last_event_context = event_context
        self._active = torch.ones(batch, dtype=torch.bool, device=features.device)

    def _initialize(
        self,
        features: torch.Tensor,
        *,
        event_context: torch.Tensor | None,
    ) -> None:
        self._first = features
        self._last = features
        self._maximum = features
        self._weighted_sum = features
        shape = (features.shape[0], 1, features.shape[-2], features.shape[-1])
        self._weight_sum = features.new_ones(shape)
        self._score_max = features.new_zeros(shape)
        self._active = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
        self._last_event_context = event_context

    def finalize(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self._last is None or self._active is None:
            raise RuntimeError("finalize called before any forward pass")
        selected = self._active if mask is None else mask.to(dtype=torch.bool)
        if not bool(selected.any()):
            return self._last.new_zeros(
                0, self.project.out_channels, self.grid_size, self.grid_size,
            )
        assert self._first is not None and self._maximum is not None
        assert self._weighted_sum is not None and self._weight_sum is not None
        weighted_mean = self._weighted_sum[selected] / self._weight_sum[selected].clamp_min(1e-6)
        last = self._last[selected]
        combined = torch.cat(
            [weighted_mean, last, last - self._first[selected], self._maximum[selected]],
            dim=1,
        )
        tokens = F.adaptive_avg_pool2d(
            self.project(combined), (self.grid_size, self.grid_size),
        )
        if mask is None or bool(selected.all()):
            self.reset()
        else:
            self._active = self._active & ~selected
        return tokens

    def reset(self) -> None:
        self._first = None
        self._last = None
        self._maximum = None
        self._weighted_sum = None
        self._weight_sum = None
        self._score_max = None
        self._active = None
        self._last_event_context = None


# ---------------------------------------------------------------------------
# Causal Chunk Memory Transformer
# ---------------------------------------------------------------------------


class ChunkMemoryTransformer(nn.Module):
    """Causal Transformer over completed chunk tokens.

    Maintains a sliding window of up to *max_chunk_memory* past chunk
    tokens.  A causal (lower-triangular) self-attention mask ensures
    each position only attends to itself and earlier positions.  The
    belief state is the output at the last (most recent) position.
    """

    def __init__(self, config: TemporalModelConfig):
        super().__init__()
        self.max_chunk_memory = config.max_chunk_memory
        self.temporal_dim = config.temporal_dim
        self.input_proj = nn.Linear(config.chunk_token_dim, config.temporal_dim)
        self.pos_embed = nn.Embedding(config.max_chunk_memory, config.temporal_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.temporal_dim,
            nhead=config.chunk_transformer_heads,
            dim_feedforward=config.chunk_transformer_ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            activation=F.gelu,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.chunk_transformer_layers,
        )
        # Causal mask: position i can attend to positions 0..i
        self._causal_mask: torch.Tensor | None = None
        # Separate histories are required when only a subset of batch rows
        # reaches a chunk boundary on a given tick.
        self._memory_by_row: list[torch.Tensor | None] = []

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return a ``(seq_len, seq_len)`` causal mask (True = masked)."""
        if self._causal_mask is not None and self._causal_mask.shape[0] >= seq_len:
            return self._causal_mask[:seq_len, :seq_len].to(device)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        self._causal_mask = mask
        return mask

    def forward(
        self,
        new_token: torch.Tensor,
        *,
        row_indices: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Append a new chunk token and return the updated belief state.

        Args:
            new_token: ``(B, chunk_token_dim)`` tensor from the accumulator.

        Returns:
            ``(B, temporal_dim)`` belief state.
        """
        selected_count = new_token.shape[0]
        if row_indices is None:
            row_indices = torch.arange(selected_count, device=new_token.device)
        if batch_size is None:
            batch_size = selected_count
        if row_indices.numel() != selected_count:
            raise ValueError("row_indices must match the number of new tokens")
        if len(self._memory_by_row) != batch_size:
            self._memory_by_row = [None] * batch_size

        projected = self.input_proj(new_token)
        beliefs: list[torch.Tensor] = []
        for local_index in range(selected_count):
            row = int(row_indices[local_index])
            if row < 0 or row >= batch_size:
                raise ValueError("row index is outside the logical batch")
            token = projected[local_index : local_index + 1]
            memory = self._memory_by_row[row]
            memory = token if memory is None else torch.cat([memory, token], dim=0)
            memory = memory[-self.max_chunk_memory :]
            self._memory_by_row[row] = memory

            seq_len = memory.shape[0]
            positions = torch.arange(seq_len, device=new_token.device)
            sequence = (memory + self.pos_embed(positions)).unsqueeze(0)
            encoded = self.transformer(
                sequence,
                mask=self._get_causal_mask(seq_len, new_token.device),
                is_causal=True,
            )
            beliefs.append(encoded[0, -1])
        if beliefs:
            output = torch.stack(beliefs, dim=0).unsqueeze(1)
        else:
            output = projected.new_zeros((0, 1, self.temporal_dim))
        return output[:, -1, :]  # (B, D) — last position

    def reset(self) -> None:
        """Clear stored chunk memory."""
        self._memory_by_row = []

    @property
    def num_chunks(self) -> int:
        """Return the number of chunks currently in memory."""
        return max(
            (0 if memory is None else memory.shape[0])
            for memory in self._memory_by_row
        ) if self._memory_by_row else 0


class ChunkMemoryGRU(nn.Module):
    """Parameter-matched non-Transformer memory over the same chunk window.

    The GRU is recomputed from a zero state over the most recent
    ``max_chunk_memory`` tokens whenever a chunk completes.  This prevents an
    unbounded recurrent history and makes its physical receptive field exactly
    match the Transformer baseline.
    """

    def __init__(self, config: TemporalModelConfig) -> None:
        super().__init__()
        self.max_chunk_memory = config.max_chunk_memory
        self.temporal_dim = config.temporal_dim
        self.input_proj = nn.Linear(config.chunk_token_dim, config.temporal_dim)
        self.pos_embed = nn.Embedding(config.max_chunk_memory, config.temporal_dim)
        self.gru = nn.GRU(
            input_size=config.temporal_dim,
            hidden_size=config.temporal_dim,
            num_layers=config.gru_layers,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(config.temporal_dim, config.temporal_dim)
        self.output_norm = nn.LayerNorm(config.temporal_dim)
        self._memory_by_row: list[torch.Tensor | None] = []

    def forward(
        self,
        new_token: torch.Tensor,
        *,
        row_indices: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        selected_count = new_token.shape[0]
        if row_indices is None:
            row_indices = torch.arange(selected_count, device=new_token.device)
        if batch_size is None:
            batch_size = selected_count
        if row_indices.numel() != selected_count:
            raise ValueError("row_indices must match the number of new tokens")
        if len(self._memory_by_row) != batch_size:
            self._memory_by_row = [None] * batch_size

        projected = self.input_proj(new_token)
        beliefs: list[torch.Tensor] = []
        for local_index in range(selected_count):
            row = int(row_indices[local_index])
            if row < 0 or row >= batch_size:
                raise ValueError("row index is outside the logical batch")
            token = projected[local_index : local_index + 1]
            memory = self._memory_by_row[row]
            memory = token if memory is None else torch.cat([memory, token], dim=0)
            memory = memory[-self.max_chunk_memory :]
            self._memory_by_row[row] = memory

            positions = torch.arange(memory.shape[0], device=memory.device)
            sequence = (memory + self.pos_embed(positions)).unsqueeze(0)
            encoded, _ = self.gru(sequence)
            latest = encoded[0, -1]
            beliefs.append(
                self.output_norm(self.output_proj(latest) + sequence[0, -1])
            )
        if not beliefs:
            return projected.new_zeros((0, self.temporal_dim))
        return torch.stack(beliefs, dim=0)

    def reset(self) -> None:
        self._memory_by_row = []

    @property
    def num_chunks(self) -> int:
        return max(
            (0 if memory is None else memory.shape[0])
            for memory in self._memory_by_row
        ) if self._memory_by_row else 0


class SpatialChunkMemoryTransformer(nn.Module):
    """Factorized spatial/temporal memory with one history per batch row.

    In addition to the latest spatial belief map, this module retains every
    causally encoded ``time x space`` token in the bounded window.  Phase 2
    decoders can therefore query the history directly instead of forcing the
    most recent token to summarize the entire sequence.
    """

    def __init__(self, config: TemporalModelConfig) -> None:
        super().__init__()
        self.grid_size = config.spatial_grid_size
        self.max_chunk_memory = config.max_chunk_memory
        self.temporal_dim = config.temporal_dim
        self.input_proj = nn.Linear(config.chunk_token_dim, config.temporal_dim)
        self.row_pos = nn.Embedding(self.grid_size, config.temporal_dim)
        self.col_pos = nn.Embedding(self.grid_size, config.temporal_dim)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=config.temporal_dim,
            nhead=config.chunk_transformer_heads,
            dim_feedforward=config.chunk_transformer_ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            activation=F.gelu,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            spatial_layer, num_layers=config.spatial_transformer_layers,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.temporal_dim,
            nhead=config.chunk_transformer_heads,
            dim_feedforward=config.chunk_transformer_ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            activation=F.gelu,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=config.chunk_transformer_layers,
        )
        self.time_pos = nn.Embedding(config.max_chunk_memory, config.temporal_dim)
        self._memory_by_row: list[torch.Tensor | None] = []
        self._encoded_by_row: list[torch.Tensor | None] = []

    def forward(
        self,
        new_map: torch.Tensor,
        *,
        row_indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Append maps for selected rows and return their updated belief maps."""
        if new_map.shape[-2:] != (self.grid_size, self.grid_size):
            raise ValueError("spatial token map does not match configured grid")
        if len(self._memory_by_row) != batch_size:
            self._memory_by_row = [None] * batch_size
            self._encoded_by_row = [None] * batch_size
        count = new_map.shape[0]
        tokens = new_map.flatten(2).transpose(1, 2)
        tokens = self.input_proj(tokens)
        rows = torch.arange(self.grid_size, device=new_map.device).repeat_interleave(self.grid_size)
        cols = torch.arange(self.grid_size, device=new_map.device).repeat(self.grid_size)
        tokens = tokens + self.row_pos(rows)[None] + self.col_pos(cols)[None]
        tokens = self.spatial_transformer(tokens)
        beliefs = []
        for local_index in range(count):
            row = int(row_indices[local_index])
            current = tokens[local_index].unsqueeze(1)  # spatial, time, dim
            memory = self._memory_by_row[row]
            memory = current if memory is None else torch.cat([memory, current], dim=1)
            memory = memory[:, -self.max_chunk_memory :, :]
            self._memory_by_row[row] = memory
            positions = torch.arange(memory.shape[1], device=memory.device)
            sequence = memory + self.time_pos(positions)[None]
            causal = torch.triu(
                torch.ones(
                    memory.shape[1], memory.shape[1],
                    dtype=torch.bool, device=memory.device,
                ),
                diagonal=1,
            )
            output = self.temporal_transformer(
                sequence, mask=causal, is_causal=True,
            )
            # ``output`` is (space, time, dim).  Store time-major history so
            # the policy decoder can flatten it into an addressable memory.
            self._encoded_by_row[row] = output.transpose(0, 1)
            belief = output[:, -1].transpose(0, 1).reshape(
                self.temporal_dim, self.grid_size, self.grid_size,
            )
            beliefs.append(belief)
        return torch.stack(beliefs, dim=0)

    def encoded_history(
        self,
        *,
        row_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded, causally encoded history and its padding mask.

        The returned memory has shape ``(B, T*S, D)`` and is suitable for
        ``nn.MultiheadAttention``.  ``padding_mask`` has shape ``(B, T*S)``
        where True entries are padding.  Rows without a completed chunk are
        rejected so callers cannot accidentally attend to an all-masked row.
        """
        if row_indices is None:
            rows = list(range(len(self._encoded_by_row)))
        else:
            rows = [int(row) for row in row_indices]
        histories = [
            self._encoded_by_row[row]
            if 0 <= row < len(self._encoded_by_row)
            else None
            for row in rows
        ]
        if not histories or any(history is None for history in histories):
            raise RuntimeError("encoded history requested before a completed chunk")
        concrete = [history for history in histories if history is not None]
        max_time = max(history.shape[0] for history in concrete)
        padded = []
        masks = []
        spatial_tokens = self.grid_size * self.grid_size
        for history in concrete:
            time = history.shape[0]
            if time < max_time:
                history = F.pad(history, (0, 0, 0, 0, 0, max_time - time))
            padded.append(history.reshape(max_time * spatial_tokens, self.temporal_dim))
            valid = torch.arange(max_time, device=history.device) < time
            masks.append(
                (~valid[:, None].expand(max_time, spatial_tokens)).reshape(-1)
            )
        return torch.stack(padded, dim=0), torch.stack(masks, dim=0)

    def reset(self) -> None:
        self._memory_by_row = []
        self._encoded_by_row = []

    @property
    def num_chunks(self) -> int:
        return max(
            (0 if memory is None else memory.shape[1])
            for memory in self._memory_by_row
        ) if self._memory_by_row else 0


# ---------------------------------------------------------------------------
# Temporal Policy Network
# ---------------------------------------------------------------------------


class TemporalPolicyNet(nn.Module):
    """U-Net policy with within-chunk accumulation and chunk-memory Transformer.

    This wraps a :class:`BCPolicyNet` spatial backbone and adds temporal
    reasoning in the bottleneck.  The spatial net is used via its
    ``encode_frame`` / ``decode_policy`` interface so that belief from the
    chunk Transformer can be injected before decoding.
    """

    MODEL_TYPE = "temporal_policy_v1"

    def __init__(
        self,
        input_channels: int,
        spatial_net: nn.Module,
        config: TemporalModelConfig,
    ):
        super().__init__()
        self.spatial = spatial_net
        self.config = config
        if config.encoder_type == "global":
            if config.chunk_token_dim != config.base_channels * 3:
                raise ValueError(
                    "global encoder requires chunk_token_dim == base_channels * 3"
                )
            self.accumulator = WithinChunkAccumulator(
                channels=config.base_channels * 3,
            )
            self.chunk_memory = (
                ChunkMemoryTransformer(config)
                if config.memory_type == "transformer"
                else ChunkMemoryGRU(config)
            )
        elif config.encoder_type == "ewsc":
            self.accumulator = EventWeightedSpatialAccumulator(
                channels=config.base_channels * 3,
                token_dim=config.chunk_token_dim,
                grid_size=config.spatial_grid_size,
                explicit_event_context=config.event_context_mode == "explicit",
            )
            self.chunk_memory = SpatialChunkMemoryTransformer(config)
        else:
            raise ValueError(f"unknown temporal encoder_type: {config.encoder_type}")
        self._belief: torch.Tensor | None = None
        self._belief_valid: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        chunk_boundary: torch.Tensor | None = None,
        reset_memory: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Per-tick forward pass.

        Args:
            x: ``(B, C, H, W)`` observation tensor.
            chunk_boundary: ``(B,)`` bool tensor — True at chunk boundaries.
            reset_memory: If True, clear accumulator and chunk memory first.

        Returns:
            Same dict as :meth:`BCPolicyNet.forward`.
        """
        if reset_memory:
            self.reset()

        s1, s2, bottleneck = self.spatial.encode_frame(x)

        if self.config.encoder_type == "ewsc":
            event_context = (
                self._build_explicit_event_context(x, bottleneck)
                if self.config.event_context_mode == "explicit"
                else None
            )
            self.accumulator(bottleneck, event_context=event_context)
            return self._forward_ewsc(
                s1, s2, bottleneck, chunk_boundary=chunk_boundary,
            )

        self.accumulator(bottleneck)

        batch = bottleneck.shape[0]
        if self._belief is not None and self._belief.shape[0] != batch:
            self._belief = None
            self._belief_valid = None
            self.chunk_memory.reset()
        if chunk_boundary is not None and bool(chunk_boundary.any()):
            selected = chunk_boundary.to(dtype=torch.bool)
            row_indices = selected.nonzero(as_tuple=False).squeeze(1)
            token = self.accumulator.finalize(mask=selected)
            updated = self.chunk_memory(
                token,
                row_indices=row_indices,
                batch_size=batch,
            )
            belief = (
                updated.new_zeros(batch, updated.shape[1])
                if self._belief is None
                else self._belief.clone()
            )
            valid = (
                torch.zeros(batch, dtype=torch.bool, device=updated.device)
                if self._belief_valid is None
                else self._belief_valid.clone()
            )
            belief[row_indices] = updated
            valid[row_indices] = True
            self._belief = belief
            self._belief_valid = valid

        return self._decode_with_persistent_belief(
            s1,
            s2,
            bottleneck,
            belief=self._belief,
            valid=self._belief_valid,
        )

    @staticmethod
    def _build_explicit_event_context(
        x: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        """Downsample visible, ownership, and army planes for EWSC scoring."""
        if x.shape[1] < 13:
            raise ValueError("explicit EWSC requires the standard 22-channel encoder")
        # ObservationEncoder channel order: own/enemy/neutral at 0:3,
        # visible at 6, and normalized per-cell army_log at 12.
        context = torch.cat(
            [x[:, 6:7], x[:, 0:3], x[:, 12:13]],
            dim=1,
        )
        return F.adaptive_avg_pool2d(context, bottleneck.shape[-2:])

    def _forward_ewsc(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        bottleneck: torch.Tensor,
        *,
        chunk_boundary: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Update selected spatial memories and decode with persistent belief."""
        batch = bottleneck.shape[0]
        if self._belief is not None and self._belief.shape[0] != batch:
            self._belief = None
            self._belief_valid = None
            self.chunk_memory.reset()
        if chunk_boundary is not None and bool(chunk_boundary.any()):
            selected = chunk_boundary.to(dtype=torch.bool)
            token_maps = self.accumulator.finalize(mask=selected)
            row_indices = selected.nonzero(as_tuple=False).squeeze(1)
            updated = self.chunk_memory(
                token_maps, row_indices=row_indices, batch_size=batch,
            )
            if self._belief is None:
                belief = updated.new_zeros(
                    batch, updated.shape[1], updated.shape[2], updated.shape[3],
                )
            else:
                belief = self._belief.clone()
            valid = (
                torch.zeros(batch, dtype=torch.bool, device=updated.device)
                if self._belief_valid is None
                else self._belief_valid.clone()
            )
            belief[row_indices] = updated
            valid[row_indices] = True
            self._belief = belief
            self._belief_valid = valid
        return self._decode_with_persistent_belief(
            s1,
            s2,
            bottleneck,
            belief=self._belief,
            valid=self._belief_valid,
        )

    def _decode_with_persistent_belief(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        bottleneck: torch.Tensor,
        *,
        belief: torch.Tensor | None,
        valid: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Fuse memory only after a row has completed its first chunk."""
        if belief is None or valid is None or not bool(valid.any()):
            return self.spatial.decode_policy(s1, s2, bottleneck, belief=None)
        if bool(valid.all()):
            return self.spatial.decode_policy(s1, s2, bottleneck, belief=belief)

        with_belief = self.spatial.decode_policy(
            s1[valid], s2[valid], bottleneck[valid], belief=belief[valid],
        )
        without_belief = self.spatial.decode_policy(
            s1[~valid], s2[~valid], bottleneck[~valid], belief=None,
        )
        output: dict[str, torch.Tensor] = {}
        for key, selected_output in with_belief.items():
            full = torch.empty(
                bottleneck.shape[0], *selected_output.shape[1:],
                device=selected_output.device,
                dtype=selected_output.dtype,
            )
            full[valid] = selected_output
            full[~valid] = without_belief[key]
            output[key] = full
        return output

    def reset(self) -> None:
        """Clear all temporal state."""
        self.accumulator.reset()
        self.chunk_memory.reset()
        self._belief = None
        self._belief_valid = None
