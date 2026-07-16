from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "torch is required for model policy. Install with: "
        ".venv/bin/python -m pip install -r requirements-train.txt"
    ) from exc

from generals_bot.sim.types import Action, Observation, PassAction
from generals_bot.training.action_codec import label_to_action, valid_action_mask
from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.model import (
    BCPolicyNet,
    POLICY_MODE,
    flatten_policy_logits,
    load_model_state_compatible,
)
from generals_bot.training.policy import apply_pass_bias


ActionKey = Hashable
ActionLike = Action | PassAction


@dataclass
class BatchedPolicyStats:
    """Timing and batch-size counters for batched policy inference."""

    timings: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    batches: int = 0
    actions: int = 0

    def add_time(self, name: str, seconds: float) -> None:
        """Add elapsed wall time for one measured section."""
        self.timings[name] = self.timings.get(name, 0.0) + seconds

    def add_batch(self, size: int) -> None:
        """Record one model batch."""
        self.batches += 1
        self.actions += size

    @property
    def avg_batch_size(self) -> float:
        """Return the mean number of observations per model batch."""
        if self.batches == 0:
            return 0.0
        return self.actions / self.batches

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable profiling counters."""
        return {
            "batches": self.batches,
            "actions": self.actions,
            "avg_batch_size": self.avg_batch_size,
            "timings": dict(self.timings),
        }


class BatchedBCPolicyRunner:
    """Run one BC checkpoint over multiple observation streams at once."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        pass_bias: float = 0.0,
        pass_bias_turn: int = 50,
        profile: bool = False,
    ) -> None:
        """Load a checkpoint and initialize batched inference state."""
        if pass_bias < 0:
            raise ValueError("pass_bias must be nonnegative")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.pass_bias = float(pass_bias)
        self.pass_bias_turn = int(pass_bias_turn)
        self.profile = bool(profile)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_mode = str(checkpoint.get("policy_mode", "joint"))
        self.history_length = int(checkpoint.get("history_length", 1))
        self.encoder = ObservationEncoder(history_length=self.history_length)
        input_channels = int(checkpoint["input_channels"])
        if input_channels != self.encoder.num_channels:
            raise ValueError(
                f"checkpoint input_channels={input_channels} does not match "
                f"encoder channels={self.encoder.num_channels}"
            )

        self._is_temporal = (
            str(checkpoint.get("model_type", "")) == "temporal_policy_v1"
        )
        self._temporal_model: Any = None
        self._chunk_controllers: dict[ActionKey, Any] = {}
        self._temporal_active_keys: tuple[ActionKey, ...] | None = None
        if self._is_temporal:
            from generals_bot.training.temporal import TemporalModelConfig, TemporalPolicyNet

            tc = checkpoint["temporal_config"]
            temporal_config = TemporalModelConfig(
                base_channels=int(checkpoint.get("base_channels", 64)),
                temporal_dim=int(tc.get("temporal_dim", 192)),
                chunk_token_dim=int(tc.get("chunk_token_dim", 192)),
                chunk_transformer_layers=int(tc.get("chunk_transformer_layers", 2)),
                chunk_transformer_heads=int(tc.get("chunk_transformer_heads", 4)),
                chunk_transformer_ffn_dim=int(tc.get("chunk_transformer_ffn_dim", 512)),
                max_chunk_memory=int(tc.get("max_chunk_memory", 32)),
                dropout=float(tc.get("dropout", 0.0)),
                chunk_size=int(tc.get("chunk_size", 8)),
                encoder_type=str(tc.get("encoder_type", "global")),
                memory_type=str(tc.get("memory_type", "transformer")),
                gru_layers=int(tc.get("gru_layers", 3)),
                event_context_mode=str(tc.get("event_context_mode", "feature")),
                spatial_grid_size=int(tc.get("spatial_grid_size", 4)),
                spatial_transformer_layers=int(tc.get("spatial_transformer_layers", 1)),
            )
            spatial = BCPolicyNet(
                input_channels,
                base_channels=temporal_config.base_channels,
                temporal_dim=temporal_config.temporal_dim,
            ).to(self.device)
            self._temporal_model = TemporalPolicyNet(
                input_channels, spatial, temporal_config,
            ).to(self.device)
            self._temporal_model.load_state_dict(checkpoint["model_state"])
            self._temporal_model.eval()
            self._chunk_size = temporal_config.chunk_size
            self.model = self._temporal_model
        else:
            self.model = BCPolicyNet(
                input_channels,
                base_channels=int(checkpoint.get("base_channels", 64)),
            ).to(self.device)
            load_model_state_compatible(self.model, checkpoint["model_state"])
            self.model.eval()

        self.histories: dict[ActionKey, deque[Observation]] = {}
        self.stats = BatchedPolicyStats()

    def reset_key(self, key: ActionKey) -> None:
        """Clear one logical stream's observation history and temporal state."""
        self.histories.pop(key, None)
        if self._is_temporal:
            self._chunk_controllers.clear()
            self._temporal_active_keys = None
            if self._temporal_model is not None:
                # Runtime memory is row-indexed.  Resetting all rows is safer
                # than allowing a newly inserted key to inherit another
                # stream's hidden state.
                self._temporal_model.reset()

    def reset_all(self) -> None:
        """Clear all stored observation histories and temporal state."""
        self.histories.clear()
        if self._is_temporal:
            self._chunk_controllers.clear()
            self._temporal_active_keys = None
            if self._temporal_model is not None:
                self._temporal_model.reset()

    def reset_stats(self) -> None:
        """Clear accumulated profiling stats."""
        self.stats = BatchedPolicyStats()

    @torch.no_grad()
    def act_many(
        self,
        items: Iterable[tuple[ActionKey, Observation]],
    ) -> dict[ActionKey, ActionLike]:
        """Select actions for many keyed observations."""
        materialized = list(items)
        requested_keys = {key for key, _ in materialized}
        if self._is_temporal:
            materialized.sort(key=lambda item: repr(item[0]))
            current_keys = tuple(key for key, _ in materialized)
            if self._temporal_active_keys is None:
                self._temporal_active_keys = current_keys
            elif current_keys != self._temporal_active_keys:
                active = set(self._temporal_active_keys)
                current = set(current_keys)
                if not current.issubset(active):
                    raise RuntimeError(
                        "temporal batched inference cannot add or replace streams; "
                        "call reset_all() before changing keys"
                    )
                observations_by_key = dict(materialized)
                for key in self._temporal_active_keys:
                    if key not in observations_by_key:
                        history = self.histories.get(key)
                        if not history:
                            raise RuntimeError(
                                f"temporal stream {key!r} has no observation for padding"
                            )
                        observations_by_key[key] = history[-1]
                # Finished streams keep their row alive with the last observation.
                # Their actions are discarded below, while active rows retain the
                # same temporal-memory index for the rest of the match batch.
                materialized = [
                    (key, observations_by_key[key])
                    for key in self._temporal_active_keys
                ]
        grouped: dict[tuple[int, int], list[tuple[ActionKey, Observation]]] = defaultdict(list)
        for key, observation in materialized:
            grouped[(observation.height, observation.width)].append((key, observation))
        if self._is_temporal and len(grouped) > 1:
            raise RuntimeError(
                "temporal batched inference requires one board size per runner"
            )

        actions: dict[ActionKey, ActionLike] = {}
        for group in grouped.values():
            actions.update(self._act_group(group))
        if self._is_temporal:
            return {key: action for key, action in actions.items() if key in requested_keys}
        return actions

    def summary(self) -> dict[str, Any]:
        """Return current runner metadata and profiling stats."""
        return {
            "device": str(self.device),
            "policy_mode": self.policy_mode,
            "history_length": self.history_length,
            **self.stats.summary(),
        }

    def _act_group(self, group: list[tuple[ActionKey, Observation]]) -> dict[ActionKey, ActionLike]:
        """Run one same-shape observation group through the model."""
        if not group:
            return {}

        t0 = perf_counter()
        encoded = []
        keys = []
        observations = []
        for key, observation in group:
            history = self.histories.setdefault(
                key,
                deque(maxlen=self.history_length),
            )
            history.append(observation)
            encoded.append(self.encoder.encode_history(tuple(history)))
            keys.append(key)
            observations.append(observation)
        x_np = np.ascontiguousarray(np.stack(encoded, axis=0))
        t1 = perf_counter()

        x = torch.from_numpy(x_np).to(self.device)
        self._sync_if_profiled()
        t2 = perf_counter()

        if self._is_temporal and self._temporal_model is not None:
            from generals_bot.training.chunking import FixedChunkController

            cb_flags: list[bool] = []
            for key in keys:
                controller = self._chunk_controllers.setdefault(
                    key,
                    FixedChunkController(self._chunk_size),
                )
                cb_flags.append(controller.should_end_chunk(key))
            cb_tensor = torch.tensor(cb_flags, device=self.device)
            output = self._temporal_model(x, chunk_boundary=cb_tensor)
        else:
            output = self.model(x)
        self._sync_if_profiled()
        t3 = perf_counter()

        valid_masks = [valid_action_mask(observation) for observation in observations]
        t4 = perf_counter()

        if self.policy_mode != POLICY_MODE:
            result = self._select_joint(output, valid_masks, keys, observations)
        else:
            result = self._select_hierarchical(output, valid_masks, keys, observations)
        self._sync_if_profiled()
        t5 = perf_counter()

        self.stats.add_batch(len(group))
        self._record_timing("encode", t1 - t0)
        self._record_timing("to_device", t2 - t1)
        self._record_timing("forward", t3 - t2)
        self._record_timing("mask", t4 - t3)
        self._record_timing("select", t5 - t4)
        self._record_timing("act_total", t5 - t0)
        return result

    def _select_joint(
        self,
        output: dict[str, torch.Tensor],
        valid_masks: list[np.ndarray],
        keys: list[ActionKey],
        observations: list[Observation],
    ) -> dict[ActionKey, ActionLike]:
        """Select actions from a legacy joint-softmax checkpoint."""
        logits = flatten_policy_logits(output["move_logits"], output["pass_logits"])
        for row, (valid_mask, observation) in enumerate(zip(valid_masks, observations, strict=True)):
            apply_pass_bias(
                logits[row : row + 1],
                valid_mask,
                observation.turn,
                pass_bias=self.pass_bias,
                pass_bias_turn=self.pass_bias_turn,
            )
        mask = torch.from_numpy(np.stack(valid_masks, axis=0)).to(self.device)
        labels = logits.masked_fill(~mask, -1e9).argmax(dim=1).detach().cpu().numpy()
        return {
            key: label_to_action(
                int(label),
                player_id=observation.player_id,
                height=observation.height,
                width=observation.width,
            )
            for key, label, observation in zip(keys, labels, observations, strict=True)
        }

    def _select_hierarchical(
        self,
        output: dict[str, torch.Tensor],
        valid_masks: list[np.ndarray],
        keys: list[ActionKey],
        observations: list[Observation],
    ) -> dict[ActionKey, ActionLike]:
        """Select actions from a hierarchical move/pass checkpoint."""
        action_logits = output["action_logits"].reshape(-1).detach().cpu().numpy()
        if self.pass_bias > 0:
            for row, observation in enumerate(observations):
                if observation.turn >= self.pass_bias_turn:
                    action_logits[row] += self.pass_bias

        move_masks = np.stack([mask[:-1] for mask in valid_masks], axis=0)
        move_logits = output["move_logits"].flatten(start_dim=1)
        mask = torch.from_numpy(move_masks).to(self.device)
        labels = move_logits.masked_fill(~mask, -1e9).argmax(dim=1).detach().cpu().numpy()

        result: dict[ActionKey, ActionLike] = {}
        for key, label, action_logit, valid_mask, observation in zip(
            keys,
            labels,
            action_logits,
            valid_masks,
            observations,
            strict=True,
        ):
            if not valid_mask[:-1].any() or float(action_logit) <= 0.0:
                result[key] = PassAction(observation.player_id)
                continue
            result[key] = label_to_action(
                int(label),
                player_id=observation.player_id,
                height=observation.height,
                width=observation.width,
            )
        return result

    def _record_timing(self, name: str, seconds: float) -> None:
        """Record timing only when profiling is enabled."""
        if self.profile:
            self.stats.add_time(name, seconds)

    def _sync_if_profiled(self) -> None:
        """Synchronize CUDA only when wall-clock profiling needs accurate sections."""
        if self.profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
