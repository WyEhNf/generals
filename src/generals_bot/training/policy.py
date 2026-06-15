from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "torch is required for model policy. Install with: "
        ".venv/bin/python -m pip install -r requirements-train.txt"
    ) from exc

from generals_bot.agents.base import EnemyKingDistribution, Policy
from generals_bot.agents.base import WinRateEstimate
from generals_bot.sim.types import Action, GameConfig, Observation, PassAction
from generals_bot.training.action_codec import label_to_action, valid_action_mask
from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.model import (
    BCPolicyNet,
    EnemyKingPredictorNet,
    POLICY_MODE,
    PREDICTOR_MODEL_TYPE,
    flatten_policy_logits,
    load_model_state_compatible,
)


class BCModelPolicy(Policy):
    """Run a trained BC checkpoint as a simulator policy."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        pass_bias: float = 0.0,
        pass_bias_turn: int = 50,
        enemy_king_predictor_path: str | Path | None = None,
    ) -> None:
        """Load a checkpoint and initialize the model policy."""
        if pass_bias < 0:
            raise ValueError("pass_bias must be nonnegative")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.pass_bias = float(pass_bias)
        self.pass_bias_turn = int(pass_bias_turn)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state = checkpoint["model_state"]
        self.has_enemy_king_head = all(
            key in model_state
            for key in ("enemy_king_head.weight", "enemy_king_head.bias")
        )
        self.policy_mode = str(checkpoint.get("policy_mode", "joint"))
        self.policy_history_length = int(checkpoint.get("history_length", 1))
        self.history_length = self.policy_history_length
        self.encoder = ObservationEncoder(history_length=self.policy_history_length)
        input_channels = int(checkpoint["input_channels"])
        if input_channels != self.encoder.num_channels:
            raise ValueError(
                f"checkpoint input_channels={input_channels} does not match "
                f"encoder channels={self.encoder.num_channels}"
            )
        self.model = BCPolicyNet(
            input_channels,
            base_channels=int(checkpoint.get("base_channels", 64)),
        ).to(self.device)
        load_model_state_compatible(self.model, model_state)
        self.model.eval()
        self.enemy_king_predictor: EnemyKingPredictorNet | None = None
        self.enemy_king_predictor_encoder: ObservationEncoder | None = None
        self.enemy_king_predictor_history_length = 0
        if enemy_king_predictor_path is not None:
            self._load_enemy_king_predictor(enemy_king_predictor_path)
        self.player_id: int | None = None
        self.history: deque[Observation] = deque(
            maxlen=max(
                self.policy_history_length,
                self.enemy_king_predictor_history_length,
            )
        )

    def reset(self, player_id: int, config: GameConfig) -> None:
        """Remember the assigned player id for a new match."""
        del config
        self.player_id = player_id
        self.history.clear()

    @torch.no_grad()
    def act(self, observation: Observation) -> Action | PassAction:
        """Select an action using the checkpoint's policy mode."""
        self.history.append(observation)
        output = self._forward_history(
            self._history_for_inference(
                observation,
                history_length=self.policy_history_length,
            )
        )
        valid_mask = valid_action_mask(observation)
        if self.policy_mode != POLICY_MODE:
            return self._act_joint(output, valid_mask, observation)
        return self._act_hierarchical(output, valid_mask, observation)

    @torch.no_grad()
    def estimate_win_rate(self, observation: Observation) -> WinRateEstimate:
        """Estimate this player's current win probability from the value head."""
        output = self._forward_history(
            self._history_for_inference(
                observation,
                history_length=self.policy_history_length,
            )
        )
        raw_value = float(output["value"].reshape(-1).item())
        win_probability = max(0.0, min(1.0, (raw_value + 1.0) / 2.0))
        return WinRateEstimate(win_probability=win_probability, raw_value=raw_value)

    @torch.no_grad()
    def estimate_enemy_king_distribution(
        self,
        observation: Observation,
    ) -> EnemyKingDistribution | None:
        """Estimate the current enemy general location probability map."""
        if self.enemy_king_predictor is not None:
            if self.enemy_king_predictor_encoder is None:  # pragma: no cover
                raise RuntimeError("enemy king predictor encoder was not initialized")
            output = self._forward_history(
                self._history_for_inference(
                    observation,
                    history_length=self.enemy_king_predictor_history_length,
                ),
                encoder=self.enemy_king_predictor_encoder,
                model=self.enemy_king_predictor,
            )
            return _enemy_king_distribution_from_output(output, observation)
        if not self.has_enemy_king_head:
            return None
        output = self._forward_history(
            self._history_for_inference(
                observation,
                history_length=self.policy_history_length,
            )
        )
        return _enemy_king_distribution_from_output(output, observation)

    def _history_for_inference(
        self,
        observation: Observation,
        *,
        history_length: int | None = None,
    ) -> tuple[Observation, ...]:
        """Return the history that would be used for observation without mutating it."""
        if history_length is None:
            history_length = self.policy_history_length
        if (
            self.history
            and self.history[-1].player_id == observation.player_id
            and self.history[-1].turn == observation.turn
        ):
            history = list(self.history)
        else:
            history = [*self.history, observation]
        if len(history) > history_length:
            history = history[-history_length:]
        return tuple(history)

    def _forward_history(
        self,
        history: tuple[Observation, ...],
        *,
        encoder: ObservationEncoder | None = None,
        model: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run one observation history through a model."""
        if encoder is None:
            encoder = self.encoder
        if model is None:
            model = self.model
        encoded = encoder.encode_history(history)[None, :, :, :]
        x = torch.from_numpy(np.ascontiguousarray(encoded)).to(self.device)
        return model(x)

    def _load_enemy_king_predictor(self, checkpoint_path: str | Path) -> None:
        """Load an optional standalone enemy king predictor checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_type = str(checkpoint.get("model_type", ""))
        if model_type != PREDICTOR_MODEL_TYPE:
            raise ValueError(
                f"enemy_king_predictor must have model_type={PREDICTOR_MODEL_TYPE!r}; "
                f"got {model_type!r}"
            )
        history_length = int(checkpoint.get("history_length", 1))
        encoder = ObservationEncoder(history_length=history_length)
        input_channels = int(checkpoint["input_channels"])
        if input_channels != encoder.num_channels:
            raise ValueError(
                f"enemy king predictor input_channels={input_channels} does not match "
                f"encoder channels={encoder.num_channels}"
            )
        model = EnemyKingPredictorNet(
            input_channels,
            base_channels=int(checkpoint.get("base_channels", 64)),
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self.enemy_king_predictor = model
        self.enemy_king_predictor_encoder = encoder
        self.enemy_king_predictor_history_length = history_length

    def _act_joint(
        self,
        output: dict[str, torch.Tensor],
        valid_mask: np.ndarray,
        observation: Observation,
    ) -> Action | PassAction:
        """Run legacy joint-softmax inference for older checkpoints."""
        logits = flatten_policy_logits(output["move_logits"], output["pass_logits"])
        apply_pass_bias(
            logits,
            valid_mask,
            observation.turn,
            pass_bias=self.pass_bias,
            pass_bias_turn=self.pass_bias_turn,
        )
        mask = torch.from_numpy(valid_mask[None, :]).to(self.device)
        label = int(logits.masked_fill(~mask, -1e9).argmax(dim=1).item())
        return label_to_action(
            label,
            player_id=observation.player_id,
            height=observation.height,
            width=observation.width,
        )

    def _act_hierarchical(
        self,
        output: dict[str, torch.Tensor],
        valid_mask: np.ndarray,
        observation: Observation,
    ) -> Action | PassAction:
        """Run staged move/pass inference for hierarchical checkpoints."""
        if not valid_mask[:-1].any():
            return PassAction(observation.player_id)
        action_logit = output["action_logits"].reshape(-1)
        if self.pass_bias > 0 and observation.turn >= self.pass_bias_turn:
            action_logit = action_logit + self.pass_bias
        if float(action_logit.item()) <= 0.0:
            return PassAction(observation.player_id)
        move_logits = output["move_logits"].flatten(start_dim=1)
        mask = torch.from_numpy(valid_mask[:-1][None, :]).to(self.device)
        label = int(move_logits.masked_fill(~mask, -1e9).argmax(dim=1).item())
        return label_to_action(
            label,
            player_id=observation.player_id,
            height=observation.height,
            width=observation.width,
        )


def _enemy_king_distribution_from_output(
    output: dict[str, torch.Tensor],
    observation: Observation,
) -> EnemyKingDistribution:
    """Convert dense enemy king logits into a normalized probability map."""
    logits = output["enemy_king_logits"][
        0,
        0,
        : observation.height,
        : observation.width,
    ]
    probabilities = torch.softmax(logits.reshape(-1), dim=0)
    probability_map = probabilities.reshape(
        observation.height,
        observation.width,
    )
    return EnemyKingDistribution(probabilities=probability_map.cpu().numpy())


def apply_pass_bias(
    logits: torch.Tensor,
    valid_mask: np.ndarray,
    turn: int,
    *,
    pass_bias: float,
    pass_bias_turn: int,
) -> None:
    """Apply inference-time pass logit calibration in place."""
    if pass_bias <= 0 or turn < pass_bias_turn:
        return
    if not valid_mask[:-1].any():
        return
    logits[:, -1] -= pass_bias
