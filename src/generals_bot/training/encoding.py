from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from generals_bot.sim.types import CellView, Observation


@dataclass(frozen=True)
class ObservationEncoder:
    """Encode simulator observations into dense model input planes."""

    history_length: int = 1
    army_scale: float = np.log1p(100.0)
    total_army_scale: float = np.log1p(1000.0)
    turn_norm_max: int = 500
    turn_phase_bucket: int = 50
    turn_phase_max: int = 10

    feature_names: ClassVar[tuple[str, ...]] = (
        "own",
        "enemy",
        "neutral",
        "mountain",
        "fog",
        "fog_obstacle",
        "visible",
        "explored",
        "city",
        "general",
        "known_general",
        "known_enemy_general",
        "army_log",
        "own_army_log",
        "enemy_army_log",
        "own_land_norm",
        "enemy_land_norm",
        "turn_parity",
        "turn_mod2",
        "turn_mod50",
        "turn_norm_500",
        "turn_phase_50",
    )

    def __post_init__(self) -> None:
        """Validate encoder history settings."""
        if self.history_length < 1:
            raise ValueError("history_length must be positive")

    @property
    def num_channels(self) -> int:
        """Return the number of encoded feature planes."""
        return len(self.feature_names) * self.history_length

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return names for every encoded channel, including history frames."""
        if self.history_length == 1:
            return self.feature_names
        return tuple(
            f"t-{self.history_length - frame_index - 1}:{name}"
            for frame_index in range(self.history_length)
            for name in self.feature_names
        )

    def encode(self, obs: Observation) -> np.ndarray:
        """Encode one observation as a float32 CxHxW array."""
        if self.history_length > 1:
            return self.encode_history((obs,))
        return self._encode_one(obs)

    def encode_history(self, history: tuple[Observation, ...]) -> np.ndarray:
        """Encode a fixed-length history by concatenating frame channels."""
        if self.history_length < 1:
            raise ValueError("history_length must be positive")
        if not history:
            raise ValueError("history cannot be empty")
        frames = history[-self.history_length:]
        if len(frames) < self.history_length:
            frames = (frames[0],) * (self.history_length - len(frames)) + frames
        return np.concatenate([self._encode_one(obs) for obs in frames], axis=0)

    def _encode_one(self, obs: Observation) -> np.ndarray:
        """Encode one observation without history stacking."""
        height, width = obs.height, obs.width
        board_cells = max(1, height * width)
        turn_norm = min(obs.turn, self.turn_norm_max) / max(1, self.turn_norm_max)
        turn_phase = min(
            obs.turn // max(1, self.turn_phase_bucket),
            self.turn_phase_max,
        ) / max(1, self.turn_phase_max)
        planes_by_name = {
            "own": obs.owner == int(CellView.OWN),
            "enemy": obs.owner == int(CellView.ENEMY),
            "neutral": obs.owner == int(CellView.NEUTRAL),
            "mountain": obs.owner == int(CellView.MOUNTAIN),
            "fog": obs.owner == int(CellView.FOG),
            "fog_obstacle": obs.owner == int(CellView.FOG_OBSTACLE),
            "visible": obs.visible,
            "explored": obs.explored,
            "city": obs.cities,
            "general": obs.generals,
            "known_general": obs.known_generals,
            "known_enemy_general": obs.known_enemy_generals,
            "army_log": np.log1p(obs.armies.astype(np.float32)) / self.army_scale,
            "own_army_log": _full_plane(
                height,
                width,
                np.log1p(obs.own_army) / self.total_army_scale,
            ),
            "enemy_army_log": _full_plane(
                height,
                width,
                np.log1p(obs.enemy_army) / self.total_army_scale,
            ),
            "own_land_norm": _full_plane(height, width, obs.own_land / board_cells),
            "enemy_land_norm": _full_plane(height, width, obs.enemy_land / board_cells),
            "turn_parity": _full_plane(height, width, obs.priority % 2),
            "turn_mod2": _full_plane(height, width, obs.turn % 2),
            "turn_mod50": _full_plane(height, width, (obs.turn % 50) / 49.0),
            "turn_norm_500": _full_plane(height, width, turn_norm),
            "turn_phase_50": _full_plane(height, width, turn_phase),
        }
        try:
            planes = [planes_by_name[name] for name in self.feature_names]
        except KeyError as exc:
            raise ValueError(f"unknown observation feature: {exc.args[0]}") from exc
        return np.stack([plane.astype(np.float32, copy=False) for plane in planes], axis=0)


def _full_plane(height: int, width: int, value: float) -> np.ndarray:
    """Create a constant feature plane."""
    return np.full((height, width), float(value), dtype=np.float32)
