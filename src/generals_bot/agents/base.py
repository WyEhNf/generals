from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from generals_bot.sim.types import Action, ActionLike, GameConfig, Observation, PassAction


@dataclass(frozen=True)
class WinRateEstimate:
    """A policy's current player-relative value estimate."""

    win_probability: float
    raw_value: float


@dataclass(frozen=True)
class EnemyKingDistribution:
    """A policy's player-relative enemy general probability map."""

    probabilities: np.ndarray


class Policy(Protocol):
    def reset(self, player_id: int, config: GameConfig) -> None:
        ...

    def act(self, observation: Observation) -> ActionLike:
        ...


class WinRateEstimator(Protocol):
    def estimate_win_rate(self, observation: Observation) -> WinRateEstimate | None:
        ...


class EnemyKingDistributionEstimator(Protocol):
    def estimate_enemy_king_distribution(
        self,
        observation: Observation,
    ) -> EnemyKingDistribution | None:
        ...


def estimate_win_rate(policy: object, observation: Observation) -> WinRateEstimate | None:
    """Return a policy's optional player-relative win-rate estimate."""
    estimator = getattr(policy, "estimate_win_rate", None)
    if estimator is None:
        return None
    estimate = estimator(observation)
    if estimate is None:
        return None
    if not isinstance(estimate, WinRateEstimate):
        raise TypeError("estimate_win_rate must return WinRateEstimate or None")
    return estimate


def estimate_enemy_king_distribution(
    policy: object,
    observation: Observation,
) -> EnemyKingDistribution | None:
    """Return a policy's optional enemy general probability map."""
    estimator = getattr(policy, "estimate_enemy_king_distribution", None)
    if estimator is None:
        return None
    estimate = estimator(observation)
    if estimate is None:
        return None
    if not isinstance(estimate, EnemyKingDistribution):
        raise TypeError(
            "estimate_enemy_king_distribution must return EnemyKingDistribution or None"
        )
    probabilities = estimate.probabilities
    if probabilities.shape != (observation.height, observation.width):
        raise ValueError(
            "enemy king probability map shape must match observation height/width"
        )
    return estimate


class PassPolicy:
    def __init__(self) -> None:
        self.player_id: int | None = None

    def reset(self, player_id: int, config: GameConfig) -> None:
        del config
        self.player_id = player_id

    def act(self, observation: Observation) -> PassAction:
        return PassAction(observation.player_id)
