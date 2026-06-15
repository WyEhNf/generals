from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from generals_bot.agents.base import EnemyKingDistribution, WinRateEstimate
from generals_bot.sim.env import GeneralsEnv
from generals_bot.sim.types import Action, ExecutedAction, Observation, PlayerScore


@dataclass(frozen=True)
class Snapshot:
    turn: int
    width: int
    height: int
    terrain: np.ndarray
    armies: np.ndarray
    cities: np.ndarray
    generals: list[int]
    alive: list[bool]
    queued_actions: dict[int, list[Action]]
    submitted_actions: dict[int, list[Action]]
    executed_actions: list[ExecutedAction]
    scores: list[PlayerScore]
    winner: int | None
    observations: dict[int, Observation]
    win_rates: dict[int, WinRateEstimate | None] = field(default_factory=dict)
    enemy_king_distributions: dict[int, EnemyKingDistribution | None] = field(
        default_factory=dict
    )


class MatchRecorder:
    def __init__(self) -> None:
        self.snapshots: list[Snapshot] = []

    def capture(
        self,
        env: GeneralsEnv,
        *,
        win_rates: Mapping[int, WinRateEstimate | None] | None = None,
        enemy_king_distributions: Mapping[int, EnemyKingDistribution | None] | None = None,
    ) -> Snapshot:
        """Record the current environment state for HTML visualization."""
        if env.state is None:
            raise RuntimeError("call env.reset() before capturing snapshots")
        state = env.state
        recorded_win_rates = {
            player: None if win_rates is None else win_rates.get(player)
            for player in (0, 1)
        }
        recorded_enemy_king_distributions = {
            player: (
                None
                if enemy_king_distributions is None
                else enemy_king_distributions.get(player)
            )
            for player in (0, 1)
        }
        snapshot = Snapshot(
            turn=state.turn,
            width=state.width,
            height=state.height,
            terrain=state.terrain.copy(),
            armies=state.armies.copy(),
            cities=state.cities.copy(),
            generals=list(state.generals),
            alive=list(state.alive),
            queued_actions=env.queue_snapshot(),
            submitted_actions={
                player: list(actions)
                for player, actions in env.last_submitted_actions.items()
            },
            executed_actions=list(env.last_executed_actions),
            scores=env.scores,
            winner=state.winner,
            observations=env.observations(),
            win_rates=recorded_win_rates,
            enemy_king_distributions=recorded_enemy_king_distributions,
        )
        self.snapshots.append(snapshot)
        return snapshot
