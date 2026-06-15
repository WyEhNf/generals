from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from generals_bot.sim.types import Action, CellView, GameConfig, Observation, PassAction


@dataclass(frozen=True)
class _Candidate:
    priority: int
    action: Action


class ExpanderPolicy:
    """Small deterministic-ish baseline that expands visible frontier tiles."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.player_id: int | None = None

    def reset(self, player_id: int, config: GameConfig) -> None:
        del config
        self.player_id = player_id

    def act(self, observation: Observation) -> Action | PassAction:
        candidates = self._candidates(observation)
        if not candidates:
            return PassAction(observation.player_id)

        best_priority = max(candidate.priority for candidate in candidates)
        best = [candidate.action for candidate in candidates if candidate.priority == best_priority]
        return self.rng.choice(best)

    def _candidates(self, obs: Observation) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        rows, cols = np.nonzero(obs.own_tiles & (obs.armies > 1))
        for row, col in zip(rows, cols, strict=True):
            start = row * obs.width + col
            army_to_move = int(obs.armies[row, col]) - 1
            for nr, nc in _neighbors(row, col, obs.height, obs.width):
                owner = int(obs.owner[nr, nc])
                if owner in (int(CellView.MOUNTAIN), int(CellView.FOG_OBSTACLE), int(CellView.OWN)):
                    continue

                target_army = int(obs.armies[nr, nc])
                if owner in (int(CellView.NEUTRAL), int(CellView.ENEMY)) and army_to_move <= target_army:
                    continue

                end = nr * obs.width + nc
                priority = _priority(owner)
                candidates.append(_Candidate(priority, Action(obs.player_id, start, end)))
        return candidates


def _neighbors(row: int, col: int, height: int, width: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    if row > 0:
        result.append((row - 1, col))
    if row + 1 < height:
        result.append((row + 1, col))
    if col > 0:
        result.append((row, col - 1))
    if col + 1 < width:
        result.append((row, col + 1))
    return result


def _priority(owner: int) -> int:
    if owner == int(CellView.ENEMY):
        return 4
    if owner == int(CellView.NEUTRAL):
        return 3
    if owner == int(CellView.FOG):
        return 2
    return 0
