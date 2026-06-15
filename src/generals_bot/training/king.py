from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from generals_bot.sim.types import Observation


@dataclass(frozen=True)
class EnemyKingTargets:
    """Dense training targets for the enemy general belief head."""

    labels: np.ndarray
    distances: np.ndarray
    valid_mask: np.ndarray


def build_enemy_king_targets(
    observations: Sequence[Observation],
    *,
    enemy_general_tiles: Sequence[int | None],
    mountains: Sequence[Iterable[int]],
    max_height: int | None = None,
    max_width: int | None = None,
) -> EnemyKingTargets:
    """Build padded labels, valid masks, and game-distance maps for observations."""
    if not (
        len(observations) == len(enemy_general_tiles) == len(mountains)
    ):
        raise ValueError("enemy king target inputs must have matching lengths")
    if not observations:
        raise ValueError("cannot build enemy king targets for an empty batch")

    resolved_height = max_height or max(observation.height for observation in observations)
    resolved_width = max_width or max(observation.width for observation in observations)
    labels = np.zeros((len(observations),), dtype=np.int64)
    distances = np.full(
        (len(observations), resolved_height * resolved_width),
        np.inf,
        dtype=np.float32,
    )
    valid_mask = np.zeros(
        (len(observations), resolved_height * resolved_width),
        dtype=bool,
    )

    for index, (observation, target_tile, mountain_tiles) in enumerate(
        zip(observations, enemy_general_tiles, mountains, strict=True)
    ):
        if observation.height > resolved_height or observation.width > resolved_width:
            raise ValueError("observation is larger than the requested target shape")
        resolved_target = _resolve_target_tile(observation, target_tile)
        target_row, target_col = divmod(resolved_target, observation.width)
        labels[index] = target_row * resolved_width + target_col

        board_mask = np.zeros((resolved_height, resolved_width), dtype=bool)
        board_mask[: observation.height, : observation.width] = True
        valid_mask[index] = board_mask.reshape(-1)

        distance_map = game_distance_map(
            width=observation.width,
            height=observation.height,
            mountains=tuple(sorted(int(tile) for tile in mountain_tiles)),
            target_tile=resolved_target,
        )
        padded_distance = np.full((resolved_height, resolved_width), np.inf, dtype=np.float32)
        padded_distance[: observation.height, : observation.width] = distance_map
        distances[index] = padded_distance.reshape(-1)

    return EnemyKingTargets(labels=labels, distances=distances, valid_mask=valid_mask)


def game_distance_map(
    *,
    width: int,
    height: int,
    mountains: Iterable[int],
    target_tile: int,
) -> np.ndarray:
    """Return shortest-path distances to target, treating mountains as blocked."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    size = width * height
    if target_tile < 0 or target_tile >= size:
        raise ValueError("target_tile is outside the board")
    cached = _cached_game_distance_map(width, height, tuple(sorted(mountains)), target_tile)
    return np.asarray(cached, dtype=np.float32).reshape((height, width)).copy()


@lru_cache(maxsize=8192)
def _cached_game_distance_map(
    width: int,
    height: int,
    mountains: tuple[int, ...],
    target_tile: int,
) -> tuple[float, ...]:
    """Compute one static-map BFS distance map and cache it as a flat tuple."""
    size = width * height
    mountain_set = {tile for tile in mountains if 0 <= tile < size}
    distances = [float("inf")] * size
    if target_tile in mountain_set:
        distances[target_tile] = 0.0
        return tuple(distances)

    distances[target_tile] = 0.0
    queue: deque[int] = deque([target_tile])
    while queue:
        tile = queue.popleft()
        next_distance = distances[tile] + 1.0
        for neighbor in _neighbors(tile, width, height):
            if neighbor in mountain_set or distances[neighbor] != float("inf"):
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return tuple(distances)


def _neighbors(tile: int, width: int, height: int) -> Iterable[int]:
    """Yield orthogonal in-bounds neighbors for a flattened tile index."""
    row, col = divmod(tile, width)
    if row > 0:
        yield tile - width
    if row + 1 < height:
        yield tile + width
    if col > 0:
        yield tile - 1
    if col + 1 < width:
        yield tile + 1


def _resolve_target_tile(observation: Observation, target_tile: int | None) -> int:
    """Return an explicit target tile or a best-effort observed enemy general tile."""
    if target_tile is not None:
        if target_tile < 0 or target_tile >= observation.width * observation.height:
            raise ValueError("enemy general tile is outside the observation board")
        return int(target_tile)
    rows, cols = np.nonzero(observation.known_enemy_generals)
    if len(rows) > 0:
        return int(rows[0] * observation.width + cols[0])
    return observation.width * observation.height - 1
