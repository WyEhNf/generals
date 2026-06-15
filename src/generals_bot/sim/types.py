from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping, Sequence

import numpy as np


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


DIRECTION_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


class Terrain(IntEnum):
    NEUTRAL = -1
    MOUNTAIN = -2


class CellView(IntEnum):
    OWN = 0
    ENEMY = 1
    NEUTRAL = -1
    MOUNTAIN = -2
    FOG = -3
    FOG_OBSTACLE = -4


@dataclass(frozen=True)
class Action:
    player_id: int
    start: int
    end: int
    split: bool = False


@dataclass(frozen=True)
class PassAction:
    player_id: int


@dataclass(frozen=True)
class DirectionMove:
    player_id: int
    row: int
    col: int
    direction: Direction
    split: bool = False


ActionLike = Action | DirectionMove | PassAction | Sequence[Action | DirectionMove | PassAction]


@dataclass(frozen=True)
class PlayerScore:
    player_id: int
    army: int
    land: int
    dead: bool
    has_kill: bool = False


@dataclass(frozen=True)
class ObservedMove:
    player_id: int
    start: int
    end: int
    split: bool
    turn: int
    visible: bool


@dataclass(frozen=True)
class ExecutedAction:
    player_id: int
    action: Action
    turn: int
    valid: bool
    reason: str | None = None
    captured_general: int | None = None
    target_owner_before: int | None = None
    target_army_before: int = 0
    moved_army: int = 0
    damage_done: int = 0


@dataclass(frozen=True)
class Observation:
    player_id: int
    turn: int
    width: int
    height: int
    visible: np.ndarray
    explored: np.ndarray
    armies: np.ndarray
    owner: np.ndarray
    own_tiles: np.ndarray
    enemy_tiles: np.ndarray
    neutral_tiles: np.ndarray
    mountains: np.ndarray
    cities: np.ndarray
    known_cities: np.ndarray
    generals: np.ndarray
    known_generals: np.ndarray
    known_enemy_generals: np.ndarray
    fog: np.ndarray
    fog_obstacles: np.ndarray
    own_army: int
    own_land: int
    enemy_army: int
    enemy_land: int
    last_moves: list[ObservedMove] = field(default_factory=list)
    priority: int = 0


@dataclass(frozen=True)
class GameConfig:
    width: int
    height: int
    generals: tuple[int, int]
    mountains: tuple[int, ...] = ()
    cities: tuple[int, ...] = ()
    city_armies: tuple[int, ...] = ()
    starting_owners: Mapping[int, int] | None = None
    starting_armies: Mapping[int, int] | None = None
    max_turns: int = 5000


def rc_to_tile(row: int, col: int, width: int) -> int:
    return row * width + col


def tile_to_rc(tile: int, width: int) -> tuple[int, int]:
    return divmod(tile, width)


def direction_move_to_action(move: DirectionMove, width: int) -> Action:
    dr, dc = DIRECTION_DELTAS[int(move.direction)]
    start = rc_to_tile(move.row, move.col, width)
    end = rc_to_tile(move.row + dr, move.col + dc, width)
    return Action(move.player_id, start, end, move.split)
