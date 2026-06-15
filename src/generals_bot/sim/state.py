from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import GameConfig, PlayerScore, Terrain


@dataclass
class PlayerMemory:
    explored: np.ndarray
    seen_cities: np.ndarray
    seen_generals: np.ndarray
    seen_obstacles: np.ndarray

    @classmethod
    def empty(cls, height: int, width: int) -> PlayerMemory:
        shape = (height, width)
        return cls(
            explored=np.zeros(shape, dtype=bool),
            seen_cities=np.zeros(shape, dtype=bool),
            seen_generals=np.zeros(shape, dtype=bool),
            seen_obstacles=np.zeros(shape, dtype=bool),
        )


@dataclass
class GameState:
    turn: int
    width: int
    height: int
    armies: np.ndarray
    terrain: np.ndarray
    cities: np.ndarray
    generals: list[int]
    alive: list[bool]
    queues: dict[int, object]
    memory: dict[int, PlayerMemory]
    max_turns: int
    has_kill: list[bool] = field(default_factory=lambda: [False, False])
    winner: int | None = None

    @classmethod
    def from_config(cls, config: GameConfig, queues: dict[int, object]) -> GameState:
        size = config.width * config.height
        if len(config.generals) != 2:
            raise ValueError("first simulator version requires exactly two generals")

        terrain = np.full((config.height, config.width), int(Terrain.NEUTRAL), dtype=int)
        armies = np.zeros((config.height, config.width), dtype=int)
        cities = np.zeros((config.height, config.width), dtype=bool)

        for tile in config.mountains:
            _validate_tile(tile, size, "mountain")
            row, col = divmod(tile, config.width)
            terrain[row, col] = int(Terrain.MOUNTAIN)

        for tile, owner in (config.starting_owners or {}).items():
            _validate_tile(tile, size, "starting owner")
            if owner not in (0, 1):
                raise ValueError(f"invalid starting owner {owner}")
            row, col = divmod(tile, config.width)
            if terrain[row, col] == int(Terrain.MOUNTAIN):
                raise ValueError("owned tile cannot be a mountain")
            terrain[row, col] = owner

        default_city_armies = (40,) * len(config.cities)
        city_armies = config.city_armies or default_city_armies
        if len(city_armies) != len(config.cities):
            raise ValueError("city_armies length must match cities length")
        for tile, army in zip(config.cities, city_armies, strict=True):
            _validate_tile(tile, size, "city")
            row, col = divmod(tile, config.width)
            if terrain[row, col] == int(Terrain.MOUNTAIN):
                raise ValueError("city cannot be a mountain")
            cities[row, col] = True
            armies[row, col] = int(army)

        for tile, army in (config.starting_armies or {}).items():
            _validate_tile(tile, size, "starting army")
            row, col = divmod(tile, config.width)
            armies[row, col] = int(army)

        for player_id, tile in enumerate(config.generals):
            _validate_tile(tile, size, "general")
            row, col = divmod(tile, config.width)
            if terrain[row, col] == int(Terrain.MOUNTAIN):
                raise ValueError("general cannot be a mountain")
            terrain[row, col] = player_id
            if armies[row, col] <= 0:
                armies[row, col] = 1

        return cls(
            turn=0,
            width=config.width,
            height=config.height,
            armies=armies,
            terrain=terrain,
            cities=cities,
            generals=list(config.generals),
            alive=[True, True],
            queues=queues,
            memory={player: PlayerMemory.empty(config.height, config.width) for player in (0, 1)},
            max_turns=config.max_turns,
        )

    @property
    def size(self) -> int:
        return self.width * self.height

    @property
    def done(self) -> bool:
        return self.winner is not None or self.turn >= self.max_turns

    @property
    def alive_players(self) -> list[int]:
        return [player for player, alive in enumerate(self.alive) if alive]

    def scores(self) -> list[PlayerScore]:
        return [
            PlayerScore(
                player_id=player,
                army=int(np.sum(self.armies[self.terrain == player])),
                land=int(np.sum(self.terrain == player)),
                dead=not self.alive[player],
                has_kill=self.has_kill[player],
            )
            for player in (0, 1)
        ]


def _validate_tile(tile: int, size: int, label: str) -> None:
    if tile < 0 or tile >= size:
        raise ValueError(f"{label} tile out of bounds: {tile}")

