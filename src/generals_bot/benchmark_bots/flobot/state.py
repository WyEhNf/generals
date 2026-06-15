from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from generals_bot.sim.types import CellView, Observation


NEUTRAL = -1
MOUNTAIN = -2
FOG = -3
FOG_STRUCTURE = -4
OUT_OF_BOUNDS = -5


@dataclass(frozen=True)
class FlobotTile:
    """One Flobot terrain value paired with its flattened tile index."""

    index: int
    value: int


@dataclass(frozen=True)
class FlobotMove:
    """A Flobot move candidate; end=-1 means pass/no-op."""

    start: int
    end: int
    weight: int = 0


@dataclass
class FlobotState:
    """Flobot's persistent player-view state reconstructed from Observations."""

    player_index: int
    width: int
    height: int
    discovered_tiles: list[bool]
    cities: list[int] = field(default_factory=list)
    generals: list[int] = field(default_factory=lambda: [-1, -1])
    turn: int = 0
    armies: list[int] = field(default_factory=list)
    terrain: list[int] = field(default_factory=list)
    own_tiles: dict[int, int] = field(default_factory=dict)
    enemy_tiles: dict[int, int] = field(default_factory=dict)
    own_general: int = -1
    enemy_general: int = -1

    @classmethod
    def from_observation(cls, observation: Observation) -> FlobotState:
        """Create Flobot state from the first local simulator observation."""
        size = observation.width * observation.height
        state = cls(
            player_index=observation.player_id,
            width=observation.width,
            height=observation.height,
            discovered_tiles=[False] * size,
        )
        state.update(observation)
        return state

    @property
    def size(self) -> int:
        """Return the number of map cells."""
        return self.width * self.height

    def update(self, observation: Observation) -> None:
        """Refresh Flobot's state from a local simulator observation."""
        self.turn = observation.turn
        self.width = observation.width
        self.height = observation.height
        self.cities = _indices(observation.known_cities)
        self.generals = _general_list(observation)
        self.armies = observation.armies.reshape(-1).astype(int).tolist()
        self.terrain = _terrain_from_observation(observation)
        self._update_player_tiles()
        self._update_discovered_tiles()
        self._update_generals()

    def _update_player_tiles(self) -> None:
        """Refresh own/enemy tile maps from the current terrain view."""
        self.own_tiles.clear()
        self.enemy_tiles.clear()
        for index, tile in enumerate(self.terrain):
            if tile >= 0:
                army = self.armies[index]
                if tile == self.player_index:
                    self.own_tiles[index] = army
                else:
                    self.enemy_tiles[index] = army

    def _update_discovered_tiles(self) -> None:
        """Match Flobot's discovered-tile memory: owned or enemy tiles only."""
        for index in range(len(self.terrain)):
            if not self.discovered_tiles[index]:
                if index in self.own_tiles or index in self.enemy_tiles:
                    self.discovered_tiles[index] = True

    def _update_generals(self) -> None:
        """Persist own and enemy general locations once Flobot sees them."""
        for general in self.generals:
            if general == -1:
                continue
            if self.own_general == -1:
                self.own_general = general
            elif general != self.own_general:
                self.enemy_general = general


def _terrain_from_observation(observation: Observation) -> list[int]:
    """Convert the project-local relative owner view into Flobot terrain values."""
    opponent = 1 - observation.player_id
    result: list[int] = []
    for owner in observation.owner.reshape(-1).astype(int):
        if owner == int(CellView.OWN):
            result.append(observation.player_id)
        elif owner == int(CellView.ENEMY):
            result.append(opponent)
        elif owner == int(CellView.NEUTRAL):
            result.append(NEUTRAL)
        elif owner == int(CellView.MOUNTAIN):
            result.append(MOUNTAIN)
        elif owner == int(CellView.FOG_OBSTACLE):
            result.append(FOG_STRUCTURE)
        else:
            result.append(FOG)
    return result


def _general_list(observation: Observation) -> list[int]:
    """Return [own_general, enemy_general_or_-1] for Flobot's update loop."""
    own_mask = observation.known_generals & observation.own_tiles
    own_indices = _indices(own_mask)
    enemy_indices = _indices(observation.known_enemy_generals)
    own_general = own_indices[0] if own_indices else -1
    enemy_general = enemy_indices[0] if enemy_indices else -1
    return [own_general, enemy_general]


def _indices(mask: np.ndarray) -> list[int]:
    """Return flattened true indices from a boolean mask."""
    rows, cols = np.nonzero(mask)
    width = mask.shape[1]
    return [int(row) * width + int(col) for row, col in zip(rows, cols, strict=True)]

