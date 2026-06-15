from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from generals_bot.sim.queue import normalize_actions
from generals_bot.sim.types import CellView, GameConfig, Observation


SERVER_NEUTRAL = -1
SERVER_MOUNTAIN = -2
SERVER_FOG = -3
SERVER_FOG_OBSTACLE = -4


@dataclass(frozen=True)
class RemoteAttack:
    """One attack command ready to send to the generals.io socket API."""

    start: int
    end: int
    split: bool = False


@dataclass
class RemoteMemory:
    """Player-relative facts remembered across server updates."""

    explored: np.ndarray
    seen_cities: np.ndarray
    seen_generals: np.ndarray
    seen_obstacles: np.ndarray

    @classmethod
    def empty(cls, height: int, width: int) -> RemoteMemory:
        """Create empty remote memory for one map size."""
        shape = (height, width)
        return cls(
            explored=np.zeros(shape, dtype=bool),
            seen_cities=np.zeros(shape, dtype=bool),
            seen_generals=np.zeros(shape, dtype=bool),
            seen_obstacles=np.zeros(shape, dtype=bool),
        )


@dataclass
class RemoteGameState:
    """Maintain patched generals.io server state and expose local Observations."""

    start_data: dict[str, Any] | None = None
    map_cache: list[int] = field(default_factory=list)
    cities_cache: list[int] = field(default_factory=list)
    stars: list[int] = field(default_factory=list)
    last_update: dict[str, Any] | None = None
    memory: RemoteMemory | None = None

    def start(self, data: dict[str, Any]) -> None:
        """Store the server game_start payload."""
        self.start_data = dict(data)
        self.map_cache = []
        self.cities_cache = []
        self.stars = []
        self.last_update = None
        self.memory = None

    def apply_update(self, data: dict[str, Any]) -> Observation:
        """Patch server state with one game_update and return an Observation."""
        if self.start_data is None:
            raise RemoteStateError("game_update received before game_start")
        apply_diff(self.map_cache, data["map_diff"])
        apply_diff(self.cities_cache, data["cities_diff"])
        if "stars" in data:
            self.stars = list(data["stars"])
        self.last_update = dict(data)
        observation = self.to_observation()
        self._update_memory(observation)
        return self.to_observation()

    def to_observation(self) -> Observation:
        """Build a local Observation from the current patched server state."""
        if self.start_data is None:
            raise RemoteStateError("missing game_start data")
        if len(self.map_cache) < 2:
            raise RemoteStateError("missing patched map data")
        if self.last_update is None:
            raise RemoteStateError("missing game_update data")

        width = int(self.map_cache[0])
        height = int(self.map_cache[1])
        size = width * height
        expected_length = 2 + 2 * size
        if len(self.map_cache) < expected_length:
            raise RemoteStateError(
                f"patched map is too short: got {len(self.map_cache)}, expected {expected_length}"
            )

        player = int(self.start_data["playerIndex"])
        if player not in (0, 1):
            raise RemoteStateError(f"only 1v1 player indices are supported, got {player}")
        opponent = 1 - player

        armies = np.array(self.map_cache[2 : 2 + size], dtype=int).reshape(height, width)
        server_tiles = np.array(
            self.map_cache[2 + size : 2 + 2 * size],
            dtype=int,
        ).reshape(height, width)
        visible = (server_tiles != SERVER_FOG) & (server_tiles != SERVER_FOG_OBSTACLE)

        memory = self._memory_for_shape(height, width)
        scores = _scores_by_player(self.last_update.get("scores", []))
        own_score = scores.get(player, {})
        enemy_score = scores.get(opponent, {})

        owner = np.full((height, width), int(CellView.FOG), dtype=int)
        owner[server_tiles == SERVER_FOG_OBSTACLE] = int(CellView.FOG_OBSTACLE)
        owner[visible & (server_tiles == SERVER_MOUNTAIN)] = int(CellView.MOUNTAIN)
        owner[visible & (server_tiles == SERVER_NEUTRAL)] = int(CellView.NEUTRAL)
        owner[visible & (server_tiles == player)] = int(CellView.OWN)
        owner[visible & (server_tiles >= 0) & (server_tiles != player)] = int(CellView.ENEMY)

        cities_mask = _mask_from_tiles(self.cities_cache, height, width)
        generals_mask = _mask_from_tiles(self.last_update.get("generals", []), height, width)
        own_general_mask = _single_general_mask(
            self.last_update.get("generals", []),
            player,
            height,
            width,
        )
        enemy_general_mask = _single_general_mask(
            self.last_update.get("generals", []),
            opponent,
            height,
            width,
        )

        mountains = visible & (server_tiles == SERVER_MOUNTAIN)
        current_generals = generals_mask & visible
        known_generals = (memory.seen_generals | current_generals) & (
            generals_mask | memory.seen_generals
        )
        known_enemy_generals = memory.seen_generals & ~own_general_mask
        if enemy_general_mask.any():
            known_enemy_generals |= enemy_general_mask

        fog_obstacles = server_tiles == SERVER_FOG_OBSTACLE
        fog = server_tiles == SERVER_FOG
        return Observation(
            player_id=player,
            turn=int(self.last_update.get("turn", 0)),
            width=width,
            height=height,
            visible=visible.copy(),
            explored=memory.explored.copy(),
            armies=np.where(visible, armies, 0),
            owner=owner,
            own_tiles=visible & (server_tiles == player),
            enemy_tiles=visible & (server_tiles >= 0) & (server_tiles != player),
            neutral_tiles=visible & (server_tiles == SERVER_NEUTRAL),
            mountains=mountains,
            cities=visible & cities_mask,
            known_cities=memory.seen_cities | cities_mask,
            generals=current_generals,
            known_generals=known_generals,
            known_enemy_generals=known_enemy_generals,
            fog=fog,
            fog_obstacles=fog_obstacles,
            own_army=int(own_score.get("total", 0)),
            own_land=int(own_score.get("tiles", 0)),
            enemy_army=int(enemy_score.get("total", 0)),
            enemy_land=int(enemy_score.get("tiles", 0)),
            last_moves=[],
            priority=int(self.last_update.get("turn", 0)),
        )

    def placeholder_config(self) -> GameConfig:
        """Return a GameConfig usable for Policy.reset after the first update."""
        if len(self.map_cache) < 2:
            raise RemoteStateError("cannot build config before first map update")
        width = int(self.map_cache[0])
        height = int(self.map_cache[1])
        return GameConfig(width=width, height=height, generals=(0, width * height - 1))

    def replay_url(self) -> str | None:
        """Return the public replay URL when the server supplied a replay id."""
        if self.start_data is None:
            return None
        replay_id = self.start_data.get("replay_id")
        if not replay_id:
            return None
        return f"http://bot.generals.io/replays/{replay_id}"

    def _memory_for_shape(self, height: int, width: int) -> RemoteMemory:
        """Return initialized memory for the current map shape."""
        if self.memory is None:
            self.memory = RemoteMemory.empty(height, width)
        return self.memory

    def _update_memory(self, observation: Observation) -> None:
        """Refresh remote memory from the latest observation."""
        memory = self._memory_for_shape(observation.height, observation.width)
        memory.explored |= observation.visible
        memory.seen_cities |= observation.cities
        memory.seen_generals |= observation.generals
        memory.seen_obstacles |= observation.mountains | observation.cities | observation.fog_obstacles


class RemoteStateError(ValueError):
    """Raised when server data cannot be converted into a local observation."""


def apply_diff(cache: list[int], diff: list[int]) -> None:
    """Apply a generals.io run-length diff patch in place."""
    i = 0
    cursor = 0
    while i < len(diff) - 1:
        cursor += int(diff[i])
        length = int(diff[i + 1])
        cache[cursor : cursor + length] = [int(value) for value in diff[i + 2 : i + 2 + length]]
        cursor += length
        i += length + 2
    if i == len(diff) - 1:
        del cache[cursor + int(diff[i]) :]
        i += 1
    if i != len(diff):
        raise RemoteStateError("invalid diff patch")


def attacks_from_action_like(action_like: Any, width: int) -> list[RemoteAttack]:
    """Normalize a Policy action-like value into remote attack commands."""
    return [
        RemoteAttack(start=action.start, end=action.end, split=action.split)
        for action in normalize_actions(action_like, width)
    ]


def summarize_observation(observation: Observation) -> dict[str, Any]:
    """Return a compact JSON-friendly observation summary for remote logs."""
    return {
        "turn": observation.turn,
        "player_id": observation.player_id,
        "width": observation.width,
        "height": observation.height,
        "own_land": observation.own_land,
        "own_army": observation.own_army,
        "enemy_land": observation.enemy_land,
        "enemy_army": observation.enemy_army,
        "visible_tiles": int(np.sum(observation.visible)),
        "own_tiles_visible": int(np.sum(observation.own_tiles)),
        "enemy_tiles_visible": int(np.sum(observation.enemy_tiles)),
        "known_enemy_generals": int(np.sum(observation.known_enemy_generals)),
    }


def _scores_by_player(scores: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index server score objects by player id."""
    return {int(score["i"]): score for score in scores if "i" in score}


def _mask_from_tiles(tiles: list[int], height: int, width: int) -> np.ndarray:
    """Return a boolean mask for valid flat tile ids."""
    result = np.zeros((height, width), dtype=bool)
    for tile in tiles:
        tile = int(tile)
        if tile >= 0 and tile < height * width:
            row, col = divmod(tile, width)
            result[row, col] = True
    return result


def _single_general_mask(
    generals: list[int],
    player: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Return one player's visible general mask."""
    if player < 0 or player >= len(generals):
        return np.zeros((height, width), dtype=bool)
    return _mask_from_tiles([int(generals[player])], height, width)
