from __future__ import annotations

import numpy as np

from .state import GameState
from .types import CellView, Observation, Terrain


def update_all_memory(state: GameState) -> None:
    """Refresh each player's remembered map facts from current visibility."""
    for player in (0, 1):
        visible = visible_mask(state, player)
        memory = state.memory[player]
        memory.explored |= visible
        memory.seen_cities |= state.cities & visible
        memory.seen_generals |= _generals_mask(state) & visible
        memory.seen_obstacles |= (state.cities | (state.terrain == int(Terrain.MOUNTAIN))) & visible


def observe(state: GameState, player: int) -> Observation:
    """Build one player-relative observation from the global game state."""
    visible = visible_mask(state, player)
    explored = state.memory[player].explored.copy()
    memory = state.memory[player]
    opponent = 1 - player
    scores = state.scores()
    own_score = scores[player]
    enemy_score = scores[opponent]

    owner = np.full((state.height, state.width), int(CellView.FOG), dtype=int)
    fog_obstacles = (~visible) & (state.cities | (state.terrain == int(Terrain.MOUNTAIN)))
    owner[fog_obstacles] = int(CellView.FOG_OBSTACLE)
    owner[visible & (state.terrain == int(Terrain.MOUNTAIN))] = int(CellView.MOUNTAIN)
    owner[visible & (state.terrain == int(Terrain.NEUTRAL))] = int(CellView.NEUTRAL)
    owner[visible & (state.terrain == player)] = int(CellView.OWN)
    owner[visible & (state.terrain == opponent)] = int(CellView.ENEMY)

    general_mask = _generals_mask(state)
    generals = general_mask & visible
    remembered_generals = memory.seen_generals | generals
    known_generals = remembered_generals & general_mask
    known_enemy_generals = remembered_generals & _general_mask_for_player(state, opponent)
    mountains = (state.terrain == int(Terrain.MOUNTAIN)) & visible
    cities = state.cities & visible
    known_cities = memory.seen_cities | cities
    fog = owner == int(CellView.FOG)

    return Observation(
        player_id=player,
        turn=state.turn,
        width=state.width,
        height=state.height,
        visible=visible.copy(),
        explored=explored,
        armies=np.where(visible, state.armies, 0),
        owner=owner,
        own_tiles=(state.terrain == player) & visible,
        enemy_tiles=(state.terrain == opponent) & visible,
        neutral_tiles=(state.terrain == int(Terrain.NEUTRAL)) & visible,
        mountains=mountains,
        cities=cities,
        known_cities=known_cities,
        generals=generals,
        known_generals=known_generals,
        known_enemy_generals=known_enemy_generals,
        fog=fog,
        fog_obstacles=fog_obstacles,
        own_army=own_score.army,
        own_land=own_score.land,
        enemy_army=enemy_score.army,
        enemy_land=enemy_score.land,
        last_moves=[],
        priority=state.turn,
    )


def visible_mask(state: GameState, player: int) -> np.ndarray:
    owned = state.terrain == player
    result = np.zeros((state.height, state.width), dtype=bool)
    rows, cols = np.nonzero(owned)
    for row, col in zip(rows, cols, strict=True):
        r0 = max(0, row - 1)
        r1 = min(state.height, row + 2)
        c0 = max(0, col - 1)
        c1 = min(state.width, col + 2)
        result[r0:r1, c0:c1] = True
    return result


def _generals_mask(state: GameState) -> np.ndarray:
    result = np.zeros((state.height, state.width), dtype=bool)
    for player, tile in enumerate(state.generals):
        if state.alive[player]:
            row, col = divmod(tile, state.width)
            result[row, col] = True
    return result


def _general_mask_for_player(state: GameState, player: int) -> np.ndarray:
    """Return a single alive player's general mask."""
    result = np.zeros((state.height, state.width), dtype=bool)
    if state.alive[player]:
        row, col = divmod(state.generals[player], state.width)
        result[row, col] = True
    return result
