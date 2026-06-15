from __future__ import annotations

from generals_bot.benchmark_bots.flobot.state import (
    FOG_STRUCTURE,
    MOUNTAIN,
    OUT_OF_BOUNDS,
    FlobotState,
    FlobotTile,
)


class FlobotMap:
    """Flobot's map helper methods over flattened tile indices."""

    def __init__(self, width: int, height: int, player_index: int) -> None:
        """Initialize geometry and the player-relative owner id."""
        self.width = width
        self.height = height
        self.size = width * height
        self.player_index = player_index

    def is_walkable(self, game_state: FlobotState, tile: FlobotTile) -> bool:
        """Return whether Flobot can route through a tile."""
        return (
            tile.value not in (FOG_STRUCTURE, OUT_OF_BOUNDS, MOUNTAIN)
            and not self.is_city(game_state, tile)
        )

    def is_city(self, game_state: FlobotState, tile: FlobotTile) -> bool:
        """Return whether a tile is a known city."""
        return tile.index in game_state.cities

    def is_enemy(self, game_state: FlobotState, index: int) -> bool:
        """Return whether an index currently contains visible enemy land."""
        return (
            0 <= index < game_state.size
            and game_state.terrain[index] >= 0
            and game_state.terrain[index] != self.player_index
        )

    def get_adjacent_tiles(self, game_state: FlobotState, index: int) -> dict[str, FlobotTile]:
        """Return adjacent tiles in Flobot's up/right/down/left order."""
        return {
            "up": self.get_adjacent_tile(game_state, index, -self.width),
            "right": self.get_adjacent_tile(game_state, index, 1),
            "down": self.get_adjacent_tile(game_state, index, self.width),
            "left": self.get_adjacent_tile(game_state, index, -1),
        }

    def get_adjacent_tile(self, game_state: FlobotState, index: int, distance: int) -> FlobotTile:
        """Return one adjacent tile or an out-of-bounds marker."""
        adjacent_index = index + distance
        cur_row = index // self.width
        adjacent_row = adjacent_index // self.width
        in_bounds = False
        if distance in (1, -1):
            in_bounds = 0 <= adjacent_index < self.size and adjacent_row == cur_row
        elif distance == self.width:
            in_bounds = 0 <= adjacent_index < self.size and adjacent_row < self.height
        elif distance == -self.width:
            in_bounds = 0 <= adjacent_index < self.size and adjacent_row >= 0
        if in_bounds:
            return FlobotTile(adjacent_index, game_state.terrain[adjacent_index])
        return FlobotTile(adjacent_index, OUT_OF_BOUNDS)

    def is_adjacent_to_fog(self, game_state: FlobotState, index: int) -> bool:
        """Return whether any adjacent in-bounds tile has not been discovered by Flobot."""
        for next_tile in self.get_adjacent_tiles(game_state, index).values():
            if 0 <= next_tile.index < game_state.size and not game_state.discovered_tiles[next_tile.index]:
                return True
        return False

    def is_adjacent_to_enemy(self, game_state: FlobotState, index: int) -> bool:
        """Return whether any adjacent tile is visible enemy-owned land."""
        for next_tile in self.get_adjacent_tiles(game_state, index).values():
            if next_tile.value >= 0 and next_tile.value != self.player_index:
                return True
        return False

    def get_moveable_tiles(self, game_state: FlobotState) -> list[int]:
        """Return own tiles with enough army to move."""
        return [index for index, army in game_state.own_tiles.items() if army > 1]

    def remaining_armies_after_attack(self, game_state: FlobotState, start: int, end: int) -> int:
        """Estimate armies remaining after attacking from start to end."""
        if 0 <= start < game_state.size and 0 <= end < game_state.size:
            return game_state.armies[start] - 1 - game_state.armies[end]
        return 0

    def get_edge_weight_for_index(self, index: int) -> int:
        """Return Flobot's center-biased edge weight for discovery choices."""
        x, y = self.get_coordinates_from_tile_index(index)
        upper_edge = y
        right_edge = self.width - 1 - x
        down_edge = self.height - 1 - y
        left_edge = x
        return min(upper_edge, down_edge) * min(left_edge, right_edge)

    def manhattan_distance(self, index1: int, index2: int) -> int:
        """Return Manhattan distance between two flattened indices."""
        x1, y1 = self.get_coordinates_from_tile_index(index1)
        x2, y2 = self.get_coordinates_from_tile_index(index2)
        return abs(x1 - x2) + abs(y1 - y2)

    def get_coordinates_from_tile_index(self, index: int) -> tuple[int, int]:
        """Return x/y coordinates for a flattened index."""
        return index % self.width, index // self.width

