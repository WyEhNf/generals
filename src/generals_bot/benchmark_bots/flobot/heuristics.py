from __future__ import annotations

from generals_bot.benchmark_bots.flobot.game_map import FlobotMap
from generals_bot.benchmark_bots.flobot.state import (
    FOG,
    NEUTRAL,
    FlobotState,
)


class FlobotHeuristics:
    """Small Flobot target-selection heuristics."""

    @staticmethod
    def choose_discover_tile(game_map: FlobotMap, tiles: list[dict[str, int]]) -> int | None:
        """Choose a discovery target from BFS tiles."""
        if not tiles:
            return None
        optimal_index = -1
        optimal_edge_weight = -1
        max_general_distance = tiles[-1]["generalDistance"]
        for tile in reversed(tiles):
            edge_weight = game_map.get_edge_weight_for_index(tile["index"])
            if tile["generalDistance"] < max_general_distance:
                return optimal_index if optimal_index != -1 else None
            if edge_weight > optimal_edge_weight:
                optimal_index = tile["index"]
                optimal_edge_weight = edge_weight
        return optimal_index if optimal_index != -1 else None

    @staticmethod
    def choose_enemy_target_tile_by_lowest_army(
        game_state: FlobotState,
        game_map: FlobotMap,
    ) -> dict[str, int] | None:
        """Choose a visible enemy tile adjacent to fog with the lowest army."""
        tiles_with_fog = [
            {"index": index, "value": army}
            for index, army in game_state.enemy_tiles.items()
            if game_map.is_adjacent_to_fog(game_state, index)
        ]
        if not tiles_with_fog:
            return None
        return min(tiles_with_fog, key=lambda tile: tile["value"])

    @staticmethod
    def calc_capture_weight(player_index: int, terrain_value: int) -> int:
        """Return Flobot's decision-tree capture weight."""
        if terrain_value == player_index:
            return 0
        if terrain_value in (NEUTRAL, FOG):
            return 1
        if terrain_value < 0:
            return 3
        return 1

