from __future__ import annotations

import heapq
from dataclasses import dataclass

from generals_bot.benchmark_bots.flobot.game_map import FlobotMap
from generals_bot.benchmark_bots.flobot.heuristics import FlobotHeuristics
from generals_bot.benchmark_bots.flobot.state import FlobotMove, FlobotState


@dataclass
class _AStarNode:
    index: int
    g: int = 0
    h: int = 0
    f: int = 0
    armies: int = 0
    visited: bool = False
    closed: bool = False
    parent: _AStarNode | None = None


class FlobotAlgorithms:
    """Graph search algorithms used by Flobot strategies."""

    @staticmethod
    def bfs(
        game_state: FlobotState,
        game_map: FlobotMap,
        node: int,
        radius: int,
    ) -> list[dict[str, int]]:
        """Run Flobot's BFS over walkable tiles up to a radius."""
        is_visited = [False] * game_map.size
        is_visited[node] = True
        queue = [node]
        cur_layer = 0
        cur_layer_tiles = 1
        next_layer_tiles = 0
        found_nodes: list[dict[str, int]] = []
        while queue:
            cur_tile = queue.pop(0)
            if cur_layer != 0:
                found_nodes.append({"index": cur_tile, "generalDistance": cur_layer})
            for next_tile in game_map.get_adjacent_tiles(game_state, cur_tile).values():
                if not (0 <= next_tile.index < game_map.size):
                    continue
                if not is_visited[next_tile.index] and game_map.is_walkable(game_state, next_tile):
                    queue.append(next_tile.index)
                    is_visited[next_tile.index] = True
                    next_layer_tiles += 1
            cur_layer_tiles -= 1
            if cur_layer_tiles == 0:
                if cur_layer == radius:
                    break
                cur_layer += 1
                cur_layer_tiles = next_layer_tiles
                next_layer_tiles = 0
        return found_nodes

    @staticmethod
    def a_star(
        game_state: FlobotState,
        game_map: FlobotMap,
        start: int,
        ends: list[int],
    ) -> list[int]:
        """Find a low-cost path to any target using Flobot's A* variant."""
        if start < 0 or start >= game_map.size or not ends:
            return []
        targets = {target for target in ends if 0 <= target < game_map.size}
        if not targets:
            return []
        search_list = [_AStarNode(index=i) for i in range(game_map.size)]
        counter = 0
        open_heap: list[tuple[int, int, int, int]] = []
        heapq.heappush(open_heap, (0, 0, counter, start))
        search_list[start].visited = True
        while open_heap:
            _, _, _, cur_index = heapq.heappop(open_heap)
            cur_tile = search_list[cur_index]
            if cur_tile.closed:
                continue
            if cur_tile.index in targets:
                return _construct_astar_path(cur_tile)
            cur_tile.closed = True
            for next_tile in game_map.get_adjacent_tiles(game_state, cur_tile.index).values():
                if not (0 <= next_tile.index < game_map.size):
                    continue
                if not game_map.is_walkable(game_state, next_tile):
                    continue
                neighbour = search_list[next_tile.index]
                if neighbour.closed:
                    continue
                g_score = cur_tile.g + 1
                if next_tile.value >= 0 and next_tile.value != game_state.player_index:
                    g_score += game_state.armies[next_tile.index]
                elif next_tile.value == -1:
                    g_score += 1
                if not neighbour.visited or g_score < neighbour.g:
                    neighbour.visited = True
                    neighbour.parent = cur_tile
                    neighbour.g = g_score
                    neighbour.h = neighbour.h or _nearest_endpoint_h_score(game_map, start, list(targets))
                    neighbour.f = neighbour.g + neighbour.h
                    neighbour.armies = game_state.armies[next_tile.index]
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (neighbour.f, -neighbour.armies, counter, neighbour.index),
                    )
        return []

    @staticmethod
    def dijkstra_moves(
        game_state: FlobotState,
        game_map: FlobotMap,
        start: int,
        end: int,
    ) -> list[FlobotMove]:
        """Return move objects along Flobot's unweighted route to a target."""
        if start < 0 or end < 0 or start >= game_map.size or end >= game_map.size:
            return []
        is_visited = [False] * game_map.size
        previous = list(range(game_map.size))
        previous[start] = -1
        queue = [start]
        while queue:
            cur_tile = queue.pop(0)
            is_visited[cur_tile] = True
            for next_tile in game_map.get_adjacent_tiles(game_state, cur_tile).values():
                if not (0 <= next_tile.index < game_map.size):
                    continue
                if (
                    not is_visited[next_tile.index]
                    and next_tile.index not in queue
                    and game_map.is_walkable(game_state, next_tile)
                ):
                    previous[next_tile.index] = cur_tile
                    if next_tile.index == end:
                        return _construct_dijkstra_moves(start, end, previous)
                    queue.append(next_tile.index)
        return []

    @classmethod
    def decision_tree_search(
        cls,
        game_state: FlobotState,
        game_map: FlobotMap,
        start_points: list[int],
        turns: int,
    ) -> FlobotMove:
        """Search short expansion plans and return the first move of the best path."""
        if not start_points:
            return FlobotMove(-1, -1, 0)
        moves = [
            cls._decision_tree_search_rec(game_state, game_map, start, turns)
            for start in start_points
        ]
        return _best_move(moves)

    @classmethod
    def _decision_tree_search_rec(
        cls,
        game_state: FlobotState,
        game_map: FlobotMap,
        start: int,
        turns: int,
        weight: int = 0,
    ) -> FlobotMove:
        """Recursive Flobot decision-tree search over adjacent walkable moves."""
        possible_moves: list[FlobotMove] = []
        if turns != 0:
            for next_tile in game_map.get_adjacent_tiles(game_state, start).values():
                if game_map.is_walkable(game_state, next_tile):
                    next_weight = FlobotHeuristics.calc_capture_weight(
                        game_map.player_index,
                        next_tile.value,
                    )
                    possible_moves.append(
                        cls._decision_tree_search_rec(
                            game_state,
                            game_map,
                            next_tile.index,
                            turns - 1,
                            next_weight,
                        )
                    )
            possible_moves.append(
                cls._decision_tree_search_rec(game_state, game_map, start, turns - 1, 0)
            )
        if not possible_moves:
            return FlobotMove(start, -1, weight)
        best_path = _best_move(possible_moves)
        return FlobotMove(start, best_path.start, weight + best_path.weight)


def _construct_astar_path(node: _AStarNode) -> list[int]:
    """Construct an index path from a terminal A* node."""
    path = []
    cur: _AStarNode | None = node
    while cur is not None:
        path.append(cur.index)
        cur = cur.parent
    return list(reversed(path))


def _construct_dijkstra_moves(start: int, end: int, previous: list[int]) -> list[FlobotMove]:
    """Construct move objects from Dijkstra predecessor links."""
    moves: list[FlobotMove] = []
    cur_index = end
    while True:
        prev_index = previous[cur_index]
        if prev_index == -1:
            break
        moves.insert(0, FlobotMove(prev_index, cur_index))
        cur_index = prev_index
    if moves and moves[0].start == start:
        return moves
    return []


def _best_move(moves: list[FlobotMove]) -> FlobotMove:
    """Return Flobot's highest-weight move candidate."""
    return max(moves, key=lambda move: move.weight)


def _nearest_endpoint_h_score(game_map: FlobotMap, start: int, ends: list[int]) -> int:
    """Return Flobot's heuristic score; original code uses start for every node."""
    return min(game_map.manhattan_distance(start, end) for end in ends)

