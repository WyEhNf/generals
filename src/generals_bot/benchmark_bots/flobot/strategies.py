from __future__ import annotations

from typing import Protocol

from generals_bot.benchmark_bots.flobot.algorithms import FlobotAlgorithms
from generals_bot.benchmark_bots.flobot.heuristics import FlobotHeuristics
from generals_bot.benchmark_bots.flobot.state import FlobotMove, FlobotState


INITIAL_WAIT_TURNS = 23
REINFORCEMENT_INTERVAL = 50
SPREADING_TIMES = 4
ATTACK_TURNS_BEFORE_REINFORCEMENTS = 10
COLLECT_TURNS = 20


class _FlobotLike(Protocol):
    """Structural protocol for strategy methods to call back into the policy."""

    game_state: FlobotState
    game_map: object
    queued_moves: int
    last_attacked_index: int
    is_collecting: bool
    collect_area: list[int]
    is_infiltrating: bool
    collect_turns_left: int

    def move(self, move: FlobotMove) -> None:
        ...

    def queue_moves(self, moves: list[FlobotMove]) -> None:
        ...


class FlobotStrategy:
    """Top-level Flobot strategy selector."""

    @classmethod
    def pick_strategy(cls, bot: _FlobotLike) -> None:
        """Select and run one Flobot strategy for the current turn."""
        turn = bot.game_state.turn
        if bot.game_state.enemy_general != -1:
            cls.end_game(bot)
        elif bot.is_infiltrating:
            FlobotInfiltrate.infiltrate(bot)
        elif (
            turn % REINFORCEMENT_INTERVAL == 0
            and (
                turn / REINFORCEMENT_INTERVAL <= SPREADING_TIMES
                or len(bot.game_state.enemy_tiles) == 0
            )
        ):
            FlobotSpread.spread(bot)
        elif turn < REINFORCEMENT_INTERVAL:
            cls.early_game(bot, turn)
        else:
            cls.mid_game(bot, turn)

    @staticmethod
    def early_game(bot: _FlobotLike, turn: int) -> None:
        """Run Flobot's initial wait and two-step discovery opening."""
        if turn <= INITIAL_WAIT_TURNS:
            return
        if turn == INITIAL_WAIT_TURNS + 1:
            FlobotDiscover.first(bot, INITIAL_WAIT_TURNS)
        elif bot.queued_moves == 0:
            FlobotDiscover.second(bot, INITIAL_WAIT_TURNS)

    @staticmethod
    def mid_game(bot: _FlobotLike, turn: int) -> None:
        """Run Flobot's collect or scheduled infiltration setup."""
        if (
            len(bot.game_state.enemy_tiles) > 0
            and len(bot.collect_area) > 1
            and (
                turn
                + ATTACK_TURNS_BEFORE_REINFORCEMENTS
                + len(bot.collect_area)
                - 1
            )
            % REINFORCEMENT_INTERVAL
            == 0
        ):
            if len(bot.collect_area) == 2:
                bot.is_infiltrating = True
            start = bot.collect_area.pop(0)
            end = bot.collect_area[0]
            bot.move(FlobotMove(start, end))
        elif not bot.is_infiltrating:
            bot.collect_area = FlobotCollect.get_collect_area(bot)
            if bot.queued_moves == 0:
                FlobotCollect.collect(bot)

    @staticmethod
    def end_game(bot: _FlobotLike) -> None:
        """Run Flobot's visible-general finishing logic."""
        if not bot.is_infiltrating:
            FlobotRushGeneral.rush(bot)
            return
        if FlobotRushGeneral.try_to_kill_general(bot):
            return
        path_to_general = FlobotAlgorithms.a_star(
            bot.game_state,
            bot.game_map,
            bot.last_attacked_index,
            [bot.game_state.enemy_general],
        )
        if (
            len(path_to_general) <= 2
            or bot.game_map.remaining_armies_after_attack(
                bot.game_state,
                path_to_general[0],
                path_to_general[1],
            )
            <= 1
        ):
            bot.is_infiltrating = False
        if len(path_to_general) > 2:
            bot.move(FlobotMove(path_to_general[0], path_to_general[1]))


class FlobotSpread:
    """Flobot's reinforcement spread strategy."""

    @staticmethod
    def spread(bot: _FlobotLike) -> None:
        """Spread movable armies into adjacent neutral non-city tiles."""
        game_map = bot.game_map
        game_state = bot.game_state
        possible_moves = []
        for tile in game_map.get_moveable_tiles(game_state):
            neighbour_moves = []
            for next_tile in game_map.get_adjacent_tiles(game_state, tile).values():
                if next_tile.value == -1 and not game_map.is_city(game_state, next_tile):
                    neighbour_moves.append(next_tile.index)
            if neighbour_moves:
                possible_moves.append({"index": tile, "moves": neighbour_moves})
        FlobotSpread._sort_by_neighbour_count(possible_moves)
        while possible_moves:
            cur_node = possible_moves.pop(0)
            if cur_node["moves"]:
                chosen_tile = cur_node["moves"].pop(0)
                bot.move(FlobotMove(cur_node["index"], chosen_tile))
                FlobotSpread._remove_already_occupied_tile(possible_moves, chosen_tile)
                FlobotSpread._sort_by_neighbour_count(possible_moves)

    @staticmethod
    def _sort_by_neighbour_count(moves: list[dict]) -> None:
        """Sort moves by ascending available neutral neighbor count."""
        moves.sort(key=lambda value: len(value["moves"]))

    @staticmethod
    def _remove_already_occupied_tile(moves: list[dict], index: int) -> None:
        """Remove a chosen target tile from all pending spread candidates."""
        for move in moves:
            move["moves"] = [value for value in move["moves"] if value != index]


class FlobotDiscover:
    """Flobot's early-game discovery strategy."""

    @staticmethod
    def first(bot: _FlobotLike, wait_turns: int) -> None:
        """Queue a first discovery route from the general."""
        radius = FlobotDiscover.armies_received_till_turn(wait_turns + 1)
        reachable_tiles = FlobotAlgorithms.bfs(
            bot.game_state,
            bot.game_map,
            bot.game_state.own_general,
            int(radius),
        )
        discover_tile = FlobotHeuristics.choose_discover_tile(bot.game_map, reachable_tiles)
        if discover_tile is None:
            return
        moves = FlobotAlgorithms.dijkstra_moves(
            bot.game_state,
            bot.game_map,
            bot.game_state.own_general,
            discover_tile,
        )
        bot.queue_moves(moves)

    @staticmethod
    def second(bot: _FlobotLike, wait_turns: int) -> None:
        """Choose the next short expansion move after the first queued path drains."""
        turns = int(((wait_turns + 1) / 2 + 1) // 2)
        moveable_tiles = bot.game_map.get_moveable_tiles(bot.game_state)
        if moveable_tiles:
            move = FlobotAlgorithms.decision_tree_search(
                bot.game_state,
                bot.game_map,
                moveable_tiles,
                turns,
            )
            bot.move(move)

    @staticmethod
    def armies_received_till_turn(turn: int) -> float:
        """Return Flobot's opening radius estimate."""
        return turn / 2 + 1


class FlobotCollect:
    """Flobot's army collection strategy."""

    @staticmethod
    def get_collect_area(bot: _FlobotLike) -> list[int]:
        """Return the path Flobot wants to collect armies onto."""
        bot.is_collecting = True
        if bot.game_state.enemy_tiles:
            enemy_target = FlobotHeuristics.choose_enemy_target_tile_by_lowest_army(
                bot.game_state,
                bot.game_map,
            )
            if enemy_target is not None:
                return FlobotAlgorithms.a_star(
                    bot.game_state,
                    bot.game_map,
                    bot.game_state.own_general,
                    [enemy_target["index"]],
                )
        return [bot.game_state.own_general]

    @staticmethod
    def collect(bot: _FlobotLike) -> None:
        """Move the highest army tile toward the current collect path."""
        highest_army_index = FlobotCollect.get_highest_army_index(
            bot.game_state.own_tiles,
            bot.collect_area,
        )
        if highest_army_index == -1:
            bot.is_collecting = False
            return
        path_to_attacking_path = FlobotAlgorithms.a_star(
            bot.game_state,
            bot.game_map,
            highest_army_index,
            bot.collect_area,
        )
        if len(path_to_attacking_path) > 1:
            bot.move(FlobotMove(highest_army_index, path_to_attacking_path[1]))

    @staticmethod
    def get_highest_army_index(tiles: dict[int, int], path: list[int]) -> int:
        """Return the largest movable own tile not already on the collect path."""
        index = -1
        armies = 0
        for tile, army in tiles.items():
            if army > armies and army > 1 and tile not in path:
                index = tile
                armies = army
        return index


class FlobotInfiltrate:
    """Flobot's enemy-territory infiltration strategy."""

    @staticmethod
    def infiltrate(bot: _FlobotLike) -> None:
        """Continue moving through enemy land adjacent to fog."""
        enemy_neighbor = None
        if not FlobotInfiltrate.last_attacked_index_is_valid(bot):
            bot.is_infiltrating = False
            return
        attack_source = bot.last_attacked_index
        for next_tile in bot.game_map.get_adjacent_tiles(bot.game_state, attack_source).values():
            if (
                bot.game_map.is_enemy(bot.game_state, next_tile.index)
                and bot.game_map.is_walkable(bot.game_state, next_tile)
                and bot.game_map.is_adjacent_to_fog(bot.game_state, next_tile.index)
            ):
                if (
                    enemy_neighbor is None
                    or bot.game_state.armies[next_tile.index] < bot.game_state.armies[enemy_neighbor.index]
                ):
                    enemy_neighbor = next_tile
        start = attack_source
        end = -1
        if enemy_neighbor is not None:
            end = enemy_neighbor.index
        else:
            path = FlobotInfiltrate.get_path_to_next_tile(bot, attack_source)
            if len(path) > 1:
                end = path[1]
        if end == -1 or bot.game_map.remaining_armies_after_attack(bot.game_state, start, end) <= 1:
            bot.is_infiltrating = False
        if end != -1 and bot.game_map.remaining_armies_after_attack(bot.game_state, start, end) >= 1:
            bot.move(FlobotMove(start, end))

    @staticmethod
    def last_attacked_index_is_valid(bot: _FlobotLike) -> bool:
        """Return whether the last attacked tile is currently owned by Flobot."""
        return (
            bot.last_attacked_index != -1
            and 0 <= bot.last_attacked_index < bot.game_state.size
            and bot.game_state.terrain[bot.last_attacked_index] == bot.game_state.player_index
        )

    @staticmethod
    def get_path_to_next_tile(bot: _FlobotLike, index: int) -> list[int]:
        """Find a path to the next visible enemy tile adjacent to fog."""
        tiles_with_fog = [
            tile
            for tile in bot.game_state.enemy_tiles
            if bot.game_map.is_adjacent_to_fog(bot.game_state, tile)
        ]
        if tiles_with_fog:
            return FlobotAlgorithms.a_star(bot.game_state, bot.game_map, index, tiles_with_fog)
        return []


class FlobotRushGeneral:
    """Flobot's visible-general attack strategy."""

    @staticmethod
    def rush(bot: _FlobotLike) -> None:
        """Try to kill a visible enemy general or gather toward it."""
        if FlobotRushGeneral.try_to_kill_general(bot):
            return
        if bot.collect_turns_left > 0:
            bot.collect_area = FlobotAlgorithms.a_star(
                bot.game_state,
                bot.game_map,
                bot.game_state.own_general,
                [bot.game_state.enemy_general],
            )
            if bot.collect_area:
                bot.collect_area.pop()
            FlobotCollect.collect(bot)
            bot.collect_turns_left -= 1
        elif bot.collect_turns_left == 0:
            FlobotRushGeneral.move_to_general(bot, bot.game_state.own_general)
            bot.collect_turns_left = -1
        else:
            FlobotRushGeneral.move_to_general(bot, bot.last_attacked_index)

    @staticmethod
    def move_to_general(bot: _FlobotLike, start: int) -> None:
        """Move along a path toward the known enemy general."""
        path = FlobotAlgorithms.a_star(
            bot.game_state,
            bot.game_map,
            start,
            [bot.game_state.enemy_general],
        )
        if len(path) > 2:
            bot.move(FlobotMove(start, path[1]))
        else:
            bot.collect_turns_left = COLLECT_TURNS

    @staticmethod
    def try_to_kill_general(bot: _FlobotLike) -> bool:
        """Attack the enemy general if adjacent armies are sufficient."""
        attackable_neighbours = []
        for next_tile in bot.game_map.get_adjacent_tiles(bot.game_state, bot.game_state.enemy_general).values():
            if next_tile.value == bot.game_state.player_index:
                if FlobotRushGeneral.has_enough_armies_to_attack_general(bot, next_tile.index):
                    bot.move(FlobotMove(next_tile.index, bot.game_state.enemy_general))
                    bot.is_infiltrating = False
                    return True
                if bot.game_state.armies[next_tile.index] > 1:
                    attackable_neighbours.append(next_tile.index)
        if len(attackable_neighbours) > 1:
            return FlobotRushGeneral.try_group_attack(bot, attackable_neighbours)
        return False

    @staticmethod
    def try_group_attack(bot: _FlobotLike, attackable_neighbours: list[int]) -> bool:
        """Try a multi-tile attack if adjacent armies can overwhelm the general."""
        highest_army = -1
        highest_army_tile = -1
        attackable_army_sum = 0
        for neighbour in attackable_neighbours:
            armies = bot.game_state.armies[neighbour]
            attackable_army_sum += armies - 1
            if armies > highest_army:
                highest_army = armies
                highest_army_tile = neighbour
        if attackable_army_sum > bot.game_state.armies[bot.game_state.enemy_general]:
            bot.move(FlobotMove(highest_army_tile, bot.game_state.enemy_general))
            return True
        return False

    @staticmethod
    def has_enough_armies_to_attack_general(bot: _FlobotLike, index: int) -> bool:
        """Return whether a single adjacent tile can capture the enemy general."""
        next_turn_army_gain = 1 if bot.game_state.turn % 2 != 0 else 0
        return (
            bot.game_state.armies[index] - 1
            > bot.game_state.armies[bot.game_state.enemy_general] + next_turn_army_gain
        )

