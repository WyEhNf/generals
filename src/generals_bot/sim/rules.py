from __future__ import annotations

from .state import GameState
from .types import Action, ExecutedAction, Terrain


def execution_order(turn: int) -> list[int]:
    first = turn % 2
    return [first, 1 - first]


def validate_action(state: GameState, action: Action) -> str | None:
    if action.player_id not in (0, 1):
        return "invalid_player"
    if not state.alive[action.player_id]:
        return "player_dead"
    if action.start < 0 or action.start >= state.size:
        return "start_out_of_bounds"
    if action.end < 0 or action.end >= state.size:
        return "end_out_of_bounds"

    start_row, start_col = divmod(action.start, state.width)
    end_row, end_col = divmod(action.end, state.width)
    distance = abs(start_row - end_row) + abs(start_col - end_col)
    if distance != 1:
        return "not_adjacent"

    if state.terrain[start_row, start_col] != action.player_id:
        return "start_not_owned"
    if state.armies[start_row, start_col] <= 1:
        return "start_army_too_small"
    if state.terrain[end_row, end_col] == int(Terrain.MOUNTAIN):
        return "end_mountain"
    return None


def apply_action(state: GameState, action: Action) -> ExecutedAction:
    invalid_reason = validate_action(state, action)
    if invalid_reason is not None:
        return ExecutedAction(
            player_id=action.player_id,
            action=action,
            turn=state.turn,
            valid=False,
            reason=invalid_reason,
        )

    player = action.player_id
    start_row, start_col = divmod(action.start, state.width)
    end_row, end_col = divmod(action.end, state.width)
    source_army = int(state.armies[start_row, start_col])
    target_owner = int(state.terrain[end_row, end_col])
    target_army = int(state.armies[end_row, end_col])

    army_to_move = source_army // 2 if action.split else source_army - 1
    army_to_move = min(army_to_move, source_army - 1)
    if army_to_move < 1:
        return ExecutedAction(
            player,
            action,
            state.turn,
            False,
            "start_army_too_small",
            target_owner_before=target_owner,
            target_army_before=target_army,
        )

    army_to_stay = source_army - army_to_move
    damage_done = (
        min(army_to_move, target_army)
        if target_owner in (0, 1) and target_owner != player
        else 0
    )

    captured_general: int | None = None
    if target_owner == player:
        state.armies[end_row, end_col] += army_to_move
        state.armies[start_row, start_col] = army_to_stay
        return ExecutedAction(
            player,
            action,
            state.turn,
            True,
            target_owner_before=target_owner,
            target_army_before=target_army,
            moved_army=army_to_move,
        )

    state.armies[start_row, start_col] = army_to_stay
    if army_to_move > target_army:
        for opponent, general_tile in enumerate(state.generals):
            if opponent != player and general_tile == action.end and state.alive[opponent]:
                captured_general = opponent
                break
        state.terrain[end_row, end_col] = player
        state.armies[end_row, end_col] = army_to_move - target_army
        if captured_general is not None:
            state.alive[captured_general] = False
            state.has_kill[player] = True
            state.winner = player
    else:
        state.armies[end_row, end_col] = target_army - army_to_move

    return ExecutedAction(
        player_id=player,
        action=action,
        turn=state.turn,
        valid=True,
        captured_general=captured_general,
        target_owner_before=target_owner,
        target_army_before=target_army,
        moved_army=army_to_move,
        damage_done=damage_done,
    )


def apply_growth(state: GameState) -> None:
    if state.winner is not None:
        return

    next_turn = state.turn + 1
    if next_turn % 2 == 0:
        growth = state.cities.copy()
        for player, general in enumerate(state.generals):
            if state.alive[player]:
                row, col = divmod(general, state.width)
                growth[row, col] = True
        state.armies[growth & (state.terrain >= 0)] += 1

    if next_turn % 50 == 0:
        state.armies[state.terrain >= 0] += 1


def direction_between(start: int, end: int, width: int) -> int | None:
    diff = end - start
    by_diff = {
        -width: 0,
        width: 1,
        -1: 2,
        1: 3,
    }
    return by_diff.get(diff)
