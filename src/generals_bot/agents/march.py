from __future__ import annotations

from generals_bot.sim.types import Action, CellView, Direction, GameConfig, Observation, PassAction


class MarchPolicy:
    """Debug policy: wait, then push the largest visible owned stack forward."""

    def __init__(self, *, wait_turns: int = 10, direction: Direction = Direction.RIGHT) -> None:
        self.wait_turns = wait_turns
        self.direction = direction
        self.player_id: int | None = None

    def reset(self, player_id: int, config: GameConfig) -> None:
        del config
        self.player_id = player_id

    def act(self, observation: Observation) -> Action | PassAction:
        if observation.turn < self.wait_turns:
            return PassAction(observation.player_id)

        start = self._largest_owned_tile(observation)
        if start is None:
            return PassAction(observation.player_id)

        row, col = divmod(start, observation.width)
        for direction in _direction_order(self.direction):
            target = _neighbor(row, col, direction, observation.height, observation.width)
            if target is None:
                continue
            nr, nc = target
            if not _target_is_usable(observation, nr, nc):
                continue
            end = nr * observation.width + nc
            return Action(observation.player_id, start, end)

        return PassAction(observation.player_id)

    def _largest_owned_tile(self, observation: Observation) -> int | None:
        rows, cols = (observation.own_tiles & (observation.armies > 1)).nonzero()
        if len(rows) == 0:
            return None

        best_tile: int | None = None
        best_army = -1
        for row, col in zip(rows, cols, strict=True):
            army = int(observation.armies[row, col])
            tile = int(row) * observation.width + int(col)
            if army > best_army or (army == best_army and (best_tile is None or tile < best_tile)):
                best_army = army
                best_tile = tile
        return best_tile


def _direction_order(preferred: Direction) -> list[Direction]:
    base = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    return [preferred] + [direction for direction in base if direction != preferred]


def _neighbor(
    row: int,
    col: int,
    direction: Direction,
    height: int,
    width: int,
) -> tuple[int, int] | None:
    if direction == Direction.UP:
        row -= 1
    elif direction == Direction.DOWN:
        row += 1
    elif direction == Direction.LEFT:
        col -= 1
    elif direction == Direction.RIGHT:
        col += 1

    if row < 0 or row >= height or col < 0 or col >= width:
        return None
    return row, col


def _target_is_usable(observation: Observation, row: int, col: int) -> bool:
    owner = int(observation.owner[row, col])
    if owner in (int(CellView.MOUNTAIN), int(CellView.FOG_OBSTACLE)):
        return False
    return True
