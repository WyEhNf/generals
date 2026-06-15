from __future__ import annotations

from collections import defaultdict

from generals_bot.replay.schema import RawReplayMove
from generals_bot.sim.types import Action, ActionLike, GameConfig, Observation, PassAction


class ReplayFollowerPolicy:
    """Policy that replays recorded raw moves for one player."""

    def __init__(self, moves: list[RawReplayMove]) -> None:
        self._moves_by_turn: dict[int, list[RawReplayMove]] = defaultdict(list)
        for move in moves:
            self._moves_by_turn[move.turn].append(move)
        self.player_id: int | None = None

    def reset(self, player_id: int, config: GameConfig) -> None:
        del config
        self.player_id = player_id

    def act(self, observation: Observation) -> ActionLike:
        if self.player_id is None:
            raise RuntimeError("call reset() before act()")
        moves = self._moves_by_turn.get(observation.turn, [])
        actions = [
            Action(move.player_id, move.start, move.end, move.split)
            for move in moves
            if move.player_id == self.player_id
        ]
        return actions if actions else PassAction(self.player_id)
