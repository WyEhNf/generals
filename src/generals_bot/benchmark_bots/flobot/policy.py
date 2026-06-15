from __future__ import annotations

from generals_bot.benchmark_bots.flobot.game_map import FlobotMap
from generals_bot.benchmark_bots.flobot.state import FlobotMove, FlobotState
from generals_bot.benchmark_bots.flobot.strategies import COLLECT_TURNS, FlobotStrategy
from generals_bot.sim.types import Action, ActionLike, GameConfig, Observation, PassAction


class FlobotPolicy:
    """Run the Flobot benchmark strategy through the local Policy interface."""

    def __init__(self) -> None:
        """Initialize empty per-game Flobot state."""
        self.player_id: int | None = None
        self.game_state: FlobotState | None = None
        self.game_map: FlobotMap | None = None
        self.queued_moves = 0
        self.last_attacked_index = -1
        self.move_count = 0
        self.is_collecting = False
        self.collect_area: list[int] = []
        self.is_infiltrating = False
        self.collect_turns_left = COLLECT_TURNS
        self._outgoing: list[Action] = []

    def reset(self, player_id: int, config: GameConfig) -> None:
        """Reset Flobot state before a new local match."""
        del config
        self.player_id = player_id
        self.game_state = None
        self.game_map = None
        self.queued_moves = 0
        self.last_attacked_index = -1
        self.move_count = 0
        self.is_collecting = False
        self.collect_area = []
        self.is_infiltrating = False
        self.collect_turns_left = COLLECT_TURNS
        self._outgoing = []

    def act(self, observation: Observation) -> ActionLike:
        """Update Flobot and return all moves emitted during this game update."""
        if self.player_id is None:
            self.player_id = observation.player_id
        if self.game_state is None:
            self.game_state = FlobotState.from_observation(observation)
            self.game_map = FlobotMap(
                observation.width,
                observation.height,
                observation.player_id,
            )
            return PassAction(observation.player_id)

        self.game_state.update(observation)
        self.queued_moves = self.queued_moves - 1 if self.queued_moves > 0 else 0
        self._outgoing = []
        FlobotStrategy.pick_strategy(self)
        if not self._outgoing:
            return PassAction(observation.player_id)
        return list(self._outgoing)

    def queue_moves(self, moves: list[FlobotMove]) -> None:
        """Emit a sequence of Flobot moves in the current update."""
        for move in moves:
            self.move(move)

    def move(self, move: FlobotMove) -> None:
        """Emit one Flobot move if it targets a concrete tile."""
        if self.game_state is None:
            return
        if move.end == -1:
            return
        if move.start < 0 or move.end < 0:
            return
        self.queued_moves += 1
        self.last_attacked_index = move.end
        self.move_count += 1
        self._outgoing.append(
            Action(
                player_id=self.game_state.player_index,
                start=move.start,
                end=move.end,
            )
        )
