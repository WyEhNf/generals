from __future__ import annotations

from typing import Any, Mapping

from .observe import observe, update_all_memory
from .queue import ActionQueue
from .rules import apply_action, apply_growth, execution_order
from .state import GameState
from .types import Action, ExecutedAction, GameConfig, Observation, PlayerScore


class GeneralsEnv:
    def __init__(self, config: GameConfig):
        self.config = config
        self.state: GameState | None = None
        self.last_executed_actions: list[ExecutedAction] = []
        self.last_submitted_actions: dict[int, list[Action]] = {0: [], 1: []}
        self._pending_submitted_actions: dict[int, list[Action]] = {0: [], 1: []}

    def reset(self, seed: int | None = None) -> dict[int, Observation]:
        del seed
        queues = {0: ActionQueue(), 1: ActionQueue()}
        self.state = GameState.from_config(self.config, queues)
        self.last_executed_actions = []
        self.last_submitted_actions = {0: [], 1: []}
        self._pending_submitted_actions = {0: [], 1: []}
        update_all_memory(self.state)
        return self.observations()

    @property
    def done(self) -> bool:
        return self._state.done

    @property
    def alive_players(self) -> list[int]:
        return self._state.alive_players

    @property
    def scores(self) -> list[PlayerScore]:
        return self._state.scores()

    def observe(self, player_id: int) -> Observation:
        return observe(self._state, player_id)

    def observations(self) -> dict[int, Observation]:
        return {player: self.observe(player) for player in (0, 1)}

    def submit(self, player_id: int, action_like) -> list[Action]:
        actions = self._state.queues[player_id].submit(
            action_like,
            current_turn=self._state.turn,
            width=self._state.width,
        )
        self._pending_submitted_actions[player_id].extend(actions)
        return actions

    def tick(self) -> None:
        self._advance({})

    def step(self, actions: Mapping[int, Any]):
        if self.done:
            return self.observations(), self.rewards(), True, self.info()
        self._advance(actions)
        return self.observations(), self.rewards(), self.done, self.info()

    def rewards(self) -> dict[int, float]:
        winner = self._state.winner
        if winner is None:
            return {0: 0.0, 1: 0.0}
        return {player: 1.0 if player == winner else -1.0 for player in (0, 1)}

    def info(self) -> dict[str, Any]:
        return {
            "turn": self._state.turn,
            "winner": self._state.winner,
            "scores": self.scores,
            "executed_actions": list(self.last_executed_actions),
            "submitted_actions": {
                player: list(actions)
                for player, actions in self.last_submitted_actions.items()
            },
        }

    def queue_snapshot(self) -> dict[int, list[Action]]:
        return {
            player: queue.snapshot()
            for player, queue in self._state.queues.items()
        }

    def _advance(self, submissions: Mapping[int, Any]) -> None:
        state = self._state
        self.last_executed_actions = []
        self.last_submitted_actions = {
            player: list(actions)
            for player, actions in self._pending_submitted_actions.items()
        }
        self._pending_submitted_actions = {0: [], 1: []}

        if not state.done:
            for player, action_like in submissions.items():
                if player in state.queues and state.alive[player]:
                    actions = state.queues[player].submit(
                        action_like,
                        current_turn=state.turn,
                        width=state.width,
                    )
                    self.last_submitted_actions[player].extend(actions)

            self._execute_ready_actions()
            if state.done:
                update_all_memory(state)
                state.turn += 1
                return

            apply_growth(state)
            update_all_memory(state)
            state.turn += 1

    def _execute_ready_actions(self) -> None:
        state = self._state
        for player in execution_order(state.turn):
            queue = state.queues[player]
            while queue.has_ready(state.turn):
                queued = queue.pop_ready(state.turn)
                if queued is None:
                    break
                executed = apply_action(state, queued.action)
                self.last_executed_actions.append(executed)
                if executed.valid or state.done:
                    break
            if state.done:
                break

    @property
    def _state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("call reset() before using the environment")
        return self.state
