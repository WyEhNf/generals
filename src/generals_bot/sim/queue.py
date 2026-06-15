from __future__ import annotations

from collections.abc import Iterable
from collections import deque
from dataclasses import dataclass

from .types import Action, DirectionMove, PassAction, direction_move_to_action


@dataclass(frozen=True)
class QueuedAction:
    action: Action
    submitted_turn: int
    available_turn: int


class ActionQueue:
    def __init__(self) -> None:
        self.pending: deque[QueuedAction] = deque()

    def submit(self, item, *, current_turn: int, width: int) -> list[Action]:
        actions = normalize_actions(item, width)
        for action in actions:
            self.pending.append(
                QueuedAction(
                    action=action,
                    submitted_turn=current_turn,
                    available_turn=current_turn,
                )
            )
        return actions

    def has_ready(self, turn: int) -> bool:
        return bool(self.pending and self.pending[0].available_turn <= turn)

    def pop_ready(self, turn: int) -> QueuedAction | None:
        if not self.has_ready(turn):
            return None
        return self.pending.popleft()

    def snapshot(self) -> list[Action]:
        return [queued.action for queued in self.pending]


def normalize_actions(item, width: int) -> list[Action]:
    if item is None:
        return []
    if isinstance(item, PassAction):
        return []
    if isinstance(item, Action):
        return [item]
    if isinstance(item, DirectionMove):
        return [direction_move_to_action(item, width)]
    if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        actions: list[Action] = []
        for value in item:
            actions.extend(normalize_actions(value, width))
        return actions
    raise TypeError(f"unsupported action type: {type(item).__name__}")
