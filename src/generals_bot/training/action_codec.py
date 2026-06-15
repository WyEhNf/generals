from __future__ import annotations

import numpy as np

from generals_bot.sim.types import Action, DIRECTION_DELTAS, Observation, PassAction

ACTION_PLANES = 8
PASS_PLANE = 8


def action_to_plane(action: Action, *, board_width: int) -> int:
    """Return the policy plane index for an adjacent move action."""
    start_row, start_col = divmod(action.start, board_width)
    end_row, end_col = divmod(action.end, board_width)
    delta = (end_row - start_row, end_col - start_col)
    try:
        direction = DIRECTION_DELTAS.index(delta)
    except ValueError as exc:
        raise ValueError(f"action is not adjacent: {action}") from exc
    return direction + (4 if action.split else 0)


def action_to_label(
    action: Action | PassAction,
    *,
    label_height: int,
    label_width: int,
    board_width: int,
) -> int:
    """Encode an action into a flat policy class on a padded board."""
    board_cells = label_height * label_width
    if isinstance(action, PassAction):
        return ACTION_PLANES * board_cells
    plane = action_to_plane(action, board_width=board_width)
    row, col = divmod(action.start, board_width)
    if row >= label_height or col >= label_width:
        raise ValueError("action start does not fit in padded label board")
    return plane * board_cells + row * label_width + col


def label_to_action(
    label: int,
    *,
    player_id: int,
    height: int,
    width: int,
) -> Action | PassAction:
    """Decode a flat policy class into an environment action."""
    board_cells = height * width
    if label == ACTION_PLANES * board_cells:
        return PassAction(player_id)
    if label < 0 or label >= ACTION_PLANES * board_cells:
        raise ValueError(f"label out of bounds: {label}")
    plane, start = divmod(label, board_cells)
    split = plane >= 4
    direction = plane % 4
    row, col = divmod(start, width)
    dr, dc = DIRECTION_DELTAS[direction]
    end_row = row + dr
    end_col = col + dc
    if end_row < 0 or end_row >= height or end_col < 0 or end_col >= width:
        raise ValueError(f"label decodes to out-of-bounds action: {label}")
    return Action(player_id, start, end_row * width + end_col, split)


def valid_action_mask(
    obs: Observation,
    *,
    label_height: int | None = None,
    label_width: int | None = None,
) -> np.ndarray:
    """Build a flat boolean mask for currently legal policy classes."""
    label_height = obs.height if label_height is None else label_height
    label_width = obs.width if label_width is None else label_width
    if label_height < obs.height or label_width < obs.width:
        raise ValueError("label board cannot be smaller than observation board")

    board_cells = label_height * label_width
    mask = np.zeros(ACTION_PLANES * board_cells + 1, dtype=bool)
    own_start = obs.own_tiles & (obs.armies > 1)
    rows, cols = np.nonzero(own_start)
    for row, col in zip(rows, cols, strict=True):
        for direction, (dr, dc) in enumerate(DIRECTION_DELTAS):
            nr = int(row) + dr
            nc = int(col) + dc
            if nr < 0 or nr >= obs.height or nc < 0 or nc >= obs.width:
                continue
            if obs.mountains[nr, nc] or obs.fog_obstacles[nr, nc]:
                continue
            start = int(row) * label_width + int(col)
            mask[direction * board_cells + start] = True
            mask[(direction + 4) * board_cells + start] = True
    mask[-1] = True
    return mask
