"""Temporal sequence data structures for chunk-based BC training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from generals_bot.sim.types import Action, Observation, PassAction
from generals_bot.training.action_codec import action_to_label, valid_action_mask
from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.king import build_enemy_king_targets


@dataclass
class TemporalStep:
    """One tick's worth of data in a temporal sequence."""

    observation: Observation
    action: Action | PassAction | None
    value_target: float
    player_id: int
    turn: int
    chunk_boundary: bool = False
    step_index: int = 0


@dataclass
class TemporalSequence:
    """A contiguous segment of one player's trajectory.

    The first ``burn_in`` steps are used only to warm up temporal memory;
    policy loss is not computed for those steps.
    """

    steps: list[TemporalStep]
    replay_id: str
    player_id: int
    enemy_general_tile: int | None
    mountains: tuple[int, ...]
    replay_version: int | None = None
    burn_in: int = 0

    @property
    def sequence_length(self) -> int:
        """Return the total number of steps in this sequence."""
        return len(self.steps)

    @property
    def trainable_steps(self) -> int:
        """Return the number of steps after burn-in."""
        return max(0, len(self.steps) - self.burn_in)


def build_sequences_from_samples(
    samples: list[dict[str, Any]],
    *,
    sequence_length: int = 64,
    burn_in: int = 16,
    chunk_size: int = 8,
    stride: int | None = None,
) -> list[TemporalSequence]:
    """Convert per-tick BC samples into temporal sequences.

    Samples must be from a single (replay_id, player_id) and ordered by turn.
    """
    if stride is None:
        stride = sequence_length - burn_in
    if stride <= 0:
        raise ValueError("stride must be positive")

    steps = []
    for index, sample in enumerate(samples):
        chunk_boundary = (index + 1) % chunk_size == 0 if chunk_size > 0 else False
        steps.append(
            TemporalStep(
                observation=sample["observation"],
                action=sample.get("action"),
                value_target=float(sample.get("value_target", 0.0)),
                player_id=int(sample.get("player_id", 0)),
                turn=int(sample.get("turn", index)),
                chunk_boundary=chunk_boundary,
                step_index=index,
            )
        )

    sequences = []
    for start in range(0, len(steps) - sequence_length + 1, stride):
        segment = steps[start : start + sequence_length]
        # Copy steps so overlapping windows don't share mutated state
        segment = [
            TemporalStep(
                observation=step.observation,
                action=step.action,
                value_target=step.value_target,
                player_id=step.player_id,
                turn=step.turn,
                chunk_boundary=step.chunk_boundary,
                step_index=i,
            )
            for i, step in enumerate(segment)
        ]
        sequences.append(
            TemporalSequence(
                steps=segment,
                replay_id=str(samples[0].get("replay_id", "")),
                player_id=int(samples[0].get("player_id", 0)),
                enemy_general_tile=samples[0].get("enemy_general_tile"),
                mountains=tuple(samples[0].get("mountains", ())),
                replay_version=samples[0].get("replay_version"),
                # The first window starts at the true episode boundary, so
                # there is no missing history to reconstruct.  Masking its
                # prefix would remove all opening-policy supervision.  Later
                # overlapping windows still need burn-in to rebuild memory.
                burn_in=0 if start == 0 else min(burn_in, len(segment)),
            )
        )

    return sequences


def temporal_collate_fn(
    sequences: list[TemporalSequence],
    *,
    encoder: ObservationEncoder,
    late_pass_turn: int = 30,
    late_pass_weight: float = 0.2,
) -> dict[str, Any]:
    """Collate a batch of TemporalSequence into training tensors.

    Returns a dict with keys:
        x: (B*T, C, max_H, max_W) — flattened frame observations
        policy_label: (B*T,) — action labels
        valid_action_mask: (B*T, num_actions) — per-frame masks
        value_target: (B*T,) — value targets
        sample_weight: (B*T,) — loss weights (0 for burn-in steps)
        burn_in_mask: (B*T,) — True for burn-in steps
        chunk_boundary: (B*T,) — True at chunk boundaries
        enemy_king_label, enemy_king_distance, enemy_king_valid_mask: king targets
        metadata: list of per-step dicts
        seq_lengths: (B,) — length of each sequence before flattening
    """
    if not sequences:
        raise ValueError("cannot collate empty sequence batch")

    all_steps: list[TemporalStep] = []
    seq_lengths: list[int] = []
    burn_in_masks: list[bool] = []
    chunk_boundaries: list[bool] = []
    step_metadata: list[dict[str, Any]] = []

    for seq in sequences:
        seq_lengths.append(len(seq.steps))
        for step in seq.steps:
            all_steps.append(step)
            is_burn_in = step.step_index < seq.burn_in
            burn_in_masks.append(is_burn_in)
            chunk_boundaries.append(step.chunk_boundary)
            step_metadata.append({
                "replay_id": seq.replay_id,
                "player_id": seq.player_id,
                "turn": step.turn,
                "step_index": step.step_index,
                "burn_in": int(is_burn_in),
                "chunk_boundary": int(step.chunk_boundary),
                "replay_version": seq.replay_version,
            })

    # Determine max board size
    max_height = max(step.observation.height for step in all_steps)
    max_width = max(step.observation.width for step in all_steps)
    num_actions = 8 * max_height * max_width + 1

    total = len(all_steps)
    channels = encoder.num_channels
    features = np.zeros((total, channels, max_height, max_width), dtype=np.float32)
    labels = np.zeros(total, dtype=np.int64)
    masks = np.zeros((total, num_actions), dtype=bool)
    value_targets = np.zeros(total, dtype=np.float32)
    turns = np.zeros(total, dtype=np.int64)
    replay_versions = np.full(total, -1, dtype=np.int64)
    sample_weights = np.ones(total, dtype=np.float32)

    observations_for_king = []
    enemy_general_tiles = []
    mountains_list = []

    offset = 0
    for sequence in sequences:
        if sequence.replay_version is not None:
            replay_versions[offset : offset + len(sequence.steps)] = sequence.replay_version
        offset += len(sequence.steps)

    for i, step in enumerate(all_steps):
        # Encode observation
        encoded = encoder.encode(step.observation)
        h, w = step.observation.height, step.observation.width
        features[i, :, :h, :w] = encoded

        # Action label
        if step.action is not None:
            labels[i] = action_to_label(
                step.action,
                label_height=max_height,
                label_width=max_width,
                board_width=step.observation.width,
            )
        else:
            labels[i] = num_actions - 1  # default to pass

        # Valid action mask
        masks[i] = valid_action_mask(
            step.observation,
            label_height=max_height,
            label_width=max_width,
        )

        # Value target
        value_targets[i] = step.value_target
        turns[i] = step.turn

        # Sample weight: 0 for burn-in, late-pass reweighting
        if burn_in_masks[i]:
            sample_weights[i] = 0.0
        elif (
            isinstance(step.action, PassAction)
            and step.turn >= late_pass_turn
            and late_pass_weight < 1.0
        ):
            sample_weights[i] = late_pass_weight

        observations_for_king.append(step.observation)
        enemy_general_tiles.append(
            sequences[0].enemy_general_tile
            if i < len(sequences[0].steps)
            else None
        )

    # Build enemy king targets
    king_targets = build_enemy_king_targets(
        observations_for_king,
        enemy_general_tiles=[
            seq.enemy_general_tile
            for seq in sequences
            for _ in seq.steps
        ],
        mountains=[
            seq.mountains
            for seq in sequences
            for _ in seq.steps
        ],
        max_height=max_height,
        max_width=max_width,
    )

    return {
        "x": torch.from_numpy(np.ascontiguousarray(features)),
        "policy_label": torch.from_numpy(labels).to(torch.long),
        "valid_action_mask": torch.from_numpy(masks),
        "value_target": torch.from_numpy(value_targets),
        "turn": torch.from_numpy(turns),
        "replay_version": torch.from_numpy(replay_versions),
        "sample_weight": torch.from_numpy(sample_weights),
        "burn_in_mask": torch.tensor(burn_in_masks, dtype=torch.bool),
        "chunk_boundary": torch.tensor(chunk_boundaries, dtype=torch.bool),
        "enemy_king_label": torch.from_numpy(king_targets.labels),
        "enemy_king_distance": torch.from_numpy(king_targets.distances),
        "enemy_king_valid_mask": torch.from_numpy(king_targets.valid_mask),
        "metadata": step_metadata,
        "seq_lengths": torch.tensor(seq_lengths, dtype=torch.long),
    }
