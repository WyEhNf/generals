from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from generals_bot.replay import RawReplay, RawReplayMove, iter_raw_replays
from generals_bot.replay.playback import config_from_raw_replay
from generals_bot.sim import GeneralsEnv
from generals_bot.sim.types import Action, Observation, PassAction
from generals_bot.training.action_codec import action_to_label, valid_action_mask
from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.king import build_enemy_king_targets

try:
    import torch
    from torch.utils.data import IterableDataset
except ModuleNotFoundError:  # pragma: no cover - exercised when train deps are absent.
    torch = None

    class IterableDataset:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class BCSample:
    """One supervised behavior cloning sample for a player-turn."""

    observation: Observation
    action: Action | PassAction
    value_target: float
    replay_id: str
    player_id: int
    turn: int
    observation_history: tuple[Observation, ...] = ()
    enemy_general_tile: int | None = None
    mountains: tuple[int, ...] = ()

    @property
    def history(self) -> tuple[Observation, ...]:
        """Return the observation history ending at this sample's current observation."""
        return self.observation_history or (self.observation,)


@dataclass(frozen=True)
class ReplayFilterEvaluation:
    """Summary of applying one replay filter rule to generated BC samples."""

    rule: str
    accepted: bool
    reason: str
    sample_count: int
    pass_count: int
    pass_ratio: float | None
    late_turn: int | None
    late_sample_count: int
    late_pass_count: int
    late_pass_ratio: float | None


REPLAY_FILTER_RULES = ("none", "whole_pass35", "late35_pass10")


def samples_from_raw_replay(
    replay: RawReplay,
    *,
    include_pass: bool = True,
    history_length: int = 1,
) -> list[BCSample]:
    """Convert a raw replay into player-turn behavior cloning samples."""
    if history_length < 1:
        raise ValueError("history_length must be positive")
    move_by_player_turn = _moves_by_player_turn(replay.moves)
    config = config_from_raw_replay(replay)
    env = GeneralsEnv(config)
    observations = env.reset()
    pending_samples: list[tuple[tuple[Observation, ...], Action | PassAction]] = []
    history_by_player: dict[int, list[Observation]] = defaultdict(list)

    while not env.done:
        current_turn = observations[0].turn
        submissions: dict[int, Action | PassAction] = {}
        for player in env.alive_players:
            history = history_by_player[player]
            history.append(observations[player])
            if len(history) > history_length:
                del history[:-history_length]
            move = move_by_player_turn.get((player, current_turn))
            action: Action | PassAction
            if move is None:
                action = PassAction(player)
            else:
                action = Action(move.player_id, move.start, move.end, move.split)
            submissions[player] = action
            if include_pass or isinstance(action, Action):
                pending_samples.append((tuple(history), action))
        observations, _, _, _ = env.step(submissions)

    winner = env.state.winner if env.state is not None else None
    samples: list[BCSample] = []
    for history, action in pending_samples:
        observation = history[-1]
        value_target = 0.0
        if winner is not None:
            value_target = 1.0 if observation.player_id == winner else -1.0
        samples.append(
            BCSample(
                observation=observation,
                action=action,
                value_target=value_target,
                replay_id=replay.replay_id,
                player_id=observation.player_id,
                turn=observation.turn,
                observation_history=history,
                enemy_general_tile=config.generals[1 - observation.player_id],
                mountains=tuple(config.mountains),
            )
        )
    return samples


class RawReplayBCDataset(IterableDataset):
    """Iterable dataset that streams BC samples from raw replay JSONL data."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str = "train",
        limit_replays: int | None = None,
        include_pass: bool = True,
        shuffle_buffer_size: int = 0,
        shuffle_shards: bool = False,
        seed: int = 0,
        replay_filter_rule: str = "none",
        replay_versions: tuple[int, ...] | None = None,
        history_length: int = 1,
    ) -> None:
        """Store dataset filtering and sampling configuration."""
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be nonnegative")
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if replay_filter_rule not in REPLAY_FILTER_RULES:
            raise ValueError(f"unknown replay_filter_rule: {replay_filter_rule}")
        if replay_versions is not None and not replay_versions:
            raise ValueError("replay_versions cannot be empty")
        self.path = Path(path)
        self.split = split
        self.limit_replays = limit_replays
        self.include_pass = include_pass
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.replay_filter_rule = replay_filter_rule
        self.replay_versions = replay_versions
        self.history_length = history_length

    def __iter__(self) -> Iterable[BCSample]:
        """Yield BC samples from replays assigned to the configured split."""
        samples = self._iter_ordered_samples()
        if self.shuffle_buffer_size <= 0:
            yield from samples
            return
        yield from _shuffle_stream(samples, buffer_size=self.shuffle_buffer_size, seed=self.seed)

    def _iter_ordered_samples(self) -> Iterable[BCSample]:
        """Yield BC samples in raw replay file order."""
        yielded_replays = 0
        shard_seed = self.seed if self.shuffle_shards else None
        for replay in iter_raw_replays(self.path, shard_seed=shard_seed):
            if not replay_in_split(replay.replay_id, self.split):
                continue
            if self.replay_versions is not None and replay.version not in self.replay_versions:
                continue
            if self.limit_replays is not None and yielded_replays >= self.limit_replays:
                break
            samples = samples_from_raw_replay(
                replay,
                include_pass=True,
                history_length=self.history_length,
            )
            filter_result = evaluate_replay_filter_rule(samples, self.replay_filter_rule)
            if not filter_result.accepted:
                continue
            yielded_replays += 1
            if self.include_pass:
                yield from samples
            else:
                yield from (
                    sample
                    for sample in samples
                    if isinstance(sample.action, Action)
                )


class MixedReplayBCDataset(IterableDataset):
    """Iterable dataset that samples from multiple replay streams by weight."""

    def __init__(
        self,
        datasets: list[RawReplayBCDataset],
        *,
        weights: list[float],
        seed: int = 0,
        stop_when_any_exhausted: bool = True,
    ) -> None:
        """Store weighted replay streams for mixed training."""
        if not datasets:
            raise ValueError("mixed replay dataset requires at least one source")
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have matching lengths")
        if any(weight < 0 for weight in weights):
            raise ValueError("mixed replay weights must be nonnegative")
        if sum(weights) <= 0:
            raise ValueError("mixed replay weights must have positive total")
        self.datasets = tuple(datasets)
        self.weights = tuple(float(weight) for weight in weights)
        self.seed = seed
        self.stop_when_any_exhausted = stop_when_any_exhausted

    def __iter__(self) -> Iterable[BCSample]:
        """Yield samples by weighted random source selection."""
        rng = random.Random(self.seed)
        iterators = [iter(dataset) for dataset in self.datasets]
        active = [weight > 0 for weight in self.weights]

        while any(active):
            source_index = _weighted_choice(
                rng,
                [
                    self.weights[index] if active[index] else 0.0
                    for index in range(len(self.weights))
                ],
            )
            try:
                yield next(iterators[source_index])
            except StopIteration:
                if self.stop_when_any_exhausted:
                    break
                active[source_index] = False


def replay_in_split(replay_id: str, split: str) -> bool:
    """Return whether a replay id belongs to a stable train/val/test split."""
    bucket = _stable_bucket(replay_id)
    if split == "train":
        return bucket < 9000
    if split == "val":
        return 9000 <= bucket < 9500
    if split == "test":
        return bucket >= 9500
    if split == "all":
        return True
    raise ValueError(f"unknown split: {split}")


def collate_bc_samples(
    samples: list[BCSample],
    *,
    encoder: ObservationEncoder | None = None,
    late_pass_turn: int = 30,
    late_pass_weight: float = 0.2,
) -> dict[str, Any]:
    """Pad variable board samples and convert them into torch tensors."""
    if torch is None:
        raise RuntimeError("torch is required for collate_bc_samples")
    if not samples:
        raise ValueError("cannot collate an empty batch")
    if late_pass_weight < 0:
        raise ValueError("late_pass_weight must be nonnegative")
    encoder = encoder or ObservationEncoder()
    max_height = max(sample.observation.height for sample in samples)
    max_width = max(sample.observation.width for sample in samples)
    channels = encoder.num_channels

    features = np.zeros((len(samples), channels, max_height, max_width), dtype=np.float32)
    labels = np.zeros((len(samples),), dtype=np.int64)
    valid_masks = np.zeros((len(samples), 8 * max_height * max_width + 1), dtype=bool)
    value_targets = np.zeros((len(samples),), dtype=np.float32)
    sample_weights = np.ones((len(samples),), dtype=np.float32)
    metadata: list[dict[str, Any]] = []
    king_targets = build_enemy_king_targets(
        [sample.observation for sample in samples],
        enemy_general_tiles=[sample.enemy_general_tile for sample in samples],
        mountains=[sample.mountains for sample in samples],
        max_height=max_height,
        max_width=max_width,
    )

    for index, sample in enumerate(samples):
        obs = sample.observation
        encoded = encoder.encode_history(sample.history)
        features[index, :, : obs.height, : obs.width] = encoded
        labels[index] = action_to_label(
            sample.action,
            label_height=max_height,
            label_width=max_width,
            board_width=obs.width,
        )
        valid_masks[index] = valid_action_mask(
            obs,
            label_height=max_height,
            label_width=max_width,
        )
        value_targets[index] = sample.value_target
        if isinstance(sample.action, PassAction) and sample.turn >= late_pass_turn:
            sample_weights[index] = late_pass_weight
        metadata.append({
            "replay_id": sample.replay_id,
            "player_id": sample.player_id,
            "turn": sample.turn,
        })

    return {
        "x": torch.from_numpy(features),
        "policy_label": torch.from_numpy(labels),
        "valid_action_mask": torch.from_numpy(valid_masks),
        "value_target": torch.from_numpy(value_targets),
        "sample_weight": torch.from_numpy(sample_weights),
        "enemy_king_label": torch.from_numpy(king_targets.labels),
        "enemy_king_distance": torch.from_numpy(king_targets.distances),
        "enemy_king_valid_mask": torch.from_numpy(king_targets.valid_mask),
        "metadata": metadata,
    }


def evaluate_replay_filter_rule(
    samples: list[BCSample],
    rule: str,
) -> ReplayFilterEvaluation:
    """Apply a named replay filter rule to generated replay samples."""
    if rule not in REPLAY_FILTER_RULES:
        raise ValueError(f"unknown replay filter rule: {rule}")
    sample_count = len(samples)
    pass_count = sum(1 for sample in samples if isinstance(sample.action, PassAction))
    pass_ratio = pass_count / sample_count if sample_count else None
    late_turn = 35
    late_samples = [sample for sample in samples if sample.turn >= late_turn]
    late_sample_count = len(late_samples)
    late_pass_count = sum(1 for sample in late_samples if isinstance(sample.action, PassAction))
    late_pass_ratio = (
        late_pass_count / late_sample_count
        if late_sample_count
        else None
    )

    accepted = True
    reason = "accepted"
    if rule == "whole_pass35":
        if pass_ratio is None:
            accepted = False
            reason = "no_samples"
        elif pass_ratio > 0.35:
            accepted = False
            reason = "whole_pass_ratio_gt_0.35"
    elif rule == "late35_pass10":
        if late_pass_ratio is None:
            accepted = False
            reason = "no_late_samples"
        elif late_pass_ratio > 0.10:
            accepted = False
            reason = "late35_pass_ratio_gt_0.10"

    return ReplayFilterEvaluation(
        rule=rule,
        accepted=accepted,
        reason=reason,
        sample_count=sample_count,
        pass_count=pass_count,
        pass_ratio=pass_ratio,
        late_turn=late_turn,
        late_sample_count=late_sample_count,
        late_pass_count=late_pass_count,
        late_pass_ratio=late_pass_ratio,
    )


def _moves_by_player_turn(moves: list[RawReplayMove]) -> dict[tuple[int, int], RawReplayMove]:
    """Index replay moves by player and turn, keeping the first move."""
    grouped: dict[tuple[int, int], list[RawReplayMove]] = defaultdict(list)
    for move in moves:
        grouped[(move.player_id, move.turn)].append(move)
    return {key: value[0] for key, value in grouped.items()}


def _shuffle_stream(
    samples: Iterable[BCSample],
    *,
    buffer_size: int,
    seed: int,
) -> Iterable[BCSample]:
    """Shuffle a sample stream using a fixed-size random buffer."""
    rng = random.Random(seed)
    buffer: list[BCSample] = []
    for sample in samples:
        if len(buffer) < buffer_size:
            buffer.append(sample)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = sample

    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer.pop(index)


def _weighted_choice(rng: random.Random, weights: list[float]) -> int:
    """Return one index sampled proportionally to nonnegative weights."""
    total = sum(weights)
    if total <= 0:
        raise ValueError("cannot sample from empty weights")
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold < cumulative:
            return index
    return len(weights) - 1


def _stable_bucket(value: str) -> int:
    """Map a string to a deterministic bucket in [0, 10000)."""
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10000
