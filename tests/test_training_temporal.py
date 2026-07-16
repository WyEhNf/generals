from __future__ import annotations

from dataclasses import replace
from collections import deque
import unittest

import torch

from generals_bot.sim import GameConfig, GeneralsEnv
from generals_bot.sim.types import Action, PassAction
from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.model import BCPolicyNet
from generals_bot.training.sequence import (
    TemporalSequence,
    TemporalStep,
    build_sequences_from_samples,
    temporal_collate_fn,
)
from generals_bot.training.temporal import (
    ChunkMemoryGRU,
    ChunkMemoryTransformer,
    EventWeightedSpatialAccumulator,
    SpatialChunkMemoryTransformer,
    TemporalModelConfig,
    TemporalPolicyNet,
    WithinChunkAccumulator,
)
from generals_bot.training.trainer import (
    TemporalTrainConfig,
    _apply_temporal_lr_schedule,
    _load_temporal_spatial_init,
    _move_batch,
    _bounded_shuffle,
    _bucket_temporal_sequences,
    _temporal_validation_health_error,
    _temporal_batch_loss,
    _temporal_sequence_forward,
    _validate_temporal_model_training_alignment,
)


def _observations(count: int, *, player_id: int = 0):
    env = GeneralsEnv(GameConfig(width=3, height=3, generals=(0, 8)))
    observation = env.reset()[player_id]
    return [replace(observation, turn=turn) for turn in range(count)]


def _samples(count: int, *, replay_id: str = "temporal", player_id: int = 0):
    observations = _observations(count, player_id=player_id)
    return [
        {
            "observation": observation,
            "action": PassAction(player_id),
            "value_target": float(index % 2),
            "replay_id": replay_id,
            "player_id": player_id,
            "turn": observation.turn,
            "enemy_general_tile": 8 if player_id == 0 else 0,
            "mountains": (),
        }
        for index, observation in enumerate(observations)
    ]


def _temporal_model(*, base_channels: int = 8, chunk_size: int = 2):
    encoder = ObservationEncoder()
    config = TemporalModelConfig(
        base_channels=base_channels,
        temporal_dim=base_channels * 3,
        chunk_token_dim=base_channels * 3,
        chunk_transformer_layers=1,
        chunk_transformer_heads=2,
        chunk_transformer_ffn_dim=64,
        max_chunk_memory=4,
        dropout=0.0,
        chunk_size=chunk_size,
    )
    spatial = BCPolicyNet(
        encoder.num_channels,
        base_channels=base_channels,
        temporal_dim=config.temporal_dim,
    )
    model = TemporalPolicyNet(encoder.num_channels, spatial, config)
    model.eval()
    return model, encoder


class TemporalSequenceTest(unittest.TestCase):
    def test_temporal_batched_runner_pads_finished_stream_rows(self) -> None:
        from generals_bot.training.batched_policy import BatchedBCPolicyRunner

        observations = _observations(1)
        observation = observations[0]
        runner = object.__new__(BatchedBCPolicyRunner)
        runner._is_temporal = True
        runner._temporal_active_keys = None
        runner.histories = {}
        seen_groups = []

        def fake_act_group(group):
            seen_groups.append([key for key, _ in group])
            return {key: PassAction(0) for key, _ in group}

        runner._act_group = fake_act_group
        first = runner.act_many([("a", observation), ("b", observation)])
        self.assertEqual(set(first), {"a", "b"})
        runner.histories = {
            "a": deque([observation]),
            "b": deque([observation]),
        }

        second = runner.act_many([("a", observation)])

        self.assertEqual(set(second), {"a"})
        self.assertEqual(set(seen_groups[-1]), {"a", "b"})

    def test_move_batch_moves_metric_grouping_tensors(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = {
            "x": torch.zeros(2, 1),
            "turn": torch.tensor([1, 2]),
            "replay_version": torch.tensor([5, 13]),
            "replay_id": ["a", "b"],
        }

        moved = _move_batch(batch, device)

        self.assertEqual(moved["x"].device.type, device.type)
        self.assertEqual(moved["turn"].device.type, device.type)
        self.assertEqual(moved["replay_version"].device.type, device.type)
        self.assertEqual(moved["replay_id"], ["a", "b"])

    """Tests for contiguous sequence construction, burn-in, and collation."""

    def test_rejects_non_positive_stride(self) -> None:
        """A sequence window must advance by at least one tick."""
        with self.assertRaisesRegex(ValueError, "stride must be positive"):
            build_sequences_from_samples(_samples(4), sequence_length=4, stride=0)

    def test_bounded_shuffle_is_deterministic_complete_and_lazy(self) -> None:
        """Shuffle buffering must preserve samples without consuming the full stream."""
        consumed: list[int] = []

        def source():
            for value in range(20):
                consumed.append(value)
                yield value

        shuffled = _bounded_shuffle(source(), buffer_size=4, seed=17)
        first = next(shuffled)
        self.assertIn(first, range(4))
        self.assertEqual(consumed, [0, 1, 2, 3])
        output = [first, *list(shuffled)]
        repeated = list(_bounded_shuffle(range(20), buffer_size=4, seed=17))

        self.assertEqual(output, repeated)
        self.assertCountEqual(output, range(20))
        self.assertNotEqual(output, list(range(20)))

    def test_bounded_shuffle_zero_preserves_source_order(self) -> None:
        """Validation streams use buffer zero and therefore stable source order."""
        self.assertEqual(
            list(_bounded_shuffle(range(6), buffer_size=0, seed=99)),
            list(range(6)),
        )

    def test_temporal_bucketing_keeps_batch_board_sizes_equal(self) -> None:
        """Each emitted DataLoader-sized group should require no board padding."""
        small = build_sequences_from_samples(
            _samples(4, replay_id="small-a"), sequence_length=4, burn_in=0,
        )[0]
        small_b = build_sequences_from_samples(
            _samples(4, replay_id="small-b"), sequence_length=4, burn_in=0,
        )[0]
        large_samples = _samples(4, replay_id="large")
        large_observations = _observations(4)
        for sample, observation in zip(large_samples, large_observations, strict=True):
            sample["observation"] = replace(observation, width=4, height=4)
        # Only the size key is inspected, so a lightweight replacement is enough.
        large = build_sequences_from_samples(
            large_samples, sequence_length=4, burn_in=0,
        )[0]

        output = list(_bucket_temporal_sequences([small, large, small_b], group_size=2))

        self.assertEqual(output, [small, small_b])

    def test_training_alignment_rejects_untrained_memory_positions(self) -> None:
        """Inference memory may not be longer than the supervised sequence horizon."""
        train_config = TemporalTrainConfig(
            input_path="unused",
            output_path="unused",
            sequence_length=64,
            burn_in=16,
            chunk_size=8,
        )
        model_config = TemporalModelConfig(
            max_chunk_memory=32,
            chunk_size=8,
        )

        with self.assertRaisesRegex(ValueError, "must cover"):
            _validate_temporal_model_training_alignment(train_config, model_config)

    def test_fair_k8_alignment_accepts_sixty_four_tick_horizon(self) -> None:
        """K=8 with eight tokens covers the same 64-tick horizon used in training."""
        train_config = TemporalTrainConfig(
            input_path="unused",
            output_path="unused",
            sequence_length=64,
            burn_in=16,
            chunk_size=8,
        )
        model_config = TemporalModelConfig(
            max_chunk_memory=8,
            chunk_size=8,
        )

        _validate_temporal_model_training_alignment(train_config, model_config)

    def test_temporal_lr_schedule_warms_up_then_decays(self) -> None:
        """Grouped rates should share one warmup/cosine scale."""
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([
            {"params": [parameter], "lr": 1e-5},
            {"params": [], "lr": 1e-4},
            {"params": [], "lr": 5e-5},
        ])
        config = TemporalTrainConfig(
            input_path="unused",
            output_path="unused",
            steps=100,
            warmup_steps=10,
            min_lr_ratio=0.1,
        )

        _apply_temporal_lr_schedule(optimizer, config, step=5)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-6)
        _apply_temporal_lr_schedule(optimizer, config, step=100)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)

    def test_validation_health_detects_pass_gate_collapse(self) -> None:
        """A checkpoint should be rejected before all-state Pass reaches deployment."""
        config = TemporalTrainConfig(
            input_path="unused",
            output_path="unused",
            max_val_predicted_pass_rate=0.6,
            max_val_move_false_pass_rate=0.5,
        )

        reason = _temporal_validation_health_error(
            config,
            {"predicted_pass_rate": 0.9, "move_false_pass_rate": 0.8},
        )

        self.assertIn("predicted_pass_rate", reason)

    def test_fixed_chunk_boundaries_follow_global_tick_order(self) -> None:
        """Chunk boundaries must remain aligned across overlapping windows."""
        sequences = build_sequences_from_samples(
            _samples(8),
            sequence_length=4,
            burn_in=1,
            chunk_size=4,
            stride=2,
        )

        observed = [
            [step.chunk_boundary for step in sequence.steps]
            for sequence in sequences
        ]

        self.assertEqual(
            observed,
            [
                [False, False, False, True],
                [False, True, False, False],
                [False, False, False, True],
            ],
        )

    def test_overlapping_windows_keep_independent_local_step_indices(self) -> None:
        """Building a later window must not mutate an earlier window's indices."""
        sequences = build_sequences_from_samples(
            _samples(6),
            sequence_length=4,
            burn_in=1,
            chunk_size=4,
            stride=2,
        )

        self.assertEqual([step.step_index for step in sequences[0].steps], [0, 1, 2, 3])
        self.assertEqual([step.step_index for step in sequences[1].steps], [0, 1, 2, 3])

    def test_collate_masks_exactly_each_sequences_burn_in_prefix(self) -> None:
        """Burn-in reconstructs memory but contributes zero policy loss."""
        sequences = build_sequences_from_samples(
            _samples(6),
            sequence_length=4,
            burn_in=2,
            chunk_size=2,
            stride=2,
        )

        batch = temporal_collate_fn(sequences, encoder=ObservationEncoder())

        self.assertEqual(
            batch["burn_in_mask"].tolist(),
            [False, False, False, False, True, True, False, False],
        )
        self.assertEqual(
            batch["sample_weight"].tolist(),
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        )

    def test_first_window_keeps_opening_supervision(self) -> None:
        """A true episode prefix must not be mistaken for truncated history."""
        sequences = build_sequences_from_samples(
            _samples(8),
            sequence_length=4,
            burn_in=2,
            chunk_size=2,
            stride=2,
        )

        self.assertEqual([sequence.burn_in for sequence in sequences], [0, 2, 2])

    def test_collate_keeps_sequence_specific_enemy_general_targets(self) -> None:
        """Auxiliary labels must come from the sequence owning each tick."""
        first = build_sequences_from_samples(
            _samples(4, replay_id="p0", player_id=0),
            sequence_length=4,
            burn_in=0,
            chunk_size=2,
        )[0]
        second = build_sequences_from_samples(
            _samples(4, replay_id="p1", player_id=1),
            sequence_length=4,
            burn_in=0,
            chunk_size=2,
        )[0]

        batch = temporal_collate_fn([first, second], encoder=ObservationEncoder())

        self.assertEqual(batch["enemy_king_label"].tolist(), [8] * 4 + [0] * 4)

    def test_collate_keeps_replay_versions_for_grouped_metrics(self) -> None:
        """Validation must be able to expose v5/v13 gate behavior separately."""
        first_samples = _samples(4, replay_id="v5")
        second_samples = _samples(4, replay_id="v13")
        for sample in first_samples:
            sample["replay_version"] = 5
        for sample in second_samples:
            sample["replay_version"] = 13
        first = build_sequences_from_samples(
            first_samples, sequence_length=4, burn_in=0,
        )[0]
        second = build_sequences_from_samples(
            second_samples, sequence_length=4, burn_in=0,
        )[0]

        batch = temporal_collate_fn([first, second], encoder=ObservationEncoder())

        self.assertEqual(batch["replay_version"].tolist(), [5] * 4 + [13] * 4)

    def test_collate_uses_each_boards_width_for_mixed_size_action_labels(self) -> None:
        """A vertical move on a small board stays valid in a larger padded batch."""
        small_env = GeneralsEnv(
            GameConfig(width=3, height=3, generals=(0, 8), starting_armies={0: 3})
        )
        large_env = GeneralsEnv(GameConfig(width=4, height=4, generals=(0, 15)))
        small = TemporalSequence(
            steps=[TemporalStep(
                observation=small_env.reset()[0],
                action=Action(0, 0, 3, False),
                value_target=0.0,
                player_id=0,
                turn=0,
            )],
            replay_id="small",
            player_id=0,
            enemy_general_tile=8,
            mountains=(),
        )
        large = TemporalSequence(
            steps=[TemporalStep(
                observation=large_env.reset()[0],
                action=PassAction(0),
                value_target=0.0,
                player_id=0,
                turn=0,
            )],
            replay_id="large",
            player_id=0,
            enemy_general_tile=15,
            mountains=(),
        )

        batch = temporal_collate_fn([small, large], encoder=ObservationEncoder())

        self.assertNotEqual(int(batch["policy_label"][0]), 8 * 4 * 4)
        self.assertTrue(bool(batch["valid_action_mask"][0, batch["policy_label"][0]]))

    def test_collate_downweights_late_pass_after_burn_in(self) -> None:
        """Late-pass weighting applies only to trainable ticks."""
        samples = _samples(6)
        samples[0]["observation"] = replace(samples[0]["observation"], turn=30)
        samples[0]["turn"] = 30
        samples[1]["observation"] = replace(samples[1]["observation"], turn=31)
        samples[1]["turn"] = 31
        samples[2]["observation"] = replace(samples[2]["observation"], turn=32)
        samples[2]["turn"] = 32
        for index in range(3, 6):
            samples[index]["observation"] = replace(
                samples[index]["observation"], turn=30 + index,
            )
            samples[index]["turn"] = 30 + index
        samples[4]["action"] = Action(0, 0, 1, False)
        sequences = build_sequences_from_samples(
            samples,
            sequence_length=4,
            burn_in=1,
            chunk_size=2,
            stride=2,
        )

        batch = temporal_collate_fn(
            [sequences[1]],
            encoder=ObservationEncoder(),
            late_pass_turn=30,
            late_pass_weight=0.2,
        )

        self.assertTrue(torch.allclose(
            batch["sample_weight"],
            torch.tensor([0.0, 0.2, 1.0, 0.2]),
        ))

    def test_training_forward_advances_sequences_tick_by_tick(self) -> None:
        """A flattened B*T batch must become T calls over B persistent streams."""
        class RecordingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

            def forward(self, x, *, chunk_boundary, reset_memory):
                self.calls.append((x.detach().clone(), chunk_boundary.clone(), reset_memory))
                marker = x[:, 0, 0, 0]
                return {"value": marker}

        model = RecordingModel()
        x = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1)
        batch = {
            "x": x,
            "seq_lengths": torch.tensor([3, 3]),
            "chunk_boundary": torch.tensor([False, True, False, True, False, True]),
        }

        output = _temporal_sequence_forward(model, batch)

        self.assertEqual(len(model.calls), 3)
        self.assertEqual([call[2] for call in model.calls], [True, False, False])
        self.assertEqual(model.calls[0][0].flatten().tolist(), [0.0, 3.0])
        self.assertEqual(model.calls[1][0].flatten().tolist(), [1.0, 4.0])
        self.assertEqual(model.calls[1][1].tolist(), [True, False])
        self.assertEqual(output["value"].tolist(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


class WithinChunkAccumulatorTest(unittest.TestCase):
    """Tests for deterministic accumulation and state lifecycle."""

    def test_zero_gate_parameters_produce_known_recurrence(self) -> None:
        """With gate=0.5, two unit frames should accumulate to 0.75."""
        accumulator = WithinChunkAccumulator(channels=2)
        with torch.no_grad():
            accumulator.gate_conv.weight.zero_()
            accumulator.gate_conv.bias.zero_()
        frame = torch.ones(1, 2, 3, 3)

        first = accumulator(frame)
        second = accumulator(frame)
        token = accumulator.finalize()

        self.assertTrue(torch.allclose(first, torch.full_like(first, 0.5)))
        self.assertTrue(torch.allclose(second, torch.full_like(second, 0.75)))
        self.assertTrue(torch.allclose(token, torch.full_like(token, 0.75)))
        self.assertIsNone(accumulator._state)

    def test_finalize_before_input_is_rejected(self) -> None:
        """An empty chunk cannot produce a meaningful token."""
        accumulator = WithinChunkAccumulator(channels=2)

        with self.assertRaisesRegex(RuntimeError, "before any forward"):
            accumulator.finalize()

    def test_reset_discards_previous_chunk(self) -> None:
        """Episode reset must prevent an old frame from changing a new chunk."""
        accumulator = WithinChunkAccumulator(channels=1)
        with torch.no_grad():
            accumulator.gate_conv.weight.zero_()
            accumulator.gate_conv.bias.zero_()
        accumulator(torch.full((1, 1, 2, 2), 10.0))

        accumulator.reset()
        state = accumulator(torch.ones(1, 1, 2, 2))

        self.assertTrue(torch.allclose(state, torch.full_like(state, 0.5)))

    def test_accumulator_parameters_receive_gradients(self) -> None:
        """The chunk compressor must be connected to the training graph."""
        accumulator = WithinChunkAccumulator(channels=2)
        token = accumulator(torch.randn(2, 2, 3, 3))
        token.mean().backward()

        self.assertIsNotNone(accumulator.gate_conv.weight.grad)
        self.assertGreater(float(accumulator.gate_conv.weight.grad.abs().sum()), 0.0)


class EventWeightedSpatialEncoderTest(unittest.TestCase):
    """Tests for the spatially preserving EWSC path."""

    def test_equal_global_means_at_different_locations_remain_distinguishable(self) -> None:
        """EWSC must not inherit global average pooling's permutation invariance."""
        left = EventWeightedSpatialAccumulator(channels=1, token_dim=1, grid_size=2)
        right = EventWeightedSpatialAccumulator(channels=1, token_dim=1, grid_size=2)
        right.load_state_dict(left.state_dict())
        with torch.no_grad():
            left.project.weight.zero_()
            left.project.bias.zero_()
            left.project.weight[0, 0, 0, 0] = 1.0
            right.load_state_dict(left.state_dict())
        a = torch.zeros(1, 1, 2, 2)
        b = torch.zeros(1, 1, 2, 2)
        a[0, 0, 0, 0] = 1.0
        b[0, 0, 1, 1] = 1.0

        left(a)
        right(b)
        token_a = left.finalize()
        token_b = right.finalize()

        self.assertEqual(float(token_a.mean().detach()), float(token_b.mean().detach()))
        self.assertFalse(torch.equal(token_a, token_b))

    def test_event_weighting_favors_a_large_change(self) -> None:
        """A large feature transition should contribute more than a quiet frame."""
        accumulator = EventWeightedSpatialAccumulator(channels=1, token_dim=1, grid_size=1)
        with torch.no_grad():
            accumulator.project.weight.zero_()
            accumulator.project.bias.zero_()
            accumulator.project.weight[0, 0, 0, 0] = 1.0
        accumulator(torch.zeros(1, 1, 1, 1))
        accumulator(torch.ones(1, 1, 1, 1))
        accumulator(torch.ones(1, 1, 1, 1))

        token = accumulator.finalize()

        self.assertGreater(float(token.item()), 2.0 / 3.0)

    def test_explicit_visibility_event_changes_temporal_weighting(self) -> None:
        """An explicit visibility transition should affect chunk compression."""
        feature_only = EventWeightedSpatialAccumulator(
            channels=1, token_dim=1, grid_size=1,
        )
        explicit = EventWeightedSpatialAccumulator(
            channels=1,
            token_dim=1,
            grid_size=1,
            explicit_event_context=True,
        )
        with torch.no_grad():
            for accumulator in (feature_only, explicit):
                accumulator.project.weight.zero_()
                accumulator.project.bias.zero_()
                accumulator.project.weight[0, 0, 0, 0] = 1.0
        context = torch.zeros(1, 5, 1, 1)
        for value in (0.0, 1.0):
            feature_only(torch.full((1, 1, 1, 1), value))
            explicit(
                torch.full((1, 1, 1, 1), value),
                event_context=context,
            )
        changed_context = context.clone()
        changed_context[:, 0] = 1.0
        feature_only(torch.zeros(1, 1, 1, 1))
        explicit(torch.zeros(1, 1, 1, 1), event_context=changed_context)

        feature_token = feature_only.finalize()
        explicit_token = explicit.finalize()

        self.assertLess(float(explicit_token.item()), float(feature_token.item()))

    def test_explicit_event_weights_receive_gradients(self) -> None:
        accumulator = EventWeightedSpatialAccumulator(
            channels=1,
            token_dim=1,
            grid_size=1,
            explicit_event_context=True,
        )
        first_context = torch.zeros(1, 5, 1, 1)
        second_context = first_context.clone()
        second_context[:, 1] = 1.0
        accumulator(torch.zeros(1, 1, 1, 1), event_context=first_context)
        accumulator(torch.ones(1, 1, 1, 1), event_context=second_context)

        accumulator.finalize().sum().backward()

        self.assertIsNotNone(accumulator.event_weight_logits.grad)
        self.assertGreater(
            float(accumulator.event_weight_logits.grad.abs().sum()), 0.0,
        )

    def test_spatial_memory_isolates_asynchronous_batch_rows(self) -> None:
        """Updating one stream may not advance another stream's chunk history."""
        config = TemporalModelConfig(
            base_channels=2,
            temporal_dim=8,
            chunk_token_dim=8,
            chunk_transformer_layers=1,
            chunk_transformer_heads=2,
            chunk_transformer_ffn_dim=16,
            max_chunk_memory=4,
            encoder_type="ewsc",
            spatial_grid_size=2,
            spatial_transformer_layers=1,
        )
        memory = SpatialChunkMemoryTransformer(config).eval()
        memory(torch.randn(1, 8, 2, 2), row_indices=torch.tensor([0]), batch_size=2)

        self.assertEqual(memory._memory_by_row[0].shape[1], 1)
        self.assertIsNone(memory._memory_by_row[1])

        memory(torch.randn(1, 8, 2, 2), row_indices=torch.tensor([1]), batch_size=2)

        self.assertEqual(memory._memory_by_row[0].shape[1], 1)
        self.assertEqual(memory._memory_by_row[1].shape[1], 1)

    def test_ewsc_belief_persists_after_boundary(self) -> None:
        """The latest spatial belief should be used on non-boundary ticks."""
        class RecordingSpatial(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.beliefs = []

            def encode_frame(self, x):
                bottleneck = x[:, :3]
                return x[:, :1], x[:, :2], bottleneck

            def decode_policy(self, s1, s2, bottleneck, *, belief=None):
                del s1, s2
                self.beliefs.append(None if belief is None else belief.detach().clone())
                return {"value": bottleneck.mean((1, 2, 3))}

        config = TemporalModelConfig(
            base_channels=1,
            temporal_dim=8,
            chunk_token_dim=8,
            chunk_transformer_layers=1,
            chunk_transformer_heads=2,
            chunk_transformer_ffn_dim=16,
            max_chunk_memory=4,
            chunk_size=2,
            encoder_type="ewsc",
            spatial_grid_size=2,
            spatial_transformer_layers=1,
        )
        spatial = RecordingSpatial()
        model = TemporalPolicyNet(3, spatial, config).eval()
        frame = torch.randn(1, 3, 4, 4)

        model(frame, chunk_boundary=torch.tensor([False]), reset_memory=True)
        model(frame, chunk_boundary=torch.tensor([True]))
        boundary_belief = spatial.beliefs[-1]
        model(frame, chunk_boundary=torch.tensor([False]))

        self.assertIsNone(spatial.beliefs[0])
        self.assertIsNotNone(boundary_belief)
        self.assertTrue(torch.equal(boundary_belief, spatial.beliefs[-1]))


class ChunkMemoryTransformerTest(unittest.TestCase):
    """Tests for bounded causal chunk memory."""

    def _memory(self, *, max_chunks: int = 3) -> ChunkMemoryTransformer:
        config = TemporalModelConfig(
            temporal_dim=8,
            chunk_token_dim=8,
            chunk_transformer_layers=1,
            chunk_transformer_heads=2,
            chunk_transformer_ffn_dim=16,
            max_chunk_memory=max_chunks,
            dropout=0.0,
        )
        memory = ChunkMemoryTransformer(config)
        memory.eval()
        return memory

    def test_memory_is_capped_and_resettable(self) -> None:
        """Only the configured number of most recent chunks may remain."""
        memory = self._memory(max_chunks=3)
        for index in range(5):
            memory(torch.full((1, 8), float(index)))

        self.assertEqual(memory.num_chunks, 3)
        memory.reset()
        self.assertEqual(memory.num_chunks, 0)

    def test_vector_memory_isolates_asynchronous_batch_rows(self) -> None:
        """A boundary for one row must not reset another row's token history."""
        memory = self._memory(max_chunks=3)
        memory(
            torch.randn(1, 8),
            row_indices=torch.tensor([0]),
            batch_size=2,
        )
        memory(
            torch.randn(1, 8),
            row_indices=torch.tensor([1]),
            batch_size=2,
        )
        memory(
            torch.randn(1, 8),
            row_indices=torch.tensor([0]),
            batch_size=2,
        )

        self.assertEqual(memory._memory_by_row[0].shape[0], 2)
        self.assertEqual(memory._memory_by_row[1].shape[0], 1)

    def test_future_token_cannot_change_already_emitted_belief(self) -> None:
        """Appending a future chunk may not retroactively alter an old output."""
        torch.manual_seed(11)
        first_run = self._memory()
        second_run = self._memory()
        second_run.load_state_dict(first_run.state_dict())
        shared_token = torch.randn(1, 8)

        first_belief = first_run(shared_token).detach().clone()
        second_belief = second_run(shared_token).detach().clone()
        first_run(torch.ones(1, 8))
        second_run(-torch.ones(1, 8))

        self.assertTrue(torch.equal(first_belief, second_belief))


class ChunkMemoryGRUTest(unittest.TestCase):
    """Tests for the parameter-matched non-Transformer memory baseline."""

    def _config(self, *, max_chunks: int = 3) -> TemporalModelConfig:
        return TemporalModelConfig(
            temporal_dim=8,
            chunk_token_dim=8,
            chunk_transformer_layers=1,
            chunk_transformer_heads=2,
            chunk_transformer_ffn_dim=16,
            max_chunk_memory=max_chunks,
            dropout=0.0,
            memory_type="gru",
            gru_layers=2,
        )

    def test_memory_is_capped_and_rows_are_isolated(self) -> None:
        memory = ChunkMemoryGRU(self._config(max_chunks=2)).eval()
        memory(torch.randn(1, 8), row_indices=torch.tensor([0]), batch_size=2)
        memory(torch.randn(1, 8), row_indices=torch.tensor([1]), batch_size=2)
        memory(torch.randn(1, 8), row_indices=torch.tensor([0]), batch_size=2)
        memory(torch.randn(1, 8), row_indices=torch.tensor([0]), batch_size=2)

        self.assertEqual(memory._memory_by_row[0].shape[0], 2)
        self.assertEqual(memory._memory_by_row[1].shape[0], 1)

    def test_dropped_token_cannot_affect_fixed_window_belief(self) -> None:
        torch.manual_seed(17)
        left = ChunkMemoryGRU(self._config(max_chunks=3)).eval()
        right = ChunkMemoryGRU(self._config(max_chunks=3)).eval()
        right.load_state_dict(left.state_dict())
        left(torch.full((1, 8), -10.0))
        right(torch.full((1, 8), 10.0))
        shared = [torch.randn(1, 8) for _ in range(3)]
        for token in shared:
            left_belief = left(token)
            right_belief = right(token)

        self.assertTrue(torch.allclose(left_belief, right_belief, atol=1e-6))

    def test_production_gru_parameter_budget_matches_transformer(self) -> None:
        transformer_config = TemporalModelConfig(max_chunk_memory=8)
        gru_config = TemporalModelConfig(max_chunk_memory=8, memory_type="gru")
        transformer = ChunkMemoryTransformer(transformer_config)
        gru = ChunkMemoryGRU(gru_config)
        transformer_parameters = sum(p.numel() for p in transformer.parameters())
        gru_parameters = sum(p.numel() for p in gru.parameters())

        relative_difference = (
            abs(gru_parameters - transformer_parameters) / transformer_parameters
        )
        self.assertLess(relative_difference, 0.05)

    def test_chunk_order_changes_belief(self) -> None:
        """Temporal memory must represent order, not just an unordered token set."""
        first = ChunkMemoryGRU(self._config()).eval()
        second = ChunkMemoryGRU(self._config()).eval()
        second.load_state_dict(first.state_dict())
        token_a = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        token_b = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        first(token_a)
        belief_ab = first(token_b)
        second(token_b)
        belief_ba = second(token_a)

        self.assertFalse(torch.allclose(belief_ab, belief_ba, atol=1e-6, rtol=1e-6))


class TemporalPolicyNetTest(unittest.TestCase):
    """Integration-level unit tests for temporal state and spatial compatibility."""

    def test_spatial_split_forward_matches_legacy_forward(self) -> None:
        """Refactoring BCPolicyNet must preserve the old non-temporal output."""
        torch.manual_seed(1)
        model = BCPolicyNet(22, base_channels=8)
        model.eval()
        x = torch.randn(2, 22, 5, 5)

        direct = model(x)
        s1, s2, bottleneck = model.encode_frame(x)
        split = model.decode_policy(s1, s2, bottleneck)

        self.assertEqual(direct.keys(), split.keys())
        for key in direct:
            with self.subTest(output=key):
                self.assertTrue(torch.equal(direct[key], split[key]))

    def test_stacked_history_checkpoint_uses_newest_frame_stem_weights(self) -> None:
        """An 8-frame checkpoint should initialize the single-frame temporal stem."""
        source = BCPolicyNet(22 * 8, base_channels=8)
        target = BCPolicyNet(22, base_channels=8, temporal_dim=24)
        with torch.no_grad():
            source.stem[0].weight.copy_(
                torch.arange(source.stem[0].weight.numel(), dtype=torch.float32)
                .reshape_as(source.stem[0].weight)
            )
        expected = source.stem[0].weight[:, -22:, :, :].detach().clone()

        _load_temporal_spatial_init(target, source.state_dict())

        self.assertTrue(torch.equal(target.stem[0].weight, expected))
        self.assertTrue(torch.equal(target.move_head.weight, source.move_head.weight))

    def test_no_completed_chunk_matches_spatial_policy(self) -> None:
        """Before the first boundary, temporal state must not alter tick policy."""
        torch.manual_seed(2)
        model, encoder = _temporal_model()
        observation = _observations(1)[0]
        x = torch.from_numpy(encoder.encode(observation)).unsqueeze(0)

        temporal = model(x, chunk_boundary=torch.tensor([False]), reset_memory=True)
        spatial = model.spatial(x)

        for key in spatial:
            with self.subTest(output=key):
                self.assertTrue(torch.equal(temporal[key], spatial[key]))

    def test_global_belief_persists_between_chunk_boundaries(self) -> None:
        """Every tick after a boundary should use the latest global belief."""
        class RecordingSpatial(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.beliefs = []

            def encode_frame(self, x):
                return x[:, :1], x[:, :2], x[:, :3]

            def decode_policy(self, s1, s2, bottleneck, *, belief=None):
                del s1, s2
                self.beliefs.append(None if belief is None else belief.detach().clone())
                return {"value": bottleneck.mean((1, 2, 3))}

        config = TemporalModelConfig(
            base_channels=1,
            temporal_dim=8,
            chunk_token_dim=3,
            chunk_transformer_layers=1,
            chunk_transformer_heads=2,
            chunk_transformer_ffn_dim=16,
            max_chunk_memory=4,
            chunk_size=2,
        )
        spatial = RecordingSpatial()
        model = TemporalPolicyNet(3, spatial, config).eval()
        frame = torch.randn(1, 3, 4, 4)

        model(frame, chunk_boundary=torch.tensor([False]), reset_memory=True)
        model(frame, chunk_boundary=torch.tensor([True]))
        boundary_belief = spatial.beliefs[-1]
        model(frame, chunk_boundary=torch.tensor([False]))

        self.assertIsNone(spatial.beliefs[0])
        self.assertIsNotNone(boundary_belief)
        self.assertTrue(torch.equal(boundary_belief, spatial.beliefs[-1]))

    def test_reset_reproduces_first_episode_output(self) -> None:
        """Reset must remove all influence from a previous episode."""
        torch.manual_seed(3)
        model, encoder = _temporal_model()
        observations = _observations(2)
        x0 = torch.from_numpy(encoder.encode(observations[0])).unsqueeze(0)
        x1 = torch.from_numpy(encoder.encode(observations[1])).unsqueeze(0)

        first = model(x0, chunk_boundary=torch.tensor([True]), reset_memory=True)
        model(x1, chunk_boundary=torch.tensor([True]))
        repeated = model(x0, chunk_boundary=torch.tensor([True]), reset_memory=True)

        for key in first:
            with self.subTest(output=key):
                self.assertTrue(torch.allclose(first[key], repeated[key], atol=1e-6, rtol=1e-6))

    def test_synchronized_batch_matches_independent_streams(self) -> None:
        """Batched inference should equal per-stream inference at shared boundaries."""
        torch.manual_seed(4)
        batched, encoder = _temporal_model()
        stream_a, _ = _temporal_model()
        stream_b, _ = _temporal_model()
        stream_a.load_state_dict(batched.state_dict())
        stream_b.load_state_dict(batched.state_dict())
        observations = _observations(2)
        xa = torch.from_numpy(encoder.encode(observations[0])).unsqueeze(0)
        xb = torch.from_numpy(encoder.encode(observations[1])).unsqueeze(0)
        x = torch.cat([xa, xb], dim=0)

        batch_output = batched(
            x,
            chunk_boundary=torch.tensor([True, True]),
            reset_memory=True,
        )
        output_a = stream_a(xa, chunk_boundary=torch.tensor([True]), reset_memory=True)
        output_b = stream_b(xb, chunk_boundary=torch.tensor([True]), reset_memory=True)

        for key in batch_output:
            expected = torch.cat([output_a[key], output_b[key]], dim=0)
            with self.subTest(output=key):
                self.assertTrue(torch.allclose(batch_output[key], expected, atol=1e-5, rtol=1e-5))

    def test_asynchronous_batch_boundary_does_not_update_other_stream(self) -> None:
        """A boundary in one stream must not finalize or fuse another stream's chunk."""
        torch.manual_seed(5)
        model, encoder = _temporal_model()
        observations = _observations(2)
        x = torch.stack([
            torch.from_numpy(encoder.encode(observations[0])),
            torch.from_numpy(encoder.encode(observations[1])),
        ])
        spatial = model.spatial(x)

        output = model(
            x,
            chunk_boundary=torch.tensor([True, False]),
            reset_memory=True,
        )

        for key in output:
            with self.subTest(output=key):
                self.assertTrue(torch.allclose(
                    output[key][1],
                    spatial[key][1],
                    atol=1e-6,
                    rtol=1e-6,
                ))
        self.assertIsNotNone(model.accumulator._state)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP16 smoke")
    def test_temporal_loss_accepts_fp16_move_logits(self) -> None:
        """AMP logits must use a representable legal-action mask value."""
        model, encoder = _temporal_model()
        model = model.cuda().half()
        observation = _observations(1)[0]
        x = torch.from_numpy(encoder.encode(observation)).unsqueeze(0).cuda().half()
        pass_label = 8 * observation.height * observation.width
        valid_mask = torch.zeros(1, pass_label + 1, dtype=torch.bool, device="cuda")
        valid_mask[:, -1] = True
        batch = {
            "x": x,
            "policy_label": torch.tensor([pass_label], device="cuda"),
            "valid_action_mask": valid_mask,
            "sample_weight": torch.ones(1, device="cuda"),
            "value_target": torch.zeros(1, device="cuda"),
            "burn_in_mask": torch.zeros(1, dtype=torch.bool, device="cuda"),
            "chunk_boundary": torch.zeros(1, dtype=torch.bool, device="cuda"),
            "seq_lengths": torch.ones(1, dtype=torch.long, device="cuda"),
        }
        config = TemporalTrainConfig(
            input_path="unused",
            output_path="unused",
            sequence_length=1,
            burn_in=0,
        )

        loss, _, _ = _temporal_batch_loss(model, batch, config)

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
