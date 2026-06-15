from __future__ import annotations

from dataclasses import replace
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generals_bot.replay import prepare_raw_replay_shards
from generals_bot.replay.schema import RawReplay, RawReplayMove, RawTerrain
from generals_bot.sim import GameConfig, GeneralsEnv
from generals_bot.sim.types import Action, PassAction
from generals_bot.training.action_codec import action_to_label, label_to_action, valid_action_mask
from generals_bot.training.dataset import (
    BCSample,
    MixedReplayBCDataset,
    RawReplayBCDataset,
    collate_bc_samples,
    evaluate_replay_filter_rule,
    samples_from_raw_replay,
)
from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.policy import apply_pass_bias

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


class TrainingBCTest(unittest.TestCase):
    """Tests for behavior cloning data, encoding, and model plumbing."""

    def test_action_codec_round_trip_and_pass_label(self) -> None:
        """Action labels should round-trip and reserve the final class for pass."""
        action = Action(0, 4, 5, False)
        label = action_to_label(action, label_height=3, label_width=3, board_width=3)
        decoded = label_to_action(label, player_id=0, height=3, width=3)
        pass_label = action_to_label(PassAction(0), label_height=3, label_width=3, board_width=3)

        self.assertEqual(decoded, action)
        self.assertEqual(pass_label, 8 * 3 * 3)

    def test_valid_action_mask_allows_pass_and_owned_starts(self) -> None:
        """The legal action mask should include pass and moves from owned tiles."""
        env = GeneralsEnv(GameConfig(width=3, height=3, generals=(0, 8), starting_armies={0: 3}))
        observations = env.reset()
        mask = valid_action_mask(observations[0])

        self.assertTrue(mask[-1])
        self.assertTrue(mask[action_to_label(Action(0, 0, 1), label_height=3, label_width=3, board_width=3)])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_apply_pass_bias_only_when_non_pass_move_exists(self) -> None:
        """Pass bias should only affect late states with legal non-pass moves."""
        logits = torch.zeros((1, 9), dtype=torch.float32)
        valid_mask = np.zeros((9,), dtype=bool)
        valid_mask[-1] = True

        apply_pass_bias(logits, valid_mask, 60, pass_bias=2.0, pass_bias_turn=50)
        self.assertEqual(float(logits[0, -1]), 0.0)

        valid_mask[0] = True
        apply_pass_bias(logits, valid_mask, 49, pass_bias=2.0, pass_bias_turn=50)
        self.assertEqual(float(logits[0, -1]), 0.0)

        apply_pass_bias(logits, valid_mask, 50, pass_bias=2.0, pass_bias_turn=50)
        self.assertEqual(float(logits[0, -1]), -2.0)

    def test_observation_encoder_shape_and_dtype(self) -> None:
        """The encoder should produce a float32 CxHxW tensor."""
        env = GeneralsEnv(GameConfig(width=3, height=3, generals=(0, 8)))
        obs = env.reset()[0]
        encoder = ObservationEncoder()

        encoded = encoder.encode(obs)

        self.assertEqual(encoded.shape, (encoder.num_channels, 3, 3))
        self.assertEqual(encoded.dtype, np.float32)

    def test_observation_encoder_absolute_turn_features(self) -> None:
        """The encoder should expose both cyclic and absolute turn features."""
        env = GeneralsEnv(GameConfig(width=3, height=3, generals=(0, 8)))
        obs = replace(env.reset()[0], turn=125)
        encoder = ObservationEncoder()

        encoded = encoder.encode(obs)
        by_name = {
            name: encoded[index]
            for index, name in enumerate(encoder.feature_names)
        }

        self.assertEqual(encoder.num_channels, 22)
        self.assertTrue(np.allclose(by_name["turn_mod50"], 25 / 49.0))
        self.assertTrue(np.allclose(by_name["turn_norm_500"], 0.25))
        self.assertTrue(np.allclose(by_name["turn_phase_50"], 0.2))

    def test_observation_encoder_stacks_history_channels(self) -> None:
        """History encoding should concatenate padded old-to-new observation frames."""
        env = GeneralsEnv(GameConfig(width=3, height=3, generals=(0, 8)))
        obs = env.reset()[0]
        old_obs = replace(obs, turn=2)
        new_obs = replace(obs, turn=3)
        encoder = ObservationEncoder(history_length=4)

        encoded = encoder.encode_history((old_obs, new_obs))
        frame_channels = len(encoder.feature_names)
        frame_turn_mod2 = encoder.feature_names.index("turn_mod2")

        self.assertEqual(encoded.shape, (frame_channels * 4, 3, 3))
        self.assertEqual(encoder.num_channels, frame_channels * 4)
        self.assertEqual(len(encoder.channel_names), encoder.num_channels)
        self.assertTrue(np.allclose(encoded[frame_turn_mod2], 0.0))
        self.assertTrue(np.allclose(encoded[frame_channels + frame_turn_mod2], 0.0))
        self.assertTrue(np.allclose(encoded[2 * frame_channels + frame_turn_mod2], 0.0))
        self.assertTrue(np.allclose(encoded[3 * frame_channels + frame_turn_mod2], 1.0))

    def test_observation_encoder_general_memory_features(self) -> None:
        """The encoder should expose remembered general locations."""
        env = GeneralsEnv(
            GameConfig(
                width=3,
                height=3,
                generals=(0, 8),
                starting_owners={7: 0},
                starting_armies={0: 1, 7: 1, 8: 5},
            )
        )
        observations = env.reset()
        observations, _, _, _ = env.step({1: Action(1, 8, 7)})
        encoder = ObservationEncoder()

        encoded = encoder.encode(observations[0])
        by_name = {
            name: encoded[index]
            for index, name in enumerate(encoder.feature_names)
        }

        self.assertEqual(by_name["general"][2, 2], 0.0)
        self.assertEqual(by_name["known_general"][2, 2], 1.0)
        self.assertEqual(by_name["known_enemy_general"][2, 2], 1.0)

    def test_samples_are_player_turn_based(self) -> None:
        """Raw replay conversion should emit one sample per alive player-turn."""
        samples = samples_from_raw_replay(_small_replay())
        by_turn = {(sample.player_id, sample.turn): sample for sample in samples}

        self.assertEqual(len(samples), 6)
        self.assertIsInstance(by_turn[(0, 0)].action, PassAction)
        self.assertEqual(by_turn[(0, 2)].action, Action(0, 0, 1, False))
        self.assertEqual(by_turn[(1, 2)].action, Action(1, 8, 5, False))

    def test_samples_include_bounded_player_history(self) -> None:
        """Raw replay conversion should attach per-player observation histories."""
        samples = samples_from_raw_replay(_small_replay(), history_length=4)
        by_turn = {(sample.player_id, sample.turn): sample for sample in samples}

        self.assertEqual(len(by_turn[(0, 0)].history), 1)
        self.assertEqual(len(by_turn[(0, 2)].history), 3)
        self.assertEqual([obs.player_id for obs in by_turn[(0, 2)].history], [0, 0, 0])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_collate_downweights_late_pass_samples(self) -> None:
        """Late pass labels should receive the configured sample weight."""
        env = GeneralsEnv(GameConfig(width=3, height=3, generals=(0, 8)))
        obs = env.reset()[0]
        samples = [
            _bc_sample(replace(obs, turn=29), PassAction(0)),
            _bc_sample(replace(obs, turn=30), PassAction(0)),
            _bc_sample(replace(obs, turn=31), Action(0, 0, 1, False)),
        ]

        batch = collate_bc_samples(samples, late_pass_turn=30, late_pass_weight=0.2)

        self.assertTrue(np.allclose(batch["sample_weight"].numpy(), [1.0, 0.2, 1.0]))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_collate_uses_history_encoder_channels(self) -> None:
        """Collation should use stacked history channels when the encoder asks for them."""
        env = GeneralsEnv(GameConfig(width=3, height=3, generals=(0, 8)))
        obs = env.reset()[0]
        sample = _bc_sample(replace(obs, turn=1), PassAction(0))
        sample = replace(
            sample,
            observation_history=(replace(obs, turn=0), replace(obs, turn=1)),
        )
        encoder = ObservationEncoder(history_length=4)

        batch = collate_bc_samples([sample], encoder=encoder)

        self.assertEqual(batch["x"].shape, (1, encoder.num_channels, 3, 3))

    def test_dataset_without_shuffle_preserves_order(self) -> None:
        """BC dataset should keep replay order when shuffle buffer is disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            dataset = RawReplayBCDataset(input_path, split="all", shuffle_buffer_size=0)

            observed = [(sample.player_id, sample.turn) for sample in dataset]

            self.assertEqual(
                observed,
                [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
            )

    def test_dataset_shuffle_is_seeded_and_complete(self) -> None:
        """Shuffle buffer should be deterministic and preserve all samples."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            ordered_dataset = RawReplayBCDataset(input_path, split="all", shuffle_buffer_size=0)
            first_dataset = RawReplayBCDataset(
                input_path,
                split="all",
                shuffle_buffer_size=3,
                seed=123,
            )
            second_dataset = RawReplayBCDataset(
                input_path,
                split="all",
                shuffle_buffer_size=3,
                seed=123,
            )
            third_dataset = RawReplayBCDataset(
                input_path,
                split="all",
                shuffle_buffer_size=3,
                seed=456,
            )

            ordered = [(sample.player_id, sample.turn) for sample in ordered_dataset]
            first = [(sample.player_id, sample.turn) for sample in first_dataset]
            second = [(sample.player_id, sample.turn) for sample in second_dataset]
            third = [(sample.player_id, sample.turn) for sample in third_dataset]

            self.assertEqual(first, second)
            self.assertCountEqual(first, ordered)
            self.assertCountEqual(third, ordered)
            self.assertNotEqual(first, ordered)
            self.assertNotEqual(first, third)

    def test_dataset_reads_replay_shard_directory(self) -> None:
        """BC dataset should accept a directory of raw replay shards."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replays_file(
                temp_dir,
                [_small_replay(), replace(_small_replay(), replay_id="bc-small-2")],
            )
            shard_dir = Path(temp_dir) / "shards"
            prepare_raw_replay_shards(input_path, shard_dir, shard_size=1, seed=5)
            dataset = RawReplayBCDataset(
                shard_dir,
                split="all",
                shuffle_buffer_size=0,
                shuffle_shards=True,
                seed=123,
            )

            replay_ids = {sample.replay_id for sample in dataset}

            self.assertEqual(replay_ids, {"bc-small", "bc-small-2"})

    def test_dataset_filters_replays_by_version(self) -> None:
        """Replay version filtering should happen before sample generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replays_file(
                temp_dir,
                [
                    replace(_small_replay(), replay_id="bc-v5", version=5),
                    replace(_small_replay(), replay_id="bc-v13", version=13),
                ],
            )
            dataset = RawReplayBCDataset(
                input_path,
                split="all",
                shuffle_buffer_size=0,
                replay_versions=(13,),
            )

            replay_ids = {sample.replay_id for sample in dataset}

            self.assertEqual(replay_ids, {"bc-v13"})

    def test_dataset_filters_replays_by_pass_ratio(self) -> None:
        """Named replay filtering rules should drop action-sparse games."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replays_file(
                temp_dir,
                [_late_pass_heavy_replay(), _dense_late_action_replay()],
            )
            dataset = RawReplayBCDataset(
                input_path,
                split="all",
                shuffle_buffer_size=0,
                replay_filter_rule="late35_pass10",
            )

            samples = list(dataset)
            rejected = evaluate_replay_filter_rule(
                samples_from_raw_replay(_late_pass_heavy_replay()),
                "late35_pass10",
            )

            self.assertTrue(samples)
            self.assertEqual({sample.replay_id for sample in samples}, {"bc-dense-late"})
            self.assertFalse(rejected.accepted)
            self.assertEqual(rejected.reason, "late35_pass_ratio_gt_0.10")

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_collate_and_model_forward(self) -> None:
        """Collated samples should run through the BC model."""
        from generals_bot.training.model import BCPolicyNet

        samples = samples_from_raw_replay(_small_replay())[:4]
        batch = collate_bc_samples(samples, encoder=ObservationEncoder())
        model = BCPolicyNet(batch["x"].shape[1], base_channels=8)

        output = model(batch["x"])

        self.assertEqual(output["move_logits"].shape[:2], (4, 8))
        self.assertEqual(output["action_logits"].shape, (4, 1))
        self.assertEqual(output["pass_logits"].shape, (4, 1))
        self.assertEqual(output["value"].shape, (4,))
        self.assertEqual(output["enemy_king_logits"].shape, (4, 1, 3, 3))
        self.assertEqual(batch["valid_action_mask"].shape[0], 4)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_model_loads_checkpoints_missing_enemy_king_head(self) -> None:
        """Model loading should tolerate old checkpoints without the auxiliary head."""
        from generals_bot.training.model import BCPolicyNet, load_model_state_compatible

        source = BCPolicyNet(ObservationEncoder().num_channels, base_channels=8)
        old_state = {
            key: value
            for key, value in source.state_dict().items()
            if not key.startswith("enemy_king_head.")
        }
        target = BCPolicyNet(ObservationEncoder().num_channels, base_channels=8)

        missing = load_model_state_compatible(target, old_state)

        self.assertTrue(missing)
        self.assertTrue(all(key.startswith("enemy_king_head.") for key in missing))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_collate_builds_enemy_king_game_distance_targets(self) -> None:
        """Enemy king targets should use shortest game distance around mountains."""
        env = GeneralsEnv(GameConfig(width=3, height=3, generals=(0, 2), mountains=(1,)))
        obs = env.reset()[0]
        sample = replace(
            _bc_sample(obs, PassAction(0)),
            enemy_general_tile=2,
            mountains=(1,),
        )

        batch = collate_bc_samples([sample], encoder=ObservationEncoder())

        self.assertEqual(int(batch["enemy_king_label"][0]), 2)
        self.assertTrue(bool(batch["enemy_king_valid_mask"][0, 2]))
        self.assertEqual(float(batch["enemy_king_distance"][0, 2]), 0.0)
        self.assertEqual(float(batch["enemy_king_distance"][0, 0]), 4.0)
        self.assertTrue(torch.isinf(batch["enemy_king_distance"][0, 1]))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_training_writes_metric_log(self) -> None:
        """Training should write one JSONL row per optimization step."""
        from generals_bot.training.trainer import TrainConfig, train_bc

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "replays.jsonl.gz"
            output_path = Path(temp_dir) / "policy.pt"
            with gzip.open(input_path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(_small_replay().to_dict()) + "\n")

            summary = train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    split="all",
                    limit_replays=1,
                    steps=1,
                    batch_size=2,
                    base_channels=8,
                    shuffle_buffer_size=0,
                    val_batches=0,
                )
            )

            metrics_path = Path(summary["metrics_output_path"])
            rows = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.with_suffix(".pt.json").exists())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], 1)
        self.assertEqual(rows[0]["phase"], "train")
        self.assertIn("loss", rows[0])
        self.assertIn("policy_loss", rows[0])
        self.assertIn("action_loss", rows[0])
        self.assertIn("move_loss", rows[0])
        self.assertIn("loss_weight_mean", rows[0])
        self.assertNotIn("value_loss", rows[0])
        self.assertNotIn("distance_loss", rows[0])
        self.assertNotIn("actionness_loss", rows[0])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_distance_targets_prefer_nearby_actions(self) -> None:
        """Distance targets should score exact actions below nearby moves and pass."""
        from generals_bot.training.trainer import _distance_targets

        label = action_to_label(
            Action(0, 0, 1, False),
            label_height=3,
            label_width=3,
            board_width=3,
        )
        nearby = action_to_label(
            Action(0, 0, 3, False),
            label_height=3,
            label_width=3,
            board_width=3,
        )
        split = action_to_label(
            Action(0, 0, 1, True),
            label_height=3,
            label_width=3,
            board_width=3,
        )
        distances = _distance_targets(
            torch.tensor([label]),
            num_classes=8 * 3 * 3 + 1,
            label_height=3,
            label_width=3,
            cap=10.0,
            split_penalty=0.5,
            device=torch.device("cpu"),
        )

        self.assertEqual(float(distances[0, label]), 0.0)
        self.assertLess(float(distances[0, nearby]), float(distances[0, -1]))
        self.assertGreater(float(distances[0, split]), 0.0)
        self.assertEqual(float(distances[0, -1]), 1.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_hierarchical_batch_loss_splits_action_and_move_terms(self) -> None:
        """Hierarchical loss should train pass samples only through the action gate."""
        from generals_bot.training.trainer import _batch_loss_and_metrics

        class DummyModel(torch.nn.Module):
            """Return fixed hierarchical policy outputs for a synthetic batch."""

            def __init__(self) -> None:
                """Create deterministic action and move logits."""
                super().__init__()
                move_logits = torch.zeros((2, 8, 1, 1), dtype=torch.float32)
                move_logits[1, 0, 0, 0] = 4.0
                self.output = {
                    "move_logits": move_logits,
                    "action_logits": torch.tensor([[-3.0], [3.0]], dtype=torch.float32),
                    "value": torch.zeros((2,), dtype=torch.float32),
                }

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                """Return the fixed outputs regardless of input."""
                del x
                return self.output

        batch = {
            "x": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
            "policy_label": torch.tensor([8, 0]),
            "valid_action_mask": torch.tensor(
                [
                    [False, False, False, False, False, False, False, False, True],
                    [True, False, False, False, False, False, False, False, True],
                ],
                dtype=torch.bool,
            ),
            "sample_weight": torch.tensor([0.2, 1.0], dtype=torch.float32),
            "value_target": torch.zeros((2,), dtype=torch.float32),
        }

        _, metrics, _ = _batch_loss_and_metrics(
            DummyModel(),
            batch,
            value_loss_weight=0.0,
            distance_loss_weight=0.05,
        )

        self.assertIn("action_loss", metrics)
        self.assertIn("move_loss", metrics)
        self.assertIn("distance_loss", metrics)
        self.assertLess(metrics["move_loss"], 0.2)
        self.assertEqual(metrics["action_accuracy"], 1.0)
        self.assertEqual(metrics["pass_accuracy"], 1.0)
        self.assertEqual(metrics["invalid_rate"], 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_policy_loss_weight_can_disable_policy_optimization(self) -> None:
        """Policy loss should be logged but excluded from total loss when weighted zero."""
        from generals_bot.training.trainer import _batch_loss_and_metrics

        class DummyModel(torch.nn.Module):
            """Return fixed policy logits for one pass sample."""

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                """Return valid tensor shapes for the batch."""
                return {
                    "move_logits": torch.zeros((x.shape[0], 8, 1, 1), dtype=torch.float32),
                    "action_logits": torch.full((x.shape[0], 1), 3.0, dtype=torch.float32),
                    "value": torch.zeros((x.shape[0],), dtype=torch.float32),
                    "enemy_king_logits": torch.zeros((x.shape[0], 1, 1, 1), dtype=torch.float32),
                }

        batch = {
            "x": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            "policy_label": torch.tensor([8]),
            "valid_action_mask": torch.tensor(
                [[False, False, False, False, False, False, False, False, True]],
                dtype=torch.bool,
            ),
            "sample_weight": torch.ones((1,), dtype=torch.float32),
            "value_target": torch.zeros((1,), dtype=torch.float32),
        }

        loss, metrics, _ = _batch_loss_and_metrics(
            DummyModel(),
            batch,
            policy_loss_weight=0.0,
            value_loss_weight=0.0,
        )

        self.assertGreater(metrics["policy_loss"], 0.0)
        self.assertEqual(metrics["policy_loss_weight"], 0.0)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(metrics["loss"], 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_hierarchical_batch_loss_has_zero_move_loss_for_pass_only_batch(self) -> None:
        """A pass-only batch should not contribute to conditional move CE."""
        from generals_bot.training.trainer import _batch_loss_and_metrics

        class PassOnlyModel(torch.nn.Module):
            """Return fixed pass-leaning action logits."""

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                """Return valid tensor shapes for a pass-only batch."""
                return {
                    "move_logits": torch.zeros((x.shape[0], 8, 1, 1), dtype=torch.float32),
                    "action_logits": torch.full((x.shape[0], 1), -3.0, dtype=torch.float32),
                    "value": torch.zeros((x.shape[0],), dtype=torch.float32),
                }

        batch = {
            "x": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            "policy_label": torch.tensor([8]),
            "valid_action_mask": torch.tensor(
                [[False, False, False, False, False, False, False, False, True]],
                dtype=torch.bool,
            ),
            "sample_weight": torch.tensor([0.2], dtype=torch.float32),
            "value_target": torch.zeros((1,), dtype=torch.float32),
        }

        _, metrics, _ = _batch_loss_and_metrics(
            PassOnlyModel(),
            batch,
            value_loss_weight=0.0,
        )

        self.assertEqual(metrics["move_loss"], 0.0)
        self.assertEqual(metrics["move_accuracy"], 0.0)
        self.assertEqual(metrics["pass_accuracy"], 1.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_training_writes_distance_loss_when_enabled(self) -> None:
        """Training should include distance loss metrics only when enabled."""
        from generals_bot.training.trainer import TrainConfig, train_bc

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            output_path = Path(temp_dir) / "policy.pt"
            summary = train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    split="all",
                    limit_replays=1,
                    steps=1,
                    batch_size=2,
                    base_channels=8,
                    shuffle_buffer_size=0,
                    val_batches=0,
                    distance_loss_weight=0.05,
                )
            )
            rows = [
                json.loads(line)
                for line in Path(summary["metrics_output_path"]).read_text(encoding="utf-8").splitlines()
            ]

            self.assertIn("distance_loss", rows[0])
            self.assertGreaterEqual(rows[0]["distance_loss"], 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_enemy_king_loss_prefers_near_predictions(self) -> None:
        """Enemy king auxiliary loss should report CE and game-distance metrics."""
        from generals_bot.training.trainer import enemy_king_prediction_loss

        logits = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        logits[0, 0, 0, 0] = 4.0
        labels = torch.tensor([2], dtype=torch.long)
        valid_mask = torch.ones((1, 9), dtype=torch.bool)
        distances = torch.tensor(
            [[4.0, float("inf"), 0.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0]],
            dtype=torch.float32,
        )

        loss, metrics = enemy_king_prediction_loss(
            logits,
            labels=labels,
            valid_mask=valid_mask,
            distances=distances,
            distance_weight=1.0,
            distance_cap=4.0,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertIn("enemy_king_ce_loss", metrics)
        self.assertIn("enemy_king_distance_loss", metrics)
        self.assertEqual(metrics["enemy_king_mean_distance"], 4.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_training_writes_enemy_king_loss_when_enabled(self) -> None:
        """Training should include enemy king metrics only when the loss is enabled."""
        from generals_bot.training.trainer import TrainConfig, train_bc

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            output_path = Path(temp_dir) / "policy.pt"
            summary = train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    split="all",
                    limit_replays=1,
                    steps=1,
                    batch_size=2,
                    base_channels=8,
                    shuffle_buffer_size=0,
                    val_batches=0,
                    enemy_king_loss_weight=0.1,
                    enemy_king_distance_loss_weight=1.0,
                )
            )
            rows = [
                json.loads(line)
                for line in Path(summary["metrics_output_path"]).read_text(encoding="utf-8").splitlines()
            ]

            self.assertIn("enemy_king_loss", rows[0])
            self.assertIn("enemy_king_top1_acc", rows[0])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_enemy_king_predictor_forward_shape(self) -> None:
        """Standalone enemy king predictor should emit one dense logit per tile."""
        from generals_bot.training.model import EnemyKingPredictorNet

        encoder = ObservationEncoder(history_length=4)
        model = EnemyKingPredictorNet(encoder.num_channels, base_channels=8)

        output = model(torch.zeros((2, encoder.num_channels, 5, 7), dtype=torch.float32))

        self.assertEqual(output["enemy_king_logits"].shape, (2, 1, 5, 7))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_enemy_king_predictor_training_writes_checkpoint(self) -> None:
        """Standalone predictor training should save predictor metadata and weights."""
        from generals_bot.training.enemy_king_predictor import (
            EnemyKingPredictorTrainConfig,
            train_enemy_king_predictor,
        )
        from generals_bot.training.model import PREDICTOR_MODEL_TYPE

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            output_path = Path(temp_dir) / "predictor.pt"
            summary = train_enemy_king_predictor(
                EnemyKingPredictorTrainConfig(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    split="all",
                    limit_replays=1,
                    steps=1,
                    batch_size=2,
                    base_channels=8,
                    history_length=2,
                    shuffle_buffer_size=0,
                    val_batches=0,
                    device="cpu",
                )
            )
            checkpoint = torch.load(output_path, map_location="cpu")

            self.assertEqual(summary["global_step"], 1)
            self.assertEqual(checkpoint["model_type"], PREDICTOR_MODEL_TYPE)
            self.assertEqual(checkpoint["history_length"], 2)
            self.assertIn("enemy_king_head.weight", checkpoint["model_state"])
            self.assertTrue(output_path.with_suffix(".pt.json").exists())

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_enemy_king_predictor_training_accepts_selfplay_mix(self) -> None:
        """Standalone predictor training can mix raw and generated replay streams."""
        from generals_bot.training.enemy_king_predictor import (
            EnemyKingPredictorTrainConfig,
            train_enemy_king_predictor,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            selfplay_dir = Path(temp_dir) / "selfplay"
            raw_dir.mkdir()
            selfplay_dir.mkdir()
            raw_path = _write_replays_file(
                str(raw_dir),
                [replace(_small_replay(), replay_id="raw-small")],
            )
            selfplay_path = _write_replays_file(
                str(selfplay_dir),
                [
                    replace(
                        _small_replay(),
                        replay_id="selfplay-small",
                        source="selfplay-test",
                    )
                ],
            )
            output_path = Path(temp_dir) / "predictor_mix.pt"

            summary = train_enemy_king_predictor(
                EnemyKingPredictorTrainConfig(
                    input_path=str(raw_path),
                    selfplay_input_path=str(selfplay_path),
                    output_path=str(output_path),
                    split="all",
                    limit_replays=1,
                    selfplay_limit_replays=1,
                    raw_source_weight=1.0,
                    selfplay_source_weight=1.0,
                    steps=1,
                    batch_size=2,
                    base_channels=8,
                    history_length=2,
                    shuffle_buffer_size=0,
                    val_batches=0,
                    device="cpu",
                )
            )

            self.assertEqual(summary["global_step"], 1)
            self.assertTrue(output_path.exists())

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_enemy_king_predictor_resume_continues_global_step(self) -> None:
        """Standalone predictor resume should load optimizer state and progress."""
        from generals_bot.training.enemy_king_predictor import (
            EnemyKingPredictorTrainConfig,
            train_enemy_king_predictor,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            first_output = Path(temp_dir) / "predictor_a.pt"
            second_output = Path(temp_dir) / "predictor_b.pt"
            first_summary = train_enemy_king_predictor(
                EnemyKingPredictorTrainConfig(
                    input_path=str(input_path),
                    output_path=str(first_output),
                    split="all",
                    limit_replays=1,
                    steps=1,
                    batch_size=2,
                    base_channels=8,
                    history_length=2,
                    shuffle_buffer_size=0,
                    val_batches=0,
                    device="cpu",
                )
            )
            second_summary = train_enemy_king_predictor(
                EnemyKingPredictorTrainConfig(
                    input_path=str(input_path),
                    output_path=str(second_output),
                    split="all",
                    limit_replays=1,
                    steps=2,
                    batch_size=2,
                    base_channels=999,
                    history_length=2,
                    resume_from=str(first_output),
                    shuffle_buffer_size=0,
                    val_batches=0,
                    device="cpu",
                )
            )
            checkpoint = torch.load(second_output, map_location="cpu")

            self.assertEqual(first_summary["global_step"], 1)
            self.assertEqual(second_summary["global_step"], 2)
            self.assertEqual(checkpoint["global_step"], 2)
            self.assertEqual(checkpoint["base_channels"], 8)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_init_checkpoint_can_train_only_enemy_king_head(self) -> None:
        """BC init checkpoints should load old policy weights with a fresh king head."""
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.trainer import TrainConfig, train_bc

        encoder = ObservationEncoder()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            init_path = Path(temp_dir) / "old.pt"
            output_path = Path(temp_dir) / "king.pt"
            model = BCPolicyNet(encoder.num_channels, base_channels=8)
            old_state = {
                key: value
                for key, value in model.state_dict().items()
                if not key.startswith("enemy_king_head.")
            }
            torch.save(
                {
                    "model_state": old_state,
                    "input_channels": encoder.num_channels,
                    "history_length": encoder.history_length,
                    "base_channels": 8,
                    "policy_mode": "hierarchical",
                },
                init_path,
            )

            train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    split="all",
                    limit_replays=1,
                    steps=1,
                    batch_size=2,
                    base_channels=8,
                    shuffle_buffer_size=0,
                    val_batches=0,
                    init_checkpoint=str(init_path),
                    freeze_non_king_head=True,
                    enemy_king_loss_weight=0.1,
                )
            )
            checkpoint = torch.load(output_path, map_location="cpu")

            self.assertIn("enemy_king_head.weight", checkpoint["model_state"])
            self.assertEqual(checkpoint["global_step"], 1)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_freeze_non_king_head_leaves_only_auxiliary_parameters_trainable(self) -> None:
        """The king-only training option should freeze policy, value, and backbone."""
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.trainer import TrainConfig, _apply_parameter_freeze

        model = BCPolicyNet(ObservationEncoder().num_channels, base_channels=8)

        _apply_parameter_freeze(
            model,
            TrainConfig(
                input_path="unused",
                output_path="unused.pt",
                freeze_non_king_head=True,
                enemy_king_loss_weight=1.0,
            ),
        )

        trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("enemy_king_head.") for name in trainable))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_actionness_loss_prefers_valid_moves_over_pass_for_move_labels(self) -> None:
        """Actionness loss should reward move labels when valid move logits beat pass."""
        from generals_bot.training.trainer import _actionness_loss

        labels = torch.tensor([0])
        valid_move_mask = torch.zeros((1, 8), dtype=torch.bool)
        valid_move_mask[0, 0] = True
        sample_weight = torch.ones((1,), dtype=torch.float32)
        move_action_logits = torch.tensor([3.0], dtype=torch.float32)
        pass_action_logits = torch.tensor([-3.0], dtype=torch.float32)

        good_loss = _actionness_loss(
            move_action_logits,
            labels=labels,
            valid_move_mask=valid_move_mask,
            sample_weight=sample_weight,
            pass_weight=0.05,
        )
        bad_loss = _actionness_loss(
            pass_action_logits,
            labels=labels,
            valid_move_mask=valid_move_mask,
            sample_weight=sample_weight,
            pass_weight=0.05,
        )

        self.assertLess(float(good_loss), float(bad_loss))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_actionness_loss_ignores_states_without_valid_moves(self) -> None:
        """Actionness loss should be zero when pass is the only legal action."""
        from generals_bot.training.trainer import _actionness_loss

        action_logits = torch.zeros((1,), dtype=torch.float32)
        labels = torch.tensor([8])
        valid_move_mask = torch.zeros((1, 8), dtype=torch.bool)

        loss = _actionness_loss(
            action_logits,
            labels=labels,
            valid_move_mask=valid_move_mask,
            sample_weight=torch.ones((1,), dtype=torch.float32),
            pass_weight=0.05,
        )

        self.assertEqual(float(loss), 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_training_writes_actionness_loss_when_enabled(self) -> None:
        """Training should include actionness loss metrics only when enabled."""
        from generals_bot.training.trainer import TrainConfig, train_bc

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            output_path = Path(temp_dir) / "policy.pt"
            summary = train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    split="all",
                    limit_replays=1,
                    steps=1,
                    batch_size=2,
                    base_channels=8,
                    shuffle_buffer_size=0,
                    val_batches=0,
                    actionness_loss_weight=0.2,
                )
            )
            rows = [
                json.loads(line)
                for line in Path(summary["metrics_output_path"]).read_text(encoding="utf-8").splitlines()
            ]

            self.assertIn("actionness_loss", rows[0])
            self.assertGreaterEqual(rows[0]["actionness_loss"], 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_training_saves_history_length_metadata(self) -> None:
        """Training with frame stacking should save the expected channel metadata."""
        from generals_bot.training.trainer import TrainConfig, train_bc

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            output_path = Path(temp_dir) / "policy.pt"
            train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    split="all",
                    limit_replays=1,
                    steps=1,
                    batch_size=2,
                    base_channels=8,
                    shuffle_buffer_size=0,
                    val_batches=0,
                    history_length=4,
                )
            )
            checkpoint = torch.load(output_path, map_location="cpu")

            self.assertEqual(checkpoint["history_length"], 4)
            self.assertEqual(checkpoint["policy_mode"], "hierarchical")
            self.assertEqual(checkpoint["input_channels"], ObservationEncoder(history_length=4).num_channels)
            self.assertEqual(len(checkpoint["feature_names"]), checkpoint["input_channels"])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_training_resume_continues_global_step(self) -> None:
        """Resume training should restore optimizer state and continue global steps."""
        from generals_bot.training.trainer import TrainConfig, train_bc

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            first_output = Path(temp_dir) / "policy_a.pt"
            legacy_output = Path(temp_dir) / "policy_legacy.pt"
            second_output = Path(temp_dir) / "policy_b.pt"
            first_summary = train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(first_output),
                    split="all",
                    limit_replays=1,
                    steps=1,
                    batch_size=2,
                    base_channels=8,
                    shuffle_buffer_size=0,
                    val_batches=0,
                )
            )
            legacy_checkpoint = torch.load(first_output, map_location="cpu")
            legacy_checkpoint.pop("global_step")
            legacy_checkpoint.pop("samples_seen")
            torch.save(legacy_checkpoint, legacy_output)
            second_summary = train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(second_output),
                    split="all",
                    limit_replays=1,
                    steps=2,
                    batch_size=2,
                    base_channels=999,
                    resume_from=str(legacy_output),
                    shuffle_buffer_size=0,
                    val_batches=0,
                )
            )
            checkpoint = torch.load(second_output, map_location="cpu")
            rows = [
                json.loads(line)
                for line in Path(second_summary["metrics_output_path"]).read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(first_summary["global_step"], 1)
            self.assertEqual(second_summary["global_step"], 2)
            self.assertEqual(checkpoint["global_step"], 2)
            self.assertEqual(checkpoint["base_channels"], 8)
            self.assertEqual(rows[0]["step"], 2)
            self.assertEqual(rows[0]["phase"], "train")

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_training_resume_can_override_learning_rate(self) -> None:
        """Resume training should optionally replace checkpoint optimizer lr."""
        from generals_bot.training.trainer import TrainConfig, train_bc

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            first_output = Path(temp_dir) / "policy_a.pt"
            second_output = Path(temp_dir) / "policy_b.pt"
            train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(first_output),
                    split="all",
                    limit_replays=1,
                    steps=1,
                    batch_size=2,
                    lr=3e-4,
                    base_channels=8,
                    shuffle_buffer_size=0,
                    val_batches=0,
                )
            )
            summary = train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(second_output),
                    split="all",
                    limit_replays=1,
                    steps=2,
                    batch_size=2,
                    lr=1e-4,
                    base_channels=8,
                    resume_from=str(first_output),
                    override_resume_lr=True,
                    shuffle_buffer_size=0,
                    val_batches=0,
                )
            )
            rows = [
                json.loads(line)
                for line in Path(summary["metrics_output_path"]).read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(rows[0]["step"], 2)
            self.assertEqual(rows[0]["lr"], 1e-4)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_training_writes_validation_metrics(self) -> None:
        """Validation should write separate metric rows when enabled."""
        from generals_bot.training.trainer import TrainConfig, train_bc

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            output_path = Path(temp_dir) / "policy.pt"
            summary = train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    split="all",
                    limit_replays=1,
                    steps=1,
                    batch_size=2,
                    base_channels=8,
                    shuffle_buffer_size=2,
                    seed=42,
                    val_split="all",
                    val_limit_replays=1,
                    val_every_steps=1,
                    val_batches=1,
                )
            )
            rows = [
                json.loads(line)
                for line in Path(summary["metrics_output_path"]).read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual([row["phase"] for row in rows], ["train", "val"])
            self.assertIn("loss", rows[1])
            self.assertIn("last_val_metrics", summary)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_training_writes_periodic_checkpoints(self) -> None:
        """Training should save step checkpoints at the configured interval."""
        from generals_bot.training.trainer import TrainConfig, train_bc

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = _write_replay_file(temp_dir)
            output_path = Path(temp_dir) / "policy.pt"
            train_bc(
                TrainConfig(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    split="all",
                    limit_replays=1,
                    steps=2,
                    batch_size=2,
                    base_channels=8,
                    shuffle_buffer_size=0,
                    val_batches=0,
                    checkpoint_every_steps=1,
                )
            )

            self.assertTrue(output_path.exists())
            self.assertTrue((Path(temp_dir) / "policy_step00001.pt").exists())
            self.assertTrue((Path(temp_dir) / "policy_step00002.pt").exists())

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_hierarchical_policy_gate_controls_pass_and_move(self) -> None:
        """BCModelPolicy should pass on negative gate logits and move on positive ones."""
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.policy import BCModelPolicy

        env = GeneralsEnv(
            GameConfig(width=3, height=3, generals=(0, 8), starting_armies={0: 3})
        )
        observation = env.reset()[0]
        encoder = ObservationEncoder()

        with tempfile.TemporaryDirectory() as temp_dir:
            pass_path = _write_hierarchical_policy_checkpoint(
                temp_dir,
                "pass.pt",
                action_bias=-3.0,
                encoder=encoder,
                model_cls=BCPolicyNet,
            )
            move_path = _write_hierarchical_policy_checkpoint(
                temp_dir,
                "move.pt",
                action_bias=3.0,
                encoder=encoder,
                model_cls=BCPolicyNet,
            )

            pass_policy = BCModelPolicy(pass_path, device="cpu")
            move_policy = BCModelPolicy(move_path, device="cpu")
            pass_policy.reset(0, env.config)
            move_policy.reset(0, env.config)

            self.assertIsInstance(pass_policy.act(observation), PassAction)
            self.assertEqual(move_policy.act(observation), Action(0, 0, 3, False))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_model_policy_estimates_win_rate_without_mutating_history(self) -> None:
        """Value-head inference should expose win rate without appending observations."""
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.policy import BCModelPolicy

        env = GeneralsEnv(
            GameConfig(width=3, height=3, generals=(0, 8), starting_armies={0: 3})
        )
        observation = env.reset()[0]
        encoder = ObservationEncoder()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = _write_hierarchical_policy_checkpoint(
                temp_dir,
                "value.pt",
                action_bias=3.0,
                value_bias=0.4,
                encoder=encoder,
                model_cls=BCPolicyNet,
            )
            policy = BCModelPolicy(checkpoint_path, device="cpu")
            policy.reset(0, env.config)

            estimate = policy.estimate_win_rate(observation)

            self.assertEqual(len(policy.history), 0)
            self.assertAlmostEqual(estimate.raw_value, 0.4, places=5)
            self.assertAlmostEqual(estimate.win_probability, 0.7, places=5)

            policy.act(observation)
            estimate_after_act = policy.estimate_win_rate(observation)

            self.assertEqual(len(policy.history), 1)
            self.assertAlmostEqual(estimate_after_act.raw_value, 0.4, places=5)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_model_policy_estimates_enemy_king_distribution(self) -> None:
        """Enemy king inference should expose a normalized map without appending history."""
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.policy import BCModelPolicy

        env = GeneralsEnv(
            GameConfig(width=3, height=3, generals=(0, 8), starting_armies={0: 3})
        )
        observation = env.reset()[0]
        encoder = ObservationEncoder()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = _write_hierarchical_policy_checkpoint(
                temp_dir,
                "king.pt",
                action_bias=3.0,
                encoder=encoder,
                model_cls=BCPolicyNet,
            )
            policy = BCModelPolicy(checkpoint_path, device="cpu")
            policy.reset(0, env.config)

            distribution = policy.estimate_enemy_king_distribution(observation)

            self.assertEqual(len(policy.history), 0)
            self.assertIsNotNone(distribution)
            self.assertEqual(distribution.probabilities.shape, (3, 3))
            self.assertAlmostEqual(float(distribution.probabilities.sum()), 1.0, places=5)

            policy.act(observation)
            distribution_after_act = policy.estimate_enemy_king_distribution(observation)

            self.assertEqual(len(policy.history), 1)
            self.assertIsNotNone(distribution_after_act)
            np.testing.assert_allclose(
                distribution.probabilities,
                distribution_after_act.probabilities,
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_model_policy_skips_missing_enemy_king_head_distribution(self) -> None:
        """Older checkpoints without enemy_king_head should not expose random maps."""
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.policy import BCModelPolicy

        env = GeneralsEnv(
            GameConfig(width=3, height=3, generals=(0, 8), starting_armies={0: 3})
        )
        observation = env.reset()[0]
        encoder = ObservationEncoder()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = _write_hierarchical_policy_checkpoint(
                temp_dir,
                "current.pt",
                action_bias=3.0,
                encoder=encoder,
                model_cls=BCPolicyNet,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            checkpoint["model_state"] = {
                key: value
                for key, value in checkpoint["model_state"].items()
                if not key.startswith("enemy_king_head.")
            }
            old_path = Path(temp_dir) / "old.pt"
            torch.save(checkpoint, old_path)
            policy = BCModelPolicy(old_path, device="cpu")
            policy.reset(0, env.config)

            self.assertIsNone(policy.estimate_enemy_king_distribution(observation))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_model_policy_can_use_external_enemy_king_predictor(self) -> None:
        """A standalone predictor can provide maps for a policy checkpoint without a head."""
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.policy import BCModelPolicy

        env = GeneralsEnv(
            GameConfig(width=3, height=3, generals=(0, 8), starting_armies={0: 3})
        )
        observation = env.reset()[0]
        policy_encoder = ObservationEncoder(history_length=1)
        predictor_encoder = ObservationEncoder(history_length=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = _write_hierarchical_policy_checkpoint(
                temp_dir,
                "policy.pt",
                action_bias=3.0,
                encoder=policy_encoder,
                model_cls=BCPolicyNet,
            )
            checkpoint = torch.load(policy_path, map_location="cpu")
            checkpoint["model_state"] = {
                key: value
                for key, value in checkpoint["model_state"].items()
                if not key.startswith("enemy_king_head.")
            }
            old_policy_path = Path(temp_dir) / "policy_without_head.pt"
            torch.save(checkpoint, old_policy_path)
            predictor_path = _write_enemy_king_predictor_checkpoint(
                temp_dir,
                "predictor.pt",
                encoder=predictor_encoder,
            )
            policy = BCModelPolicy(
                old_policy_path,
                device="cpu",
                enemy_king_predictor_path=predictor_path,
            )
            policy.reset(0, env.config)

            distribution = policy.estimate_enemy_king_distribution(observation)

            self.assertEqual(policy.history.maxlen, 2)
            self.assertEqual(len(policy.history), 0)
            self.assertIsNotNone(distribution)
            self.assertEqual(distribution.probabilities.shape, (3, 3))
            self.assertAlmostEqual(float(distribution.probabilities.sum()), 1.0, places=5)

            self.assertEqual(policy.act(observation), Action(0, 0, 3, False))
            distribution_after_act = policy.estimate_enemy_king_distribution(observation)

            self.assertEqual(len(policy.history), 1)
            self.assertIsNotNone(distribution_after_act)
            np.testing.assert_allclose(
                distribution.probabilities,
                distribution_after_act.probabilities,
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_batched_policy_matches_serial_policy(self) -> None:
        """Batched BC inference should match the legacy one-observation policy."""
        from generals_bot.training.batched_policy import BatchedBCPolicyRunner
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.policy import BCModelPolicy

        env = GeneralsEnv(
            GameConfig(width=3, height=3, generals=(0, 8), starting_armies={0: 3, 1: 3})
        )
        observations = env.reset()
        encoder = ObservationEncoder()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = _write_hierarchical_policy_checkpoint(
                temp_dir,
                "move.pt",
                action_bias=3.0,
                encoder=encoder,
                model_cls=BCPolicyNet,
            )
            serial_policies = {
                0: BCModelPolicy(checkpoint_path, device="cpu"),
                1: BCModelPolicy(checkpoint_path, device="cpu"),
            }
            for player, policy in serial_policies.items():
                policy.reset(player, env.config)
            serial_actions = {
                player: serial_policies[player].act(observations[player])
                for player in (0, 1)
            }

            runner = BatchedBCPolicyRunner(checkpoint_path, device="cpu")
            batched_actions = runner.act_many(
                [((0, player), observations[player]) for player in (0, 1)]
            )

            self.assertEqual(batched_actions[(0, 0)], serial_actions[0])
            self.assertEqual(batched_actions[(0, 1)], serial_actions[1])
            self.assertEqual(runner.stats.actions, 2)
            self.assertEqual(runner.stats.batches, 1)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_batched_policy_groups_mixed_map_sizes(self) -> None:
        """Batched BC inference should handle mixed board sizes by grouping shapes."""
        from generals_bot.training.batched_policy import BatchedBCPolicyRunner
        from generals_bot.training.model import BCPolicyNet

        small_env = GeneralsEnv(
            GameConfig(width=3, height=3, generals=(0, 8), starting_armies={0: 3})
        )
        large_env = GeneralsEnv(
            GameConfig(width=4, height=4, generals=(0, 15), starting_armies={0: 3})
        )
        small_observation = small_env.reset()[0]
        large_observation = large_env.reset()[0]
        encoder = ObservationEncoder()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = _write_hierarchical_policy_checkpoint(
                temp_dir,
                "move.pt",
                action_bias=3.0,
                encoder=encoder,
                model_cls=BCPolicyNet,
            )
            runner = BatchedBCPolicyRunner(checkpoint_path, device="cpu")
            actions = runner.act_many(
                [
                    (("small", 0), small_observation),
                    (("large", 0), large_observation),
                ]
            )

            self.assertEqual(actions[("small", 0)], Action(0, 0, 3, False))
            self.assertEqual(actions[("large", 0)], Action(0, 0, 4, False))
            self.assertEqual(runner.stats.actions, 2)
            self.assertEqual(runner.stats.batches, 2)
            runner.reset_key(("small", 0))
            self.assertNotIn(("small", 0), runner.histories)

    def test_mixed_replay_dataset_can_read_multiple_sources(self) -> None:
        """Mixed replay datasets should preserve samples from each configured source."""
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            selfplay_dir = Path(temp_dir) / "selfplay"
            raw_dir.mkdir()
            selfplay_dir.mkdir()
            raw_path = _write_replays_file(
                str(raw_dir),
                [replace(_small_replay(), replay_id="raw-small")],
            )
            selfplay_path = _write_replays_file(
                str(selfplay_dir),
                [replace(_small_replay(), replay_id="selfplay-small")],
            )
            dataset = MixedReplayBCDataset(
                [
                    RawReplayBCDataset(raw_path, split="all"),
                    RawReplayBCDataset(selfplay_path, split="all"),
                ],
                weights=[1.0, 1.0],
                seed=3,
                stop_when_any_exhausted=False,
            )

            replay_ids = {sample.replay_id for sample in dataset}

            self.assertEqual(replay_ids, {"raw-small", "selfplay-small"})


def _small_replay() -> RawReplay:
    """Create a tiny two-player replay for deterministic unit tests."""
    return RawReplay(
        schema_version=1,
        source="test",
        replay_id="bc-small",
        version=1,
        width=3,
        height=3,
        players=["p0", "p1"],
        stars=[None, None],
        terrain=RawTerrain(
            generals=[0, 8],
            mountains=[],
            cities=[],
            city_armies=[],
        ),
        moves=[
            RawReplayMove(0, 0, 1, False, 2),
            RawReplayMove(1, 8, 5, False, 2),
        ],
        afks=[],
    )


def _late_pass_heavy_replay() -> RawReplay:
    """Create a replay whose late turns are mostly pass labels."""
    return RawReplay(
        schema_version=1,
        source="test",
        replay_id="bc-late-pass-heavy",
        version=1,
        width=3,
        height=3,
        players=["p0", "p1"],
        stars=[None, None],
        terrain=RawTerrain(
            generals=[0, 8],
            mountains=[],
            cities=[],
            city_armies=[],
        ),
        moves=[
            RawReplayMove(0, 0, 1, False, 40),
        ],
        afks=[],
    )


def _dense_late_action_replay() -> RawReplay:
    """Create a replay with actions on every player-turn through late turns."""
    return RawReplay(
        schema_version=1,
        source="test",
        replay_id="bc-dense-late",
        version=1,
        width=3,
        height=3,
        players=["p0", "p1"],
        stars=[None, None],
        terrain=RawTerrain(
            generals=[0, 8],
            mountains=[],
            cities=[],
            city_armies=[],
        ),
        moves=[
            move
            for turn in range(41)
            for move in (
                RawReplayMove(0, 0, 1, False, turn),
                RawReplayMove(1, 8, 5, False, turn),
            )
        ],
        afks=[],
    )


def _write_replay_file(temp_dir: str) -> Path:
    """Write a tiny raw replay JSONL file for training tests."""
    return _write_replays_file(temp_dir, [_small_replay()])


def _write_replays_file(temp_dir: str, replays: list[RawReplay]) -> Path:
    """Write raw replay JSONL rows for training tests."""
    input_path = Path(temp_dir) / "replays.jsonl.gz"
    with gzip.open(input_path, "wt", encoding="utf-8") as handle:
        for replay in replays:
            handle.write(json.dumps(replay.to_dict()) + "\n")
    return input_path


def _bc_sample(observation, action) -> BCSample:
    """Create a minimal BC sample for collate tests."""
    return BCSample(
        observation=observation,
        action=action,
        value_target=0.0,
        replay_id="test",
        player_id=observation.player_id,
        turn=observation.turn,
    )


def _write_hierarchical_policy_checkpoint(
    temp_dir: str,
    filename: str,
    *,
    action_bias: float,
    value_bias: float = 0.0,
    encoder: ObservationEncoder,
    model_cls,
) -> Path:
    """Write a tiny deterministic hierarchical policy checkpoint."""
    model = model_cls(encoder.num_channels, base_channels=8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.move_head.bias[1] = 5.0
        model.pass_head[-1].bias.fill_(action_bias)
        model.value_head[-1].bias.fill_(value_bias)
    path = Path(temp_dir) / filename
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_channels": encoder.num_channels,
            "history_length": encoder.history_length,
            "base_channels": 8,
            "policy_mode": "hierarchical",
        },
        path,
    )
    return path


def _write_enemy_king_predictor_checkpoint(
    temp_dir: str,
    filename: str,
    *,
    encoder: ObservationEncoder,
) -> Path:
    """Write a tiny standalone enemy king predictor checkpoint."""
    from generals_bot.training.model import EnemyKingPredictorNet, PREDICTOR_MODEL_TYPE

    model = EnemyKingPredictorNet(encoder.num_channels, base_channels=8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    path = Path(temp_dir) / filename
    torch.save(
        {
            "model_type": PREDICTOR_MODEL_TYPE,
            "model_state": model.state_dict(),
            "input_channels": encoder.num_channels,
            "history_length": encoder.history_length,
            "base_channels": 8,
        },
        path,
    )
    return path


if __name__ == "__main__":
    unittest.main()
