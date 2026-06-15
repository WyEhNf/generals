from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generals_bot.sim import Action, ExecutedAction, GameConfig, GeneralsEnv
from generals_bot.training.action_codec import valid_action_mask
from generals_bot.training.encoding import ObservationEncoder

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

_BaseModule = torch.nn.Module if TORCH_AVAILABLE else object


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class PPOTrainingTest(unittest.TestCase):
    """Tests for self-play PPO sampling, returns, and checkpoint plumbing."""

    def test_terminal_reward_is_only_win_loss_draw(self) -> None:
        """Terminal reward should not include any shaping terms."""
        from generals_bot.training.ppo import terminal_reward

        self.assertEqual(terminal_reward(0, 0), 1.0)
        self.assertEqual(terminal_reward(0, 1), -1.0)
        self.assertEqual(terminal_reward(1, 0), -1.0)
        self.assertEqual(terminal_reward(None, 0), 0.0)

    def test_compute_gae_handles_interleaved_terminal_streams(self) -> None:
        """GAE should reset at terminal samples even when streams are interleaved."""
        from generals_bot.training.ppo import compute_gae

        rewards = torch.tensor([0.0, 0.0, 1.0, -1.0])
        values = torch.tensor([0.5, -0.5, 0.2, -0.2])
        done = torch.tensor([False, False, True, True])
        stream_ids = [(0, 0), (0, 1), (0, 0), (0, 1)]

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            done=done,
            stream_ids=stream_ids,
            last_values_by_stream={},
            gamma=1.0,
            gae_lambda=1.0,
        )

        self.assertTrue(torch.allclose(returns, torch.tensor([1.0, -1.0, 1.0, -1.0])))
        self.assertTrue(torch.allclose(advantages, returns - values))

    def test_sampler_masks_invalid_high_logit_moves(self) -> None:
        """PPO sampling should never return a move outside the valid action mask."""
        from generals_bot.training.ppo import sample_policy_actions

        env = GeneralsEnv(
            GameConfig(width=3, height=3, generals=(0, 8), starting_armies={0: 3})
        )
        observation = env.reset()[0]
        encoder = ObservationEncoder()
        x = torch.from_numpy(encoder.encode(observation)[None, :, :, :])
        mask = torch.from_numpy(valid_action_mask(observation)[None, :])
        model = _FixedPolicyModel(
            move_logits=torch.full((1, 8, 3, 3), -5.0),
            action_logits=torch.tensor([[10.0]]),
            values=torch.zeros((1,)),
        )
        model.move_logits[0, 0, 2, 2] = 50.0
        torch.manual_seed(0)

        sampled = sample_policy_actions(model, x, mask, observations=[observation])

        self.assertTrue(bool(mask[0, int(sampled.labels[0])]))
        self.assertNotEqual(int(sampled.labels[0]), 0 * 9 + 8)

    def test_sampled_log_prob_recomputes_consistently(self) -> None:
        """The PPO update path should reproduce rollout log-probabilities."""
        from generals_bot.training.ppo import evaluate_policy_actions, sample_policy_actions

        env = GeneralsEnv(
            GameConfig(width=3, height=3, generals=(0, 8), starting_armies={0: 3})
        )
        observation = env.reset()[0]
        encoder = ObservationEncoder()
        x = torch.from_numpy(encoder.encode(observation)[None, :, :, :])
        mask = torch.from_numpy(valid_action_mask(observation)[None, :])
        model = _FixedPolicyModel(
            move_logits=torch.zeros((1, 8, 3, 3)),
            action_logits=torch.tensor([[3.0]]),
            values=torch.tensor([0.25]),
        )
        torch.manual_seed(4)

        sampled = sample_policy_actions(model, x, mask, observations=[observation])
        log_probs, _, values = evaluate_policy_actions(model, x, mask, sampled.labels)

        self.assertTrue(torch.allclose(log_probs, sampled.log_probs))
        self.assertTrue(torch.allclose(values, sampled.values))

    def test_train_ppo_smoke_writes_checkpoint_and_metrics(self) -> None:
        """A tiny PPO run should write a resumable checkpoint and JSONL metrics."""
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.ppo import PPOConfig, train_ppo

        encoder = ObservationEncoder(history_length=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            init_path = Path(temp_dir) / "init.pt"
            output_path = Path(temp_dir) / "ppo.pt"
            _write_init_checkpoint(init_path, encoder, BCPolicyNet)

            summary = train_ppo(
                PPOConfig(
                    init_checkpoint=str(init_path),
                    output_path=str(output_path),
                    device="cpu",
                    updates=1,
                    num_envs=1,
                    rollout_steps=2,
                    ppo_epochs=1,
                    minibatch_size=2,
                    map_size=18,
                    max_turns=4,
                    seed=123,
                )
            )

            checkpoint = torch.load(output_path, map_location="cpu")
            metrics_text = Path(summary["metrics_output_path"]).read_text(encoding="utf-8")
            metrics_rows = [json.loads(line) for line in metrics_text.splitlines()]

            self.assertEqual(summary["global_update"], 1)
            self.assertEqual(checkpoint["algo"], "ppo")
            self.assertEqual(checkpoint["global_update"], 1)
            self.assertEqual(checkpoint["history_length"], 1)
            self.assertTrue(_value_head_has_trainable_weights(checkpoint["model_state"]))
            self.assertEqual(len(metrics_rows), 1)
            self.assertIn("policy_loss", metrics_rows[0])
            self.assertIn("value_loss", metrics_rows[0])

            resumed_output = Path(temp_dir) / "ppo_resumed.pt"
            resumed_summary = train_ppo(
                PPOConfig(
                    init_checkpoint=str(init_path),
                    resume_from=str(output_path),
                    output_path=str(resumed_output),
                    device="cpu",
                    updates=2,
                    num_envs=1,
                    rollout_steps=2,
                    ppo_epochs=1,
                    minibatch_size=2,
                    map_size=18,
                    max_turns=4,
                    seed=123,
                )
            )
            resumed_checkpoint = torch.load(resumed_output, map_location="cpu")

            self.assertEqual(resumed_summary["global_update"], 2)
            self.assertEqual(resumed_checkpoint["global_update"], 2)

    def test_opponent_pool_config_resolves_relative_paths(self) -> None:
        """Opponent pool JSON paths should resolve relative to the config file."""
        from generals_bot.training.ppo import load_opponent_pool_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            opponent_path = root / "opponent.pt"
            opponent_path.write_bytes(b"placeholder")
            config_path = root / "pool.json"
            config_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "self_play_weight": 0.25,
                        "opponent_device": "same_as_training",
                        "opponent_pass_bias": 0.0,
                        "opponents": [
                            {"id": "opp", "path": "opponent.pt", "weight": 0.75},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            pool = load_opponent_pool_config(config_path, training_device="cpu")

            self.assertEqual(pool.self_play_weight, 0.25)
            self.assertEqual(pool.opponent_device, "cpu")
            self.assertEqual(len(pool.opponents), 1)
            self.assertEqual(pool.opponents[0].id, "opp")
            self.assertEqual(pool.opponents[0].path, opponent_path.resolve())

    def test_train_ppo_with_opponent_pool_records_only_learner_samples(self) -> None:
        """Pool matchups should train only the learner side, not the frozen opponent."""
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.ppo import PPOConfig, train_ppo

        encoder = ObservationEncoder(history_length=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_path = root / "init.pt"
            output_path = root / "ppo.pt"
            pool_path = root / "pool.json"
            _write_init_checkpoint(init_path, encoder, BCPolicyNet)
            pool_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "self_play_weight": 0.0,
                        "opponent_device": "cpu",
                        "opponent_pass_bias": 0.0,
                        "opponents": [
                            {"id": "fixed", "path": "init.pt", "weight": 1.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = train_ppo(
                PPOConfig(
                    init_checkpoint=str(init_path),
                    output_path=str(output_path),
                    opponent_pool_config=str(pool_path),
                    device="cpu",
                    updates=1,
                    num_envs=1,
                    rollout_steps=2,
                    ppo_epochs=1,
                    minibatch_size=2,
                    map_size=18,
                    max_turns=4,
                    seed=123,
                )
            )
            metrics_text = Path(summary["metrics_output_path"]).read_text(encoding="utf-8")
            metrics_rows = [json.loads(line) for line in metrics_text.splitlines()]

            self.assertEqual(summary["rollout_samples"], 2.0)
            self.assertEqual(metrics_rows[0]["rollout_samples"], 2.0)
            self.assertIn("learner_pool_win_rate", metrics_rows[0])
            self.assertIn("opponent_fixed_episodes", metrics_rows[0])

    def test_distributed_global_env_assignment_splits_all_envs(self) -> None:
        """Global env scope should shard the requested env count across ranks."""
        from generals_bot.training.ppo import _rank_env_assignment

        assignments = [
            _rank_env_assignment(17, world_size=4, rank=rank, scope="global")
            for rank in range(4)
        ]

        self.assertEqual(assignments, [(5, 0), (4, 5), (4, 9), (4, 13)])
        self.assertEqual(sum(count for count, _ in assignments), 17)

    def test_distributed_per_rank_env_assignment_repeats_env_count(self) -> None:
        """Per-rank env scope should give every rank the full requested env count."""
        from generals_bot.training.ppo import _rank_env_assignment

        assignments = [
            _rank_env_assignment(16, world_size=4, rank=rank, scope="per-rank")
            for rank in range(4)
        ]

        self.assertEqual(assignments, [(16, 0), (16, 16), (16, 32), (16, 48)])

    def test_distributed_global_env_assignment_rejects_empty_ranks(self) -> None:
        """Global env scope should reject fewer envs than ranks."""
        from generals_bot.training.ppo import _rank_env_assignment

        with self.assertRaisesRegex(ValueError, "at least world_size"):
            _rank_env_assignment(2, world_size=4, rank=0, scope="global")

    def test_local_minibatch_size_requires_world_size_divisibility(self) -> None:
        """Distributed minibatch splitting should preserve the global batch contract."""
        from generals_bot.training.ppo import _local_minibatch_size

        self.assertEqual(_local_minibatch_size(1024, 4), 256)
        with self.assertRaisesRegex(ValueError, "divisible"):
            _local_minibatch_size(1025, 4)

    def test_distributed_wrapper_allows_unused_prediction_heads(self) -> None:
        """DDP wrapping should allow disabled auxiliary heads to be unused."""
        from generals_bot.training.ppo import DistributedContext, _wrap_model_for_distributed

        captured_kwargs = {}
        original_ddp = torch.nn.parallel.DistributedDataParallel

        class _FakeDDP(torch.nn.Module):
            """Capture DDP constructor kwargs without initializing a process group."""

            def __init__(self, module, **kwargs):
                super().__init__()
                self.module = module
                captured_kwargs.update(kwargs)

        try:
            torch.nn.parallel.DistributedDataParallel = _FakeDDP
            model = torch.nn.Linear(1, 1)
            wrapped = _wrap_model_for_distributed(
                model,
                DistributedContext(
                    enabled=True,
                    rank=0,
                    world_size=2,
                    local_rank=0,
                    device=torch.device("cpu"),
                ),
            )
        finally:
            torch.nn.parallel.DistributedDataParallel = original_ddp

        self.assertIs(wrapped.module, model)
        self.assertTrue(captured_kwargs["find_unused_parameters"])

    def test_advantage_normalization_non_distributed_matches_standardization(self) -> None:
        """Non-distributed advantage normalization should keep the legacy behavior."""
        from generals_bot.training.ppo import _normalize_advantages

        advantages = torch.tensor([1.0, 2.0, 4.0, 8.0])
        normalized = _normalize_advantages(advantages)

        expected = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        self.assertTrue(torch.allclose(normalized, expected))

    def test_rollout_padding_adds_zero_weight_dummy_samples(self) -> None:
        """Distributed padding should keep real samples and mark dummy samples weight 0."""
        from generals_bot.training.ppo import _pad_rollout_batch

        batch = _make_rollout_batch(3)

        padded = _pad_rollout_batch(batch, 5)

        self.assertEqual(padded.size, 5)
        self.assertEqual(padded.real_size, 3)
        self.assertTrue(torch.equal(padded.sample_weight, torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)))
        self.assertEqual(padded.stream_ids[-1], (-1, -1))
        self.assertEqual(padded.metadata[-1]["padding"], 1)
        self.assertTrue(torch.equal(padded.x[3], batch.x[0]))

    def test_distributed_alignment_pads_to_max_rank_count(self) -> None:
        """DDP rollout alignment should pad to the max count instead of truncating."""
        from generals_bot.training.ppo import (
            DistributedContext,
            _align_rollout_batch_for_distributed,
        )

        batch = _make_rollout_batch(3)
        original_all_reduce = torch.distributed.all_reduce

        def fake_all_reduce(tensor: torch.Tensor, op=None) -> None:
            """Pretend another rank has five rollout samples."""
            tensor.fill_(5)

        try:
            torch.distributed.all_reduce = fake_all_reduce
            aligned, padded_samples = _align_rollout_batch_for_distributed(
                batch,
                DistributedContext(
                    enabled=True,
                    rank=0,
                    world_size=2,
                    local_rank=0,
                    device=torch.device("cpu"),
                ),
            )
        finally:
            torch.distributed.all_reduce = original_all_reduce

        self.assertEqual(padded_samples, 2)
        self.assertEqual(aligned.size, 5)
        self.assertEqual(aligned.real_size, 3)
        self.assertEqual(float(aligned.sample_weight[-1]), 0.0)

    def test_weighted_advantage_normalization_ignores_padding(self) -> None:
        """Advantage normalization should compute stats only from real samples."""
        from generals_bot.training.ppo import _normalize_advantages

        advantages = torch.tensor([2.0, 4.0, 1000.0])
        weights = torch.tensor([1.0, 1.0, 0.0])

        normalized = _normalize_advantages(advantages, sample_weight=weights)

        self.assertTrue(torch.allclose(normalized[:2], torch.tensor([-1.0, 1.0])))

    def test_explained_variance_ignores_padding(self) -> None:
        """Explained variance should ignore dummy padding targets."""
        from generals_bot.training.ppo import explained_variance

        targets = torch.tensor([1.0, 3.0, 1000.0])
        predictions = torch.tensor([1.0, 3.0, -1000.0])
        weights = torch.tensor([1.0, 1.0, 0.0])

        self.assertAlmostEqual(
            explained_variance(targets, predictions, sample_weight=weights),
            1.0,
        )

    def test_reward_config_parses_segments_and_zero_defaults(self) -> None:
        """Reward configs should load segment weights and default omitted values to zero."""
        from generals_bot.training.rewards import load_reward_config

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "reward.json"
            config_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "terminal_bonuses": {"city_capture_win": 0.25},
                        "city_capture_lifecycle": {
                            "max_reward_turns_per_capture": 50,
                            "survive_drip_per_turn": 0.002,
                            "hold_drip_per_turn": 0.003,
                            "survive_milestone": 0.04,
                            "hold_milestone": 0.06,
                            "death_penalty": -0.05,
                            "enemy_distance": {
                                "enabled": True,
                                "min_distance": 3,
                                "full_distance": 10,
                                "min_multiplier": 0.25,
                            },
                        },
                        "city_attack_outcome": {
                            "window_turns": 10,
                            "success_reward": 0.02,
                            "failure_penalty": -0.02,
                        },
                        "metrics": {"land": {"mode": "milestone_absolute", "interval": 50}},
                        "segments": [
                            {
                                "start_turn": 0,
                                "end_turn": 100,
                                "weights": {"land": 0.04},
                            },
                            {
                                "start_turn": 100,
                                "end_turn": None,
                                "weights": {"attack": 0.01},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_reward_config(config_path)

            self.assertEqual(config.metrics["land"].interval, 50)
            self.assertEqual(config.terminal_bonuses.city_capture_win, 0.25)
            self.assertEqual(
                config.city_capture_lifecycle.max_reward_turns_per_capture,
                50,
            )
            self.assertEqual(config.city_capture_lifecycle.hold_drip_per_turn, 0.003)
            self.assertTrue(config.city_capture_lifecycle.enemy_distance.enabled)
            self.assertEqual(config.city_capture_lifecycle.enemy_distance.full_distance, 10)
            self.assertEqual(config.city_attack_outcome.window_turns, 10)
            self.assertEqual(config.city_attack_outcome.success_reward, 0.02)
            self.assertEqual(config.city_attack_outcome.failure_penalty, -0.02)
            self.assertEqual(config.weights_for_turn(99).land, 0.04)
            self.assertEqual(config.weights_for_turn(99).city, 0.0)
            self.assertEqual(config.weights_for_turn(99).city_capture, 0.0)
            self.assertEqual(config.weights_for_turn(100).land, 0.0)
            self.assertEqual(config.weights_for_turn(100).attack, 0.01)

            legacy_path = Path(temp_dir) / "legacy_reward.json"
            legacy_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "city_attack_failure": {
                            "window_turns": 10,
                            "penalty": -0.02,
                        },
                    }
                ),
                encoding="utf-8",
            )
            legacy = load_reward_config(legacy_path)
            self.assertEqual(legacy.city_attack_outcome.window_turns, 10)
            self.assertEqual(legacy.city_attack_outcome.success_reward, 0.0)
            self.assertEqual(legacy.city_attack_outcome.failure_penalty, -0.02)

    def test_shaped_reward_uses_relative_log_scores_and_milestones(self) -> None:
        """Land, city, and army rewards should use relative log scores."""
        from generals_bot.training.rewards import (
            RewardCalculator,
            RewardConfig,
            RewardSegment,
            RewardState,
            RewardWeights,
            default_reward_config,
        )

        default = default_reward_config()
        config = RewardConfig(
            terminal=default.terminal,
            metrics=default.metrics,
            segments=(
                RewardSegment(
                    start_turn=0,
                    end_turn=100,
                    weights=RewardWeights(land=0.5, city=0.25, army=0.1),
                ),
            ),
        )
        calculator = RewardCalculator(config)
        before = RewardState(turn=49, army=(10, 10), land=(4, 4), cities=(0, 0))
        after = RewardState(turn=50, army=(15, 10), land=(9, 4), cities=(1, 0))

        rewards = calculator.rewards(
            before=before,
            after=after,
            executed_actions=[],
            winner=None,
            done=False,
        )

        expected_land = 0.5 * (math.log1p(9) - math.log1p(4))
        expected_city = 0.25 * (math.log1p(1) - math.log1p(0))
        expected_army = 0.1 * (math.log1p(15) - math.log1p(10))
        self.assertAlmostEqual(rewards[0].land, expected_land)
        self.assertAlmostEqual(rewards[0].city, expected_city)
        self.assertAlmostEqual(rewards[0].army, expected_army)
        self.assertAlmostEqual(rewards[1].land, -expected_land)
        self.assertAlmostEqual(rewards[1].city, -expected_city)
        self.assertAlmostEqual(rewards[1].army, -expected_army)

        non_milestone = calculator.rewards(
            before=RewardState(turn=50, army=(15, 10), land=(9, 4), cities=(1, 0)),
            after=RewardState(turn=51, army=(15, 10), land=(9, 4), cities=(1, 0)),
            executed_actions=[],
            winner=None,
            done=False,
        )
        self.assertEqual(non_milestone[0].land, 0.0)

    def test_attack_reward_counts_only_enemy_damage(self) -> None:
        """Attack reward should ignore neutral attacks and reward enemy damage."""
        from generals_bot.training.rewards import (
            RewardCalculator,
            RewardConfig,
            RewardSegment,
            RewardState,
            RewardWeights,
            default_reward_config,
        )

        default = default_reward_config()
        config = RewardConfig(
            terminal=default.terminal,
            metrics=default.metrics,
            segments=(
                RewardSegment(
                    start_turn=0,
                    end_turn=None,
                    weights=RewardWeights(attack=0.2),
                ),
            ),
        )
        calculator = RewardCalculator(config)
        state = RewardState(turn=10, army=(10, 10), land=(2, 2), cities=(0, 0))
        enemy_attack = ExecutedAction(
            player_id=0,
            action=Action(0, 0, 1),
            turn=10,
            valid=True,
            target_owner_before=1,
            target_army_before=5,
            moved_army=3,
            damage_done=3,
        )
        neutral_attack = ExecutedAction(
            player_id=0,
            action=Action(0, 0, 2),
            turn=10,
            valid=True,
            target_owner_before=-1,
            target_army_before=5,
            moved_army=3,
            damage_done=0,
        )

        rewards = calculator.rewards(
            before=state,
            after=state,
            executed_actions=[enemy_attack, neutral_attack],
            winner=None,
            done=False,
        )

        expected = 0.2 * math.log1p(3)
        self.assertAlmostEqual(rewards[0].attack, expected)
        self.assertAlmostEqual(rewards[1].attack, -expected)

    def test_city_capture_reward_counts_only_successful_neutral_city_captures(self) -> None:
        """City capture reward should ignore failed city attacks and non-city moves."""
        from generals_bot.training.rewards import (
            RewardCalculator,
            RewardConfig,
            RewardSegment,
            RewardState,
            TerminalBonusConfig,
            RewardWeights,
            default_reward_config,
        )

        default = default_reward_config()
        config = RewardConfig(
            terminal=default.terminal,
            metrics=default.metrics,
            segments=(
                RewardSegment(
                    start_turn=0,
                    end_turn=None,
                    weights=RewardWeights(city_capture=0.2),
                ),
            ),
            terminal_bonuses=TerminalBonusConfig(city_capture_win=0.25),
        )
        calculator = RewardCalculator(config)
        state = RewardState(
            turn=10,
            army=(10, 10),
            land=(2, 2),
            cities=(0, 0),
            city_tiles=(4,),
        )
        captured_city = ExecutedAction(
            player_id=0,
            action=Action(0, 0, 4),
            turn=10,
            valid=True,
            target_owner_before=-1,
            target_army_before=20,
            moved_army=21,
            damage_done=0,
        )
        failed_city_attack = ExecutedAction(
            player_id=0,
            action=Action(0, 1, 4),
            turn=10,
            valid=True,
            target_owner_before=-1,
            target_army_before=20,
            moved_army=20,
            damage_done=0,
        )
        non_city_capture = ExecutedAction(
            player_id=0,
            action=Action(0, 2, 5),
            turn=10,
            valid=True,
            target_owner_before=-1,
            target_army_before=1,
            moved_army=2,
            damage_done=0,
        )
        enemy_city_attack = ExecutedAction(
            player_id=0,
            action=Action(0, 3, 4),
            turn=10,
            valid=True,
            target_owner_before=1,
            target_army_before=20,
            moved_army=21,
            damage_done=20,
        )

        rewards = calculator.rewards(
            before=state,
            after=state,
            executed_actions=[
                captured_city,
                failed_city_attack,
                non_city_capture,
                enemy_city_attack,
            ],
            winner=None,
            done=False,
        )

        expected = 0.2 * (1.0 + math.log1p(20) / math.log1p(50))
        self.assertAlmostEqual(rewards[0].city_capture, expected)
        self.assertAlmostEqual(rewards[1].city_capture, -expected)

        terminal_rewards = calculator.rewards(
            before=state,
            after=state,
            executed_actions=[],
            winner=0,
            done=True,
            episode_city_captures={0: 3, 1: 2},
        )

        self.assertAlmostEqual(
            terminal_rewards[0].city_capture_win,
            0.25 * math.log1p(3),
        )
        self.assertEqual(terminal_rewards[1].city_capture_win, 0.0)
        self.assertAlmostEqual(
            terminal_rewards[0].total,
            1.0 + 0.25 * math.log1p(3),
        )

    def test_city_capture_distance_modifier_reduces_close_enemy_rewards(self) -> None:
        """City capture reward should be reduced when enemy tiles are nearby."""
        from generals_bot.training.rewards import (
            CityCaptureLifecycleConfig,
            EnemyDistanceConfig,
            RewardCalculator,
            RewardConfig,
            RewardSegment,
            RewardState,
            RewardWeights,
            default_reward_config,
        )

        default = default_reward_config()
        config = RewardConfig(
            terminal=default.terminal,
            metrics=default.metrics,
            segments=(
                RewardSegment(
                    start_turn=0,
                    end_turn=None,
                    weights=RewardWeights(city_capture=0.2),
                ),
            ),
            city_capture_lifecycle=CityCaptureLifecycleConfig(
                enemy_distance=EnemyDistanceConfig(
                    enabled=True,
                    min_distance=3,
                    full_distance=10,
                    min_multiplier=0.25,
                ),
            ),
        )
        calculator = RewardCalculator(config)
        before = RewardState(
            turn=10,
            army=(10, 10),
            land=(2, 2),
            cities=(0, 0),
            city_tiles=(12,),
        )
        terrain = np.full((5, 5), -1, dtype=int)
        armies = np.zeros((5, 5), dtype=int)
        terrain[2, 2] = 0
        terrain[2, 3] = 1
        armies[2, 3] = 5
        after = RewardState(
            turn=11,
            army=(10, 10),
            land=(2, 2),
            cities=(1, 0),
            city_tiles=(12,),
            width=5,
            height=5,
            terrain=terrain,
            armies_grid=armies,
        )
        captured_city = ExecutedAction(
            player_id=0,
            action=Action(0, 0, 12),
            turn=10,
            valid=True,
            target_owner_before=-1,
            target_army_before=20,
            moved_army=21,
        )

        rewards = calculator.rewards(
            before=before,
            after=after,
            executed_actions=[captured_city],
            winner=None,
            done=False,
        )

        base_reward = 0.2 * (1.0 + math.log1p(20) / math.log1p(50))
        self.assertAlmostEqual(rewards[0].city_capture, base_reward * 0.25)
        self.assertEqual(rewards[0].city_capture_distance_multiplier, 0.25)

    def test_city_capture_lifecycle_rewards_drip_milestones_and_death(self) -> None:
        """City capture lifecycle should reward survival/holding and penalize early death."""
        from generals_bot.training.rewards import (
            CityCaptureLifecycleConfig,
            CityCaptureLifecycleEvent,
            RewardCalculator,
            RewardConfig,
            RewardState,
            default_reward_config,
        )

        default = default_reward_config()
        config = RewardConfig(
            terminal=default.terminal,
            metrics=default.metrics,
            segments=(),
            city_capture_lifecycle=CityCaptureLifecycleConfig(
                max_reward_turns_per_capture=3,
                survive_drip_per_turn=0.2,
                hold_drip_per_turn=0.3,
                survive_milestone=0.4,
                hold_milestone=0.6,
                death_penalty=-0.5,
            ),
        )
        calculator = RewardCalculator(config)
        terrain = np.full((3, 3), -1, dtype=int)
        terrain[1, 1] = 0
        alive_state = RewardState(
            turn=1,
            army=(10, 10),
            land=(2, 2),
            cities=(1, 0),
            width=3,
            height=3,
            terrain=terrain,
        )
        events = [
            CityCaptureLifecycleEvent(
                player_id=0,
                city_tile=4,
                capture_turn=0,
                end_turn=3,
            )
        ]

        rewards, events = calculator.city_capture_lifecycle_rewards(
            events,
            alive_state,
        )
        self.assertAlmostEqual(rewards[0].city_capture_survive_drip, 0.2)
        self.assertAlmostEqual(rewards[0].city_capture_hold_drip, 0.3)
        self.assertEqual(len(events), 1)

        due_state = RewardState(
            turn=3,
            army=(10, 10),
            land=(2, 2),
            cities=(1, 0),
            width=3,
            height=3,
            terrain=terrain,
        )
        due_rewards, events = calculator.city_capture_lifecycle_rewards(
            events,
            due_state,
        )
        self.assertAlmostEqual(due_rewards[0].city_capture_survive_drip, 0.2)
        self.assertAlmostEqual(due_rewards[0].city_capture_hold_drip, 0.3)
        self.assertAlmostEqual(due_rewards[0].city_capture_survive_milestone, 0.4)
        self.assertAlmostEqual(due_rewards[0].city_capture_hold_milestone, 0.6)
        self.assertEqual(events, [])

        death_state = RewardState(
            turn=2,
            army=(10, 10),
            land=(2, 2),
            cities=(1, 0),
            alive=(False, True),
            width=3,
            height=3,
            terrain=terrain,
        )
        death_rewards, events = calculator.city_capture_lifecycle_rewards(
            [
                CityCaptureLifecycleEvent(
                    player_id=0,
                    city_tile=4,
                    capture_turn=0,
                    end_turn=3,
                )
            ],
            death_state,
        )
        self.assertAlmostEqual(death_rewards[0].city_capture_death_penalty, -0.5)
        self.assertEqual(events, [])

    def test_city_attack_outcome_rewards_resolve_success_and_failure(self) -> None:
        """City attack outcome events should reward ownership and penalize failure."""
        from generals_bot.training.rewards import (
            CityAttackOutcomeConfig,
            RewardCalculator,
            RewardConfig,
            RewardState,
            default_reward_config,
        )

        default = default_reward_config()
        config = RewardConfig(
            terminal=default.terminal,
            metrics=default.metrics,
            segments=(),
            city_attack_outcome=CityAttackOutcomeConfig(
                window_turns=10,
                success_reward=0.02,
                failure_penalty=-0.02,
            ),
        )
        calculator = RewardCalculator(config)
        before = RewardState(
            turn=10,
            army=(10, 10),
            land=(2, 2),
            cities=(0, 0),
            city_tiles=(4,),
        )
        failed_city_attack = ExecutedAction(
            player_id=0,
            action=Action(0, 0, 4),
            turn=10,
            valid=True,
            target_owner_before=-1,
            target_army_before=20,
            moved_army=20,
        )
        captured_city = ExecutedAction(
            player_id=0,
            action=Action(0, 1, 4),
            turn=10,
            valid=True,
            target_owner_before=-1,
            target_army_before=20,
            moved_army=21,
        )
        enemy_city_attack = ExecutedAction(
            player_id=0,
            action=Action(0, 2, 4),
            turn=10,
            valid=True,
            target_owner_before=1,
            target_army_before=20,
            moved_army=20,
        )

        events = calculator.city_attack_outcome_events(
            [failed_city_attack, captured_city, enemy_city_attack],
            before.city_tiles,
            attack_turn=before.turn,
            reward_indices_by_player={0: 42},
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].due_turn, 20)
        self.assertEqual(events[0].reward_index, 42)

        neutral_terrain = np.full((3, 3), -1, dtype=int)
        pending_rewards, pending_events = calculator.city_attack_outcome_rewards(
            events,
            RewardState(
                turn=15,
                army=(10, 10),
                land=(2, 2),
                cities=(0, 0),
                width=3,
                height=3,
                terrain=neutral_terrain,
            ),
        )
        self.assertEqual(pending_rewards, [])
        self.assertEqual(len(pending_events), 1)

        owned_terrain = np.full((3, 3), -1, dtype=int)
        owned_terrain[1, 1] = 0
        success_rewards, success_events = calculator.city_attack_outcome_rewards(
            pending_events,
            RewardState(
                turn=19,
                army=(10, 10),
                land=(2, 2),
                cities=(1, 0),
                width=3,
                height=3,
                terrain=owned_terrain,
            ),
        )
        self.assertEqual(len(success_rewards), 1)
        self.assertEqual(success_rewards[0][0].reward_index, 42)
        self.assertAlmostEqual(success_rewards[0][1].city_attack_success, 0.02)
        self.assertEqual(success_events, [])

        due_rewards, due_events = calculator.city_attack_outcome_rewards(
            events,
            RewardState(
                turn=20,
                army=(10, 10),
                land=(2, 2),
                cities=(0, 0),
                width=3,
                height=3,
                terrain=neutral_terrain,
            ),
        )
        self.assertEqual(len(due_rewards), 1)
        self.assertEqual(due_rewards[0][0].reward_index, 42)
        self.assertAlmostEqual(due_rewards[0][1].city_attack_failure, -0.02)
        self.assertEqual(due_events, [])

    def test_train_ppo_with_reward_config_records_components(self) -> None:
        """A tiny shaped PPO run should record reward component metrics."""
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.ppo import PPOConfig, train_ppo

        encoder = ObservationEncoder(history_length=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_path = root / "init.pt"
            output_path = root / "ppo.pt"
            reward_config_path = root / "reward.json"
            _write_init_checkpoint(init_path, encoder, BCPolicyNet)
            reward_config_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "segments": [
                            {
                                "start_turn": 0,
                                "end_turn": None,
                                "weights": {
                                    "land": 0.01,
                                    "city": 0.01,
                                    "city_capture": 0.01,
                                    "army": 0.01,
                                    "attack": 0.01,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = train_ppo(
                PPOConfig(
                    init_checkpoint=str(init_path),
                    output_path=str(output_path),
                    reward_config_path=str(reward_config_path),
                    device="cpu",
                    updates=1,
                    num_envs=1,
                    rollout_steps=2,
                    ppo_epochs=1,
                    minibatch_size=2,
                    map_size=18,
                    max_turns=4,
                    seed=123,
                )
            )

            metrics_text = Path(summary["metrics_output_path"]).read_text(encoding="utf-8")
            metrics_rows = [json.loads(line) for line in metrics_text.splitlines()]
            checkpoint = torch.load(output_path, map_location="cpu")

            self.assertIn("reward_total_mean", metrics_rows[0])
            self.assertIn("reward_land_mean", metrics_rows[0])
            self.assertIn("reward_city_mean", metrics_rows[0])
            self.assertIn("reward_city_capture_mean", metrics_rows[0])
            self.assertIn("reward_city_capture_win_mean", metrics_rows[0])
            self.assertIn("reward_city_capture_survive_drip_mean", metrics_rows[0])
            self.assertIn("reward_city_capture_hold_drip_mean", metrics_rows[0])
            self.assertIn("reward_city_capture_survive_milestone_mean", metrics_rows[0])
            self.assertIn("reward_city_capture_hold_milestone_mean", metrics_rows[0])
            self.assertIn("reward_city_capture_death_penalty_mean", metrics_rows[0])
            self.assertIn("reward_city_capture_lifecycle_mean", metrics_rows[0])
            self.assertIn("reward_city_capture_distance_multiplier_mean", metrics_rows[0])
            self.assertIn("reward_city_attack_success_mean", metrics_rows[0])
            self.assertIn("reward_city_attack_failure_mean", metrics_rows[0])
            self.assertIn("reward_army_mean", metrics_rows[0])
            self.assertIn("reward_attack_mean", metrics_rows[0])
            self.assertEqual(
                checkpoint["ppo_config"]["reward_config_path"],
                str(reward_config_path),
            )

    def test_train_ppo_with_enemy_king_loss_records_metrics(self) -> None:
        """A tiny PPO run should train and log the enemy king auxiliary head."""
        from generals_bot.training.model import BCPolicyNet
        from generals_bot.training.ppo import PPOConfig, train_ppo

        encoder = ObservationEncoder(history_length=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            init_path = Path(temp_dir) / "init.pt"
            output_path = Path(temp_dir) / "ppo.pt"
            _write_init_checkpoint(init_path, encoder, BCPolicyNet)

            summary = train_ppo(
                PPOConfig(
                    init_checkpoint=str(init_path),
                    output_path=str(output_path),
                    device="cpu",
                    updates=1,
                    num_envs=1,
                    rollout_steps=2,
                    ppo_epochs=1,
                    minibatch_size=2,
                    map_size=18,
                    max_turns=4,
                    seed=123,
                    enemy_king_loss_weight=0.1,
                )
            )

            metrics_text = Path(summary["metrics_output_path"]).read_text(encoding="utf-8")
            metrics_rows = [json.loads(line) for line in metrics_text.splitlines()]

            self.assertIn("enemy_king_loss", metrics_rows[0])
            self.assertIn("enemy_king_distance_loss", metrics_rows[0])


class _FixedPolicyModel(_BaseModule):
    """Return deterministic policy tensors for PPO unit tests."""

    def __init__(
        self,
        *,
        move_logits: torch.Tensor,
        action_logits: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Store fixed outputs with the same keys as BCPolicyNet."""
        super().__init__()
        self.move_logits = move_logits
        self.action_logits = action_logits
        self.values = values

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return fixed outputs expanded to the input batch size."""
        batch = x.shape[0]
        return {
            "move_logits": self.move_logits[:batch].to(x.device),
            "action_logits": self.action_logits[:batch].to(x.device),
            "value": self.values[:batch].to(x.device),
        }


def _write_init_checkpoint(path: Path, encoder: ObservationEncoder, model_cls) -> None:
    """Write a tiny hierarchical checkpoint usable as a PPO initialization."""
    model = model_cls(encoder.num_channels, base_channels=8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.pass_head[-1].bias.fill_(3.0)
        model.value_head[-1].bias.fill_(7.0)
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


def _make_rollout_batch(size: int):
    """Create a minimal rollout batch for PPO helper tests."""
    from generals_bot.training.ppo import RolloutBatch

    return RolloutBatch(
        x=torch.arange(size, dtype=torch.float32).reshape(size, 1),
        valid_action_mask=torch.ones((size, 2), dtype=torch.bool),
        action_label=torch.zeros(size, dtype=torch.long),
        old_log_prob=torch.zeros(size, dtype=torch.float32),
        old_value=torch.zeros(size, dtype=torch.float32),
        enemy_king_label=torch.zeros(size, dtype=torch.long),
        enemy_king_distance=torch.zeros((size, 2), dtype=torch.float32),
        enemy_king_valid_mask=torch.ones((size, 2), dtype=torch.bool),
        rewards=torch.zeros(size, dtype=torch.float32),
        done=torch.zeros(size, dtype=torch.bool),
        returns=torch.zeros(size, dtype=torch.float32),
        advantages=torch.arange(size, dtype=torch.float32),
        sample_weight=torch.ones(size, dtype=torch.float32),
        stream_ids=[(index, 0) for index in range(size)],
        metadata=[
            {"env_id": index, "player_id": 0, "turn": index}
            for index in range(size)
        ],
    )


def _value_head_has_trainable_weights(state: dict[str, torch.Tensor]) -> bool:
    """Return whether the saved value head has nonzero hidden weights."""
    return any(
        not torch.allclose(value, torch.zeros_like(value))
        for key, value in state.items()
        if key.startswith("value_head.") and key.endswith(".weight")
    )


if __name__ == "__main__":
    unittest.main()
