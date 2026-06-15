from __future__ import annotations

import unittest

import numpy as np

from generals_bot.agents import (
    EnemyKingDistribution,
    ExpanderPolicy,
    PassPolicy,
    WinRateEstimate,
)
from generals_bot.matches import make_policy, parse_agent_spec, parse_agent_specs, run_match
from generals_bot.sim import GameConfig
from generals_bot.viz import render_html


class MatchRunnerTest(unittest.TestCase):
    """Tests for the common local match runner and policy factory."""

    def test_run_match_without_snapshots_reports_metrics(self) -> None:
        """A match can run without retaining visualization snapshots."""
        config = GameConfig(width=3, height=1, generals=(0, 2), max_turns=4)
        policies = {0: ExpanderPolicy(seed=0), 1: ExpanderPolicy(seed=1)}

        result = run_match(
            config=config,
            policies=policies,
            side_labels={0: "a", 1: "b"},
            seed=7,
            record_snapshots=False,
        )

        self.assertEqual(result.snapshots, [])
        self.assertEqual(result.report["seed"], 7)
        self.assertEqual(result.report["side_labels"], {0: "a", 1: "b"})
        self.assertIn("submitted_actions", result.report)
        self.assertIn("invalid_actions", result.report)
        self.assertIn("max_both_pass_streak", result.report)

    def test_run_match_with_snapshots_renders_html(self) -> None:
        """Recorded snapshots should be directly renderable by the visualizer."""
        config = GameConfig(width=3, height=1, generals=(0, 2), max_turns=2)
        policies = {0: PassPolicy(), 1: PassPolicy()}

        result = run_match(
            config=config,
            policies=policies,
            side_labels={0: "pass0", 1: "pass1"},
            seed=0,
            record_snapshots=True,
        )
        html = render_html(result.snapshots)

        self.assertGreaterEqual(len(result.snapshots), 1)
        self.assertIn("Generals Local Replay", html)
        self.assertIn("P0 POV", html)
        self.assertIn("P1 POV", html)

    def test_run_match_records_optional_win_rate_estimates(self) -> None:
        """Policies with a value estimate should write it into snapshots."""
        config = GameConfig(width=3, height=1, generals=(0, 2), max_turns=1)
        policies = {0: _EstimatingPassPolicy(), 1: PassPolicy()}

        result = run_match(
            config=config,
            policies=policies,
            side_labels={0: "estimate", 1: "pass"},
            record_snapshots=True,
        )

        self.assertEqual(result.snapshots[0].win_rates[0], WinRateEstimate(0.75, 0.5))
        self.assertIsNone(result.snapshots[0].win_rates[1])

    def test_run_match_records_optional_enemy_king_distributions(self) -> None:
        """Policies with an enemy king distribution should write it into snapshots."""
        config = GameConfig(width=3, height=1, generals=(0, 2), max_turns=1)
        policies = {0: _KingDistributionPassPolicy(), 1: PassPolicy()}

        result = run_match(
            config=config,
            policies=policies,
            side_labels={0: "king", 1: "pass"},
            record_snapshots=True,
        )

        distribution = result.snapshots[0].enemy_king_distributions[0]
        self.assertIsNotNone(distribution)
        self.assertEqual(distribution.probabilities.shape, (1, 3))
        self.assertIsNone(result.snapshots[0].enemy_king_distributions[1])

    def test_parse_agent_specs_requires_unique_labels(self) -> None:
        """Agent specs should infer labels and reject duplicates."""
        specs = parse_agent_specs([
            "expander",
            "flobot",
            "righty=march-right",
            "ppo=checkpoint:checkpoints/ppo_upd270/model.pt",
        ])

        self.assertEqual(specs["expander"].kind, "expander")
        self.assertEqual(specs["flobot"].kind, "flobot")
        self.assertEqual(specs["righty"].kind, "march-right")
        self.assertEqual(specs["ppo"].kind, "checkpoint")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_agent_specs(["same=expander", "same=march"])

    def test_checkpoint_agent_spec_accepts_enemy_king_predictor_option(self) -> None:
        """Checkpoint specs can attach a standalone enemy king predictor path."""
        spec = parse_agent_spec(
            "checkpoint:checkpoints/policy/model.pt,"
            "enemy_king_predictor=checkpoints/predictor/model.pt"
        )
        labeled = parse_agent_spec(
            "policy=checkpoint:checkpoints/policy/model.pt,"
            "enemy_king_predictor=checkpoints/predictor/model.pt"
        )

        self.assertEqual(spec.label, "policy")
        self.assertEqual(spec.value, "checkpoints/policy/model.pt")
        self.assertEqual(
            spec.enemy_king_predictor_path,
            "checkpoints/predictor/model.pt",
        )
        self.assertEqual(labeled.label, "policy")
        self.assertEqual(labeled.value, "checkpoints/policy/model.pt")
        self.assertEqual(
            labeled.enemy_king_predictor_path,
            "checkpoints/predictor/model.pt",
        )

    def test_router_agent_spec_is_batched_capable(self) -> None:
        """Router specs should parse as batched-capable model agents."""
        spec = parse_agent_spec("route=router:checkpoints/router/model.pt")

        self.assertEqual(spec.label, "route")
        self.assertEqual(spec.kind, "router")
        self.assertEqual(spec.value, "checkpoints/router/model.pt")
        self.assertTrue(spec.supports_batched)

    def test_make_policy_creates_rule_based_policy(self) -> None:
        """Rule-based specs should instantiate current Policy implementations."""
        config = GameConfig(width=3, height=1, generals=(0, 2))
        spec = parse_agent_spec("march-right")

        policy = make_policy(spec, player=0, seed=0, config=config)
        result = run_match(
            config=config,
            policies={0: policy, 1: PassPolicy()},
            side_labels={0: "march-right", 1: "pass"},
            record_snapshots=False,
        )

        self.assertIsNotNone(policy)
        self.assertEqual(result.report["side_labels"][0], "march-right")


class _EstimatingPassPolicy(PassPolicy):
    """Pass policy with a fixed value estimate for snapshot tests."""

    def estimate_win_rate(self, observation) -> WinRateEstimate:
        """Return a deterministic estimate for the current observation."""
        del observation
        return WinRateEstimate(win_probability=0.75, raw_value=0.5)


class _KingDistributionPassPolicy(PassPolicy):
    """Pass policy with a fixed enemy king distribution for snapshot tests."""

    def estimate_enemy_king_distribution(self, observation) -> EnemyKingDistribution:
        """Return a deterministic distribution for the current observation."""
        return EnemyKingDistribution(
            probabilities=np.full(
                (observation.height, observation.width),
                1.0 / (observation.height * observation.width),
                dtype=np.float32,
            )
        )


if __name__ == "__main__":
    unittest.main()
