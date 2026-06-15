from __future__ import annotations

import unittest

import numpy as np

from generals_bot.agents import (
    EnemyKingDistribution,
    ExpanderPolicy,
    MarchPolicy,
    PassPolicy,
    WinRateEstimate,
    estimate_enemy_king_distribution,
    estimate_win_rate,
)
from generals_bot.sim import Action, Direction, GameConfig, GeneralsEnv, PassAction
from generals_bot.viz import MatchRecorder, render_html
from generals_bot.viz.html import _snapshot_to_dict


class AgentsAndVizTest(unittest.TestCase):
    def test_expander_returns_action_for_frontier(self) -> None:
        config = GameConfig(
            width=3,
            height=1,
            generals=(0, 2),
            starting_armies={0: 5, 2: 1},
        )
        env = GeneralsEnv(config)
        obs = env.reset()[0]
        policy = ExpanderPolicy(seed=0)
        policy.reset(0, config)

        action = policy.act(obs)

        self.assertIsInstance(action, Action)
        self.assertEqual(action.player_id, 0)
        self.assertEqual(action.start, 0)
        self.assertEqual(action.end, 1)

    def test_march_waits_then_moves_largest_stack(self) -> None:
        config = GameConfig(
            width=4,
            height=1,
            generals=(0, 3),
            starting_armies={0: 5, 1: 20, 3: 1},
            starting_owners={1: 0},
        )
        env = GeneralsEnv(config)
        obs = env.reset()[0]
        policy = MarchPolicy(wait_turns=10, direction=Direction.RIGHT)
        policy.reset(0, config)

        waiting_action = policy.act(obs)
        self.assertIsInstance(waiting_action, PassAction)

        while env.state.turn < 10:
            obs = env.step({})[0][0]

        action = policy.act(obs)
        self.assertIsInstance(action, Action)
        self.assertEqual(action.start, 1)
        self.assertEqual(action.end, 2)

    def test_visualization_smoke(self) -> None:
        config = GameConfig(
            width=3,
            height=1,
            generals=(0, 2),
            starting_armies={0: 5, 2: 1},
            max_turns=4,
        )
        env = GeneralsEnv(config)
        observations = env.reset()
        policy = ExpanderPolicy(seed=0)
        policy.reset(0, config)
        recorder = MatchRecorder()
        recorder.capture(env)

        observations, _, _, _ = env.step({0: policy.act(observations[0])})
        recorder.capture(env)
        env.step({})
        recorder.capture(env)

        html = render_html(recorder.snapshots)
        self.assertIn("Generals Local Replay", html)
        self.assertIn("snapshots", html)
        self.assertIn("playback-rate", html)
        self.assertIn("current-turn", html)
        self.assertIn("p0-army", html)
        self.assertIn("p1-land", html)
        self.assertIn("Global", html)
        self.assertIn("P0 POV", html)
        self.assertIn("P1 POV", html)
        self.assertIn('"observations"', html)
        self.assertIn("N/A", html)
        self.assertIn('"turn":2', html)

    def test_rule_based_policy_win_rate_estimate_defaults_to_none(self) -> None:
        config = GameConfig(width=3, height=1, generals=(0, 2))
        env = GeneralsEnv(config)
        observation = env.reset()[0]
        policy = PassPolicy()
        policy.reset(0, config)

        self.assertIsNone(estimate_win_rate(policy, observation))

    def test_rule_based_policy_enemy_king_distribution_defaults_to_none(self) -> None:
        """Rule-based policies should not expose an enemy king probability map."""
        config = GameConfig(width=3, height=1, generals=(0, 2))
        env = GeneralsEnv(config)
        observation = env.reset()[0]
        policy = PassPolicy()
        policy.reset(0, config)

        self.assertIsNone(estimate_enemy_king_distribution(policy, observation))

    def test_visualization_renders_win_rate_estimates(self) -> None:
        config = GameConfig(width=3, height=1, generals=(0, 2))
        env = GeneralsEnv(config)
        env.reset()
        recorder = MatchRecorder()
        recorder.capture(
            env,
            win_rates={
                0: WinRateEstimate(win_probability=0.75, raw_value=0.5),
                1: None,
            },
        )

        data = _snapshot_to_dict(recorder.snapshots[0])
        html = render_html(recorder.snapshots)

        self.assertEqual(data["win_rates"]["0"]["win_probability"], 0.75)
        self.assertIsNone(data["win_rates"]["1"])
        self.assertIn('"win_probability":0.75', html)
        self.assertIn("formatWinRate", html)
        self.assertIn("N/A", html)

    def test_visualization_renders_enemy_king_probability_maps(self) -> None:
        """Recorded enemy king distributions should be serialized and rendered."""
        config = GameConfig(width=3, height=1, generals=(0, 2))
        env = GeneralsEnv(config)
        env.reset()
        probabilities = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
        recorder = MatchRecorder()
        recorder.capture(
            env,
            enemy_king_distributions={
                0: EnemyKingDistribution(probabilities=probabilities),
                1: None,
            },
        )

        data = _snapshot_to_dict(recorder.snapshots[0])
        html = render_html(recorder.snapshots)

        self.assertEqual(
            data["enemy_king_distributions"]["0"]["probabilities"],
            [[0.10000000149011612, 0.20000000298023224, 0.699999988079071]],
        )
        self.assertIsNone(data["enemy_king_distributions"]["1"])
        self.assertIn('"enemy_king_distributions"', html)
        self.assertIn("renderProbabilityMap", html)
        self.assertIn("formatProbability", html)
        self.assertIn("max-probability", html)

    def test_snapshot_serializes_player_observations_without_hidden_armies(self) -> None:
        config = GameConfig(
            width=5,
            height=1,
            generals=(0, 4),
            starting_armies={0: 5, 4: 9},
            max_turns=4,
        )
        env = GeneralsEnv(config)
        env.reset()
        recorder = MatchRecorder()
        recorder.capture(env)

        data = _snapshot_to_dict(recorder.snapshots[0])
        p0 = data["observations"]["0"]
        p1 = data["observations"]["1"]

        self.assertEqual(p0["owner"][0][4], -3)
        self.assertEqual(p0["armies"][0][4], 0)
        self.assertEqual(p1["owner"][0][0], -3)
        self.assertEqual(p1["armies"][0][0], 0)
        self.assertEqual(p0["owner"][0][0], 0)
        self.assertEqual(p1["owner"][0][4], 0)


if __name__ == "__main__":
    unittest.main()
