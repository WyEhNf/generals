from __future__ import annotations

import unittest

from generals_bot.agents import ExpanderPolicy
from generals_bot.sim import GameConfig, GeneralsEnv, generate_1v1_map
from generals_bot.viz import MatchRecorder, render_html


class ExpanderMatchTest(unittest.TestCase):
    def test_expander_vs_expander_100_ticks_visualization(self) -> None:
        config = GameConfig(
            width=9,
            height=9,
            generals=(0, 80),
            mountains=(20, 22, 38, 42, 58, 60),
            cities=(40,),
            city_armies=(20,),
            max_turns=100,
        )
        env = GeneralsEnv(config)
        observations = env.reset(seed=0)
        policies = {
            0: ExpanderPolicy(seed=0),
            1: ExpanderPolicy(seed=1),
        }
        for player, policy in policies.items():
            policy.reset(player, config)

        recorder = MatchRecorder()
        recorder.capture(env)
        while not env.done:
            actions = {
                player: policies[player].act(observations[player])
                for player in env.alive_players
            }
            observations, _, _, _ = env.step(actions)
            recorder.capture(env)

        self.assertEqual(env.state.turn, 100)
        self.assertEqual(len(recorder.snapshots), 101)
        html = render_html(recorder.snapshots)
        self.assertIn('"turn":100', html)
        self.assertIn("Generals Local Replay", html)
        self.assertIn("playback-rate", html)
        self.assertIn("p0-army", html)

    def test_expander_vs_expander_generated_map_smoke(self) -> None:
        config = generate_1v1_map(seed=0, max_turns=100)
        env = GeneralsEnv(config)
        observations = env.reset(seed=0)
        policies = {
            0: ExpanderPolicy(seed=0),
            1: ExpanderPolicy(seed=1),
        }
        for player, policy in policies.items():
            policy.reset(player, config)

        recorder = MatchRecorder()
        recorder.capture(env)
        while not env.done:
            actions = {
                player: policies[player].act(observations[player])
                for player in env.alive_players
            }
            observations, _, _, _ = env.step(actions)
            recorder.capture(env)

        self.assertLessEqual(env.state.turn, 100)
        html = render_html(recorder.snapshots)
        self.assertIn("Generals Local Replay", html)
        self.assertIn("playback-rate", html)


if __name__ == "__main__":
    unittest.main()
