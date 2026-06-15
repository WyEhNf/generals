from __future__ import annotations

import unittest

from generals_bot.agents import PassPolicy
from generals_bot.benchmark_bots.flobot import FlobotPolicy
from generals_bot.benchmark_bots.flobot.algorithms import FlobotAlgorithms
from generals_bot.benchmark_bots.flobot.game_map import FlobotMap
from generals_bot.benchmark_bots.flobot.state import FlobotState
from generals_bot.matches import run_match
from generals_bot.sim import Action, GameConfig, GeneralsEnv, PassAction


class FlobotPolicyTest(unittest.TestCase):
    """Tests for the Flobot benchmark bot port."""

    def test_known_cities_are_exposed_in_observation(self) -> None:
        """Observation should remember seen cities for Flobot's city list."""
        env = GeneralsEnv(
            GameConfig(
                width=3,
                height=3,
                generals=(0, 8),
                cities=(4,),
                city_armies=(10,),
            )
        )

        observation = env.reset()[0]

        self.assertTrue(observation.cities[1, 1])
        self.assertTrue(observation.known_cities[1, 1])

    def test_flobot_can_run_in_match_runner(self) -> None:
        """Flobot should satisfy the local Policy interface."""
        config = GameConfig(width=5, height=5, generals=(0, 24), max_turns=10)

        result = run_match(
            config=config,
            policies={0: FlobotPolicy(), 1: PassPolicy()},
            side_labels={0: "flobot", 1: "pass"},
            seed=0,
            record_snapshots=False,
        )

        self.assertEqual(result.report["side_labels"][0], "flobot")
        self.assertEqual(result.report["turns"], 10)

    def test_spread_can_emit_action_sequence(self) -> None:
        """Flobot's spread strategy may emit a sequence of moves in one act call."""
        env = GeneralsEnv(GameConfig(width=5, height=5, generals=(12, 24)))
        observations = env.reset()
        policy = FlobotPolicy()
        policy.reset(0, env.config)
        self.assertIsInstance(policy.act(observations[0]), PassAction)
        assert env.state is not None
        env.state.turn = 50
        env.state.armies[2, 2] = 10

        action_like = policy.act(env.observations()[0])

        self.assertIsInstance(action_like, list)
        self.assertGreaterEqual(len(action_like), 1)
        self.assertTrue(all(isinstance(action, Action) for action in action_like))

    def test_visible_enemy_general_is_attacked_when_killable(self) -> None:
        """Flobot should attack an adjacent visible enemy general with enough army."""
        env = GeneralsEnv(
            GameConfig(
                width=3,
                height=3,
                generals=(4, 5),
                starting_armies={4: 10, 5: 1},
            )
        )
        observations = env.reset()
        policy = FlobotPolicy()
        policy.reset(0, env.config)
        self.assertIsInstance(policy.act(observations[0]), PassAction)

        action_like = policy.act(env.observations()[0])

        self.assertIsInstance(action_like, list)
        self.assertEqual(action_like[0], Action(0, 4, 5))

    def test_astar_returns_path_to_target(self) -> None:
        """Flobot A* should return flattened tile paths."""
        env = GeneralsEnv(GameConfig(width=3, height=3, generals=(0, 8)))
        observation = env.reset()[0]
        state = FlobotState.from_observation(observation)
        game_map = FlobotMap(3, 3, 0)

        path = FlobotAlgorithms.a_star(state, game_map, 0, [2])

        self.assertEqual(path, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

