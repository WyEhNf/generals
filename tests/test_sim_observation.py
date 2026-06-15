from __future__ import annotations

import unittest

from generals_bot.sim import Action, CellView, GameConfig, GeneralsEnv


class SimObservationTest(unittest.TestCase):
    def test_fog_observation_hides_armies_and_marks_obstacles(self) -> None:
        env = GeneralsEnv(
            GameConfig(
                width=5,
                height=5,
                generals=(12, 24),
                mountains=(0,),
                cities=(4,),
                city_armies=(45,),
                starting_armies={12: 3, 24: 9},
            )
        )
        observations = env.reset()
        obs = observations[0]

        self.assertTrue(obs.visible[2, 2])
        self.assertEqual(int(obs.owner[2, 2]), int(CellView.OWN))
        self.assertEqual(int(obs.armies[4, 4]), 0)
        self.assertEqual(int(obs.owner[4, 4]), int(CellView.FOG))
        self.assertTrue(obs.fog_obstacles[0, 0])
        self.assertTrue(obs.fog_obstacles[0, 4])
        self.assertEqual(int(obs.owner[0, 0]), int(CellView.FOG_OBSTACLE))
        self.assertEqual(int(obs.owner[0, 4]), int(CellView.FOG_OBSTACLE))

    def test_visible_enemy_is_exposed(self) -> None:
        env = GeneralsEnv(
            GameConfig(
                width=3,
                height=3,
                generals=(4, 8),
                starting_owners={5: 1},
                starting_armies={4: 3, 5: 7, 8: 1},
            )
        )
        obs = env.reset()[0]

        self.assertTrue(obs.visible[1, 2])
        self.assertTrue(obs.enemy_tiles[1, 2])
        self.assertEqual(int(obs.owner[1, 2]), int(CellView.ENEMY))
        self.assertEqual(int(obs.armies[1, 2]), 7)

    def test_public_score_fields_are_player_relative(self) -> None:
        env = GeneralsEnv(
            GameConfig(
                width=3,
                height=3,
                generals=(0, 8),
                starting_owners={1: 0, 7: 1},
                starting_armies={0: 3, 1: 4, 7: 5, 8: 6},
            )
        )
        observations = env.reset()

        self.assertEqual(observations[0].own_army, 7)
        self.assertEqual(observations[0].own_land, 2)
        self.assertEqual(observations[0].enemy_army, 11)
        self.assertEqual(observations[0].enemy_land, 2)

        self.assertEqual(observations[1].own_army, 11)
        self.assertEqual(observations[1].own_land, 2)
        self.assertEqual(observations[1].enemy_army, 7)
        self.assertEqual(observations[1].enemy_land, 2)

    def test_seen_enemy_general_stays_known_after_leaving_vision(self) -> None:
        """Enemy general memory should persist after the tile returns to fog."""
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

        self.assertTrue(observations[0].generals[2, 2])
        self.assertTrue(observations[0].known_enemy_generals[2, 2])

        observations, _, _, _ = env.step({1: Action(1, 8, 7)})
        obs = observations[0]

        self.assertFalse(obs.visible[2, 2])
        self.assertFalse(obs.generals[2, 2])
        self.assertTrue(obs.known_generals[2, 2])
        self.assertTrue(obs.known_enemy_generals[2, 2])


if __name__ == "__main__":
    unittest.main()
