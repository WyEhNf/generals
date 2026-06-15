from __future__ import annotations

import unittest

from generals_bot.sim import (
    MapGenerationConfig,
    are_passable_tiles_connected,
    generate_1v1_map,
    manhattan,
)


class MapGenerationTest(unittest.TestCase):
    def test_generated_maps_match_basic_constraints(self) -> None:
        options = MapGenerationConfig(max_attempts=2000)
        for seed in range(50):
            config = generate_1v1_map(seed=seed, config=options, max_turns=100)

            self.assertGreaterEqual(config.width, options.min_size)
            self.assertLessEqual(config.width, options.max_size)
            self.assertGreaterEqual(config.height, options.min_size)
            self.assertLessEqual(config.height, options.max_size)
            self.assertEqual(config.max_turns, 100)

            occupied = set(config.generals) | set(config.mountains) | set(config.cities)
            self.assertEqual(
                len(occupied),
                len(config.generals) + len(config.mountains) + len(config.cities),
            )
            self.assertGreaterEqual(
                manhattan(config.generals[0], config.generals[1], config.width),
                options.min_general_distance,
            )
            self.assertTrue(
                are_passable_tiles_connected(config.width, config.height, config.mountains)
            )
            self.assertTrue(
                all(
                    options.city_army_range[0] <= army <= options.city_army_range[1]
                    for army in config.city_armies
                )
            )

    def test_generation_is_seed_reproducible(self) -> None:
        first = generate_1v1_map(seed=123, max_turns=100)
        second = generate_1v1_map(seed=123, max_turns=100)

        self.assertEqual(first, second)

    def test_connectivity_rejects_isolated_passable_region(self) -> None:
        width = 3
        height = 3
        mountains = {1, 3}

        self.assertFalse(are_passable_tiles_connected(width, height, mountains))


if __name__ == "__main__":
    unittest.main()
