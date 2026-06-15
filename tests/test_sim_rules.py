from __future__ import annotations

import unittest

import numpy as np

from generals_bot.sim import Action, GameConfig, GeneralsEnv, Terrain
from generals_bot.sim.queue import QueuedAction


class SimRulesTest(unittest.TestCase):
    def test_action_visible_next_observation(self) -> None:
        env = GeneralsEnv(
            GameConfig(
                width=3,
                height=1,
                generals=(0, 2),
                starting_armies={0: 5, 2: 1},
                max_turns=20,
            )
        )
        observations = env.reset()
        self.assertEqual(observations[0].turn, 0)

        observations, _, _, _ = env.step({0: Action(0, 0, 1)})
        self.assertEqual(observations[0].turn, 1)
        self.assertEqual(int(env.state.terrain[0, 1]), 0)
        self.assertTrue(observations[0].own_tiles[0, 1])

    def test_action_list_advances_one_action_per_tick(self) -> None:
        env = GeneralsEnv(
            GameConfig(
                width=4,
                height=1,
                generals=(0, 3),
                starting_armies={0: 5, 3: 1},
                max_turns=20,
            )
        )
        env.reset()

        env.step({0: [Action(0, 0, 1), Action(0, 1, 2)]})
        self.assertEqual(int(env.state.terrain[0, 1]), 0)
        self.assertEqual(int(env.state.terrain[0, 2]), int(Terrain.NEUTRAL))
        self.assertEqual(len(env.queue_snapshot()[0]), 1)

        env.step({})
        self.assertEqual(int(env.state.terrain[0, 2]), 0)
        self.assertEqual(len(env.queue_snapshot()[0]), 0)

    def test_move_capture_neutral_and_reinforce(self) -> None:
        env = _ready_env(width=4, height=1, generals=(0, 3), armies={0: 5, 1: 2, 3: 1})
        env.state.queues[0].pending.append(QueuedAction(Action(0, 0, 1), 0, 0))

        env.tick()

        self.assertEqual(int(env.state.terrain[0, 1]), 0)
        self.assertEqual(int(env.state.armies[0, 0]), 1)
        self.assertEqual(int(env.state.armies[0, 1]), 2)

        env.state.queues[0].pending.append(QueuedAction(Action(0, 1, 0), env.state.turn, env.state.turn))
        env.tick()

        self.assertEqual(int(env.state.terrain[0, 0]), 0)
        self.assertGreaterEqual(int(env.state.armies[0, 0]), 2)

    def test_split_move(self) -> None:
        env = _ready_env(width=3, height=1, generals=(0, 2), armies={0: 7, 2: 1})
        env.state.queues[0].pending.append(QueuedAction(Action(0, 0, 1, split=True), 0, 0))

        env.tick()

        self.assertEqual(int(env.state.armies[0, 0]), 4)
        self.assertEqual(int(env.state.armies[0, 1]), 3)
        self.assertEqual(int(env.state.terrain[0, 1]), 0)

    def test_illegal_move_skip_continues_to_next_ready_action(self) -> None:
        env = _ready_env(width=4, height=1, generals=(0, 3), armies={0: 5, 3: 1})
        env.state.queues[0].pending.append(QueuedAction(Action(0, 2, 1), 0, 0))
        env.state.queues[0].pending.append(QueuedAction(Action(0, 0, 1), 0, 0))

        env.tick()

        self.assertEqual(len(env.last_executed_actions), 2)
        self.assertFalse(env.last_executed_actions[0].valid)
        self.assertEqual(env.last_executed_actions[0].reason, "start_not_owned")
        self.assertTrue(env.last_executed_actions[1].valid)
        self.assertEqual(int(env.state.terrain[0, 1]), 0)

    def test_priority_alternates_by_turn(self) -> None:
        env = _ready_env(width=3, height=1, generals=(0, 2), armies={0: 5, 2: 5})
        env.state.queues[0].pending.append(QueuedAction(Action(0, 0, 1), 0, 0))
        env.state.queues[1].pending.append(QueuedAction(Action(1, 2, 1), 0, 0))
        env.tick()
        self.assertEqual(int(env.state.terrain[0, 1]), 0)

        env = _ready_env(width=3, height=1, generals=(0, 2), armies={0: 5, 2: 5})
        env.state.turn = 1
        env.state.queues[0].pending.append(QueuedAction(Action(0, 0, 1), 1, 1))
        env.state.queues[1].pending.append(QueuedAction(Action(1, 2, 1), 1, 1))
        env.tick()
        self.assertEqual(int(env.state.terrain[0, 1]), 1)

    def test_growth_timing(self) -> None:
        env = GeneralsEnv(
            GameConfig(
                width=3,
                height=1,
                generals=(0, 2),
                starting_armies={0: 1, 1: 2, 2: 1},
                starting_owners={1: 0},
                cities=(1,),
                city_armies=(2,),
                max_turns=60,
            )
        )
        env.reset()

        env.step({})
        self.assertEqual(int(env.state.armies[0, 0]), 1)
        self.assertEqual(int(env.state.armies[0, 1]), 2)

        env.step({})
        self.assertEqual(int(env.state.armies[0, 0]), 2)
        self.assertEqual(int(env.state.armies[0, 1]), 3)

        while env.state.turn < 49:
            env.step({})
        before = env.state.armies.copy()
        env.step({})
        self.assertTrue(np.all(env.state.armies[env.state.terrain >= 0] >= before[env.state.terrain >= 0]))
        self.assertGreaterEqual(int(env.state.armies[0, 0]), int(before[0, 0]) + 1)

    def test_general_capture_ends_game(self) -> None:
        env = _ready_env(width=2, height=1, generals=(0, 1), armies={0: 5, 1: 1})
        env.state.queues[0].pending.append(QueuedAction(Action(0, 0, 1), 0, 0))

        env.tick()

        self.assertTrue(env.done)
        self.assertEqual(env.state.winner, 0)
        self.assertFalse(env.state.alive[1])
        self.assertEqual(env.rewards(), {0: 1.0, 1: -1.0})

    def test_executed_action_records_attack_damage(self) -> None:
        """Executed actions should expose pre-attack damage fields for reward shaping."""
        env = _ready_env(width=3, height=1, generals=(0, 2), armies={0: 5, 1: 3, 2: 5})
        env.state.terrain[0, 1] = 1
        env.state.queues[0].pending.append(QueuedAction(Action(0, 0, 1), 0, 0))

        env.tick()

        executed = env.last_executed_actions[0]
        self.assertTrue(executed.valid)
        self.assertEqual(executed.target_owner_before, 1)
        self.assertEqual(executed.target_army_before, 3)
        self.assertEqual(executed.moved_army, 4)
        self.assertEqual(executed.damage_done, 3)


def _ready_env(
    *,
    width: int,
    height: int,
    generals: tuple[int, int],
    armies: dict[int, int],
) -> GeneralsEnv:
    env = GeneralsEnv(
        GameConfig(width=width, height=height, generals=generals, starting_armies=armies, max_turns=20)
    )
    env.reset()
    return env


if __name__ == "__main__":
    unittest.main()
