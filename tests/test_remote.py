from __future__ import annotations

import unittest

from generals_bot.remote.client import GeneralsSocketClient, RemoteClientError, SocketEvent
from scripts.run_remote_agent import _event_data_dict, _queue_update_shows_bot_forced
from generals_bot.remote.state import (
    SERVER_FOG,
    SERVER_FOG_OBSTACLE,
    SERVER_MOUNTAIN,
    SERVER_NEUTRAL,
    RemoteGameState,
    apply_diff,
    attacks_from_action_like,
)
from generals_bot.sim import Action, CellView, PassAction


class RemoteStateTest(unittest.TestCase):
    """Tests for translating generals.io websocket state into local observations."""

    def test_apply_diff_replaces_and_truncates_cache(self) -> None:
        """Generals.io run-length diffs should patch and truncate the cached array."""
        cache: list[int] = []
        apply_diff(cache, [0, 5, 1, 2, 3, 4, 5])
        self.assertEqual(cache, [1, 2, 3, 4, 5])

        apply_diff(cache, [1, 2, 8, 9, 1])
        self.assertEqual(cache, [1, 8, 9, 4])

    def test_apply_update_builds_player_relative_observation(self) -> None:
        """A server update should become the same player-relative Observation used locally."""
        state = RemoteGameState()
        state.start({"playerIndex": 0, "usernames": ["bot", "enemy"], "replay_id": "abc"})

        armies = [
            10,
            2,
            0,
            0,
            4,
            40,
            0,
            3,
            7,
        ]
        tiles = [
            0,
            SERVER_NEUTRAL,
            SERVER_FOG,
            SERVER_MOUNTAIN,
            0,
            SERVER_NEUTRAL,
            SERVER_FOG_OBSTACLE,
            1,
            1,
        ]
        observation = state.apply_update(
            {
                "turn": 12,
                "map_diff": _full_diff([3, 3, *armies, *tiles]),
                "cities_diff": _full_diff([5]),
                "generals": [0, 8],
                "scores": [
                    {"i": 0, "total": 14, "tiles": 2, "dead": False},
                    {"i": 1, "total": 10, "tiles": 2, "dead": False},
                ],
            }
        )

        self.assertEqual(observation.turn, 12)
        self.assertEqual((observation.height, observation.width), (3, 3))
        self.assertTrue(observation.visible[0, 0])
        self.assertFalse(observation.visible[0, 2])
        self.assertEqual(observation.owner[0, 0], int(CellView.OWN))
        self.assertEqual(observation.owner[0, 1], int(CellView.NEUTRAL))
        self.assertEqual(observation.owner[1, 0], int(CellView.MOUNTAIN))
        self.assertEqual(observation.owner[2, 1], int(CellView.ENEMY))
        self.assertEqual(observation.owner[2, 0], int(CellView.FOG_OBSTACLE))
        self.assertEqual(observation.armies[0, 2], 0)
        self.assertEqual(observation.armies[2, 2], 7)
        self.assertTrue(observation.cities[1, 2])
        self.assertTrue(observation.known_cities[1, 2])
        self.assertTrue(observation.generals[0, 0])
        self.assertTrue(observation.known_enemy_generals[2, 2])
        self.assertEqual(observation.own_army, 14)
        self.assertEqual(observation.own_land, 2)
        self.assertEqual(observation.enemy_army, 10)
        self.assertEqual(observation.enemy_land, 2)
        self.assertEqual(state.replay_url(), "http://bot.generals.io/replays/abc")

    def test_seen_enemy_general_remains_known_after_fog(self) -> None:
        """The remote adapter should remember enemy generals after they leave vision."""
        state = RemoteGameState()
        state.start({"playerIndex": 0})
        armies = [10, 1, 1, 1, 1, 1, 1, 1, 9]
        first_tiles = [
            0,
            SERVER_NEUTRAL,
            SERVER_NEUTRAL,
            SERVER_NEUTRAL,
            SERVER_NEUTRAL,
            SERVER_NEUTRAL,
            SERVER_NEUTRAL,
            1,
            1,
        ]
        first = state.apply_update(
            {
                "turn": 1,
                "map_diff": _full_diff([3, 3, *armies, *first_tiles]),
                "cities_diff": _full_diff([]),
                "generals": [0, 8],
                "scores": [],
            }
        )
        self.assertTrue(first.known_enemy_generals[2, 2])

        second_tiles = [
            0,
            SERVER_NEUTRAL,
            SERVER_NEUTRAL,
            SERVER_NEUTRAL,
            SERVER_NEUTRAL,
            SERVER_NEUTRAL,
            SERVER_NEUTRAL,
            SERVER_FOG,
            SERVER_FOG,
        ]
        second = state.apply_update(
            {
                "turn": 2,
                "map_diff": _full_diff([3, 3, *armies, *second_tiles]),
                "cities_diff": _full_diff([]),
                "generals": [0, -1],
                "scores": [],
            }
        )

        self.assertFalse(second.generals[2, 2])
        self.assertTrue(second.known_enemy_generals[2, 2])
        self.assertTrue(second.explored[2, 2])
        self.assertFalse(second.visible[2, 2])

    def test_attacks_from_action_like_normalizes_policy_outputs(self) -> None:
        """Remote attacks should preserve legal action fields and drop pass actions."""
        attacks = attacks_from_action_like(
            [Action(player_id=0, start=1, end=2, split=True), PassAction(player_id=0)],
            width=3,
        )

        self.assertEqual(len(attacks), 1)
        self.assertEqual(attacks[0].start, 1)
        self.assertEqual(attacks[0].end, 2)
        self.assertTrue(attacks[0].split)

    def test_start_resets_previous_game_cache(self) -> None:
        """A reused state object should not leak map memory across games."""
        state = RemoteGameState()
        state.start({"playerIndex": 0})
        state.apply_update(
            {
                "turn": 1,
                "map_diff": _full_diff([1, 1, 10, 0]),
                "cities_diff": _full_diff([]),
                "generals": [0, -1],
                "scores": [],
            }
        )

        state.start({"playerIndex": 1})

        self.assertEqual(state.map_cache, [])
        self.assertEqual(state.cities_cache, [])
        self.assertIsNone(state.memory)


class GeneralsSocketClientTest(unittest.TestCase):
    """Tests for the Socket.IO wrapper used by the remote client."""

    def test_emit_sends_tuple_payload(self) -> None:
        """Socket emits should preserve generals.io's positional event arguments."""
        fake = _FakeSimpleClient()
        client = GeneralsSocketClient(client_factory=lambda: fake)
        client.connect()

        client.attack(1, 2, False, 9)

        self.assertEqual(fake.emits, [("attack", (1, 2, False, 9))])

    def test_matchmaking_helpers_emit_expected_events(self) -> None:
        """Remote client helpers should emit botserver matchmaking events."""
        fake = _FakeSimpleClient()
        client = GeneralsSocketClient(client_factory=lambda: fake)
        client.connect()

        client.join_1v1("user_123")
        client.cancel()

        self.assertEqual(fake.emits, [("join_1v1", ("user_123",)), ("cancel", ())])

    def test_events_unpack_single_and_multi_payloads(self) -> None:
        """Socket receives should expose one payload directly and many payloads as a list."""
        fake = _FakeSimpleClient()
        fake.receives = [
            ["game_update", {"turn": 3}],
            ["game_won", {"score": 1}, "extra"],
        ]
        client = GeneralsSocketClient(client_factory=lambda: fake)
        client.connect()

        events = client.events()

        self.assertEqual(next(events), SocketEvent("game_update", {"turn": 3}))
        self.assertEqual(next(events), SocketEvent("game_won", [{"score": 1}, "extra"]))

    def test_events_yield_timeout_ticks(self) -> None:
        """Socket receive timeouts should surface as timer events for force-start retries."""
        fake = _FakeSimpleClient()
        fake.timeout_once = True
        fake.receives = [["queue_update", {"numPlayers": 2}]]
        client = GeneralsSocketClient(
            client_factory=lambda: fake,
            receive_timeout_seconds=0.1,
        )
        client.connect()

        events = client.events()

        self.assertEqual(next(events), SocketEvent("timeout"))
        self.assertEqual(next(events), SocketEvent("queue_update", {"numPlayers": 2}))

    def test_emit_requires_connection(self) -> None:
        """Remote clients should fail clearly when used before connect."""
        client = GeneralsSocketClient()

        with self.assertRaises(RemoteClientError):
            client.attack(1, 2, False)

    def test_queue_update_detects_own_force_start(self) -> None:
        """Queue updates should reveal whether our username is in numForce."""
        self.assertTrue(
            _queue_update_shows_bot_forced(
                {"usernames": ["player", "[Bot] release_bot"], "numForce": [1]},
                "[Bot] release_bot",
            )
        )
        self.assertFalse(
            _queue_update_shows_bot_forced(
                {"usernames": ["player", "[Bot] release_bot"], "numForce": [0]},
                "[Bot] release_bot",
            )
        )

    def test_event_data_dict_uses_first_socketio_argument(self) -> None:
        """Remote script should accept server events with extra positional payloads."""
        payload = {"playerIndex": 1}

        self.assertEqual(_event_data_dict("game_start", [payload, None]), payload)
        with self.assertRaisesRegex(ValueError, "must be a dict"):
            _event_data_dict("game_start", ["not-dict", None])


def _full_diff(values: list[int]) -> list[int]:
    """Return a full replacement generals.io diff for tests."""
    return [0, len(values), *values]


class _FakeSimpleClient:
    """Tiny stand-in for socketio.SimpleClient in unit tests."""

    def __init__(self) -> None:
        """Initialize recorded emits and queued receives."""
        self.endpoint: str | None = None
        self.transports: list[str] | None = None
        self.emits: list[tuple[str, tuple[object, ...]]] = []
        self.receives: list[list[object]] = []
        self.disconnected = False
        self.timeout_once = False

    def connect(self, endpoint: str, transports: list[str]) -> None:
        """Record connection arguments."""
        self.endpoint = endpoint
        self.transports = transports

    def disconnect(self) -> None:
        """Record that the fake connection was closed."""
        self.disconnected = True

    def emit(self, event_name: str, data: tuple[object, ...]) -> None:
        """Record one emitted event."""
        self.emits.append((event_name, data))

    def receive(self, timeout: float | None = None) -> list[object]:
        """Return the next queued event."""
        del timeout
        if self.timeout_once:
            self.timeout_once = False
            import socketio

            raise socketio.exceptions.TimeoutError
        if not self.receives:
            raise StopIteration
        return self.receives.pop(0)


if __name__ == "__main__":
    unittest.main()
