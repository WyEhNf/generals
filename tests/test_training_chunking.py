from __future__ import annotations

import unittest

from generals_bot.training.chunking import FixedChunkController


class FixedChunkControllerTest(unittest.TestCase):
    """Unit tests for Phase 1 fixed-size chunk boundaries."""

    def test_rejects_non_positive_chunk_size(self) -> None:
        """A fixed chunk must contain at least one tick."""
        for invalid_size in (0, -1, -8):
            with self.subTest(chunk_size=invalid_size):
                with self.assertRaisesRegex(ValueError, "positive"):
                    FixedChunkController(invalid_size)

    def test_k1_ends_every_tick(self) -> None:
        """K=1 is the no-compression control and must end every tick."""
        controller = FixedChunkController(1)

        observed = [controller.should_end_chunk("stream") for _ in range(5)]

        self.assertEqual(observed, [True] * 5)

    def test_fixed_k_boundaries_have_no_gap_or_duplicate(self) -> None:
        """A stream should emit exactly one boundary after each K ticks."""
        controller = FixedChunkController(4)

        observed = [controller.should_end_chunk("stream") for _ in range(10)]

        self.assertEqual(
            observed,
            [False, False, False, True, False, False, False, True, False, False],
        )

    def test_stream_counters_are_isolated(self) -> None:
        """Advancing one player/environment stream must not age another stream."""
        controller = FixedChunkController(3)

        self.assertFalse(controller.should_end_chunk((0, 0)))
        self.assertFalse(controller.should_end_chunk((0, 0)))
        self.assertFalse(controller.should_end_chunk((0, 1)))
        self.assertTrue(controller.should_end_chunk((0, 0)))
        self.assertFalse(controller.should_end_chunk((0, 1)))
        self.assertTrue(controller.should_end_chunk((0, 1)))

    def test_reset_only_clears_the_requested_stream(self) -> None:
        """Episode reset for one stream must preserve every other stream."""
        controller = FixedChunkController(3)
        controller.should_end_chunk("reset-me")
        controller.should_end_chunk("keep-me")
        controller.should_end_chunk("keep-me")

        controller.reset("reset-me")

        self.assertFalse(controller.should_end_chunk("reset-me"))
        self.assertTrue(controller.should_end_chunk("keep-me"))

    def test_reset_all_clears_every_stream(self) -> None:
        """A global reset must start every stream at chunk age zero."""
        controller = FixedChunkController(2)
        controller.should_end_chunk("a")
        controller.should_end_chunk("b")

        controller.reset_all()

        self.assertFalse(controller.should_end_chunk("a"))
        self.assertFalse(controller.should_end_chunk("b"))


if __name__ == "__main__":
    unittest.main()
