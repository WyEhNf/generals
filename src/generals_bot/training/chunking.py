"""Fixed-chunk boundary controller for temporal policy inference."""

from __future__ import annotations

from typing import Hashable


class FixedChunkController:
    """Marks chunk boundaries at fixed intervals per stream key.

    Each stream (identified by an opaque key) maintains its own step
    counter.  A boundary is signalled every *chunk_size* steps.
    """

    def __init__(self, chunk_size: int):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self._counters: dict[Hashable, int] = {}

    def should_end_chunk(self, key: Hashable) -> bool:
        """Advance the counter for *key* and return True on a boundary."""
        count = self._counters.get(key, 0) + 1
        if count >= self.chunk_size:
            self._counters[key] = 0
            return True
        self._counters[key] = count
        return False

    def reset(self, key: Hashable) -> None:
        """Reset the counter for one stream."""
        self._counters.pop(key, None)

    def reset_all(self) -> None:
        """Reset all stream counters."""
        self._counters.clear()
