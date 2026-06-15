from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

try:
    import socketio
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "python-socketio[client] is required for remote play. Install with: "
        ".venv/bin/python -m pip install -r requirements.txt"
    ) from exc


DEFAULT_ENDPOINT = "https://botws.generals.io/"


@dataclass(frozen=True)
class SocketEvent:
    """One parsed Socket.IO event from generals.io."""

    name: str
    data: Any = None


class GeneralsSocketClient:
    """Small Socket.IO client for the generals.io bot websocket API."""

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        client_factory: Callable[[], Any] | None = None,
        receive_timeout_seconds: float = 2.0,
    ) -> None:
        """Create an unconnected Socket.IO client."""
        self.endpoint = endpoint
        self._client_factory = client_factory or (
            lambda: socketio.SimpleClient(reconnection=False)
        )
        self.receive_timeout_seconds = receive_timeout_seconds
        self._client: Any | None = None

    def connect(self) -> None:
        """Open the Socket.IO connection."""
        self._client = self._client_factory()
        self._client.connect(self.endpoint, transports=["websocket"])

    def close(self) -> None:
        """Close the Socket.IO connection."""
        if self._client is not None:
            self._client.disconnect()
            self._client = None

    def set_username(self, user_id: str, username: str) -> None:
        """Set the display name for this bot user id."""
        self.emit("set_username", user_id, username)

    def join_private(self, room_id: str, user_id: str) -> None:
        """Join a private game room."""
        self.emit("join_private", room_id, user_id)

    def join_1v1(self, user_id: str) -> None:
        """Join the bot server 1v1 matchmaking queue."""
        self.emit("join_1v1", user_id)

    def force_start(self, room_id: str, enabled: bool = True) -> None:
        """Request force-start for a private room."""
        self.emit("set_force_start", room_id, enabled)

    def attack(self, start: int, end: int, split: bool, move_id: int | None = None) -> None:
        """Send one attack command to the server."""
        if move_id is None:
            self.emit("attack", start, end, split)
            return
        self.emit("attack", start, end, split, move_id)

    def leave_game(self) -> None:
        """Leave the current game."""
        self.emit("leave_game")

    def cancel(self) -> None:
        """Leave the current matchmaking queue."""
        self.emit("cancel")

    def emit(self, event_name: str, *args: Any) -> None:
        """Send one Socket.IO event packet."""
        client = self._require_client()
        client.emit(event_name, args)

    def events(self) -> Iterator[SocketEvent]:
        """Yield Socket.IO events until the connection closes."""
        client = self._require_client()
        while True:
            try:
                message = client.receive(timeout=self.receive_timeout_seconds)
            except socketio.exceptions.TimeoutError:
                yield SocketEvent("timeout")
                continue
            except socketio.exceptions.DisconnectedError:
                break
            if not message:
                continue
            name = str(message[0])
            payload = message[1:]
            if not payload:
                yield SocketEvent(name)
            elif len(payload) == 1:
                yield SocketEvent(name, payload[0])
            else:
                yield SocketEvent(name, list(payload))

    def _require_client(self) -> Any:
        """Return the connected client or raise a clearer error."""
        if self._client is None:
            raise RemoteClientError("connect before using the remote client")
        return self._client


class RemoteClientError(RuntimeError):
    """Raised when the remote websocket client is used incorrectly."""
