#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generals_bot.matches import make_policy, parse_agent_spec
from generals_bot.remote import GeneralsSocketClient, RemoteGameState, attacks_from_action_like
from generals_bot.remote.client import DEFAULT_ENDPOINT
from generals_bot.remote.state import summarize_observation


DEFAULT_LOG_ROOT = "artifacts/remote_matches"
DEFAULT_ENV_FILE = str(ROOT / ".env")
DEFAULT_USERNAME_WAIT_SEC = 5.0
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def main(argv: Sequence[str] | None = None) -> int:
    """Run a local Policy in a generals.io online game."""
    args = parse_args(argv)
    spec = parse_agent_spec(args.agent)
    log_dir = _resolve_log_dir(args.log_dir)
    logger = RemoteMatchLogger(log_dir / "events.jsonl")
    client = GeneralsSocketClient(endpoint=args.endpoint)
    remote_state = RemoteGameState()
    policy = None
    move_id = 1
    updates_seen = 0
    game_started = False
    queued_for_match = False

    try:
        client.connect()
        logger.write("connect", {"endpoint": args.endpoint})
        if args.username and args.set_username:
            client.set_username(args.user_id, args.username)
            logger.write("set_username", {"username": args.username})
            _wait_for_set_username(client, logger, args.username_wait_sec)
        if args.mode == "private":
            client.join_private(args.room_id, args.user_id)
            logger.write("join_private", {"room_id": args.room_id})
        elif args.mode == "1v1":
            client.join_1v1(args.user_id)
            queued_for_match = True
            logger.write("join_1v1", {})
        else:  # pragma: no cover - argparse validates this.
            raise ValueError(f"unsupported remote mode: {args.mode}")
        if _force_start_enabled(args):
            if not _send_force_start(client, logger, args, reason="join"):
                return 2

        for event in client.events():
            if event.name == "timeout":
                if not game_started and _force_start_enabled(args):
                    if not _send_force_start(client, logger, args, reason="timeout"):
                        return 2
                continue
            if event.name == "engine_open":
                logger.write("engine_open", event.data)
                continue
            if event.name == "error_user_id":
                logger.write("error_user_id", event.data)
                return 2
            if event.name == "game_start":
                game_started = True
                queued_for_match = False
                start_data = _event_data_dict(event.name, event.data)
                remote_state.start(start_data)
                logger.write("game_start", start_data)
                continue
            if event.name == "queue_update":
                queue_data = _event_data_dict(event.name, event.data)
                logger.write("queue_update", queue_data)
                if _force_start_enabled(args) and not _queue_update_shows_bot_forced(
                    queue_data,
                    args.username,
                ):
                    if not _send_force_start(client, logger, args, reason="queue_update"):
                        return 2
                continue
            if event.name == "game_update":
                update_data = _event_data_dict(event.name, event.data)
                observation = remote_state.apply_update(update_data)
                if policy is None:
                    policy = make_policy(
                        spec,
                        player=observation.player_id,
                        seed=args.seed,
                        config=remote_state.placeholder_config(),
                        device=args.device,
                        pass_bias=args.pass_bias,
                        pass_bias_turn=args.pass_bias_turn,
                    )
                    logger.write(
                        "policy_ready",
                        {
                            "agent": spec.raw,
                            "label": spec.label,
                            "player_id": observation.player_id,
                            "width": observation.width,
                            "height": observation.height,
                        },
                    )

                action_like = policy.act(observation)
                attacks = attacks_from_action_like(action_like, observation.width)
                sent_attacks = []
                for attack in attacks:
                    client.attack(attack.start, attack.end, attack.split, move_id)
                    sent_attacks.append({
                        "start": attack.start,
                        "end": attack.end,
                        "split": attack.split,
                        "move_id": move_id,
                    })
                    move_id += 1
                updates_seen += 1
                logger.write("game_update", {
                    "observation": summarize_observation(observation),
                    "scores": update_data.get("scores", []),
                    "attacks": sent_attacks,
                    "replay_url": remote_state.replay_url(),
                })
                if args.max_updates is not None and updates_seen >= args.max_updates:
                    client.leave_game()
                    logger.write("leave_game", {"reason": "max_updates"})
                    break
                continue
            if event.name in {"game_won", "game_lost"}:
                logger.write(event.name, {
                    "data": event.data,
                    "replay_url": remote_state.replay_url(),
                })
                break
            logger.write("unhandled_event", {"name": event.name, "data": event.data})
    except Exception as exc:
        logger.write("error", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if queued_for_match and not game_started:
            try:
                client.cancel()
                logger.write("cancel", {"reason": "shutdown_before_game_start"})
            except Exception as exc:  # pragma: no cover - best-effort cleanup.
                logger.write(
                    "cancel_error",
                    {"type": type(exc).__name__, "message": str(exc)},
                )
        client.close()
        logger.close()
    print(log_dir / "events.jsonl")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse remote-agent arguments after loading the selected dotenv file."""
    env_file = _preparse_env_file(argv)
    _load_env_file(env_file)
    parser = _build_parser(env_file)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _build_parser(env_file: str) -> argparse.ArgumentParser:
    """Create the argparse parser with defaults sourced from the environment."""
    parser = argparse.ArgumentParser(description="Run a local agent on generals.io.")
    parser.add_argument("--env-file", default=env_file)
    parser.add_argument("--agent", default=_env("GENERALS_AGENT"))
    parser.add_argument("--user-id", default=_env("GENERALS_USER_ID"))
    parser.add_argument("--username", default=_env("GENERALS_USERNAME"))
    try:
        set_username_default = _env_bool("GENERALS_SET_USERNAME", True)
    except ValueError as exc:
        parser.error(str(exc))
    parser.add_argument("--set-username", dest="set_username", action="store_true")
    parser.add_argument("--no-set-username", dest="set_username", action="store_false")
    parser.set_defaults(set_username=set_username_default)
    parser.add_argument("--room-id", default=_env("GENERALS_ROOM_ID"))
    parser.add_argument(
        "--mode",
        choices=["private", "1v1"],
        default=_env("GENERALS_MODE", "private"),
    )
    parser.add_argument("--device", default=_env("GENERALS_DEVICE", "auto"))
    parser.add_argument("--seed", type=int, default=_env("GENERALS_SEED", 0))
    parser.add_argument("--pass-bias", type=float, default=_env("GENERALS_PASS_BIAS", 0.0))
    parser.add_argument(
        "--pass-bias-turn",
        type=int,
        default=_env("GENERALS_PASS_BIAS_TURN", 50),
    )
    parser.add_argument("--endpoint", default=_env("GENERALS_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--log-dir", default=_env("GENERALS_LOG_DIR"))
    parser.add_argument("--max-updates", type=int, default=_env("GENERALS_MAX_UPDATES"))
    parser.add_argument(
        "--username-wait-sec",
        type=float,
        default=_env("GENERALS_USERNAME_WAIT_SEC", DEFAULT_USERNAME_WAIT_SEC),
    )
    try:
        force_start_default = _env_bool("GENERALS_FORCE_START", True)
    except ValueError as exc:
        parser.error(str(exc))
    parser.add_argument("--force-start", dest="force_start", action="store_true")
    parser.add_argument("--no-force-start", dest="force_start", action="store_false")
    parser.set_defaults(force_start=force_start_default)
    return parser


def _preparse_env_file(argv: Sequence[str] | None) -> str:
    """Return the dotenv path before constructing the full parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--env-file",
        default=os.environ.get("GENERALS_ENV_FILE", DEFAULT_ENV_FILE),
    )
    args, _ = parser.parse_known_args(argv)
    return args.env_file or DEFAULT_ENV_FILE


def _load_env_file(raw_path: str) -> None:
    """Load KEY=VALUE entries from a dotenv file without overriding process env."""
    if not raw_path:
        return
    path = Path(raw_path)
    if not path.exists():
        return
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not _ENV_KEY_RE.match(key):
            raise ValueError(f"{path}:{line_number}: invalid env key {key!r}")
        os.environ.setdefault(key, _parse_env_file_value(path, line_number, raw_value))


def _parse_env_file_value(path: Path, line_number: int, raw_value: str) -> str:
    """Parse one dotenv value, supporting quotes and trailing comments."""
    lexer = shlex.shlex(raw_value, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        parts = list(lexer)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not parts:
        return ""
    if len(parts) > 1:
        raise ValueError(f"{path}:{line_number}: quote values that contain spaces")
    return parts[0]


def _env(name: str, default: Any = None) -> Any:
    """Return a non-empty environment variable or a default."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    """Return an environment variable parsed as a boolean."""
    value = _env(name)
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate required remote-play settings after env defaults are applied."""
    required = [
        ("agent", "--agent", "GENERALS_AGENT"),
        ("user_id", "--user-id", "GENERALS_USER_ID"),
    ]
    if args.mode == "private":
        required.append(("room_id", "--room-id", "GENERALS_ROOM_ID"))
    for attr, option, env_name in required:
        if not getattr(args, attr):
            parser.error(f"pass {option} or set {env_name}")
    try:
        parse_agent_spec(args.agent)
    except ValueError as exc:
        parser.error(str(exc))
    if args.username_wait_sec < 0:
        parser.error("--username-wait-sec must be nonnegative")


def _wait_for_set_username(
    client: GeneralsSocketClient,
    logger: "RemoteMatchLogger",
    timeout_seconds: float,
) -> None:
    """Wait briefly for the server username acknowledgement before joining."""
    if timeout_seconds <= 0:
        return
    deadline = time.monotonic() + timeout_seconds
    events = client.events()
    while time.monotonic() < deadline:
        try:
            event = next(events)
        except StopIteration:
            return
        if event.name == "timeout":
            continue
        if event.name == "error_set_username":
            logger.write("error_set_username", event.data)
            return
        if event.name == "engine_open":
            logger.write("engine_open", event.data)
            continue
        logger.write(
            "prejoin_event",
            {"name": event.name, "data": event.data},
        )
    logger.write(
        "set_username_wait_timeout",
        {"timeout_seconds": timeout_seconds},
    )


def _force_start_enabled(args: argparse.Namespace) -> bool:
    """Return whether private-room force-start should be used."""
    return args.mode == "private" and bool(args.force_start)


def _send_force_start(
    client: GeneralsSocketClient,
    logger: "RemoteMatchLogger",
    args: argparse.Namespace,
    *,
    reason: str,
) -> bool:
    """Send force-start and report whether the socket accepted the emit."""
    try:
        client.force_start(args.room_id, True)
    except Exception as exc:
        logger.write(
            "force_start_error",
            {"type": type(exc).__name__, "message": str(exc), "reason": reason},
        )
        return False
    logger.write(
        "force_start",
        {"room_id": args.room_id, "enabled": True, "reason": reason},
    )
    return True


class RemoteMatchLogger:
    """Write remote match events as JSON lines."""

    def __init__(self, path: Path) -> None:
        """Open one JSONL log file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a", encoding="utf-8")

    def write(self, event: str, data: Any) -> None:
        """Append one event to the log."""
        self._handle.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    "data": data,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying log file."""
        self._handle.close()


def _resolve_log_dir(raw_path: str | None) -> Path:
    """Return the log directory for one remote run."""
    if raw_path:
        return Path(raw_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(DEFAULT_LOG_ROOT) / stamp


def _event_data_dict(event_name: str, data: Any) -> dict[str, Any]:
    """Return the primary dict payload from a Socket.IO event."""
    payload = data
    if isinstance(data, list) and data:
        payload = data[0]
    if not isinstance(payload, dict):
        raise ValueError(f"{event_name} payload must be a dict, got {type(payload).__name__}")
    return payload


def _queue_update_shows_bot_forced(data: Any, username: str | None) -> bool:
    """Return whether the queue update says our bot has already force-started."""
    if not isinstance(data, dict) or not username:
        return False
    usernames = data.get("usernames")
    forced = data.get("numForce", [])
    if not isinstance(usernames, list) or not isinstance(forced, list):
        return False
    try:
        bot_index = usernames.index(username)
    except ValueError:
        return False
    return bot_index in forced


if __name__ == "__main__":
    raise SystemExit(main())
