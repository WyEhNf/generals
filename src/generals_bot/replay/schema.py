from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RAW_SCHEMA_VERSION = 1
SOURCE_DATASET = "strakammm/generals_io_replays"


class ReplayFormatError(ValueError):
    """Raised when a replay row cannot be normalized at all."""


@dataclass(frozen=True)
class RawReplayMove:
    player_id: int
    start: int
    end: int
    split: bool
    turn: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "start": self.start,
            "end": self.end,
            "split": self.split,
            "turn": self.turn,
        }


@dataclass(frozen=True)
class RawTerrain:
    generals: list[int]
    mountains: list[int]
    cities: list[int]
    city_armies: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generals": self.generals,
            "mountains": self.mountains,
            "cities": self.cities,
            "city_armies": self.city_armies,
        }


@dataclass(frozen=True)
class RawReplay:
    schema_version: int
    source: str
    replay_id: str
    version: int
    width: int
    height: int
    players: list[str]
    stars: list[int | None]
    terrain: RawTerrain
    moves: list[RawReplayMove]
    afks: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "replay_id": self.replay_id,
            "version": self.version,
            "width": self.width,
            "height": self.height,
            "players": self.players,
            "stars": self.stars,
            "terrain": self.terrain.to_dict(),
            "moves": [move.to_dict() for move in self.moves],
            "afks": self.afks,
        }


def raw_replay_from_dict(data: dict[str, Any]) -> RawReplay:
    terrain = data["terrain"]
    return RawReplay(
        schema_version=int(data["schema_version"]),
        source=str(data["source"]),
        replay_id=str(data["replay_id"]),
        version=int(data["version"]),
        width=int(data["width"]),
        height=int(data["height"]),
        players=_str_list(data.get("players")),
        stars=_optional_int_list(data.get("stars")),
        terrain=RawTerrain(
            generals=_int_list(terrain.get("generals")),
            mountains=_int_list(terrain.get("mountains")),
            cities=_int_list(terrain.get("cities")),
            city_armies=_int_list(terrain.get("city_armies")),
        ),
        moves=[
            RawReplayMove(
                player_id=int(move["player_id"]),
                start=int(move["start"]),
                end=int(move["end"]),
                split=bool(move["split"]),
                turn=int(move["turn"]),
            )
            for move in data.get("moves", [])
        ],
        afks=_plain_list(data.get("afks")),
    )


@dataclass(frozen=True)
class NormalizedReplay:
    replay: RawReplay
    warnings: list[str]


def normalize_hf_row(row: dict[str, Any], *, source: str = SOURCE_DATASET) -> NormalizedReplay:
    warnings: list[str] = []
    replay_id = _required_str(row, "id")
    version = _required_int(row, "version")
    width = _required_int(row, "mapWidth")
    height = _required_int(row, "mapHeight")
    if width <= 0 or height <= 0:
        raise ReplayFormatError(f"{replay_id}: invalid map size {width}x{height}")

    players = _str_list(row.get("usernames"))
    stars = _optional_int_list(row.get("stars"))
    generals = _int_list(row.get("generals"))
    mountains = _int_list(row.get("mountains"))
    cities = _int_list(row.get("cities"))
    city_armies = _int_list(row.get("cityArmies"))
    afks = _plain_list(row.get("afks"))

    if len(generals) != 2:
        warnings.append(f"{replay_id}: expected 2 generals, got {len(generals)}")
    if len(cities) != len(city_armies):
        warnings.append(
            f"{replay_id}: cities/city_armies length mismatch "
            f"{len(cities)} != {len(city_armies)}"
        )
    if len(players) != len(generals):
        warnings.append(
            f"{replay_id}: player/general length mismatch {len(players)} != {len(generals)}"
        )

    size = width * height
    for field_name, tiles in (
        ("generals", generals),
        ("mountains", mountains),
        ("cities", cities),
    ):
        for tile in tiles:
            if not _tile_in_bounds(tile, size):
                warnings.append(f"{replay_id}: {field_name} tile out of bounds: {tile}")

    moves = _normalize_moves(row.get("moves"), replay_id=replay_id, width=width, height=height)
    warnings.extend(moves.warnings)

    replay = RawReplay(
        schema_version=RAW_SCHEMA_VERSION,
        source=source,
        replay_id=replay_id,
        version=version,
        width=width,
        height=height,
        players=players,
        stars=stars,
        terrain=RawTerrain(
            generals=generals,
            mountains=mountains,
            cities=cities,
            city_armies=city_armies,
        ),
        moves=moves.moves,
        afks=afks,
    )
    return NormalizedReplay(replay=replay, warnings=warnings)


@dataclass(frozen=True)
class _MovesResult:
    moves: list[RawReplayMove]
    warnings: list[str]


def _normalize_moves(value: Any, *, replay_id: str, width: int, height: int) -> _MovesResult:
    warnings: list[str] = []
    moves: list[RawReplayMove] = []
    size = width * height
    for index, raw_move in enumerate(_plain_list(value)):
        parts = _plain_list(raw_move)
        if len(parts) != 5:
            warnings.append(f"{replay_id}: skipped malformed move {index}: expected 5 fields")
            continue
        player_id, start, end, split, turn = parts
        try:
            move = RawReplayMove(
                player_id=int(player_id),
                start=int(start),
                end=int(end),
                split=bool(int(split)),
                turn=int(turn),
            )
        except (TypeError, ValueError) as exc:
            warnings.append(f"{replay_id}: skipped malformed move {index}: {exc}")
            continue

        if move.player_id not in (0, 1):
            warnings.append(f"{replay_id}: move {index} has invalid player_id {move.player_id}")
        if not _tile_in_bounds(move.start, size):
            warnings.append(f"{replay_id}: move {index} start out of bounds: {move.start}")
        if not _tile_in_bounds(move.end, size):
            warnings.append(f"{replay_id}: move {index} end out of bounds: {move.end}")
        if _tile_in_bounds(move.start, size) and _tile_in_bounds(move.end, size):
            if not _tiles_adjacent(move.start, move.end, width):
                warnings.append(
                    f"{replay_id}: move {index} is not adjacent: {move.start}->{move.end}"
                )
        moves.append(move)
    return _MovesResult(moves=moves, warnings=warnings)


def _required_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        raise ReplayFormatError(f"missing required field: {key}")
    return str(value)


def _required_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if value is None:
        raise ReplayFormatError(f"missing required field: {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ReplayFormatError(f"invalid integer field {key}: {value!r}") from exc


def _int_list(value: Any) -> list[int]:
    return [int(item) for item in _plain_list(value)]


def _optional_int_list(value: Any) -> list[int | None]:
    result: list[int | None] = []
    for item in _plain_list(value):
        result.append(None if item is None else int(item))
    return result


def _str_list(value: Any) -> list[str]:
    return [str(item) for item in _plain_list(value)]


def _plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return list(value)


def _tile_in_bounds(tile: int, size: int) -> bool:
    return 0 <= tile < size


def _tiles_adjacent(start: int, end: int, width: int) -> bool:
    start_row, start_col = divmod(start, width)
    end_row, end_col = divmod(end, width)
    return abs(start_row - end_row) + abs(start_col - end_col) == 1
