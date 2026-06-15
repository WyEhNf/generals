from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from generals_bot.replay.schema import (
    RAW_SCHEMA_VERSION,
    RawReplay,
    ReplayFormatError,
    normalize_hf_row,
    raw_replay_from_dict,
)


@dataclass(frozen=True)
class ProcessingResult:
    manifest: dict[str, Any]
    warnings: list[str]


@dataclass(frozen=True)
class ShardingResult:
    manifest: dict[str, Any]


def prepare_raw_replays(
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    limit: int | None = None,
) -> ProcessingResult:
    input_path = Path(input_path)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    replay_count = 0
    skipped_rows = 0
    move_count = 0
    warnings: list[str] = []

    with gzip.open(output_path, "wt", encoding="utf-8") as output:
        for row in iter_hf_rows(input_path):
            if limit is not None and replay_count >= limit:
                break
            row_count += 1
            try:
                normalized = normalize_hf_row(row)
            except ReplayFormatError as exc:
                skipped_rows += 1
                warnings.append(str(exc))
                continue
            replay = normalized.replay
            warnings.extend(normalized.warnings)
            output.write(json.dumps(replay.to_dict(), ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
            replay_count += 1
            move_count += len(replay.moves)

    manifest = {
        "schema_version": RAW_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "row_count_read": row_count,
        "replay_count": replay_count,
        "skipped_rows": skipped_rows,
        "move_count": move_count,
        "warning_count": len(warnings),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ProcessingResult(manifest=manifest, warnings=warnings)


def prepare_raw_replay_shards(
    input_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    *,
    shard_size: int = 512,
    seed: int = 0,
    limit: int | None = None,
    overwrite: bool = False,
) -> ShardingResult:
    """Shuffle raw replay rows into bounded replay shards."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json" if manifest_path is None else Path(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing_shards = sorted(output_dir.glob("shard_*.jsonl.gz"))
    if existing_shards and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains replay shards; pass overwrite=True to replace them"
        )
    if overwrite:
        for shard_path in existing_shards:
            shard_path.unlink()

    rows = list(_iter_raw_json_lines(input_path, limit=limit))
    rng = random.Random(seed)
    rng.shuffle(rows)

    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start : start + shard_size]
        shard_path = output_dir / f"shard_{shard_index:05d}.jsonl.gz"
        with gzip.open(shard_path, "wt", encoding="utf-8") as output:
            for row in shard_rows:
                output.write(row)
                output.write("\n")
        shards.append({
            "file": shard_path.name,
            "replay_count": len(shard_rows),
        })

    manifest = {
        "schema_version": RAW_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "shard_size": shard_size,
        "seed": seed,
        "limit": limit,
        "replay_count": len(rows),
        "shard_count": len(shards),
        "shards": shards,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ShardingResult(manifest=manifest)


def iter_hf_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        yield from _iter_jsonl_rows(path)
        return
    if path.suffix == ".parquet":
        yield from _iter_parquet_rows(path)
        return
    raise ValueError(f"unsupported replay input format: {path}")


def iter_raw_replays(
    path: str | Path,
    *,
    shard_seed: int | None = None,
) -> Iterable[RawReplay]:
    """Yield raw replays from one file or a directory of replay shard files."""
    path = Path(path)
    if path.is_dir():
        shard_paths = list_raw_replay_files(path)
        if shard_seed is not None:
            random.Random(shard_seed).shuffle(shard_paths)
        for shard_path in shard_paths:
            yield from iter_raw_replays(shard_path)
        return

    opener = gzip.open if "".join(path.suffixes).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield raw_replay_from_dict(json.loads(line))


def list_raw_replay_files(path: str | Path) -> list[Path]:
    """List replay jsonl files represented by a file or shard directory."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"raw replay path does not exist: {path}")
    files = sorted(
        file_path
        for file_path in path.iterdir()
        if file_path.is_file()
        and (
            "".join(file_path.suffixes).endswith(".jsonl")
            or "".join(file_path.suffixes).endswith(".jsonl.gz")
        )
    )
    if not files:
        raise ValueError(f"raw replay directory contains no jsonl files: {path}")
    return files


def load_raw_replay(
    path: str | Path,
    *,
    replay_id: str | None = None,
    index: int = 0,
) -> RawReplay:
    if replay_id is None and index < 0:
        raise ValueError("index must be nonnegative")
    for current_index, replay in enumerate(iter_raw_replays(path)):
        if replay_id is not None:
            if replay.replay_id == replay_id:
                return replay
        elif current_index == index:
            return replay
    if replay_id is not None:
        raise ValueError(f"replay id not found: {replay_id}")
    raise ValueError(f"replay index not found: {index}")


def _iter_raw_json_lines(
    path: Path,
    *,
    limit: int | None,
) -> Iterable[str]:
    """Yield canonical raw replay JSON lines from one prepared replay file."""
    for index, row in enumerate(_iter_jsonl_rows(path)):
        if limit is not None and index >= limit:
            break
        replay = raw_replay_from_dict(row)
        yield json.dumps(replay.to_dict(), ensure_ascii=False, separators=(",", ":"))


def _iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if "".join(path.suffixes).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required to read parquet files. "
            "Install data dependencies with: "
            ".venv/bin/python -m pip install -r requirements-data.txt"
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches():
        yield from batch.to_pylist()
