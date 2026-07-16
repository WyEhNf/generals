#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generals_bot.replay import prepare_raw_replay_shards


def main() -> int:
    """Globally shuffle replay rows and write bounded gzip shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = prepare_raw_replay_shards(
        args.input,
        args.output_dir,
        shard_size=args.shard_size,
        seed=args.seed,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
