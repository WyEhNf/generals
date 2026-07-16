#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze fixed-validation trend for a Phase 1 temporal run."
    )
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", default="loss")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--minimum-improvement", type=float, default=0.001)
    args = parser.parse_args()

    if args.window < 2:
        raise ValueError("window must be at least 2")
    if args.minimum_improvement < 0:
        raise ValueError("minimum-improvement must be nonnegative")

    rows_by_step: dict[int, dict] = {}
    for raw_line in Path(args.metrics).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if row.get("phase") == "val" and args.metric in row:
            rows_by_step[int(row["step"])] = row
    rows = [rows_by_step[step] for step in sorted(rows_by_step)]
    required = args.window * 2
    if len(rows) < required:
        raise ValueError(
            f"need at least {required} validation points, found {len(rows)}"
        )

    values = [float(row[args.metric]) for row in rows]
    previous = values[-required : -args.window]
    recent = values[-args.window :]
    previous_mean = fmean(previous)
    recent_mean = fmean(recent)
    improvement = previous_mean - recent_mean
    recent_steps = [int(row["step"]) for row in rows[-args.window :]]
    recent_slope_per_1k = _linear_slope(recent_steps, recent) * 1000.0
    best_index = min(range(len(values)), key=values.__getitem__)
    best_step = int(rows[best_index]["step"])
    recent_start_step = int(rows[-args.window]["step"])

    # Continue only when two independent signs agree: the recent window beats
    # the preceding fixed-validation window by a meaningful margin, and the
    # best observed checkpoint belongs to the recent window.
    continue_to_50k = (
        improvement >= args.minimum_improvement
        and best_step >= recent_start_step
    )
    report = {
        "metrics": args.metrics,
        "metric": args.metric,
        "validation_points": len(rows),
        "window": args.window,
        "minimum_improvement": args.minimum_improvement,
        "previous_window": {
            "steps": [int(row["step"]) for row in rows[-required : -args.window]],
            "mean": previous_mean,
        },
        "recent_window": {
            "steps": recent_steps,
            "mean": recent_mean,
            "slope_per_1000_steps": recent_slope_per_1k,
        },
        "improvement": improvement,
        "best": {"step": best_step, "value": values[best_index]},
        "decision": {
            "continue_to_50k": continue_to_50k,
            "rule": (
                "recent mean improves over previous mean by at least the "
                "configured margin, and the global best lies in the recent window"
            ),
        },
        "validation_rows": [
            {"step": int(row["step"]), args.metric: float(row[args.metric])}
            for row in rows
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _linear_slope(xs: list[int], ys: list[float]) -> float:
    x_mean = fmean(xs)
    y_mean = fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)
    ) / denominator


if __name__ == "__main__":
    raise SystemExit(main())
