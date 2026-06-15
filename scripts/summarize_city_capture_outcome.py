#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "artifacts/evaluations/city_behavior/city_capture_outcome.json"


def main() -> int:
    """Summarize how successful neutral-city captures correlate with outcomes."""
    parser = argparse.ArgumentParser(
        description="Summarize win/loss probabilities after neutral city captures.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Input records.jsonl, a city_behavior output directory, or a summary.json "
            "containing records_path. Can be repeated."
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--agent-label",
        action="append",
        default=None,
        help="Only include records for this agent label. Can be repeated.",
    )
    parser.add_argument(
        "--min-captures",
        type=int,
        default=1,
        help="Minimum successful neutral-city captures required to count as captured_city.",
    )
    args = parser.parse_args()

    if args.min_captures <= 0:
        raise ValueError("--min-captures must be positive")

    records_paths = resolve_records_paths([Path(value) for value in args.input])
    records = load_records(records_paths)
    if args.agent_label is not None:
        allowed = set(args.agent_label)
        records = [
            record
            for record in records
            if str(record.get("agent_label")) in allowed
        ]

    summary = summarize_records(
        records,
        min_captures=args.min_captures,
        config={
            "inputs": [str(path) for path in records_paths],
            "agent_labels": args.agent_label,
            "min_captures": args.min_captures,
        },
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(output_path)
    return 0


def resolve_records_paths(inputs: list[Path]) -> list[Path]:
    """Resolve CLI inputs into concrete city-behavior records files."""
    records_paths = []
    for path in inputs:
        if path.is_dir():
            resolved = path / "records.jsonl"
        elif path.name == "summary.json":
            resolved = _records_path_from_summary(path)
        else:
            resolved = path
        if not resolved.exists():
            raise FileNotFoundError(f"records file not found: {resolved}")
        records_paths.append(resolved)
    return records_paths


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Load city-behavior JSONL records and annotate their source path."""
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_source_path"] = str(path)
                record["_source_line"] = line_number
                records.append(record)
    return records


def summarize_records(
    records: list[dict[str, Any]],
    *,
    min_captures: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate capture/outcome probabilities overall and per agent."""
    if not records:
        raise ValueError("cannot summarize zero records")

    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_agent[str(record.get("agent_label"))].append(record)

    agents = {
        label: _summarize_group(agent_records, min_captures=min_captures)
        for label, agent_records in sorted(by_agent.items())
    }
    ranked = sorted(
        (
            {
                "agent_label": label,
                "p_win_given_capture": metrics["p_win_given_capture"],
                "p_loss_given_capture": metrics["p_loss_given_capture"],
                "captured_samples": metrics["captured_samples"],
                "samples": metrics["samples"],
            }
            for label, metrics in agents.items()
        ),
        key=lambda row: (
            _sort_probability(row["p_win_given_capture"]),
            row["captured_samples"],
        ),
        reverse=True,
    )
    return {
        "config": config,
        "overall": _summarize_group(records, min_captures=min_captures),
        "agents": agents,
        "ranked_by_p_win_given_capture": ranked,
    }


def _summarize_group(
    records: list[dict[str, Any]],
    *,
    min_captures: int,
) -> dict[str, Any]:
    """Summarize one group of player-game records."""
    outcomes = [_outcome(record) for record in records]
    capture_counts = [
        int(record.get("neutral_city_capture_count", 0))
        for record in records
    ]
    captured = [count >= min_captures for count in capture_counts]
    no_capture = [not value for value in captured]

    samples = len(records)
    captured_samples = sum(captured)
    no_capture_samples = samples - captured_samples
    wins = sum(outcome == "win" for outcome in outcomes)
    losses = sum(outcome == "loss" for outcome in outcomes)
    draws = sum(outcome == "draw_or_unfinished" for outcome in outcomes)

    capture_wins = _count_matching(captured, outcomes, "win")
    capture_losses = _count_matching(captured, outcomes, "loss")
    capture_draws = _count_matching(captured, outcomes, "draw_or_unfinished")
    no_capture_wins = _count_matching(no_capture, outcomes, "win")
    no_capture_losses = _count_matching(no_capture, outcomes, "loss")
    no_capture_draws = _count_matching(no_capture, outcomes, "draw_or_unfinished")

    return {
        "samples": samples,
        "min_captures": min_captures,
        "wins": wins,
        "losses": losses,
        "draws_or_unfinished": draws,
        "captured_samples": captured_samples,
        "no_capture_samples": no_capture_samples,
        "capture_win_count": capture_wins,
        "capture_loss_count": capture_losses,
        "capture_draw_or_unfinished_count": capture_draws,
        "no_capture_win_count": no_capture_wins,
        "no_capture_loss_count": no_capture_losses,
        "no_capture_draw_or_unfinished_count": no_capture_draws,
        "p_capture": _safe_div(captured_samples, samples),
        "p_capture_and_win": _safe_div(capture_wins, samples),
        "p_capture_and_loss": _safe_div(capture_losses, samples),
        "p_capture_and_draw_or_unfinished": _safe_div(capture_draws, samples),
        "p_win_given_capture": _safe_div(capture_wins, captured_samples),
        "p_loss_given_capture": _safe_div(capture_losses, captured_samples),
        "p_draw_or_unfinished_given_capture": _safe_div(
            capture_draws,
            captured_samples,
        ),
        "p_win_given_no_capture": _safe_div(no_capture_wins, no_capture_samples),
        "p_loss_given_no_capture": _safe_div(no_capture_losses, no_capture_samples),
        "p_draw_or_unfinished_given_no_capture": _safe_div(
            no_capture_draws,
            no_capture_samples,
        ),
        "p_capture_given_win": _safe_div(capture_wins, wins),
        "p_capture_given_loss": _safe_div(capture_losses, losses),
        "mean_captures": _mean(capture_counts),
        "mean_captures_when_win": _mean_for_outcome(capture_counts, outcomes, "win"),
        "mean_captures_when_loss": _mean_for_outcome(capture_counts, outcomes, "loss"),
        "mean_captures_when_draw_or_unfinished": _mean_for_outcome(
            capture_counts,
            outcomes,
            "draw_or_unfinished",
        ),
        "mean_captures_given_capture": _mean(
            [count for count in capture_counts if count >= min_captures],
        ),
    }


def _records_path_from_summary(path: Path) -> Path:
    """Read records_path from a city-behavior summary file."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    raw_records_path = summary.get("records_path")
    if not isinstance(raw_records_path, str):
        raise ValueError(f"summary has no records_path: {path}")
    records_path = Path(raw_records_path)
    if records_path.exists():
        return records_path
    return path.parent / records_path


def _outcome(record: dict[str, Any]) -> str:
    """Classify one player-game record as win, loss, or draw/unfinished."""
    winner_side = record.get("winner_side")
    if winner_side is None or winner_side == "None":
        return "draw_or_unfinished"
    player_id = int(record["player_id"])
    return "win" if int(winner_side) == player_id else "loss"


def _count_matching(
    masks: list[bool],
    outcomes: list[str],
    target_outcome: str,
) -> int:
    """Count records where both the mask and outcome match."""
    return sum(mask and outcome == target_outcome for mask, outcome in zip(masks, outcomes))


def _mean(values: list[int]) -> float | None:
    """Return a float mean for non-empty integer lists."""
    if not values:
        return None
    return float(statistics.fmean(values))


def _mean_for_outcome(
    values: list[int],
    outcomes: list[str],
    target_outcome: str,
) -> float | None:
    """Return the mean over values with a matching outcome."""
    return _mean([
        value
        for value, outcome in zip(values, outcomes)
        if outcome == target_outcome
    ])


def _safe_div(numerator: int, denominator: int) -> float | None:
    """Return numerator / denominator or None when the condition has no samples."""
    if denominator == 0:
        return None
    return numerator / denominator


def _sort_probability(value: float | None) -> float:
    """Sort missing probabilities behind real values."""
    if value is None:
        return -1.0
    return value


if __name__ == "__main__":
    raise SystemExit(main())
