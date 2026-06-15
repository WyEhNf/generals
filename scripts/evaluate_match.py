#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generals_bot.matches import (
    build_game_config,
    make_policy,
    parse_agent_specs,
    run_match,
    summarize_match_reports,
)
from generals_bot.matches.policy_factory import PolicySpec
from generals_bot.sim import GeneralsEnv
from generals_bot.sim.types import PassAction
from generals_bot.training.batched_policy import BatchedBCPolicyRunner
from generals_bot.training.router import BatchedRouterPolicyRunner
from generals_bot.viz import write_html


DEFAULT_OUTPUT_DIR = "artifacts/evaluations/matches"
DEFAULT_TURNS = 800


def main() -> int:
    """Evaluate local agent specs over many seeds."""
    parser = argparse.ArgumentParser(description="Evaluate local agent-vs-agent matches.")
    parser.add_argument("--agent", action="append", required=True)
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help="Optional pair formatted as LABEL_A:LABEL_B. Defaults to all unordered pairs.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=24)
    parser.add_argument("--map", choices=["generated", "demo"], default="generated")
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--pass-bias", type=float, default=0.0)
    parser.add_argument("--pass-bias-turn", type=int, default=50)
    parser.add_argument("--late-start", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-html", action="store_true")
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--no-swapped-side", action="store_true")
    args = parser.parse_args()

    specs = parse_agent_specs(args.agent)
    pairs = _requested_pairs(list(specs), args.pair)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for label_a, label_b in pairs:
        summary = _evaluate_pair(
            label_a=label_a,
            spec_a=specs[label_a],
            label_b=label_b,
            spec_b=specs[label_b],
            output_dir=output_dir / f"{_slug(label_a)}__vs__{_slug(label_b)}",
            args=args,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    return 0


def _evaluate_pair(
    *,
    label_a: str,
    spec_a: PolicySpec,
    label_b: str,
    spec_b: PolicySpec,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Evaluate one requested agent pair over all requested seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.batched and args.write_html:
        raise ValueError("--batched does not support --write-html")
    swapped_side = not args.no_swapped_side and label_a != label_b
    if args.batched:
        reports = _run_pair_batched(
            label_a=label_a,
            spec_a=spec_a,
            label_b=label_b,
            spec_b=spec_b,
            swapped_side=swapped_side,
            args=args,
        )
        for report in reports:
            _write_report(output_dir / f"{_report_suffix(report)}.json", report)
    else:
        reports = []
        policy_cache = {}
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            for side_specs in _side_assignments(
                spec_a=spec_a,
                spec_b=spec_b,
                swapped_side=swapped_side,
            ):
                suffix = _report_suffix({
                    "seed": seed,
                    "side_labels": {player: spec.label for player, spec in side_specs.items()},
                })
                report, snapshots = _run_pair_serial(
                    side_specs=side_specs,
                    seed=seed,
                    args=args,
                    record_snapshots=args.write_html,
                    policy_cache=policy_cache,
                )
                _write_report(output_dir / f"{suffix}.json", report)
                if args.write_html:
                    write_html(snapshots, output_dir / f"{suffix}.html")
                reports.append(report)

    summary = summarize_match_reports(
        reports,
        label_a=label_a,
        label_b=label_b,
        agent_a=spec_a.raw,
        agent_b=spec_b.raw,
        turns=args.turns,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        pass_bias=args.pass_bias,
        pass_bias_turn=args.pass_bias_turn,
        inference_mode="batched" if args.batched else "serial",
    )
    _write_report(output_dir / "summary.json", summary)
    return summary


def _run_pair_serial(
    *,
    side_specs: dict[int, PolicySpec],
    seed: int,
    args: argparse.Namespace,
    record_snapshots: bool,
    policy_cache: dict[tuple[str, int], object] | None = None,
) -> tuple[dict, list]:
    """Run one serial match from concrete player-side specs."""
    config = build_game_config(
        args.map,
        seed=seed,
        max_turns=args.turns,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    policies = {
        player: _make_serial_policy(
            spec=spec,
            player=player,
            seed=seed + player,
            config=config,
            args=args,
            policy_cache=policy_cache,
        )
        for player, spec in side_specs.items()
    }
    result = run_match(
        config=config,
        policies=policies,
        side_labels={player: spec.label for player, spec in side_specs.items()},
        seed=seed,
        record_snapshots=record_snapshots,
        late_start=args.late_start,
    )
    report = {
        "agent_specs": {str(player): spec.raw for player, spec in side_specs.items()},
        "inference_mode": "serial",
        **result.report,
    }
    return report, result.snapshots


def _make_serial_policy(
    *,
    spec: PolicySpec,
    player: int,
    seed: int,
    config,
    args: argparse.Namespace,
    policy_cache: dict[tuple[str, int], object] | None,
):
    """Instantiate or reuse a serial policy for one player side."""
    if spec.is_checkpoint and policy_cache is not None:
        key = (spec.label, player)
        if key not in policy_cache:
            policy_cache[key] = make_policy(
                spec,
                player=player,
                seed=seed,
                config=config,
                device=args.device,
                pass_bias=args.pass_bias,
                pass_bias_turn=args.pass_bias_turn,
            )
        else:
            policy_cache[key].reset(player, config)
        return policy_cache[key]
    return make_policy(
        spec,
        player=player,
        seed=seed,
        config=config,
        device=args.device,
        pass_bias=args.pass_bias,
        pass_bias_turn=args.pass_bias_turn,
    )


def _run_pair_batched(
    *,
    label_a: str,
    spec_a: PolicySpec,
    label_b: str,
    spec_b: PolicySpec,
    swapped_side: bool,
    args: argparse.Namespace,
) -> list[dict]:
    """Run one checkpoint pair with batched model inference across all games."""
    if not spec_a.supports_batched or not spec_b.supports_batched:
        raise ValueError(
            "--batched currently supports checkpoint:<path> and router:<path> agents only"
        )

    runners = {
        label: _make_batched_runner(spec, args)
        for label, spec in {label_a: spec_a, label_b: spec_b}.items()
    }
    for runner in runners.values():
        runner.reset_all()

    games = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for side_specs in _side_assignments(
            spec_a=spec_a,
            spec_b=spec_b,
            swapped_side=swapped_side,
        ):
            config = build_game_config(
                args.map,
                seed=seed,
                max_turns=args.turns,
                min_size=args.min_size,
                max_size=args.max_size,
            )
            env = GeneralsEnv(config)
            games.append({
                "seed": seed,
                "side_labels": {player: spec.label for player, spec in side_specs.items()},
                "side_specs": side_specs,
                "env": env,
                "observations": env.reset(seed=seed),
                "submitted_actions": 0,
                "invalid_actions": 0,
                "late_turns": 0,
                "late_both_pass": 0,
                "max_both_pass_streak": 0,
                "current_streak": 0,
            })

    while True:
        items_by_label: dict[str, list[tuple[tuple[int, int], object]]] = {
            label: []
            for label in runners
        }
        for game_index, game in enumerate(games):
            env = game["env"]
            if env.done:
                continue
            observations = game["observations"]
            for player in env.alive_players:
                label = game["side_labels"][player]
                items_by_label[label].append(((game_index, player), observations[player]))

        if not any(items_by_label.values()):
            break

        actions_by_key = {}
        for label, items in items_by_label.items():
            if items:
                actions_by_key.update(runners[label].act_many(items))

        for game_index, game in enumerate(games):
            env = game["env"]
            if env.done:
                continue
            turn = env.state.turn if env.state is not None else 0
            actions = {
                player: actions_by_key[(game_index, player)]
                for player in env.alive_players
            }
            if turn >= args.late_start and len(actions) == 2:
                game["late_turns"] += 1
                if all(isinstance(action, PassAction) for action in actions.values()):
                    game["late_both_pass"] += 1
                    game["current_streak"] += 1
                    game["max_both_pass_streak"] = max(
                        game["max_both_pass_streak"],
                        game["current_streak"],
                    )
                else:
                    game["current_streak"] = 0
            game["observations"], _, _, info = env.step(actions)
            game["submitted_actions"] += sum(
                len(value)
                for value in info["submitted_actions"].values()
            )
            game["invalid_actions"] += sum(
                1 for executed in info["executed_actions"]
                if not executed.valid
            )

    reports = []
    for game in games:
        env = game["env"]
        winner_side = env.state.winner if env.state is not None else None
        side_labels = game["side_labels"]
        side_specs = game["side_specs"]
        reports.append({
            "agent_specs": {str(player): spec.raw for player, spec in side_specs.items()},
            "seed": game["seed"],
            "side_labels": side_labels,
            "turns": env.state.turn if env.state is not None else None,
            "winner_side": winner_side,
            "winner_label": side_labels[winner_side] if winner_side is not None else None,
            "submitted_actions": game["submitted_actions"],
            "invalid_actions": game["invalid_actions"],
            "late_start": args.late_start,
            "late_turns": game["late_turns"],
            "late_both_pass": game["late_both_pass"],
            "late_both_pass_rate": (
                game["late_both_pass"] / game["late_turns"]
                if game["late_turns"]
                else None
            ),
            "max_both_pass_streak": game["max_both_pass_streak"],
            "inference_mode": "batched",
            "scores": [score.__dict__ for score in env.scores],
        })
    return reports


def _make_batched_runner(spec: PolicySpec, args: argparse.Namespace):
    """Create a batched runner for a checkpoint or router policy spec."""
    if spec.kind == "checkpoint":
        return BatchedBCPolicyRunner(
            spec.value,
            device=args.device,
            pass_bias=args.pass_bias,
            pass_bias_turn=args.pass_bias_turn,
        )
    if spec.kind == "router":
        return BatchedRouterPolicyRunner(
            spec.value,
            device=args.device,
        )
    raise ValueError(f"unsupported batched policy kind: {spec.kind}")


def _requested_pairs(labels: list[str], requested: list[str] | None) -> list[tuple[str, str]]:
    """Resolve requested label pairs or default pairs."""
    label_set = set(labels)
    if requested is None:
        if len(labels) == 1:
            return [(labels[0], labels[0])]
        return list(itertools.combinations(labels, 2))
    pairs = []
    for raw_pair in requested:
        if ":" not in raw_pair:
            raise ValueError("--pair must be formatted as LABEL_A:LABEL_B")
        label_a, label_b = raw_pair.split(":", 1)
        if label_a not in label_set or label_b not in label_set:
            raise ValueError(f"unknown agent label in pair: {raw_pair}")
        pairs.append((label_a, label_b))
    return pairs


def _side_assignments(
    *,
    spec_a: PolicySpec,
    spec_b: PolicySpec,
    swapped_side: bool,
) -> list[dict[int, PolicySpec]]:
    """Return player-side assignments for one seed."""
    assignments = [{0: spec_a, 1: spec_b}]
    if swapped_side:
        assignments.append({0: spec_b, 1: spec_a})
    return assignments


def _write_report(path: Path, data: dict) -> None:
    """Write one JSON report, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    """Create a filesystem-friendly label."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "agent"


def _report_suffix(report: dict) -> str:
    """Build a stable report filename suffix from a match report."""
    side_labels = report["side_labels"]
    return (
        f"seed{int(report['seed']):03d}_"
        f"{_slug(side_labels[0])}_p0_"
        f"{_slug(side_labels[1])}_p1"
    )


if __name__ == "__main__":
    raise SystemExit(main())
