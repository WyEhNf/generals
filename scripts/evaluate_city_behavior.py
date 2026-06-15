#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generals_bot.matches import build_game_config, make_policy, parse_agent_specs
from generals_bot.matches.policy_factory import PolicySpec
from generals_bot.sim import GeneralsEnv
from generals_bot.sim.types import Terrain


DEFAULT_OUTPUT_DIR = "artifacts/evaluations/city_behavior"
DEFAULT_TARGET_TURN = 1200


def main() -> int:
    """Evaluate city attack and capture behavior for local agents."""
    parser = argparse.ArgumentParser(
        description="Measure neutral city attack and capture behavior.",
    )
    parser.add_argument("--agent", action="append", required=True)
    parser.add_argument("--mode", choices=["selfplay", "h2h"], default="selfplay")
    parser.add_argument(
        "--pair",
        default=None,
        help="H2H pair formatted as LABEL_A:LABEL_B. Defaults to the two agents if unambiguous.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-turn", type=int, default=DEFAULT_TARGET_TURN)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--map", choices=["generated", "demo"], default="generated")
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--pass-bias", type=float, default=0.0)
    parser.add_argument("--pass-bias-turn", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Disable automatic batched checkpoint inference.",
    )
    parser.add_argument("--no-swapped-side", action="store_true")
    args = parser.parse_args()

    if args.target_turn < 0:
        raise ValueError("--target-turn must be nonnegative")
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive")

    specs = parse_agent_specs(args.agent)
    pair = None
    if args.mode == "h2h":
        pair = _requested_pair(list(specs), args.pair)
    elif args.pair is not None:
        raise ValueError("--pair is only supported with --mode h2h")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "h2h":
        records, runner_summaries = evaluate_h2h(specs, pair, args)
    else:
        records, runner_summaries = evaluate_selfplay(specs, args)
    summary = summarize_records(
        records,
        config={
            "mode": args.mode,
            "pair": None if pair is None else list(pair),
            "swapped_side": None if pair is None else not args.no_swapped_side,
            "target_turn": args.target_turn,
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "map": args.map,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "pass_bias": args.pass_bias,
            "pass_bias_turn": args.pass_bias_turn,
            "device": args.device,
            "inference_mode": (
                "serial"
                if args.serial
                else _inference_mode(specs, pair=pair)
            ),
        },
        runner_summaries=runner_summaries,
    )

    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary["records_path"] = str(records_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(summary_path)
    return 0


def evaluate_selfplay(
    specs: dict[str, PolicySpec],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate every agent spec against itself."""
    if not args.serial and all(spec.supports_batched for spec in specs.values()):
        return _evaluate_selfplay_batched(specs, args)
    return _evaluate_selfplay_serial(specs, args), {}


def evaluate_h2h(
    specs: dict[str, PolicySpec],
    pair: tuple[str, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one head-to-head pair for city behavior."""
    label_a, label_b = pair
    selected_specs = {label_a: specs[label_a], label_b: specs[label_b]}
    if not args.serial and all(spec.supports_batched for spec in selected_specs.values()):
        return _evaluate_h2h_batched(selected_specs, pair, args)
    return _evaluate_h2h_serial(selected_specs, pair, args), {}


def summarize_records(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    runner_summaries: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate city behavior records into per-agent metrics."""
    if not records:
        raise ValueError("cannot summarize zero city behavior records")
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_agent[record["agent_label"]].append(record)
    agents = {
        label: _summarize_agent_records(agent_records)
        for label, agent_records in sorted(by_agent.items())
    }
    ranked = sorted(
        (
            {
                "agent_label": label,
                "mean_neutral_city_captures": data["mean_neutral_city_captures"],
                "mean_owned_cities": data["mean_owned_cities"],
                "city_attack_success_rate": data["city_attack_success_rate"],
                "samples": data["samples"],
            }
            for label, data in agents.items()
        ),
        key=lambda row: (
            row["mean_neutral_city_captures"],
            row["mean_owned_cities"],
        ),
        reverse=True,
    )
    return {
        "config": config,
        "agents": agents,
        "ranked_by_city_captures": ranked,
        "records": len(records),
        "runner_summaries": runner_summaries,
    }


def _evaluate_selfplay_batched(
    specs: dict[str, PolicySpec],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run batched model self-play games with one runner per agent."""
    runners = {
        label: _make_batched_runner(spec, args)
        for label, spec in specs.items()
    }
    for runner in runners.values():
        runner.reset_all()

    games = []
    for label, spec in specs.items():
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            config = _build_config(args, seed=seed)
            env = GeneralsEnv(config)
            games.append({
                "agent_label": label,
                "agent_spec": spec.raw,
                "seed": seed,
                "game_id": f"{label}:seed{seed}:selfplay",
                "config": config,
                "env": env,
                "observations": env.reset(seed=seed),
                "city_stats": _empty_game_city_stats(),
            })

    _run_batched_games(games, runners, args)
    records = []
    for game in games:
        records.extend(_score_records(game, target_turn=args.target_turn))
    return records, {label: runner.summary() for label, runner in runners.items()}


def _evaluate_selfplay_serial(
    specs: dict[str, PolicySpec],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Run self-play games with separately instantiated policies."""
    records = []
    for label, spec in specs.items():
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            config = _build_config(args, seed=seed)
            env = GeneralsEnv(config)
            observations = env.reset(seed=seed)
            policies = {
                player: make_policy(
                    spec,
                    player=player,
                    seed=seed + player,
                    config=config,
                    device=args.device,
                    pass_bias=args.pass_bias,
                    pass_bias_turn=args.pass_bias_turn,
                )
                for player in (0, 1)
            }
            game = {
                "agent_label": label,
                "agent_spec": spec.raw,
                "seed": seed,
                "game_id": f"{label}:seed{seed}:selfplay",
                "config": config,
                "env": env,
                "city_stats": _empty_game_city_stats(),
            }
            _run_serial_game(game, observations, policies, args)
            records.extend(_score_records(game, target_turn=args.target_turn))
    return records


def _evaluate_h2h_batched(
    specs: dict[str, PolicySpec],
    pair: tuple[str, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run batched model head-to-head games with one runner per agent."""
    runners = {
        label: _make_batched_runner(spec, args)
        for label, spec in specs.items()
    }
    for runner in runners.values():
        runner.reset_all()
    games = _build_h2h_games(specs, pair, args)
    _run_batched_games(games, runners, args)
    records = []
    for game in games:
        records.extend(_score_records(game, target_turn=args.target_turn))
    return records, {label: runner.summary() for label, runner in runners.items()}


def _evaluate_h2h_serial(
    specs: dict[str, PolicySpec],
    pair: tuple[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Run head-to-head games with serial policy inference."""
    records = []
    for game in _build_h2h_games(specs, pair, args):
        policies = {
            player: make_policy(
                game["side_specs"][player],
                player=player,
                seed=game["seed"] + player,
                config=game["config"],
                device=args.device,
                pass_bias=args.pass_bias,
                pass_bias_turn=args.pass_bias_turn,
            )
            for player in (0, 1)
        }
        _run_serial_game(game, game["observations"], policies, args)
        records.extend(_score_records(game, target_turn=args.target_turn))
    return records


def _run_batched_games(
    games: list[dict[str, Any]],
    runners: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Advance a group of games with batched checkpoint inference."""
    while True:
        items_by_label: dict[str, list[tuple[tuple[int, int], object]]] = {
            label: []
            for label in runners
        }
        for game_index, game in enumerate(games):
            env = game["env"]
            if env.done or _turn(env) >= args.target_turn:
                continue
            observations = game["observations"]
            for player in env.alive_players:
                label = _player_label(game, player)
                items_by_label[label].append(((game_index, player), observations[player]))

        if not any(items_by_label.values()):
            break

        actions_by_key = {}
        for label, items in items_by_label.items():
            if items:
                actions_by_key.update(runners[label].act_many(items))

        for game_index, game in enumerate(games):
            env = game["env"]
            if env.done or _turn(env) >= args.target_turn:
                continue
            actions = {
                player: actions_by_key[(game_index, player)]
                for player in env.alive_players
            }
            game["observations"], _, _, info = env.step(actions)
            _record_city_events(game["city_stats"], game["config"], info["executed_actions"])


def _run_serial_game(
    game: dict[str, Any],
    observations,
    policies,
    args: argparse.Namespace,
) -> None:
    """Advance one game with serial policy inference and record city events."""
    env = game["env"]
    while not env.done and _turn(env) < args.target_turn:
        actions = {
            player: policies[player].act(observations[player])
            for player in env.alive_players
        }
        observations, _, _, info = env.step(actions)
        _record_city_events(game["city_stats"], game["config"], info["executed_actions"])


def _build_h2h_games(
    specs: dict[str, PolicySpec],
    pair: tuple[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Build head-to-head game records for all seeds and side assignments."""
    label_a, label_b = pair
    games = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for side_specs in _side_assignments(
            spec_a=specs[label_a],
            spec_b=specs[label_b],
            swapped_side=not args.no_swapped_side and label_a != label_b,
        ):
            config = _build_config(args, seed=seed)
            env = GeneralsEnv(config)
            side_labels = {player: spec.label for player, spec in side_specs.items()}
            games.append({
                "seed": seed,
                "game_id": (
                    f"seed{seed}:"
                    f"{side_labels[0]}_p0:"
                    f"{side_labels[1]}_p1"
                ),
                "config": config,
                "env": env,
                "observations": env.reset(seed=seed),
                "side_labels": side_labels,
                "side_specs": side_specs,
                "city_stats": _empty_game_city_stats(),
            })
    return games


def _record_city_events(
    city_stats: dict[int, dict[str, Any]],
    config,
    executed_actions,
) -> None:
    """Record neutral city attacks and captures from executed actions."""
    city_tiles = set(config.cities)
    for executed in executed_actions:
        if not executed.valid:
            continue
        if executed.action.end not in city_tiles:
            continue
        if executed.target_owner_before != int(Terrain.NEUTRAL):
            continue
        stats = city_stats[executed.player_id]
        stats["neutral_city_attack_count"] += 1
        stats["target_city_army_before_sum"] += int(executed.target_army_before)
        stats["target_city_army_before_count"] += 1
        if stats["first_neutral_city_attack_turn"] is None:
            stats["first_neutral_city_attack_turn"] = int(executed.turn)
        if executed.moved_army > executed.target_army_before:
            stats["neutral_city_capture_count"] += 1
            if stats["first_neutral_city_capture_turn"] is None:
                stats["first_neutral_city_capture_turn"] = int(executed.turn)
        else:
            stats["failed_neutral_city_attack_count"] += 1


def _score_records(game: dict[str, Any], *, target_turn: int) -> list[dict[str, Any]]:
    """Convert one evaluated game into per-player city behavior records."""
    env: GeneralsEnv = game["env"]
    config = game["config"]
    measured_turn = _turn(env)
    winner_side = env.state.winner if env.state is not None else None
    side_labels = game.get("side_labels")
    side_specs = game.get("side_specs")
    city_counts = _owned_city_counts(env)
    records = []
    for score in env.scores:
        stats = game["city_stats"][score.player_id]
        attack_count = int(stats["neutral_city_attack_count"])
        capture_count = int(stats["neutral_city_capture_count"])
        records.append({
            "agent_label": (
                side_labels[score.player_id]
                if side_labels is not None
                else game["agent_label"]
            ),
            "agent_spec": (
                side_specs[score.player_id].raw
                if side_specs is not None
                else game["agent_spec"]
            ),
            "opponent_label": (
                side_labels[1 - score.player_id]
                if side_labels is not None
                else game["agent_label"]
            ),
            "seed": game["seed"],
            "game_id": game.get("game_id", str(game["seed"])),
            "player_id": score.player_id,
            "target_turn": target_turn,
            "measured_turn": measured_turn,
            "ended_before_target": measured_turn < target_turn,
            "winner_side": winner_side,
            "owned_city_count_at_target": city_counts[score.player_id],
            "neutral_city_attack_count": attack_count,
            "neutral_city_capture_count": capture_count,
            "failed_neutral_city_attack_count": int(
                stats["failed_neutral_city_attack_count"]
            ),
            "first_neutral_city_attack_turn": stats["first_neutral_city_attack_turn"],
            "first_neutral_city_capture_turn": stats["first_neutral_city_capture_turn"],
            "city_attack_success_rate": (
                capture_count / attack_count
                if attack_count
                else None
            ),
            "mean_target_city_army_before": (
                stats["target_city_army_before_sum"]
                / stats["target_city_army_before_count"]
                if stats["target_city_army_before_count"]
                else None
            ),
            "land": score.land,
            "army": score.army,
            "dead": score.dead,
            "has_kill": score.has_kill,
            "width": config.width,
            "height": config.height,
        })
    return records


def _summarize_agent_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all per-player city behavior records for one agent label."""
    attacks = [int(record["neutral_city_attack_count"]) for record in records]
    captures = [int(record["neutral_city_capture_count"]) for record in records]
    failed = [int(record["failed_neutral_city_attack_count"]) for record in records]
    owned_cities = [int(record["owned_city_count_at_target"]) for record in records]
    target_armies = [
        float(record["mean_target_city_army_before"])
        for record in records
        if record["mean_target_city_army_before"] is not None
    ]
    attack_turns = [
        int(record["first_neutral_city_attack_turn"])
        for record in records
        if record["first_neutral_city_attack_turn"] is not None
    ]
    capture_turns = [
        int(record["first_neutral_city_capture_turn"])
        for record in records
        if record["first_neutral_city_capture_turn"] is not None
    ]
    total_attacks = sum(attacks)
    total_captures = sum(captures)
    return {
        "agent_spec": records[0]["agent_spec"],
        "samples": len(records),
        "games": len({record.get("game_id", record["seed"]) for record in records}),
        "total_neutral_city_attacks": total_attacks,
        "total_neutral_city_captures": total_captures,
        "total_failed_neutral_city_attacks": sum(failed),
        "mean_neutral_city_attacks": statistics.fmean(attacks),
        "mean_neutral_city_captures": statistics.fmean(captures),
        "mean_failed_neutral_city_attacks": statistics.fmean(failed),
        "city_attack_success_rate": (
            total_captures / total_attacks
            if total_attacks
            else None
        ),
        "mean_owned_cities": statistics.fmean(owned_cities),
        "median_owned_cities": statistics.median(owned_cities),
        "max_owned_cities": max(owned_cities),
        "mean_first_neutral_city_attack_turn": (
            statistics.fmean(attack_turns) if attack_turns else None
        ),
        "mean_first_neutral_city_capture_turn": (
            statistics.fmean(capture_turns) if capture_turns else None
        ),
        "mean_target_city_army_before": (
            statistics.fmean(target_armies) if target_armies else None
        ),
        "ended_before_target": sum(1 for record in records if record["ended_before_target"]),
    }


def _empty_game_city_stats() -> dict[int, dict[str, Any]]:
    """Return per-player mutable counters for city behavior events."""
    return {
        player: {
            "neutral_city_attack_count": 0,
            "neutral_city_capture_count": 0,
            "failed_neutral_city_attack_count": 0,
            "first_neutral_city_attack_turn": None,
            "first_neutral_city_capture_turn": None,
            "target_city_army_before_sum": 0,
            "target_city_army_before_count": 0,
        }
        for player in (0, 1)
    }


def _owned_city_counts(env: GeneralsEnv) -> dict[int, int]:
    """Return the number of city tiles owned by each player."""
    if env.state is None:
        raise RuntimeError("environment has not been reset")
    return {
        player: int((env.state.cities & (env.state.terrain == player)).sum())
        for player in (0, 1)
    }


def _build_config(args: argparse.Namespace, *, seed: int):
    """Build a match config whose max turn reaches the requested target turn."""
    return build_game_config(
        args.map,
        seed=seed,
        max_turns=max(1, args.target_turn),
        min_size=args.min_size,
        max_size=args.max_size,
    )


def _requested_pair(labels: list[str], raw_pair: str | None) -> tuple[str, str]:
    """Resolve the requested head-to-head pair."""
    label_set = set(labels)
    if raw_pair is None:
        if len(labels) != 2:
            raise ValueError("--mode h2h needs --pair unless exactly two agents are provided")
        return labels[0], labels[1]
    if ":" not in raw_pair:
        raise ValueError("--pair must be formatted as LABEL_A:LABEL_B")
    label_a, label_b = raw_pair.split(":", 1)
    if label_a not in label_set or label_b not in label_set:
        raise ValueError(f"unknown agent label in pair: {raw_pair}")
    return label_a, label_b


def _side_assignments(
    *,
    spec_a: PolicySpec,
    spec_b: PolicySpec,
    swapped_side: bool,
) -> list[dict[int, PolicySpec]]:
    """Return player-side assignments for a seed."""
    assignments = [{0: spec_a, 1: spec_b}]
    if swapped_side:
        assignments.append({0: spec_b, 1: spec_a})
    return assignments


def _turn(env: GeneralsEnv) -> int:
    """Return the current simulator turn."""
    if env.state is None:
        raise RuntimeError("environment has not been reset")
    return int(env.state.turn)


def _player_label(game: dict[str, Any], player: int) -> str:
    """Return the agent label controlling one player in a game record."""
    side_labels = game.get("side_labels")
    if side_labels is not None:
        return side_labels[player]
    return game["agent_label"]


def _inference_mode(
    specs: dict[str, PolicySpec],
    *,
    pair: tuple[str, str] | None = None,
) -> str:
    """Return the default inference mode selected for the provided specs."""
    selected = specs.values() if pair is None else (specs[label] for label in pair)
    return "batched" if all(spec.supports_batched for spec in selected) else "serial"


def _make_batched_runner(spec: PolicySpec, args: argparse.Namespace):
    """Create a batched checkpoint or router runner lazily."""
    if spec.kind == "checkpoint":
        from generals_bot.training.batched_policy import BatchedBCPolicyRunner

        return BatchedBCPolicyRunner(
            spec.value,
            device=args.device,
            pass_bias=args.pass_bias,
            pass_bias_turn=args.pass_bias_turn,
        )
    if spec.kind == "router":
        from generals_bot.training.router import BatchedRouterPolicyRunner

        return BatchedRouterPolicyRunner(spec.value, device=args.device)
    raise ValueError(f"unsupported batched policy kind: {spec.kind}")


if __name__ == "__main__":
    raise SystemExit(main())
