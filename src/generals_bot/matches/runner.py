from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from generals_bot.agents import estimate_enemy_king_distribution, estimate_win_rate
from generals_bot.agents.base import Policy
from generals_bot.sim import GameConfig, GeneralsEnv
from generals_bot.sim.queue import normalize_actions
from generals_bot.viz import MatchRecorder, Snapshot


@dataclass(frozen=True)
class MatchResult:
    """Result of one local agent-vs-agent match."""

    snapshots: list[Snapshot]
    report: dict[str, Any]


def run_match(
    *,
    config: GameConfig,
    policies: Mapping[int, Policy],
    side_labels: Mapping[int, str] | None = None,
    seed: int | None = None,
    record_snapshots: bool = False,
    late_start: int = 50,
) -> MatchResult:
    """Run one local match and optionally record snapshots for visualization."""
    env = GeneralsEnv(config)
    observations = env.reset(seed=seed)
    resolved_labels = {
        player: (side_labels[player] if side_labels is not None else f"p{player}")
        for player in (0, 1)
    }
    for player, policy in policies.items():
        policy.reset(player, config)

    recorder = MatchRecorder() if record_snapshots else None
    if recorder is not None:
        recorder.capture(
            env,
            win_rates=_estimate_win_rates(policies, observations),
            enemy_king_distributions=_estimate_enemy_king_distributions(
                policies,
                observations,
            ),
        )

    submitted_actions = 0
    invalid_actions = 0
    late_turns = 0
    late_both_pass = 0
    max_both_pass_streak = 0
    current_both_pass_streak = 0

    while not env.done:
        turn = env.state.turn if env.state is not None else 0
        alive_players = list(env.alive_players)
        actions = {
            player: policies[player].act(observations[player])
            for player in alive_players
        }
        if turn >= late_start and len(actions) == 2:
            late_turns += 1
            if all(_is_pass_like(action, config.width) for action in actions.values()):
                late_both_pass += 1
                current_both_pass_streak += 1
                max_both_pass_streak = max(max_both_pass_streak, current_both_pass_streak)
            else:
                current_both_pass_streak = 0

        observations, _, _, info = env.step(actions)
        submitted_actions += sum(len(value) for value in info["submitted_actions"].values())
        invalid_actions += sum(1 for executed in info["executed_actions"] if not executed.valid)
        if recorder is not None:
            recorder.capture(
                env,
                win_rates=_estimate_win_rates(policies, observations),
                enemy_king_distributions=_estimate_enemy_king_distributions(
                    policies,
                    observations,
                ),
            )

    winner_side = env.state.winner if env.state is not None else None
    report = {
        "seed": seed,
        "side_labels": dict(resolved_labels),
        "turns": env.state.turn if env.state is not None else None,
        "winner_side": winner_side,
        "winner_label": resolved_labels[winner_side] if winner_side is not None else None,
        "submitted_actions": submitted_actions,
        "invalid_actions": invalid_actions,
        "late_start": late_start,
        "late_turns": late_turns,
        "late_both_pass": late_both_pass,
        "late_both_pass_rate": late_both_pass / late_turns if late_turns else None,
        "max_both_pass_streak": max_both_pass_streak,
        "scores": [score.__dict__ for score in env.scores],
    }
    return MatchResult(
        snapshots=[] if recorder is None else recorder.snapshots,
        report=report,
    )


def summarize_match_reports(
    reports: list[dict[str, Any]],
    *,
    label_a: str,
    label_b: str,
    agent_a: str,
    agent_b: str,
    turns: int,
    seed_start: int,
    seed_count: int,
    pass_bias: float,
    pass_bias_turn: int,
    inference_mode: str,
) -> dict[str, Any]:
    """Aggregate per-game reports into a head-to-head summary."""
    if not reports:
        raise ValueError("cannot summarize zero match reports")
    winner_counts = Counter(str(report["winner_label"]) for report in reports)
    side_winner_counts = Counter(str(report["winner_side"]) for report in reports)
    return {
        "label_a": label_a,
        "agent_a": agent_a,
        "label_b": label_b,
        "agent_b": agent_b,
        "turns": turns,
        "seed_start": seed_start,
        "seed_count": seed_count,
        "games": len(reports),
        "pass_bias": pass_bias,
        "pass_bias_turn": pass_bias_turn,
        "inference_mode": inference_mode,
        "winner_counts": dict(winner_counts),
        "side_winner_counts": dict(side_winner_counts),
        "unfinished_count": winner_counts.get("None", 0),
        "mean_turns": sum(report["turns"] for report in reports) / len(reports),
        "mean_submitted_actions": sum(
            report["submitted_actions"]
            for report in reports
        )
        / len(reports),
        "mean_invalid_actions": sum(
            report["invalid_actions"]
            for report in reports
        )
        / len(reports),
        "mean_late_both_pass_rate": sum(
            report["late_both_pass_rate"] or 0
            for report in reports
        )
        / len(reports),
        "max_both_pass_streak": max(report["max_both_pass_streak"] for report in reports),
        "bad_seed_count_streak_ge_100": sum(
            1 for report in reports
            if report["max_both_pass_streak"] >= 100
        ),
    }


def _estimate_win_rates(
    policies: Mapping[int, Policy],
    observations,
):
    """Return optional value-head estimates for current player observations."""
    return {
        player: estimate_win_rate(policy, observations[player])
        for player, policy in policies.items()
    }


def _estimate_enemy_king_distributions(
    policies: Mapping[int, Policy],
    observations,
):
    """Return optional enemy king probability maps for current observations."""
    return {
        player: estimate_enemy_king_distribution(policy, observations[player])
        for player, policy in policies.items()
    }


def _is_pass_like(action_like, width: int) -> bool:
    """Return whether an action-like value submits no concrete actions."""
    return len(normalize_actions(action_like, width)) == 0
