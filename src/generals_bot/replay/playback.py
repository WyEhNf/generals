from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from generals_bot.agents.replay import ReplayFollowerPolicy
from generals_bot.replay.schema import RawReplay, RawReplayMove
from generals_bot.sim import GameConfig, GeneralsEnv
from generals_bot.sim.types import Action, ExecutedAction
from generals_bot.viz import MatchRecorder
from generals_bot.viz.recorder import Snapshot

MATCH_FIELDS = (
    "submitted_matches_raw",
    "valid_executed_matches_raw",
)


@dataclass(frozen=True)
class ReplayPlaybackResult:
    replay: RawReplay
    snapshots: list[Snapshot]
    report: dict[str, Any]


@dataclass(frozen=True)
class SubmittedReplayAction:
    turn: int
    action: Action


def config_from_raw_replay(replay: RawReplay, *, max_turns: int | None = None) -> GameConfig:
    last_turn = max((move.turn for move in replay.moves), default=0)
    return GameConfig(
        width=replay.width,
        height=replay.height,
        generals=tuple(replay.terrain.generals),
        mountains=tuple(replay.terrain.mountains),
        cities=tuple(replay.terrain.cities),
        city_armies=tuple(replay.terrain.city_armies),
        max_turns=max_turns if max_turns is not None else last_turn + 1,
    )


def simulate_raw_replay(replay: RawReplay, *, max_turns: int | None = None) -> ReplayPlaybackResult:
    config = config_from_raw_replay(replay, max_turns=max_turns)
    env = GeneralsEnv(config)
    observations = env.reset()
    policies = {
        0: ReplayFollowerPolicy([move for move in replay.moves if move.player_id == 0]),
        1: ReplayFollowerPolicy([move for move in replay.moves if move.player_id == 1]),
    }
    for player, policy in policies.items():
        policy.reset(player, config)

    recorder = MatchRecorder()
    recorder.capture(env)
    submitted: list[SubmittedReplayAction] = []
    executed: list[ExecutedAction] = []

    while not env.done:
        current_turn = observations[0].turn
        actions = {
            player: policies[player].act(observations[player])
            for player in env.alive_players
        }
        observations, _, _, info = env.step(actions)
        for player_actions in info["submitted_actions"].values():
            for action in player_actions:
                submitted.append(SubmittedReplayAction(current_turn, action))
        executed.extend(info["executed_actions"])
        recorder.capture(env)

    report = compare_replay_to_simulation(replay, submitted, executed, env)
    return ReplayPlaybackResult(replay=replay, snapshots=recorder.snapshots, report=report)


def compare_replay_to_simulation(
    replay: RawReplay,
    submitted: list[SubmittedReplayAction],
    executed: list[ExecutedAction],
    env: GeneralsEnv,
) -> dict[str, Any]:
    raw_counter = Counter(_move_key(move) for move in replay.moves)
    submitted_counter = Counter(_submitted_key(item) for item in submitted)
    valid_executed = [item for item in executed if item.valid]
    invalid_executed = [item for item in executed if not item.valid]
    valid_executed_counter = Counter(_executed_key(item) for item in valid_executed)
    missing_submissions = raw_counter - submitted_counter
    extra_submissions = submitted_counter - raw_counter
    missing_valid_executions = raw_counter - valid_executed_counter
    extra_valid_executions = valid_executed_counter - raw_counter

    state = env.state
    if state is None:
        raise RuntimeError("environment has no state")

    return {
        "replay_id": replay.replay_id,
        "players": replay.players,
        "stars": replay.stars,
        "version": replay.version,
        "width": replay.width,
        "height": replay.height,
        "raw_move_count": len(replay.moves),
        "submitted_move_count": len(submitted),
        "valid_executed_move_count": len(valid_executed),
        "invalid_executed_move_count": len(invalid_executed),
        "submitted_matches_raw": not missing_submissions and not extra_submissions,
        "valid_executed_matches_raw": (
            not missing_valid_executions and not extra_valid_executions
        ),
        "simulated_turns": state.turn,
        "max_raw_turn": max((move.turn for move in replay.moves), default=0),
        "winner": state.winner,
        "final_scores": [
            {
                "player_id": score.player_id,
                "army": score.army,
                "land": score.land,
                "dead": score.dead,
                "has_kill": score.has_kill,
            }
            for score in env.scores
        ],
        "missing_submissions": _counter_to_entries(missing_submissions),
        "extra_submissions": _counter_to_entries(extra_submissions),
        "missing_valid_executions": _counter_to_entries(missing_valid_executions),
        "extra_valid_executions": _counter_to_entries(extra_valid_executions),
        "first_invalid_executions": [
            _executed_to_report(item)
            for item in invalid_executed[:20]
        ],
    }


def summarize_replay_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    exact_matches = [
        report
        for report in reports
        if all(report[field] for field in MATCH_FIELDS)
        and report["invalid_executed_move_count"] == 0
    ]
    failed_reports = [report for report in reports if report not in exact_matches]
    winner_counts = Counter(str(report["winner"]) for report in reports)
    return {
        "checked_replays": len(reports),
        "exact_match_count": len(exact_matches),
        "submitted_mismatch_count": sum(
            1 for report in reports if not report["submitted_matches_raw"]
        ),
        "valid_execution_mismatch_count": sum(
            1 for report in reports if not report["valid_executed_matches_raw"]
        ),
        "invalid_executed_move_count": sum(
            int(report["invalid_executed_move_count"]) for report in reports
        ),
        "winner_counts": dict(sorted(winner_counts.items())),
        "first_failed_replay_ids": [report["replay_id"] for report in failed_reports[:20]],
    }


def compact_replay_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "replay_id": report["replay_id"],
        "players": report["players"],
        "raw_move_count": report["raw_move_count"],
        "submitted_move_count": report["submitted_move_count"],
        "valid_executed_move_count": report["valid_executed_move_count"],
        "invalid_executed_move_count": report["invalid_executed_move_count"],
        "submitted_matches_raw": report["submitted_matches_raw"],
        "valid_executed_matches_raw": report["valid_executed_matches_raw"],
        "simulated_turns": report["simulated_turns"],
        "max_raw_turn": report["max_raw_turn"],
        "winner": report["winner"],
        "first_invalid_executions": report["first_invalid_executions"],
    }


def _move_key(move: RawReplayMove) -> tuple[int, int, int, int, bool]:
    return (move.turn, move.player_id, move.start, move.end, move.split)


def _submitted_key(item: SubmittedReplayAction) -> tuple[int, int, int, int, bool]:
    action = item.action
    return (item.turn, action.player_id, action.start, action.end, action.split)


def _executed_key(item: ExecutedAction) -> tuple[int, int, int, int, bool]:
    action = item.action
    return (item.turn, action.player_id, action.start, action.end, action.split)


def _counter_to_entries(counter: Counter) -> list[dict[str, Any]]:
    return [
        {
            "turn": key[0],
            "player_id": key[1],
            "start": key[2],
            "end": key[3],
            "split": key[4],
            "count": count,
        }
        for key, count in sorted(counter.items())
    ]


def _executed_to_report(item: ExecutedAction) -> dict[str, Any]:
    return {
        "turn": item.turn,
        "player_id": item.player_id,
        "start": item.action.start,
        "end": item.action.end,
        "split": item.action.split,
        "reason": item.reason,
    }
