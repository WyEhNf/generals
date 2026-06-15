from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generals_bot.agents.base import Policy
from generals_bot.matches import build_game_config, make_policy, parse_agent_spec
from generals_bot.matches.policy_factory import PolicySpec
from generals_bot.replay.schema import RAW_SCHEMA_VERSION, RawReplay, RawReplayMove, RawTerrain
from generals_bot.sim import GameConfig, GeneralsEnv


SELFPLAY_SOURCE = "selfplay_enemy_king_v1"
SELFPLAY_REPLAY_VERSION = 13


@dataclass(frozen=True)
class SelfPlayOpponentEntry:
    """One sampled opponent choice for generated self-play replays."""

    id: str
    spec: str
    weight: float
    device: str
    pass_bias: float


@dataclass(frozen=True)
class SelfPlayOpponentPool:
    """Resolved opponent pool for replay generation."""

    entries: tuple[SelfPlayOpponentEntry, ...]


@dataclass(frozen=True)
class SelfPlayReplayResult:
    """One generated replay and the side labels used to create it."""

    replay: RawReplay
    side_labels: dict[int, str]
    winner: int | None
    turns: int
    submitted_actions: int
    invalid_actions: int


def generate_selfplay_replay(
    *,
    replay_id: str,
    config: GameConfig,
    side_specs: dict[int, PolicySpec],
    seed: int,
    device_by_player: dict[int, str],
    pass_bias_by_player: dict[int, float] | None = None,
    policy_cache: dict[tuple[str, int, str, float], Policy] | None = None,
) -> SelfPlayReplayResult:
    """Run one local match and return it as a RawReplay-compatible record."""
    env = GeneralsEnv(config)
    observations = env.reset(seed=seed)
    policies: dict[int, Policy] = {}
    for player, spec in side_specs.items():
        policies[player] = _make_or_reset_policy(
            spec=spec,
            player=player,
            seed=seed + player,
            config=config,
            device=device_by_player[player],
            pass_bias=(pass_bias_by_player or {}).get(player, 0.0),
            policy_cache=policy_cache,
        )

    moves: list[RawReplayMove] = []
    submitted_actions = 0
    invalid_actions = 0
    while not env.done:
        turn = env.state.turn if env.state is not None else observations[0].turn
        actions = {
            player: policies[player].act(observations[player])
            for player in env.alive_players
        }
        observations, _, _, info = env.step(actions)
        for player, player_actions in info["submitted_actions"].items():
            for action in player_actions:
                moves.append(
                    RawReplayMove(
                        player_id=int(player),
                        start=int(action.start),
                        end=int(action.end),
                        split=bool(action.split),
                        turn=int(turn),
                    )
                )
        submitted_actions += sum(len(value) for value in info["submitted_actions"].values())
        invalid_actions += sum(1 for item in info["executed_actions"] if not item.valid)

    state = env.state
    if state is None:
        raise RuntimeError("environment has no final state")
    side_labels = {
        player: side_specs[player].label
        for player in sorted(side_specs)
    }
    replay = RawReplay(
        schema_version=RAW_SCHEMA_VERSION,
        source=SELFPLAY_SOURCE,
        replay_id=replay_id,
        version=SELFPLAY_REPLAY_VERSION,
        width=int(config.width),
        height=int(config.height),
        players=[side_labels[0], side_labels[1]],
        stars=[None, None],
        terrain=RawTerrain(
            generals=[int(tile) for tile in config.generals],
            mountains=[int(tile) for tile in config.mountains],
            cities=[int(tile) for tile in config.cities],
            city_armies=[int(army) for army in config.city_armies],
        ),
        moves=moves,
        afks=[],
    )
    return SelfPlayReplayResult(
        replay=replay,
        side_labels=side_labels,
        winner=state.winner,
        turns=state.turn,
        submitted_actions=submitted_actions,
        invalid_actions=invalid_actions,
    )


def load_selfplay_opponent_pool(
    path: str | Path,
    *,
    learner_spec: str,
    training_device: str,
) -> SelfPlayOpponentPool:
    """Load an opponent-pool JSON file without importing PPO training code."""
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("opponent pool config must contain version=1")
    opponent_device = str(data.get("opponent_device", "same_as_training"))
    if opponent_device == "same_as_training":
        opponent_device = training_device
    opponent_pass_bias = float(data.get("opponent_pass_bias", 0.0))
    if opponent_pass_bias < 0:
        raise ValueError("opponent_pass_bias must be nonnegative")

    entries: list[SelfPlayOpponentEntry] = []
    self_play_weight = float(data.get("self_play_weight", 0.0))
    if self_play_weight < 0:
        raise ValueError("self_play_weight must be nonnegative")
    if self_play_weight > 0:
        entries.append(
            SelfPlayOpponentEntry(
                id="self_play",
                spec=learner_spec,
                weight=self_play_weight,
                device=training_device,
                pass_bias=0.0,
            )
        )

    for raw in data.get("opponents", []):
        if not isinstance(raw, dict):
            raise ValueError("each opponent entry must be an object")
        opponent_id = str(raw.get("id", "")).strip()
        if not opponent_id:
            raise ValueError("each opponent must have a nonempty id")
        weight = float(raw.get("weight", 1.0))
        if weight < 0:
            raise ValueError(f"opponent {opponent_id} weight must be nonnegative")
        if weight == 0:
            continue
        opponent_path = Path(str(raw["path"]))
        if not opponent_path.is_absolute():
            opponent_path = config_path.parent / opponent_path
        opponent_path = opponent_path.resolve()
        if not opponent_path.exists():
            raise FileNotFoundError(f"opponent checkpoint not found: {opponent_path}")
        entries.append(
            SelfPlayOpponentEntry(
                id=opponent_id,
                spec=f"checkpoint:{opponent_path}",
                weight=weight,
                device=opponent_device,
                pass_bias=opponent_pass_bias,
            )
        )

    if sum(entry.weight for entry in entries) <= 0:
        raise ValueError("opponent pool must have positive total weight")
    return SelfPlayOpponentPool(entries=tuple(entries))


def sample_opponent_entry(
    pool: SelfPlayOpponentPool,
    *,
    rng: random.Random,
) -> SelfPlayOpponentEntry:
    """Sample one opponent entry by configured weight."""
    total = sum(entry.weight for entry in pool.entries)
    threshold = rng.random() * total
    cumulative = 0.0
    for entry in pool.entries:
        cumulative += entry.weight
        if threshold < cumulative:
            return entry
    return pool.entries[-1]


def write_selfplay_replays(
    results: list[SelfPlayReplayResult],
    *,
    output_dir: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write generated self-play replays and a JSON manifest."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replay_path = output / "replays.jsonl.gz"
    with gzip.open(replay_path, "wt", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    result.replay.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")

    resolved_manifest = {
        "schema_version": RAW_SCHEMA_VERSION,
        "source": SELFPLAY_SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": str(replay_path),
        "replay_count": len(results),
        "move_count": sum(len(result.replay.moves) for result in results),
        "mean_turns": (
            sum(result.turns for result in results) / len(results)
            if results
            else 0.0
        ),
        "mean_submitted_actions": (
            sum(result.submitted_actions for result in results) / len(results)
            if results
            else 0.0
        ),
        "invalid_action_count": sum(result.invalid_actions for result in results),
        **manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(resolved_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_manifest


def side_specs_for_match(
    *,
    learner: PolicySpec,
    opponent: PolicySpec,
    learner_side: int,
) -> dict[int, PolicySpec]:
    """Return player-side specs for one learner/opponent match."""
    if learner_side not in (0, 1):
        raise ValueError("learner_side must be 0 or 1")
    return {
        learner_side: learner,
        1 - learner_side: opponent,
    }


def replay_id_for_generated_match(
    *,
    seed: int,
    learner_label: str,
    opponent_label: str,
    learner_side: int,
) -> str:
    """Return a stable replay id for a generated match."""
    return (
        f"selfplay-v1-{seed}-"
        f"{_slug(learner_label)}-p{learner_side}-vs-{_slug(opponent_label)}"
    )


def map_manifest(config: GameConfig) -> dict[str, Any]:
    """Return a compact JSON description of a generated map."""
    return {
        "width": config.width,
        "height": config.height,
        "generals": list(config.generals),
        "mountain_count": len(config.mountains),
        "city_count": len(config.cities),
    }


def build_generated_match_config(
    *,
    map_name: str,
    seed: int,
    turns: int,
    min_size: int | None,
    max_size: int | None,
) -> GameConfig:
    """Build a map config for self-play data generation."""
    return build_game_config(
        map_name,
        seed=seed,
        max_turns=turns,
        min_size=min_size,
        max_size=max_size,
    )


def _slug(value: str) -> str:
    """Return a stable id fragment."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "agent"


def _make_or_reset_policy(
    *,
    spec: PolicySpec,
    player: int,
    seed: int,
    config: GameConfig,
    device: str,
    pass_bias: float,
    policy_cache: dict[tuple[str, int, str, float], Policy] | None,
) -> Policy:
    """Instantiate or reuse one policy object for a player side."""
    if policy_cache is None:
        return make_policy(
            spec,
            player=player,
            seed=seed,
            config=config,
            device=device,
            pass_bias=pass_bias,
        )
    key = (spec.raw, player, device, pass_bias)
    if key not in policy_cache:
        policy_cache[key] = make_policy(
            spec,
            player=player,
            seed=seed,
            config=config,
            device=device,
            pass_bias=pass_bias,
        )
    else:
        policy_cache[key].reset(player, config)
    return policy_cache[key]
