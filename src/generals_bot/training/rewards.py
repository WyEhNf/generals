from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from generals_bot.sim.env import GeneralsEnv
from generals_bot.sim.types import ExecutedAction, Terrain


REWARD_COMPONENTS = (
    "terminal",
    "land",
    "city",
    "city_capture",
    "city_capture_survive_drip",
    "city_capture_hold_drip",
    "city_capture_survive_milestone",
    "city_capture_hold_milestone",
    "city_capture_death_penalty",
    "city_capture_lifecycle",
    "city_capture_win",
    "city_capture_distance_multiplier",
    "city_attack_success",
    "city_attack_failure",
    "army",
    "attack",
    "total",
)
SHAPING_COMPONENTS = ("land", "city", "city_capture", "army", "attack")


@dataclass(frozen=True)
class RewardWeights:
    """Shaping coefficients active for one turn segment."""

    land: float = 0.0
    city: float = 0.0
    city_capture: float = 0.0
    army: float = 0.0
    attack: float = 0.0


@dataclass(frozen=True)
class RewardSegment:
    """One half-open turn range with its shaping coefficients."""

    start_turn: int
    end_turn: int | None
    weights: RewardWeights

    def contains(self, turn: int) -> bool:
        """Return whether this segment applies to the given pre-step turn."""
        return turn >= self.start_turn and (
            self.end_turn is None or turn < self.end_turn
        )


@dataclass(frozen=True)
class TerminalRewardConfig:
    """Terminal win/loss/draw reward values."""

    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0


@dataclass(frozen=True)
class TerminalBonusConfig:
    """Terminal bonuses gated on episode-level events."""

    city_capture_win: float = 0.0


@dataclass(frozen=True)
class EnemyDistanceConfig:
    """Distance-based modifier for immediate city-capture rewards."""

    enabled: bool = False
    min_distance: int = 3
    full_distance: int = 10
    min_multiplier: float = 0.25


@dataclass(frozen=True)
class CityCaptureLifecycleConfig:
    """Delayed city-capture reward parameters."""

    max_reward_turns_per_capture: int = 0
    survive_drip_per_turn: float = 0.0
    hold_drip_per_turn: float = 0.0
    survive_milestone: float = 0.0
    hold_milestone: float = 0.0
    death_penalty: float = 0.0
    enemy_distance: EnemyDistanceConfig = field(default_factory=EnemyDistanceConfig)

    @property
    def enabled(self) -> bool:
        """Return whether lifecycle events should be tracked."""
        return self.max_reward_turns_per_capture > 0 and any(
            value != 0.0
            for value in (
                self.survive_drip_per_turn,
                self.hold_drip_per_turn,
                self.survive_milestone,
                self.hold_milestone,
                self.death_penalty,
            )
        )


@dataclass(frozen=True)
class CityAttackOutcomeConfig:
    """Delayed outcome reward for neutral-city attacks."""

    window_turns: int = 0
    success_reward: float = 0.0
    failure_penalty: float = 0.0
    attribution: str = "resolution"

    @property
    def enabled(self) -> bool:
        """Return whether neutral-city attack outcomes should be tracked."""
        return self.window_turns > 0 and (
            self.success_reward != 0.0 or self.failure_penalty != 0.0
        )


@dataclass(frozen=True)
class MetricRewardConfig:
    """Mode-specific configuration for one reward metric."""

    mode: str
    interval: int | None = None


@dataclass(frozen=True)
class RewardConfig:
    """Parsed reward configuration used by PPO rollouts."""

    terminal: TerminalRewardConfig
    metrics: dict[str, MetricRewardConfig]
    segments: tuple[RewardSegment, ...]
    terminal_bonuses: TerminalBonusConfig = field(default_factory=TerminalBonusConfig)
    city_capture_lifecycle: CityCaptureLifecycleConfig = field(
        default_factory=CityCaptureLifecycleConfig
    )
    city_attack_outcome: CityAttackOutcomeConfig = field(
        default_factory=CityAttackOutcomeConfig
    )
    source_path: str | None = None

    def weights_for_turn(self, turn: int) -> RewardWeights:
        """Return the active shaping weights for a pre-step turn."""
        for segment in self.segments:
            if segment.contains(turn):
                return segment.weights
        return RewardWeights()


@dataclass(frozen=True)
class RewardState:
    """Compact score snapshot used for shaping deltas."""

    turn: int
    army: tuple[int, int]
    land: tuple[int, int]
    cities: tuple[int, int]
    city_tiles: tuple[int, ...] = ()
    alive: tuple[bool, bool] = (True, True)
    width: int = 0
    height: int = 0
    terrain: np.ndarray | None = None
    armies_grid: np.ndarray | None = None

    @classmethod
    def from_env(cls, env: GeneralsEnv) -> RewardState:
        """Capture public score and owned-city counts from an environment."""
        state = env.state
        if state is None:
            raise RuntimeError("cannot build reward state before env.reset()")
        scores = env.scores
        cities = tuple(
            int(np.sum(state.cities & (state.terrain == player)))
            for player in (0, 1)
        )
        return cls(
            turn=state.turn,
            army=(scores[0].army, scores[1].army),
            land=(scores[0].land, scores[1].land),
            cities=(cities[0], cities[1]),
            city_tiles=tuple(int(tile) for tile in np.flatnonzero(state.cities)),
            alive=(bool(state.alive[0]), bool(state.alive[1])),
            width=state.width,
            height=state.height,
            terrain=np.array(state.terrain, copy=True),
            armies_grid=np.array(state.armies, copy=True),
        )


@dataclass
class CityCaptureLifecycleEvent:
    """A pending delayed reward event created by one neutral-city capture."""

    player_id: int
    city_tile: int
    capture_turn: int
    end_turn: int
    survive_milestone_paid: bool = False
    hold_milestone_paid: bool = False
    death_penalty_paid: bool = False


@dataclass
class CityAttackOutcomeEvent:
    """A pending outcome event created by one failed neutral-city attack."""

    player_id: int
    city_tile: int
    attack_turn: int
    due_turn: int
    reward_index: int | None = None


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-player reward components for one environment step."""

    terminal: float = 0.0
    land: float = 0.0
    city: float = 0.0
    city_capture: float = 0.0
    city_capture_survive_drip: float = 0.0
    city_capture_hold_drip: float = 0.0
    city_capture_survive_milestone: float = 0.0
    city_capture_hold_milestone: float = 0.0
    city_capture_death_penalty: float = 0.0
    city_capture_win: float = 0.0
    city_capture_distance_multiplier: float = 0.0
    city_attack_success: float = 0.0
    city_attack_failure: float = 0.0
    army: float = 0.0
    attack: float = 0.0

    @property
    def city_capture_lifecycle(self) -> float:
        """Return the sum of delayed city-capture lifecycle rewards."""
        return (
            self.city_capture_survive_drip
            + self.city_capture_hold_drip
            + self.city_capture_survive_milestone
            + self.city_capture_hold_milestone
            + self.city_capture_death_penalty
        )

    @property
    def total(self) -> float:
        """Return the sum of terminal and shaping reward components."""
        return (
            self.terminal
            + self.land
            + self.city
            + self.city_capture
            + self.city_capture_lifecycle
            + self.city_capture_win
            + self.city_attack_success
            + self.city_attack_failure
            + self.army
            + self.attack
        )

    def __add__(self, other: RewardBreakdown) -> RewardBreakdown:
        """Return the component-wise sum of two reward breakdowns."""
        return RewardBreakdown(
            terminal=self.terminal + other.terminal,
            land=self.land + other.land,
            city=self.city + other.city,
            city_capture=self.city_capture + other.city_capture,
            city_capture_survive_drip=(
                self.city_capture_survive_drip + other.city_capture_survive_drip
            ),
            city_capture_hold_drip=(
                self.city_capture_hold_drip + other.city_capture_hold_drip
            ),
            city_capture_survive_milestone=(
                self.city_capture_survive_milestone
                + other.city_capture_survive_milestone
            ),
            city_capture_hold_milestone=(
                self.city_capture_hold_milestone + other.city_capture_hold_milestone
            ),
            city_capture_death_penalty=(
                self.city_capture_death_penalty + other.city_capture_death_penalty
            ),
            city_capture_win=self.city_capture_win + other.city_capture_win,
            city_capture_distance_multiplier=(
                self.city_capture_distance_multiplier
                + other.city_capture_distance_multiplier
            ),
            city_attack_success=(
                self.city_attack_success + other.city_attack_success
            ),
            city_attack_failure=(
                self.city_attack_failure + other.city_attack_failure
            ),
            army=self.army + other.army,
            attack=self.attack + other.attack,
        )

    def as_metrics(self) -> dict[str, float]:
        """Return component values keyed by metric suffix."""
        return {
            "terminal": self.terminal,
            "land": self.land,
            "city": self.city,
            "city_capture": self.city_capture,
            "city_capture_survive_drip": self.city_capture_survive_drip,
            "city_capture_hold_drip": self.city_capture_hold_drip,
            "city_capture_survive_milestone": self.city_capture_survive_milestone,
            "city_capture_hold_milestone": self.city_capture_hold_milestone,
            "city_capture_death_penalty": self.city_capture_death_penalty,
            "city_capture_lifecycle": self.city_capture_lifecycle,
            "city_capture_win": self.city_capture_win,
            "city_capture_distance_multiplier": self.city_capture_distance_multiplier,
            "city_attack_success": self.city_attack_success,
            "city_attack_failure": self.city_attack_failure,
            "army": self.army,
            "attack": self.attack,
            "total": self.total,
        }


class RewardCalculator:
    """Compute terminal and configured shaping rewards for PPO."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        """Create a calculator, defaulting to terminal-only reward."""
        self.config = config or default_reward_config()

    def rewards(
        self,
        *,
        before: RewardState,
        after: RewardState,
        executed_actions: list[ExecutedAction],
        winner: int | None,
        done: bool,
        episode_city_captures: Mapping[int, int] | None = None,
    ) -> dict[int, RewardBreakdown]:
        """Return reward components for both players after one simulator step."""
        weights = self.config.weights_for_turn(before.turn)
        terminal = {
            player: self._terminal_reward(winner, player) if done else 0.0
            for player in (0, 1)
        }
        land = self._land_rewards(after, weights)
        city = self._potential_delta_rewards(
            before.cities,
            after.cities,
            weights.city,
            metric_name="city",
        )
        city_capture = self._city_capture_rewards(
            executed_actions,
            before.city_tiles,
            weights.city_capture,
            after,
        )
        city_capture_distance_multipliers = self.city_capture_distance_multipliers(
            executed_actions,
            before.city_tiles,
            after,
        )
        city_capture_win = self._city_capture_win_rewards(
            winner=winner,
            done=done,
            episode_city_captures=episode_city_captures,
        )
        army = self._potential_delta_rewards(
            before.army,
            after.army,
            weights.army,
            metric_name="army",
        )
        attack = self._attack_rewards(executed_actions, weights.attack)
        return {
            player: RewardBreakdown(
                terminal=terminal[player],
                land=land[player],
                city=city[player],
                city_capture=city_capture[player],
                city_capture_win=city_capture_win[player],
                city_capture_distance_multiplier=(
                    city_capture_distance_multipliers[player]
                ),
                army=army[player],
                attack=attack[player],
            )
            for player in (0, 1)
        }

    def city_capture_counts(
        self,
        executed_actions: list[ExecutedAction],
        city_tiles: tuple[int, ...],
    ) -> dict[int, int]:
        """Count successful neutral-city captures by player for one step."""
        counts = {0: 0, 1: 0}
        for executed in self._successful_neutral_city_captures(
            executed_actions,
            city_tiles,
        ):
            counts[executed.player_id] += 1
        return counts

    def city_capture_lifecycle_events(
        self,
        executed_actions: list[ExecutedAction],
        city_tiles: tuple[int, ...],
        *,
        capture_turn: int,
    ) -> list[CityCaptureLifecycleEvent]:
        """Return new delayed reward events created by city captures."""
        config = self.config.city_capture_lifecycle
        if not config.enabled:
            return []
        return [
            CityCaptureLifecycleEvent(
                player_id=executed.player_id,
                city_tile=executed.action.end,
                capture_turn=capture_turn,
                end_turn=capture_turn + config.max_reward_turns_per_capture,
            )
            for executed in self._successful_neutral_city_captures(
                executed_actions,
                city_tiles,
            )
        ]

    def city_capture_lifecycle_rewards(
        self,
        events: list[CityCaptureLifecycleEvent],
        after: RewardState,
    ) -> tuple[dict[int, RewardBreakdown], list[CityCaptureLifecycleEvent]]:
        """Return lifecycle rewards and remaining pending events."""
        config = self.config.city_capture_lifecycle
        rewards = {0: RewardBreakdown(), 1: RewardBreakdown()}
        if not config.enabled or not events:
            return rewards, events

        remaining = []
        for event in events:
            player = event.player_id
            alive = after.alive[player]
            owns_city = alive and _tile_owner(after, event.city_tile) == player
            due = after.turn >= event.end_turn

            if alive and event.capture_turn < after.turn <= event.end_turn:
                rewards[player] += RewardBreakdown(
                    city_capture_survive_drip=config.survive_drip_per_turn,
                )
                if owns_city:
                    rewards[player] += RewardBreakdown(
                        city_capture_hold_drip=config.hold_drip_per_turn,
                    )

            if due and alive and not event.survive_milestone_paid:
                rewards[player] += RewardBreakdown(
                    city_capture_survive_milestone=config.survive_milestone,
                )
                event.survive_milestone_paid = True
            if due and owns_city and not event.hold_milestone_paid:
                rewards[player] += RewardBreakdown(
                    city_capture_hold_milestone=config.hold_milestone,
                )
                event.hold_milestone_paid = True

            if not alive and after.turn <= event.end_turn and not event.death_penalty_paid:
                rewards[player] += RewardBreakdown(
                    city_capture_death_penalty=config.death_penalty,
                )
                event.death_penalty_paid = True

            if after.turn < event.end_turn and not event.death_penalty_paid:
                remaining.append(event)
        return rewards, remaining

    def city_attack_outcome_events(
        self,
        executed_actions: list[ExecutedAction],
        city_tiles: tuple[int, ...],
        *,
        attack_turn: int,
        reward_indices_by_player: Mapping[int, int] | None = None,
    ) -> list[CityAttackOutcomeEvent]:
        """Return delayed outcome events created by failed neutral-city attacks.

        When ``attribution`` is ``"retroactive"``, each event carries the
        rollout reward index of the step where the attack was initiated so
        that the resolved reward can be added to the original sample row.
        When ``attribution`` is ``"resolution"`` (the default), the reward
        index is left as ``None`` and the caller should add the resolved
        reward to the current step.
        """
        config = self.config.city_attack_outcome
        if not config.enabled:
            return []
        events = []
        use_retroactive = config.attribution == "retroactive"
        for executed in self._failed_neutral_city_attacks(
            executed_actions,
            city_tiles,
        ):
            reward_index = None
            if use_retroactive and reward_indices_by_player is not None:
                reward_index = reward_indices_by_player.get(executed.player_id)
            if use_retroactive and reward_indices_by_player is not None and reward_index is None:
                continue
            events.append(
                CityAttackOutcomeEvent(
                    player_id=executed.player_id,
                    city_tile=executed.action.end,
                    attack_turn=attack_turn,
                    due_turn=attack_turn + config.window_turns,
                    reward_index=reward_index,
                )
            )
        return events

    def city_attack_outcome_rewards(
        self,
        events: list[CityAttackOutcomeEvent],
        after: RewardState,
    ) -> tuple[list[tuple[CityAttackOutcomeEvent, RewardBreakdown]], list[CityAttackOutcomeEvent]]:
        """Return resolved city-attack outcomes and remaining pending events."""
        config = self.config.city_attack_outcome
        if not config.enabled or not events:
            return [], events

        resolved = []
        remaining = []
        for event in events:
            owner = _tile_owner(after, event.city_tile)
            if owner == event.player_id:
                resolved.append(
                    (
                        event,
                        RewardBreakdown(
                            city_attack_success=config.success_reward,
                        ),
                    )
                )
                continue
            if after.turn >= event.due_turn:
                resolved.append(
                    (
                        event,
                        RewardBreakdown(
                            city_attack_failure=config.failure_penalty,
                        ),
                    )
                )
                continue
            remaining.append(event)
        return resolved, remaining

    def city_attack_outcome_rewards_for_step(
        self,
        events: list[CityAttackOutcomeEvent],
        after: RewardState,
    ) -> tuple[dict[int, RewardBreakdown], list[CityAttackOutcomeEvent]]:
        """Return resolved outcome rewards aggregated by player for the current step.

        Unlike :meth:`city_attack_outcome_rewards`, this method does not
        attach a ``reward_index`` to each resolved event.  It is intended
        for ``"resolution"``-mode attribution where the reward should be
        added to the *current* environment step rather than retroactively
        applied to a past rollout row.
        """
        resolved, remaining = self.city_attack_outcome_rewards(events, after)
        rewards = {0: RewardBreakdown(), 1: RewardBreakdown()}
        for event, breakdown in resolved:
            rewards[event.player_id] += breakdown
        return rewards, remaining

    def city_attack_failure_events(
        self,
        executed_actions: list[ExecutedAction],
        city_tiles: tuple[int, ...],
        *,
        attack_turn: int,
    ) -> list[CityAttackOutcomeEvent]:
        """Return legacy failure-only events for compatibility."""
        return self.city_attack_outcome_events(
            executed_actions,
            city_tiles,
            attack_turn=attack_turn,
        )

    def city_attack_failure_rewards(
        self,
        events: list[CityAttackOutcomeEvent],
        after: RewardState,
    ) -> tuple[dict[int, RewardBreakdown], list[CityAttackOutcomeEvent]]:
        """Return legacy failure-only rewards by player for compatibility."""
        resolved, remaining = self.city_attack_outcome_rewards(events, after)
        rewards = {0: RewardBreakdown(), 1: RewardBreakdown()}
        for event, breakdown in resolved:
            rewards[event.player_id] += breakdown
        return rewards, remaining

    def _terminal_reward(self, winner: int | None, player_id: int) -> float:
        """Return configured terminal reward for one player."""
        if winner is None:
            return self.config.terminal.draw
        return self.config.terminal.win if player_id == winner else self.config.terminal.loss

    def _land_rewards(self, after: RewardState, weights: RewardWeights) -> dict[int, float]:
        """Return land milestone rewards for both players."""
        metric = self.config.metrics["land"]
        if metric.mode != "milestone_absolute":
            raise ValueError(f"unsupported land reward mode: {metric.mode}")
        interval = metric.interval or 0
        if weights.land == 0.0 or interval <= 0 or after.turn % interval != 0:
            return {0: 0.0, 1: 0.0}
        score = _relative_log_score(after.land, 0)
        return {
            0: weights.land * score,
            1: -weights.land * score,
        }

    def _potential_delta_rewards(
        self,
        before_values: tuple[int, int],
        after_values: tuple[int, int],
        weight: float,
        *,
        metric_name: str,
    ) -> dict[int, float]:
        """Return potential-difference reward for a relative score."""
        metric = self.config.metrics[metric_name]
        if metric.mode != "potential_delta":
            raise ValueError(f"unsupported {metric_name} reward mode: {metric.mode}")
        if weight == 0.0:
            return {0: 0.0, 1: 0.0}
        before_score = _relative_log_score(before_values, 0)
        after_score = _relative_log_score(after_values, 0)
        delta = after_score - before_score
        return {
            0: weight * delta,
            1: -weight * delta,
        }

    def _city_capture_rewards(
        self,
        executed_actions: list[ExecutedAction],
        city_tiles: tuple[int, ...],
        weight: float,
        after: RewardState,
    ) -> dict[int, float]:
        """Return event rewards for successfully capturing neutral cities."""
        metric = self.config.metrics["city_capture"]
        if metric.mode != "event_city_capture":
            raise ValueError(f"unsupported city_capture reward mode: {metric.mode}")
        rewards = {0: 0.0, 1: 0.0}
        if weight == 0.0:
            return rewards
        for executed in self._successful_neutral_city_captures(
            executed_actions,
            city_tiles,
        ):
            multiplier = self._city_capture_distance_multiplier(executed, after)
            reward = weight * (
                1.0 + math.log1p(max(0, executed.target_army_before)) / math.log1p(50.0)
            ) * multiplier
            rewards[executed.player_id] += reward
            rewards[1 - executed.player_id] -= reward
        return rewards

    def city_capture_distance_multipliers(
        self,
        executed_actions: list[ExecutedAction],
        city_tiles: tuple[int, ...],
        after: RewardState,
    ) -> dict[int, float]:
        """Return summed capture-distance multipliers by player for metrics."""
        multipliers = {0: 0.0, 1: 0.0}
        for executed in self._successful_neutral_city_captures(
            executed_actions,
            city_tiles,
        ):
            multipliers[executed.player_id] += self._city_capture_distance_multiplier(
                executed,
                after,
            )
        return multipliers

    def _city_capture_distance_multiplier(
        self,
        executed: ExecutedAction,
        after: RewardState,
    ) -> float:
        """Return immediate city-capture multiplier based on enemy distance."""
        config = self.config.city_capture_lifecycle.enemy_distance
        if not config.enabled:
            return 1.0
        distance = _nearest_enemy_distance(
            after,
            start_tile=executed.action.end,
            player=executed.player_id,
        )
        if distance is None or distance >= config.full_distance:
            return 1.0
        if distance <= config.min_distance:
            return config.min_multiplier
        span = config.full_distance - config.min_distance
        progress = (distance - config.min_distance) / span
        return config.min_multiplier + progress * (1.0 - config.min_multiplier)

    def _successful_neutral_city_captures(
        self,
        executed_actions: list[ExecutedAction],
        city_tiles: tuple[int, ...],
    ) -> list[ExecutedAction]:
        """Return valid successful captures of neutral city tiles."""
        city_tile_set = set(city_tiles)
        captures = []
        for executed in executed_actions:
            if not executed.valid:
                continue
            if executed.action.end not in city_tile_set:
                continue
            if executed.target_owner_before != int(Terrain.NEUTRAL):
                continue
            if executed.moved_army <= executed.target_army_before:
                continue
            captures.append(executed)
        return captures

    def _failed_neutral_city_attacks(
        self,
        executed_actions: list[ExecutedAction],
        city_tiles: tuple[int, ...],
    ) -> list[ExecutedAction]:
        """Return valid neutral-city attacks that did not capture immediately."""
        city_tile_set = set(city_tiles)
        failures = []
        for executed in executed_actions:
            if not executed.valid:
                continue
            if executed.action.end not in city_tile_set:
                continue
            if executed.target_owner_before != int(Terrain.NEUTRAL):
                continue
            if executed.moved_army > executed.target_army_before:
                continue
            failures.append(executed)
        return failures

    def _city_capture_win_rewards(
        self,
        *,
        winner: int | None,
        done: bool,
        episode_city_captures: Mapping[int, int] | None,
    ) -> dict[int, float]:
        """Return terminal bonuses for winners that captured neutral cities."""
        rewards = {0: 0.0, 1: 0.0}
        weight = self.config.terminal_bonuses.city_capture_win
        if not done or winner is None or weight == 0.0:
            return rewards
        count = 0 if episode_city_captures is None else episode_city_captures.get(winner, 0)
        if count <= 0:
            return rewards
        rewards[winner] = weight * math.log1p(count)
        return rewards

    def _attack_rewards(
        self,
        executed_actions: list[ExecutedAction],
        weight: float,
    ) -> dict[int, float]:
        """Return event-based enemy damage rewards."""
        metric = self.config.metrics["attack"]
        if metric.mode != "event_damage":
            raise ValueError(f"unsupported attack reward mode: {metric.mode}")
        rewards = {0: 0.0, 1: 0.0}
        if weight == 0.0:
            return rewards
        for executed in executed_actions:
            if not executed.valid:
                continue
            target_owner = executed.target_owner_before
            damage = executed.damage_done
            if target_owner not in (0, 1) or target_owner == executed.player_id:
                continue
            if damage <= 0:
                continue
            reward = weight * math.log1p(damage)
            rewards[executed.player_id] += reward
            rewards[target_owner] -= reward
        return rewards


def default_reward_config() -> RewardConfig:
    """Return terminal-only reward with known metric modes and zero weights."""
    return RewardConfig(
        terminal=TerminalRewardConfig(),
        metrics={
            "land": MetricRewardConfig(mode="milestone_absolute", interval=50),
            "city": MetricRewardConfig(mode="potential_delta"),
            "city_capture": MetricRewardConfig(mode="event_city_capture"),
            "army": MetricRewardConfig(mode="potential_delta"),
            "attack": MetricRewardConfig(mode="event_damage"),
        },
        segments=(),
        terminal_bonuses=TerminalBonusConfig(),
        city_capture_lifecycle=CityCaptureLifecycleConfig(),
        city_attack_outcome=CityAttackOutcomeConfig(),
        source_path=None,
    )


def load_reward_config(config_path: str | Path | None) -> RewardConfig:
    """Load a JSON reward config, returning terminal-only defaults for None."""
    if config_path is None:
        return default_reward_config()
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("reward config must be a JSON object")
    if data.get("version") != 1:
        raise ValueError("reward config must contain version=1")

    terminal = _parse_terminal_config(data.get("terminal", {}))
    terminal_bonuses = _parse_terminal_bonus_config(data.get("terminal_bonuses", {}))
    city_capture_lifecycle = _parse_city_capture_lifecycle_config(
        data.get("city_capture_lifecycle", {})
    )
    city_attack_outcome = _parse_city_attack_outcome_config(
        data.get("city_attack_outcome"),
        legacy_failure=data.get("city_attack_failure"),
    )
    metrics = _parse_metric_configs(data.get("metrics", {}))
    segments = _parse_segments(data.get("segments", []))
    return RewardConfig(
        terminal=terminal,
        metrics=metrics,
        segments=tuple(segments),
        terminal_bonuses=terminal_bonuses,
        city_capture_lifecycle=city_capture_lifecycle,
        city_attack_outcome=city_attack_outcome,
        source_path=str(path),
    )


def _parse_terminal_config(raw: Any) -> TerminalRewardConfig:
    """Parse terminal reward values with documented defaults."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("terminal reward config must be an object")
    return TerminalRewardConfig(
        win=float(raw.get("win", 1.0)),
        loss=float(raw.get("loss", -1.0)),
        draw=float(raw.get("draw", 0.0)),
    )


def _parse_terminal_bonus_config(raw: Any) -> TerminalBonusConfig:
    """Parse terminal bonuses with zero defaults."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("terminal_bonuses reward config must be an object")
    unknown = set(raw) - {"city_capture_win"}
    if unknown:
        raise ValueError(f"unknown terminal bonus(es): {sorted(unknown)}")
    return TerminalBonusConfig(
        city_capture_win=float(raw.get("city_capture_win", 0.0)),
    )


def _parse_city_capture_lifecycle_config(raw: Any) -> CityCaptureLifecycleConfig:
    """Parse delayed city-capture reward settings."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("city_capture_lifecycle reward config must be an object")
    unknown = set(raw) - {
        "max_reward_turns_per_capture",
        "survive_drip_per_turn",
        "hold_drip_per_turn",
        "survive_milestone",
        "hold_milestone",
        "death_penalty",
        "enemy_distance",
    }
    if unknown:
        raise ValueError(f"unknown city_capture_lifecycle field(s): {sorted(unknown)}")
    max_turns = int(raw.get("max_reward_turns_per_capture", 0))
    if max_turns < 0:
        raise ValueError("max_reward_turns_per_capture must be nonnegative")
    return CityCaptureLifecycleConfig(
        max_reward_turns_per_capture=max_turns,
        survive_drip_per_turn=float(raw.get("survive_drip_per_turn", 0.0)),
        hold_drip_per_turn=float(raw.get("hold_drip_per_turn", 0.0)),
        survive_milestone=float(raw.get("survive_milestone", 0.0)),
        hold_milestone=float(raw.get("hold_milestone", 0.0)),
        death_penalty=float(raw.get("death_penalty", 0.0)),
        enemy_distance=_parse_enemy_distance_config(raw.get("enemy_distance", {})),
    )


def _parse_enemy_distance_config(raw: Any) -> EnemyDistanceConfig:
    """Parse enemy-distance capture reward modifier settings."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("enemy_distance reward config must be an object")
    unknown = set(raw) - {
        "enabled",
        "min_distance",
        "full_distance",
        "min_multiplier",
    }
    if unknown:
        raise ValueError(f"unknown enemy_distance field(s): {sorted(unknown)}")
    enabled = bool(raw.get("enabled", False))
    min_distance = int(raw.get("min_distance", 3))
    full_distance = int(raw.get("full_distance", 10))
    min_multiplier = float(raw.get("min_multiplier", 0.25))
    if min_distance < 0:
        raise ValueError("enemy_distance min_distance must be nonnegative")
    if full_distance <= min_distance:
        raise ValueError("enemy_distance full_distance must exceed min_distance")
    if not 0.0 <= min_multiplier <= 1.0:
        raise ValueError("enemy_distance min_multiplier must be in [0, 1]")
    return EnemyDistanceConfig(
        enabled=enabled,
        min_distance=min_distance,
        full_distance=full_distance,
        min_multiplier=min_multiplier,
    )


def _parse_city_attack_outcome_config(
    raw: Any,
    *,
    legacy_failure: Any = None,
) -> CityAttackOutcomeConfig:
    """Parse delayed city-attack outcome settings."""
    if raw is None and legacy_failure is not None:
        return _parse_legacy_city_attack_failure_config(legacy_failure)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("city_attack_outcome reward config must be an object")
    unknown = set(raw) - {"window_turns", "success_reward", "failure_penalty", "attribution"}
    if unknown:
        raise ValueError(f"unknown city_attack_outcome field(s): {sorted(unknown)}")
    window_turns = int(raw.get("window_turns", 0))
    if window_turns < 0:
        raise ValueError("city_attack_outcome window_turns must be nonnegative")
    attribution = str(raw.get("attribution", "resolution"))
    if attribution not in ("resolution", "retroactive"):
        raise ValueError(
            f"city_attack_outcome attribution must be 'resolution' or 'retroactive', "
            f"got {attribution!r}"
        )
    return CityAttackOutcomeConfig(
        window_turns=window_turns,
        success_reward=float(raw.get("success_reward", 0.0)),
        failure_penalty=float(raw.get("failure_penalty", 0.0)),
        attribution=attribution,
    )


def _parse_legacy_city_attack_failure_config(raw: Any) -> CityAttackOutcomeConfig:
    """Parse legacy failure-only city-attack settings as outcome settings."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("city_attack_failure reward config must be an object")
    unknown = set(raw) - {"window_turns", "penalty"}
    if unknown:
        raise ValueError(f"unknown city_attack_failure field(s): {sorted(unknown)}")
    window_turns = int(raw.get("window_turns", 0))
    if window_turns < 0:
        raise ValueError("city_attack_failure window_turns must be nonnegative")
    return CityAttackOutcomeConfig(
        window_turns=window_turns,
        success_reward=0.0,
        failure_penalty=float(raw.get("penalty", 0.0)),
    )


def _parse_metric_configs(raw: Any) -> dict[str, MetricRewardConfig]:
    """Parse metric modes, filling in defaults for omitted metrics."""
    defaults = default_reward_config().metrics
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("metrics reward config must be an object")
    unknown = set(raw) - set(SHAPING_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown reward metric(s): {sorted(unknown)}")
    metrics = dict(defaults)
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"metric {name} config must be an object")
        mode = str(value.get("mode", defaults[name].mode))
        interval = value.get("interval", defaults[name].interval)
        parsed_interval = None if interval is None else int(interval)
        if name == "land":
            if mode != "milestone_absolute":
                raise ValueError("land mode must be milestone_absolute")
            if parsed_interval is None or parsed_interval <= 0:
                raise ValueError("land interval must be positive")
        elif name in {"city", "army"}:
            if mode != "potential_delta":
                raise ValueError(f"{name} mode must be potential_delta")
            parsed_interval = None
        elif name == "city_capture":
            if mode != "event_city_capture":
                raise ValueError("city_capture mode must be event_city_capture")
            parsed_interval = None
        elif name == "attack":
            if mode != "event_damage":
                raise ValueError("attack mode must be event_damage")
            parsed_interval = None
        metrics[name] = MetricRewardConfig(mode=mode, interval=parsed_interval)
    return metrics


def _parse_segments(raw: Any) -> list[RewardSegment]:
    """Parse half-open turn segments and reject ambiguous overlaps."""
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("segments must be a list")
    segments: list[RewardSegment] = []
    previous_end: int | None = None
    saw_open_ended = False
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each reward segment must be an object")
        if saw_open_ended:
            raise ValueError("open-ended reward segment must be last")
        start_turn = int(item.get("start_turn", 0))
        end_raw = item.get("end_turn")
        end_turn = None if end_raw is None else int(end_raw)
        if start_turn < 0:
            raise ValueError("segment start_turn must be nonnegative")
        if end_turn is not None and end_turn <= start_turn:
            raise ValueError("segment end_turn must be greater than start_turn")
        if previous_end is not None and start_turn < previous_end:
            raise ValueError("reward segments must not overlap")
        weights = _parse_weights(item.get("weights", {}))
        segments.append(
            RewardSegment(
                start_turn=start_turn,
                end_turn=end_turn,
                weights=weights,
            )
        )
        previous_end = end_turn
        saw_open_ended = end_turn is None
    return segments


def _parse_weights(raw: Any) -> RewardWeights:
    """Parse shaping weights with zero defaults."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("segment weights must be an object")
    unknown = set(raw) - set(SHAPING_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown reward weight(s): {sorted(unknown)}")
    return RewardWeights(
        land=float(raw.get("land", 0.0)),
        city=float(raw.get("city", 0.0)),
        city_capture=float(raw.get("city_capture", 0.0)),
        army=float(raw.get("army", 0.0)),
        attack=float(raw.get("attack", 0.0)),
    )


def _relative_log_score(values: tuple[int, int], player: int) -> float:
    """Return log(self+1)-log(enemy+1) for one player's perspective."""
    opponent = 1 - player
    return math.log1p(max(0, values[player])) - math.log1p(max(0, values[opponent]))


def _tile_owner(state: RewardState, tile: int) -> int | None:
    """Return owner for a tile in a reward state when terrain is available."""
    if state.terrain is None or state.width <= 0 or state.height <= 0:
        return None
    if tile < 0 or tile >= state.width * state.height:
        return None
    row, col = divmod(tile, state.width)
    return int(state.terrain[row, col])


def _nearest_enemy_distance(
    state: RewardState,
    *,
    start_tile: int,
    player: int,
) -> int | None:
    """Return shortest grid distance from a tile to nearest enemy-owned tile."""
    if state.terrain is None or state.width <= 0 or state.height <= 0:
        return None
    if start_tile < 0 or start_tile >= state.width * state.height:
        return None
    opponent = 1 - player
    terrain = state.terrain
    enemy_mask = terrain == opponent
    if state.armies_grid is not None:
        enemy_mask = enemy_mask & (state.armies_grid > 0)
    if not bool(np.any(enemy_mask)):
        return None

    start_row, start_col = divmod(start_tile, state.width)
    visited = np.zeros((state.height, state.width), dtype=bool)
    queue: list[tuple[int, int, int]] = [(start_row, start_col, 0)]
    visited[start_row, start_col] = True
    index = 0
    while index < len(queue):
        row, col, distance = queue[index]
        index += 1
        if enemy_mask[row, col]:
            return distance
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if next_row < 0 or next_row >= state.height:
                continue
            if next_col < 0 or next_col >= state.width:
                continue
            if visited[next_row, next_col]:
                continue
            if terrain[next_row, next_col] == int(Terrain.MOUNTAIN):
                continue
            visited[next_row, next_col] = True
            queue.append((next_row, next_col, distance + 1))
    return None
