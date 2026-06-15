from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from generals_bot.agents import ExpanderPolicy, MarchPolicy
from generals_bot.agents.base import Policy
from generals_bot.benchmark_bots.flobot import FlobotPolicy
from generals_bot.sim import Direction, GameConfig


_RULE_BASED_SPECS = {
    "expander",
    "flobot",
    "march",
    "march-up",
    "march-down",
    "march-left",
    "march-right",
}


@dataclass(frozen=True)
class PolicySpec:
    """Parsed CLI description of one match policy."""

    label: str
    raw: str
    kind: str
    value: str
    enemy_king_predictor_path: str | None = None

    @property
    def is_checkpoint(self) -> bool:
        """Return whether this spec runs a model checkpoint."""
        return self.kind == "checkpoint"

    @property
    def supports_batched(self) -> bool:
        """Return whether this spec supports batched local evaluation."""
        return self.kind in {"checkpoint", "router"}


def parse_agent_spec(raw_value: str) -> PolicySpec:
    """Parse one agent spec, optionally prefixed by LABEL=."""
    label, spec = _split_agent_label(raw_value)

    if spec.startswith("checkpoint:"):
        value = spec.split(":", 1)[1]
        path, enemy_king_predictor_path = _parse_checkpoint_value(value)
        if not path:
            raise ValueError("checkpoint spec must include a path")
        resolved_label = label or _checkpoint_label(Path(path))
        return PolicySpec(
            label=_slug(resolved_label),
            raw=spec,
            kind="checkpoint",
            value=path,
            enemy_king_predictor_path=enemy_king_predictor_path,
        )

    if spec.startswith("router:"):
        path = spec.split(":", 1)[1]
        if not path:
            raise ValueError("router spec must include a path")
        resolved_label = label or _checkpoint_label(Path(path))
        return PolicySpec(
            label=_slug(resolved_label),
            raw=spec,
            kind="router",
            value=path,
        )

    if spec in _RULE_BASED_SPECS:
        return PolicySpec(
            label=_slug(label or spec),
            raw=spec,
            kind=spec,
            value=spec,
        )

    raise ValueError(
        "unknown agent spec. Expected expander, flobot, march*, checkpoint:<path>, "
        "or router:<path>; "
        f"got {raw_value!r}"
    )


def parse_agent_specs(raw_values: list[str]) -> dict[str, PolicySpec]:
    """Parse many agent specs and ensure labels are unique."""
    specs = [parse_agent_spec(value) for value in raw_values]
    result: dict[str, PolicySpec] = {}
    for spec in specs:
        if spec.label in result:
            raise ValueError(f"duplicate agent label: {spec.label}")
        result[spec.label] = spec
    return result


def make_policy(
    spec: PolicySpec,
    *,
    player: int,
    seed: int,
    config: GameConfig,
    device: str = "auto",
    pass_bias: float = 0.0,
    pass_bias_turn: int = 50,
) -> Policy:
    """Instantiate a policy for one player from a parsed spec."""
    if spec.kind == "expander":
        policy: Policy = ExpanderPolicy(seed=seed)
    elif spec.kind == "flobot":
        policy = FlobotPolicy()
    elif spec.kind == "march":
        policy = MarchPolicy(direction=Direction.RIGHT if player == 0 else Direction.LEFT)
    elif spec.kind == "march-up":
        policy = MarchPolicy(direction=Direction.UP)
    elif spec.kind == "march-down":
        policy = MarchPolicy(direction=Direction.DOWN)
    elif spec.kind == "march-left":
        policy = MarchPolicy(direction=Direction.LEFT)
    elif spec.kind == "march-right":
        policy = MarchPolicy(direction=Direction.RIGHT)
    elif spec.kind == "checkpoint":
        from generals_bot.training.policy import BCModelPolicy

        policy = BCModelPolicy(
            spec.value,
            device=device,
            pass_bias=pass_bias,
            pass_bias_turn=pass_bias_turn,
            enemy_king_predictor_path=spec.enemy_king_predictor_path,
        )
    elif spec.kind == "router":
        from generals_bot.training.router import RouterModelPolicy

        policy = RouterModelPolicy(
            spec.value,
            device=device,
        )
    else:  # pragma: no cover - parse_agent_spec prevents this branch.
        raise ValueError(f"unknown policy kind: {spec.kind}")
    policy.reset(player, config)
    return policy


def _checkpoint_label(path: Path) -> str:
    """Infer a readable label for a checkpoint path."""
    if path.name == "model.pt" and path.parent.name:
        return path.parent.name
    return path.stem


def _split_agent_label(raw_value: str) -> tuple[str | None, str]:
    """Split LABEL=SPEC while leaving checkpoint options untouched."""
    if raw_value.startswith("checkpoint:"):
        return None, raw_value
    if "=" not in raw_value:
        return None, raw_value
    label, spec = raw_value.split("=", 1)
    if not label:
        raise ValueError("agent label cannot be empty")
    if not spec:
        raise ValueError("agent spec cannot be empty")
    return label, spec


def _parse_checkpoint_value(value: str) -> tuple[str, str | None]:
    """Parse checkpoint path and optional checkpoint-specific options."""
    option = ",enemy_king_predictor="
    if option not in value:
        return value, None
    path, predictor_path = value.split(option, 1)
    if not predictor_path:
        raise ValueError("enemy_king_predictor option must include a path")
    return path, predictor_path


def _slug(value: str) -> str:
    """Return a stable label safe for JSON keys and filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "agent"
