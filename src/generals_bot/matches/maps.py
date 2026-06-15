from __future__ import annotations

from dataclasses import replace

from generals_bot.sim import GameConfig, MapGenerationConfig, generate_1v1_map


def build_game_config(
    map_name: str,
    *,
    seed: int,
    max_turns: int,
    min_size: int | None = None,
    max_size: int | None = None,
) -> GameConfig:
    """Build a GameConfig for match scripts from common map arguments."""
    if map_name == "generated":
        return _generated_config(
            seed=seed,
            max_turns=max_turns,
            min_size=min_size,
            max_size=max_size,
        )
    if map_name == "demo":
        return demo_config(max_turns=max_turns)
    raise ValueError(f"unknown map: {map_name}")


def demo_config(*, max_turns: int) -> GameConfig:
    """Return the small deterministic demo map used by local smoke checks."""
    return GameConfig(
        width=9,
        height=9,
        generals=(0, 80),
        mountains=(20, 22, 38, 42, 58, 60),
        cities=(40,),
        city_armies=(20,),
        max_turns=max_turns,
    )


def _generated_config(
    *,
    seed: int,
    max_turns: int,
    min_size: int | None,
    max_size: int | None,
) -> GameConfig:
    """Return a generated 1v1 map with optional fixed-size overrides."""
    map_options = None
    if min_size is not None or max_size is not None:
        default_options = MapGenerationConfig()
        resolved_min_size = default_options.min_size if min_size is None else min_size
        resolved_max_size = default_options.max_size if max_size is None else max_size
        max_possible_distance = resolved_min_size + resolved_max_size - 2
        map_options = replace(
            default_options,
            min_size=resolved_min_size,
            max_size=resolved_max_size,
            size_weights=None,
            min_general_distance=min(default_options.min_general_distance, max_possible_distance),
        )
    return generate_1v1_map(seed=seed, config=map_options, max_turns=max_turns)
