from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

from .types import GameConfig

T = TypeVar("T")


class MapGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MapGenerationConfig:
    min_size: int = 17
    max_size: int = 23
    size_weights: tuple[tuple[int, float], ...] | None = (
        (17, 0.04),
        (18, 0.60),
        (19, 0.15),
        (20, 0.08),
        (21, 0.06),
        (22, 0.05),
        (23, 0.02),
    )
    min_general_distance: int = 15
    mountain_density_range: tuple[float, float] = (0.16, 0.24)
    city_density_range: tuple[float, float] = (0.025, 0.05)
    city_army_range: tuple[int, int] = (40, 50)
    min_spawn_open_neighbors: int = 2
    local_fairness_radius: int = 8
    max_local_open_diff_ratio: float = 0.35
    city_fairness_radius: int = 10
    max_near_city_count_diff: int = 2
    max_near_city_army_diff: int = 80
    spawn_edge_bias: float = 1.2
    enemy_distance_bias: float = 2.0
    max_attempts: int = 1000
    max_turns: int = 5000


def generate_1v1_map(
    seed: int | None = None,
    config: MapGenerationConfig | None = None,
    *,
    max_turns: int | None = None,
) -> GameConfig:
    """Generate a 1v1 map using the configured local approximation."""
    options = config or MapGenerationConfig()
    _validate_options(options)
    rng = random.Random(seed)

    for _ in range(options.max_attempts):
        width = _sample_dimension(rng, options)
        height = _sample_dimension(rng, options)
        size = width * height

        generals = _sample_generals(width, height, rng, options)
        if generals is None:
            continue

        mountain_count = round(size * rng.uniform(*options.mountain_density_range))
        mountains = _sample_mountains(width, height, generals, mountain_count, rng)

        if not _spawn_has_open_neighbors(width, height, mountains, generals, options):
            continue
        if not are_passable_tiles_connected(width, height, mountains):
            continue
        if not _local_area_fair_enough(width, height, mountains, generals, options):
            continue

        city_count = max(1, round(size * rng.uniform(*options.city_density_range)))
        cities, city_armies = _sample_cities(
            width,
            height,
            mountains,
            generals,
            city_count,
            rng,
            options,
        )
        if not _cities_fair_enough(width, cities, city_armies, generals, options):
            continue

        return GameConfig(
            width=width,
            height=height,
            generals=generals,
            mountains=tuple(sorted(mountains)),
            cities=tuple(cities),
            city_armies=tuple(city_armies),
            max_turns=max_turns if max_turns is not None else options.max_turns,
        )

    raise MapGenerationError(f"failed to generate a valid map after {options.max_attempts} attempts")


def are_passable_tiles_connected(width: int, height: int, mountains: Iterable[int]) -> bool:
    """Return whether every non-mountain tile is in one connected component."""
    mountain_set = set(mountains)
    size = width * height
    start = next((tile for tile in range(size) if tile not in mountain_set), None)
    if start is None:
        return False

    visited = {start}
    queue: deque[int] = deque([start])
    while queue:
        tile = queue.popleft()
        for neighbor in _neighbors(tile, width, height):
            if neighbor in mountain_set or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)

    return len(visited) == size - len(mountain_set)


def manhattan(a: int, b: int, width: int) -> int:
    """Compute Manhattan distance between two flat tile indexes."""
    ar, ac = divmod(a, width)
    br, bc = divmod(b, width)
    return abs(ar - br) + abs(ac - bc)


def _sample_generals(
    width: int,
    height: int,
    rng: random.Random,
    options: MapGenerationConfig,
) -> tuple[int, int] | None:
    """Sample two generals that satisfy the configured distance rule."""
    tiles = list(range(width * height))
    first = _weighted_choice(
        tiles,
        lambda tile: _spawn_edge_weight(tile, width, height, options),
        rng,
    )
    candidates = [
        tile
        for tile in tiles
        if tile != first and manhattan(first, tile, width) >= options.min_general_distance
    ]
    if not candidates:
        return None

    second = _weighted_choice(
        candidates,
        lambda tile: _enemy_spawn_weight(first, tile, width, height, options),
        rng,
    )
    return (int(first), int(second))


def _sample_dimension(rng: random.Random, options: MapGenerationConfig) -> int:
    """Sample one board dimension from configured game-size weights."""
    if options.size_weights is None:
        return rng.randint(options.min_size, options.max_size)

    candidates = [
        (size, weight)
        for size, weight in options.size_weights
        if options.min_size <= size <= options.max_size
    ]
    if not candidates:
        return rng.randint(options.min_size, options.max_size)
    return int(_weighted_choice(candidates, lambda item: item[1], rng)[0])


def _sample_mountains(
    width: int,
    height: int,
    generals: tuple[int, int],
    count: int,
    rng: random.Random,
) -> set[int]:
    """Sample mountain tiles without covering generals."""
    blocked = set(generals)
    candidates = [tile for tile in range(width * height) if tile not in blocked]
    rng.shuffle(candidates)
    return set(candidates[: min(count, len(candidates))])


def _sample_cities(
    width: int,
    height: int,
    mountains: set[int],
    generals: tuple[int, int],
    count: int,
    rng: random.Random,
    options: MapGenerationConfig,
) -> tuple[list[int], list[int]]:
    """Sample neutral city locations and their starting armies."""
    blocked = set(generals) | mountains
    candidates = [
        tile
        for tile in range(width * height)
        if tile not in blocked
        and manhattan(tile, generals[0], width) > 2
        and manhattan(tile, generals[1], width) > 2
    ]
    rng.shuffle(candidates)
    cities = candidates[: min(count, len(candidates))]
    low, high = options.city_army_range
    city_armies = [rng.randint(low, high) for _ in cities]
    return cities, city_armies


def _weighted_choice(
    items: list[T],
    weight_fn: Callable[[T], float],
    rng: random.Random,
) -> T:
    """Choose one item with non-negative weights."""
    weights = [max(0.0, float(weight_fn(item))) for item in items]
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)

    mark = rng.random() * total
    running = 0.0
    for item, weight in zip(items, weights, strict=True):
        running += weight
        if running >= mark:
            return item
    return items[-1]


def _spawn_edge_weight(tile: int, width: int, height: int, options: MapGenerationConfig) -> float:
    row, col = divmod(tile, width)
    edge_distance = min(row, col, height - 1 - row, width - 1 - col)
    max_distance = max(1, min(width, height) // 2)
    edge_score = 1.0 - min(edge_distance, max_distance) / max_distance
    return 1.0 + options.spawn_edge_bias * edge_score


def _enemy_spawn_weight(
    first: int,
    candidate: int,
    width: int,
    height: int,
    options: MapGenerationConfig,
) -> float:
    distance = manhattan(first, candidate, width)
    close_score = 1.0 / max(1, distance - options.min_general_distance + 1)
    return _spawn_edge_weight(candidate, width, height, options) * (
        1.0 + options.enemy_distance_bias * close_score
    )


def _spawn_has_open_neighbors(
    width: int,
    height: int,
    mountains: set[int],
    generals: tuple[int, int],
    options: MapGenerationConfig,
) -> bool:
    return all(
        sum(1 for neighbor in _neighbors(general, width, height) if neighbor not in mountains)
        >= options.min_spawn_open_neighbors
        for general in generals
    )


def _local_area_fair_enough(
    width: int,
    height: int,
    mountains: set[int],
    generals: tuple[int, int],
    options: MapGenerationConfig,
) -> bool:
    areas = [
        _open_tiles_in_radius(width, height, mountains, general, options.local_fairness_radius)
        for general in generals
    ]
    larger = max(areas)
    if larger == 0:
        return False
    return abs(areas[0] - areas[1]) <= max(4, round(larger * options.max_local_open_diff_ratio))


def _cities_fair_enough(
    width: int,
    cities: list[int],
    city_armies: list[int],
    generals: tuple[int, int],
    options: MapGenerationConfig,
) -> bool:
    count_and_army = []
    for general in generals:
        near = [
            army
            for tile, army in zip(cities, city_armies, strict=True)
            if manhattan(tile, general, width) <= options.city_fairness_radius
        ]
        count_and_army.append((len(near), sum(near)))

    count_diff = abs(count_and_army[0][0] - count_and_army[1][0])
    army_diff = abs(count_and_army[0][1] - count_and_army[1][1])
    return (
        count_diff <= options.max_near_city_count_diff
        and army_diff <= options.max_near_city_army_diff
    )


def _open_tiles_in_radius(
    width: int,
    height: int,
    mountains: set[int],
    center: int,
    radius: int,
) -> int:
    center_row, center_col = divmod(center, width)
    total = 0
    for row in range(max(0, center_row - radius), min(height, center_row + radius + 1)):
        for col in range(max(0, center_col - radius), min(width, center_col + radius + 1)):
            if abs(row - center_row) + abs(col - center_col) > radius:
                continue
            tile = row * width + col
            if tile not in mountains:
                total += 1
    return total


def _neighbors(tile: int, width: int, height: int) -> list[int]:
    row, col = divmod(tile, width)
    result: list[int] = []
    if row > 0:
        result.append(tile - width)
    if row + 1 < height:
        result.append(tile + width)
    if col > 0:
        result.append(tile - 1)
    if col + 1 < width:
        result.append(tile + 1)
    return result


def _validate_options(options: MapGenerationConfig) -> None:
    """Validate map generation parameters before sampling."""
    if options.min_size <= 0 or options.max_size < options.min_size:
        raise ValueError("invalid size range")
    if options.min_general_distance < 1:
        raise ValueError("min_general_distance must be positive")
    if options.size_weights is not None:
        if not options.size_weights:
            raise ValueError("size_weights cannot be empty")
        for size, weight in options.size_weights:
            if size <= 0:
                raise ValueError("size_weights contains a non-positive size")
            if weight < 0:
                raise ValueError("size_weights contains a negative weight")
    _validate_density_range(options.mountain_density_range, "mountain_density_range")
    _validate_density_range(options.city_density_range, "city_density_range")
    if options.city_army_range[1] < options.city_army_range[0]:
        raise ValueError("invalid city_army_range")
    if options.max_attempts <= 0:
        raise ValueError("max_attempts must be positive")


def _validate_density_range(value: tuple[float, float], label: str) -> None:
    low, high = value
    if low < 0 or high < low or high >= 1:
        raise ValueError(f"invalid {label}")
