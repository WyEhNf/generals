#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generals_bot.matches import build_game_config, make_policy, parse_agent_spec
from generals_bot.sim import GeneralsEnv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end serial temporal-policy inference latency."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--turns", type=int, default=800)
    args = parser.parse_args()

    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("warmup must be nonnegative and samples must be positive")

    checkpoint = Path(args.checkpoint)
    target_spec = parse_agent_spec(f"target=checkpoint:{checkpoint}")
    opponent_spec = parse_agent_spec("flobot")
    latencies_ms: list[float] = []
    actions_seen = 0
    games = 0

    while len(latencies_ms) < args.samples:
        seed = args.seed_start + games
        config = build_game_config("generated", seed=seed, max_turns=args.turns)
        env = GeneralsEnv(config)
        observations = env.reset(seed=seed)
        target = make_policy(
            target_spec, player=0, seed=seed, config=config, device=args.device,
        )
        opponent = make_policy(
            opponent_spec, player=1, seed=seed + 1, config=config, device="cpu",
        )
        games += 1

        while not env.done and len(latencies_ms) < args.samples:
            actions = {}
            for player in env.alive_players:
                observation = observations[player]
                if player == 0:
                    _synchronize(args.device)
                    started = perf_counter_ns()
                    action = target.act(observation)
                    _synchronize(args.device)
                    elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0
                    if actions_seen >= args.warmup:
                        latencies_ms.append(elapsed_ms)
                    actions_seen += 1
                else:
                    action = opponent.act(observation)
                actions[player] = action
            observations, _, _, _ = env.step(actions)

    values = np.asarray(latencies_ms, dtype=np.float64)
    report = {
        "checkpoint": str(checkpoint),
        "device": args.device,
        "warmup": args.warmup,
        "samples": int(values.size),
        "games": games,
        "latency_ms": {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        },
        "acceptance": {
            "threshold_p95_ms": 20.0,
            "passed": bool(np.percentile(values, 95) < 20.0),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["acceptance"]["passed"] else 1


def _synchronize(device: str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
