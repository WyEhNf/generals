#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generals_bot.matches import build_game_config, make_policy, parse_agent_spec, run_match
from generals_bot.viz import write_html


DEFAULT_OUTPUT = "artifacts/matches/render_match.html"
DEFAULT_REPORT = "artifacts/matches/render_match.json"
DEFAULT_TURNS = 800


def main() -> int:
    """Run one local agent-vs-agent match and write HTML plus a JSON report."""
    parser = argparse.ArgumentParser(description="Render one local agent-vs-agent match.")
    parser.add_argument("--agent0", default="expander")
    parser.add_argument("--agent1", default="expander")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map", choices=["generated", "demo"], default="generated")
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--pass-bias", type=float, default=0.0)
    parser.add_argument("--pass-bias-turn", type=int, default=50)
    parser.add_argument("--late-start", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    spec0 = parse_agent_spec(args.agent0)
    spec1 = parse_agent_spec(args.agent1)
    config = build_game_config(
        args.map,
        seed=args.seed,
        max_turns=args.turns,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    policies = {
        0: make_policy(
            spec0,
            player=0,
            seed=args.seed,
            config=config,
            device=args.device,
            pass_bias=args.pass_bias,
            pass_bias_turn=args.pass_bias_turn,
        ),
        1: make_policy(
            spec1,
            player=1,
            seed=args.seed + 1,
            config=config,
            device=args.device,
            pass_bias=args.pass_bias,
            pass_bias_turn=args.pass_bias_turn,
        ),
    }
    result = run_match(
        config=config,
        policies=policies,
        side_labels={0: spec0.label, 1: spec1.label},
        seed=args.seed,
        record_snapshots=True,
        late_start=args.late_start,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_html(result.snapshots, output_path)

    report = {
        "agent0": {"label": spec0.label, "spec": spec0.raw},
        "agent1": {"label": spec1.label, "spec": spec1.raw},
        "map": {
            "kind": args.map,
            "seed": args.seed,
            "width": config.width,
            "height": config.height,
            "generals": list(config.generals),
            "mountain_count": len(config.mountains),
            "city_count": len(config.cities),
            "min_size": args.min_size,
            "max_size": args.max_size,
        },
        **result.report,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(output_path)
    print(report_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
