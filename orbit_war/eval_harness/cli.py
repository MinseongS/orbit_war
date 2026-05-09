"""CLI: run the stratified gate against the W1 zoo and print a report."""

from __future__ import annotations

import argparse
import importlib
from typing import Callable

from orbit_war.eval_harness.gate import evaluate_gate

ZOO_BOT_PATHS: dict[str, str] = {
    "random": "orbit_war.bots.random_bot:agent",
    "starter": "orbit_war.bots.starter_bot:agent",
    "greedy": "orbit_war.bots.greedy_baseline:agent",
    "public_tactical": "orbit_war.bots.public_tactical:agent",
    "heuristic_v1": "orbit_war.bots.heuristic_v1:agent",
}


def _load(spec: str) -> Callable:
    module_path, attr = spec.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run W1 stratified evaluation gate.")
    ap.add_argument("challenger", help="dotted path:attr of the challenger agent")
    ap.add_argument("--champion", default="orbit_war.bots.greedy_baseline:agent")
    ap.add_argument("--seeds", type=int, default=20, help="seeds per pool")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    challenger = _load(args.challenger)
    champion = _load(args.champion)

    sanity = {
        "random": _load(ZOO_BOT_PATHS["random"]),
        "starter": _load(ZOO_BOT_PATHS["starter"]),
    }
    diversity = {
        "greedy": _load(ZOO_BOT_PATHS["greedy"]),
        "public_tactical": _load(ZOO_BOT_PATHS["public_tactical"]),
    }

    report = evaluate_gate(
        challenger=challenger,
        sanity_pool=sanity,
        diversity_pool=diversity,
        champion=champion,
        seeds_per_pool=tuple(range(args.seeds)),
        workers=args.workers,
    )

    for tier in report.tiers:
        print(
            f"{tier.name:30s} "
            f"win_rate={tier.summary.win_rate_a:.3f} "
            f"games={tier.summary.games_played} "
            f"req={tier.required_win_rate:.2f} "
            f"{'PASS' if tier.passed else 'FAIL'}"
        )
    print(f"\nOVERALL: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
