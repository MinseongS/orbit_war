"""Stratified evaluation gate.

A challenger must clear all configured tiers (sanity, diversity, champion)
before it may be submitted to the ladder. The gate fails closed: any tier
that does not meet its minimum win rate blocks the submission as a whole."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

from orbit_war.eval_harness.parallel import PairSummary, run_mirrored_pairs

Agent = Callable[[dict], list]


@dataclass(frozen=True)
class GateTier:
    name: str
    passed: bool
    summary: PairSummary
    required_win_rate: float


@dataclass(frozen=True)
class GateReport:
    passed: bool
    tiers: tuple[GateTier, ...]


def evaluate_gate(
    challenger: Agent,
    sanity_pool: Mapping[str, Agent],
    diversity_pool: Mapping[str, Agent],
    champion: Agent,
    sanity_min_win_rate: float = 0.95,
    diversity_min_win_rate: float = 0.55,
    champion_min_win_rate: float = 0.55,
    seeds_per_pool: Iterable[int] = tuple(range(20)),
    workers: int = 4,
) -> GateReport:
    """Run the challenger through three tiers and report pass/fail per tier."""
    tier_results: list[GateTier] = []
    seeds = tuple(seeds_per_pool)

    for name, opponent in sanity_pool.items():
        summary = run_mirrored_pairs(challenger, opponent, seeds, workers=workers)
        tier_results.append(
            GateTier(
                name=f"sanity:{name}",
                passed=summary.win_rate_a >= sanity_min_win_rate,
                summary=summary,
                required_win_rate=sanity_min_win_rate,
            )
        )

    for name, opponent in diversity_pool.items():
        summary = run_mirrored_pairs(challenger, opponent, seeds, workers=workers)
        tier_results.append(
            GateTier(
                name=f"diversity:{name}",
                passed=summary.win_rate_a >= diversity_min_win_rate,
                summary=summary,
                required_win_rate=diversity_min_win_rate,
            )
        )

    champ_summary = run_mirrored_pairs(challenger, champion, seeds, workers=workers)
    tier_results.append(
        GateTier(
            name="champion",
            passed=champ_summary.win_rate_a >= champion_min_win_rate,
            summary=champ_summary,
            required_win_rate=champion_min_win_rate,
        )
    )

    overall = all(t.passed for t in tier_results)
    return GateReport(passed=overall, tiers=tuple(tier_results))
