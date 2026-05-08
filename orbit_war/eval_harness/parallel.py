"""Parallel mirrored-pair self-play runner.

Each seed plays *both* bots on each side; results are aggregated from the
perspective of bot_a. Uses `multiprocessing` so we scale across all CPU cores
on the host."""

from __future__ import annotations

import multiprocessing as mp
import pickle
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from orbit_war.eval_harness.match import MatchResult, play_match
from orbit_war.eval_harness.stats import wilson_lower_bound, wilson_upper_bound

Agent = Callable[[dict], list]


@dataclass(frozen=True)
class PairSummary:
    games_played: int
    wins_a: int
    wins_b: int
    draws: int
    error_rate_a: float
    error_rate_b: float
    avg_score_margin_a: float
    win_rate_lower_a: float
    win_rate_upper_a: float

    @property
    def win_rate_a(self) -> float:
        denom = self.wins_a + self.wins_b
        if denom == 0:
            return 0.5
        return self.wins_a / denom


# Workers cannot pickle local functions, so we route through module-level helpers.
def _worker(args: tuple) -> MatchResult:
    bot_a_pkl, bot_b_pkl, seed, side = args
    bot_a = pickle.loads(bot_a_pkl)
    bot_b = pickle.loads(bot_b_pkl)
    if side == 0:
        return play_match(bot_a, bot_b, seed=seed)
    flipped = play_match(bot_b, bot_a, seed=seed)
    # Re-frame from bot_a's perspective.
    return MatchResult(
        seed=seed,
        winner=(None if flipped.winner is None else 1 - flipped.winner),
        score_a=flipped.score_b,
        score_b=flipped.score_a,
        error_a=flipped.error_b,
        error_b=flipped.error_a,
        turns_played=flipped.turns_played,
    )


def run_mirrored_pairs(
    bot_a: Agent,
    bot_b: Agent,
    seeds: Iterable[int],
    workers: int = 4,
) -> PairSummary:
    """Run each seed twice — once with bot_a as player 0, once as player 1."""
    seed_list = list(seeds)
    bot_a_pkl = pickle.dumps(bot_a)
    bot_b_pkl = pickle.dumps(bot_b)
    jobs = [
        (bot_a_pkl, bot_b_pkl, seed, side)
        for seed in seed_list
        for side in (0, 1)
    ]

    if workers <= 1 or len(jobs) <= 1:
        results: Sequence[MatchResult] = [_worker(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            results = pool.map(_worker, jobs)

    return _summarize(results)


def _summarize(results: Sequence[MatchResult]) -> PairSummary:
    wins_a = sum(1 for r in results if r.winner == 0)
    wins_b = sum(1 for r in results if r.winner == 1)
    draws = sum(1 for r in results if r.winner is None)
    err_a = sum(1 for r in results if r.error_a)
    err_b = sum(1 for r in results if r.error_b)
    margin = sum(r.score_margin for r in results) / max(1, len(results))

    decisive = wins_a + wins_b
    n = max(1, len(results))
    return PairSummary(
        games_played=len(results),
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        error_rate_a=err_a / n,
        error_rate_b=err_b / n,
        avg_score_margin_a=margin,
        win_rate_lower_a=wilson_lower_bound(wins_a, max(1, decisive)),
        win_rate_upper_a=wilson_upper_bound(wins_a, max(1, decisive)),
    )
