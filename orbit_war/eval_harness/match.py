"""Run a single Orbit Wars game between two callable bots and report the result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kaggle_environments import make

Agent = Callable[[dict], list]


@dataclass(frozen=True)
class MatchResult:
    seed: int
    winner: int | None  # 0, 1, or None on draw
    score_a: float
    score_b: float
    error_a: bool
    error_b: bool
    turns_played: int

    @property
    def score_margin(self) -> float:
        return self.score_a - self.score_b


def play_match(
    bot_a: Agent,
    bot_b: Agent,
    seed: int,
    episode_steps: int = 500,
    act_timeout: int = 1,
) -> MatchResult:
    """Run a single 1v1 episode with bot_a as player 0 and bot_b as player 1."""
    error_a = False
    error_b = False

    def safe_bot_a(obs):
        nonlocal error_a
        try:
            return bot_a(obs)
        except Exception:
            error_a = True
            return []

    def safe_bot_b(obs):
        nonlocal error_b
        try:
            return bot_b(obs)
        except Exception:
            error_b = True
            return []

    env = make(
        "orbit_wars",
        configuration={
            "seed": seed,
            "episodeSteps": episode_steps,
            "actTimeout": act_timeout,
        },
        debug=True,
    )
    env.run([safe_bot_a, safe_bot_b])
    final = env.steps[-1]

    score_a = float(final[0].reward) if final[0].reward is not None else 0.0
    score_b = float(final[1].reward) if final[1].reward is not None else 0.0

    if error_a and not error_b:
        winner: int | None = 1
    elif error_b and not error_a:
        winner = 0
    elif score_a > score_b:
        winner = 0
    elif score_b > score_a:
        winner = 1
    else:
        winner = None

    return MatchResult(
        seed=seed,
        winner=winner,
        score_a=score_a,
        score_b=score_b,
        error_a=error_a,
        error_b=error_b,
        turns_played=len(env.steps),
    )
