"""Tests for the W3 heuristic_v3 bot."""

from kaggle_environments import make

from orbit_war.bots import (
    heuristic_v1,
    heuristic_v3,
    public_tactical,
    random_bot,
)
from orbit_war.eval_harness.parallel import run_mirrored_pairs


def test_heuristic_v3_runs_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([heuristic_v3.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_heuristic_v3_beats_random_decisively():
    summary = run_mirrored_pairs(
        bot_a=heuristic_v3.agent,
        bot_b=random_bot.agent,
        seeds=tuple(range(8)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.85, (
        f"heuristic_v3 only beat random {summary.win_rate_a:.0%}"
    )


def test_heuristic_v3_at_least_matches_v1():
    """v3 should NOT regress against v1 — orbit-aware aiming + new templates
    should be at least as good as the W2 baseline."""
    summary = run_mirrored_pairs(
        bot_a=heuristic_v3.agent,
        bot_b=heuristic_v1.agent,
        seeds=tuple(range(10)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.50, (
        f"heuristic_v3 regressed to {summary.win_rate_a:.0%} vs v1"
    )


def test_heuristic_v3_improves_against_public_tactical():
    """Against public_tactical, v3 should lift from v1's 10% to at least 25%
    — orbit-aware aiming alone tends to be a 10-15pp lift; new templates may
    add more."""
    summary = run_mirrored_pairs(
        bot_a=heuristic_v3.agent,
        bot_b=public_tactical.agent,
        seeds=tuple(range(8)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.25, (
        f"heuristic_v3 only at {summary.win_rate_a:.0%} vs public_tactical — "
        f"orbit-aware aim should lift this above 25%"
    )
