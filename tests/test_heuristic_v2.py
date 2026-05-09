"""Tests for heuristic_v2 (fitted weights)."""

from kaggle_environments import make

from orbit_war.bots import heuristic_v1, heuristic_v2, random_bot
from orbit_war.eval_harness.parallel import run_mirrored_pairs


def test_heuristic_v2_runs_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([heuristic_v2.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_heuristic_v2_does_not_regress_against_v1():
    """Soft check: v2 should at least match v1. A clear loss means the fit
    overfit or the score-derivation function from weights is wrong."""
    summary = run_mirrored_pairs(
        bot_a=heuristic_v2.agent,
        bot_b=heuristic_v1.agent,
        seeds=tuple(range(10)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.45, (
        f"heuristic_v2 only beat heuristic_v1 {summary.win_rate_a:.0%} — fit may be bad"
    )
