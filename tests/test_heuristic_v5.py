"""Tests for heuristic_v5 (per-step regression weights)."""

from kaggle_environments import make

from orbit_war.bots import heuristic_v4, heuristic_v5, random_bot
from orbit_war.eval_harness.parallel import run_mirrored_pairs


def test_heuristic_v5_runs_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([heuristic_v5.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_heuristic_v5_does_not_regress_against_v4():
    """Soft check: v5 should at least match v4."""
    summary = run_mirrored_pairs(
        bot_a=heuristic_v5.agent,
        bot_b=heuristic_v4.agent,
        seeds=tuple(range(10)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.45, (
        f"heuristic_v5 only beat heuristic_v4 {summary.win_rate_a:.0%} — fit may be bad"
    )
