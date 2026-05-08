"""Smoke tests for the translated public Tactical Heuristic bot."""

from kaggle_environments import make

from orbit_war.bots import public_tactical, random_bot
from orbit_war.eval_harness.parallel import run_mirrored_pairs


def test_public_tactical_runs_without_errors():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([public_tactical.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_public_tactical_beats_random_majority():
    summary = run_mirrored_pairs(
        bot_a=public_tactical.agent,
        bot_b=random_bot.agent,
        seeds=tuple(range(10)),
        workers=2,
    )
    # Public floor must comfortably beat random.
    assert summary.win_rate_a >= 0.85, f"public_tactical only won {summary.win_rate_a:.0%} vs random"
