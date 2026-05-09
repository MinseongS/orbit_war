"""Tests for the W2 heuristic_v1 bot."""

from kaggle_environments import make

from orbit_war.bots import greedy_baseline, heuristic_v1, random_bot, starter_bot
from orbit_war.eval_harness.parallel import run_mirrored_pairs


def test_heuristic_v1_runs_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([heuristic_v1.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_heuristic_v1_beats_random_decisively():
    summary = run_mirrored_pairs(
        bot_a=heuristic_v1.agent,
        bot_b=random_bot.agent,
        seeds=tuple(range(8)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.85, (
        f"heuristic_v1 only beat random {summary.win_rate_a:.0%} — composition is broken"
    )


def test_heuristic_v1_at_least_matches_starter():
    summary = run_mirrored_pairs(
        bot_a=heuristic_v1.agent,
        bot_b=starter_bot.agent,
        seeds=tuple(range(8)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.55, (
        f"heuristic_v1 only beat starter {summary.win_rate_a:.0%} — should clearly outperform"
    )


def test_heuristic_v1_at_least_matches_greedy():
    summary = run_mirrored_pairs(
        bot_a=heuristic_v1.agent,
        bot_b=greedy_baseline.agent,
        seeds=tuple(range(6)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.55, (
        f"heuristic_v1 only beat greedy {summary.win_rate_a:.0%} — composer not adding value"
    )
