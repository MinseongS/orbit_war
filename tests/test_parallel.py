"""Tests for the parallel mirrored-pair runner."""

from orbit_war.bots import greedy_baseline, random_bot
from orbit_war.eval_harness.parallel import PairSummary, run_mirrored_pairs


def _crashing_agent_module_level(obs):  # noqa: ARG001
    raise RuntimeError("boom")


def test_run_mirrored_pairs_against_random():
    summary = run_mirrored_pairs(
        bot_a=greedy_baseline.agent,
        bot_b=random_bot.agent,
        seeds=(1, 2, 3, 4, 5),
        workers=2,
    )
    assert isinstance(summary, PairSummary)
    assert summary.games_played == 10  # 5 seeds * 2 sides
    assert 0.0 <= summary.win_rate_a <= 1.0
    assert summary.win_rate_a > 0.7  # greedy clearly beats random


def test_pair_summary_breaks_out_errors_per_bot():
    summary = run_mirrored_pairs(
        bot_a=_crashing_agent_module_level,
        bot_b=random_bot.agent,
        seeds=(1, 2),
        workers=2,
    )
    assert summary.error_rate_a == 1.0
    assert summary.error_rate_b == 0.0
