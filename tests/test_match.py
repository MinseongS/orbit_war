"""Tests for the single-game match runner."""

from orbit_war.bots import greedy_baseline, random_bot, starter_bot
from orbit_war.eval_harness.match import MatchResult, play_match


def test_play_match_returns_result_with_correct_seed():
    result = play_match(starter_bot.agent, random_bot.agent, seed=42)
    assert isinstance(result, MatchResult)
    assert result.seed == 42
    assert result.winner in (0, 1, None)


def test_play_match_records_score_margin():
    result = play_match(greedy_baseline.agent, random_bot.agent, seed=1)
    assert result.score_a + result.score_b >= 0
    assert result.score_margin == result.score_a - result.score_b


def test_play_match_detects_errors_per_side():
    def crashing_agent(obs):
        raise RuntimeError("boom")

    result = play_match(crashing_agent, random_bot.agent, seed=1)
    assert result.error_a is True
    assert result.error_b is False
