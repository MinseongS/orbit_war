"""Tests for the stratified submission gate."""

from orbit_war.bots import greedy_baseline, random_bot, starter_bot
from orbit_war.eval_harness.gate import GateReport, GateTier, evaluate_gate


def test_gate_passes_for_a_clearly_strong_challenger():
    # Greedy crushes random; we use a small-N config so the test runs in seconds.
    report = evaluate_gate(
        challenger=greedy_baseline.agent,
        sanity_pool={"random": random_bot.agent},
        diversity_pool={},
        champion=random_bot.agent,
        sanity_min_win_rate=0.80,
        diversity_min_win_rate=0.55,
        champion_min_win_rate=0.55,
        seeds_per_pool=(1, 2, 3),
    )
    assert isinstance(report, GateReport)
    assert report.passed is True
    sanity_results = [t for t in report.tiers if t.name == "sanity:random"]
    assert sanity_results and sanity_results[0].passed


def test_gate_fails_when_challenger_is_weaker_than_champion():
    # Random vs greedy: random will lose decisively. Champion tier must fail.
    report = evaluate_gate(
        challenger=random_bot.agent,
        sanity_pool={},
        diversity_pool={},
        champion=greedy_baseline.agent,
        sanity_min_win_rate=0.80,
        diversity_min_win_rate=0.55,
        champion_min_win_rate=0.55,
        seeds_per_pool=(1, 2, 3),
    )
    assert report.passed is False
    failed = [t for t in report.tiers if not t.passed]
    assert any(t.name == "champion" for t in failed)


def test_gate_marks_sanity_failure_explicitly():
    # Starter vs starter: rough parity, sanity demand at 80% should fail.
    report = evaluate_gate(
        challenger=starter_bot.agent,
        sanity_pool={"twin": starter_bot.agent},
        diversity_pool={},
        champion=starter_bot.agent,
        sanity_min_win_rate=0.80,
        diversity_min_win_rate=0.55,
        champion_min_win_rate=0.55,
        seeds_per_pool=(1, 2),
    )
    sanity = next(t for t in report.tiers if t.name == "sanity:twin")
    assert sanity.passed is False
    assert report.passed is False
