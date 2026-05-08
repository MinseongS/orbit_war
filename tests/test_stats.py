"""Tests for Wilson CI and sample-size helpers."""

import math

from orbit_war.eval_harness.stats import (
    samples_needed_for_two_proportion,
    wilson_lower_bound,
    wilson_upper_bound,
)


def test_wilson_lower_bound_zero_wins():
    # 0 wins out of 10 — lower bound must be < 0.31 (rule of three roughly).
    assert wilson_lower_bound(wins=0, n=10) == 0.0
    assert wilson_upper_bound(wins=0, n=10) > 0.0
    assert wilson_upper_bound(wins=0, n=10) < 0.31


def test_wilson_lower_bound_centered():
    # 50 wins out of 100 — lower bound should be a touch under 0.5.
    lo = wilson_lower_bound(wins=50, n=100)
    hi = wilson_upper_bound(wins=50, n=100)
    assert 0.39 < lo < 0.5
    assert 0.5 < hi < 0.61


def test_wilson_lower_bound_high_confidence():
    # 95 wins out of 100 — lower bound must clearly exceed 0.5.
    assert wilson_lower_bound(wins=95, n=100) > 0.85


def test_samples_needed_decreases_with_larger_effect():
    n_small = samples_needed_for_two_proportion(p1=0.53, p2=0.50, alpha=0.05, power=0.80)
    n_med = samples_needed_for_two_proportion(p1=0.60, p2=0.50, alpha=0.05, power=0.80)
    n_large = samples_needed_for_two_proportion(p1=0.70, p2=0.50, alpha=0.05, power=0.80)
    assert n_small > n_med > n_large
    # Sanity bounds
    assert 800 < n_small < 5000
    assert 100 < n_med < 800
    assert 30 < n_large < 200


def test_sample_size_handles_equal_proportions():
    n = samples_needed_for_two_proportion(p1=0.5, p2=0.5, alpha=0.05, power=0.80)
    assert math.isinf(n)
