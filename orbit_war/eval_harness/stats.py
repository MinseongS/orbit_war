"""Confidence intervals and sample-size calculations for win-rate comparisons.

We use:
  - Wilson score interval (better than normal-approximation, especially
    near 0 and 1) for the per-comparison CI.
  - Two-proportion z-test for sample-size targets.
"""

from __future__ import annotations

import math

# 95% two-sided z, 80% power one-sided z
Z_95 = 1.959963984540054
Z_80 = 0.8416212335729143


def _wilson(wins: int, n: int, z: float, lower: bool) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return centre - margin if lower else centre + margin


def wilson_lower_bound(wins: int, n: int, z: float = Z_95) -> float:
    """Lower bound of the Wilson score interval at confidence z (default 95%)."""
    return max(0.0, _wilson(wins, n, z, lower=True))


def wilson_upper_bound(wins: int, n: int, z: float = Z_95) -> float:
    """Upper bound of the Wilson score interval at confidence z."""
    return min(1.0, _wilson(wins, n, z, lower=False))


def samples_needed_for_two_proportion(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Return N (per group) needed to detect p1 vs p2 in a two-proportion z-test.

    Returns +inf if the proportions are equal.
    """
    if p1 == p2:
        return math.inf

    z_alpha = Z_95 if alpha == 0.05 else _z_two_sided(alpha)
    z_power = Z_80 if power == 0.80 else _z_one_sided(power)

    p_bar = (p1 + p2) / 2.0
    numerator = (
        z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_power * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    denominator = (p1 - p2) ** 2
    return math.ceil(numerator / denominator)


def _z_two_sided(alpha: float) -> float:
    # Erf-based inverse normal for the (1 - alpha/2) quantile.
    return math.sqrt(2.0) * _inv_erf(1.0 - alpha)


def _z_one_sided(power: float) -> float:
    return math.sqrt(2.0) * _inv_erf(2.0 * power - 1.0)


def _inv_erf(y: float) -> float:
    # Winitzki approximation, accurate to ~4e-3.
    a = 0.147
    ln = math.log(1.0 - y * y)
    first = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(
        math.sqrt(math.sqrt(first * first - ln / a) - first), y
    )
