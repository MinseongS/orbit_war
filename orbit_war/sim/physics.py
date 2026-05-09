"""Fleet trajectory physics — straight-line, no sun avoidance.

Sun-avoidance pathfinding belongs in W3 (alongside the lookahead search).
For W2's greedy composer, straight-line arrival math is enough: each step
template already self-prunes routes that would obviously cross the sun.

Constants match the official simulator
(`kaggle_environments.envs.orbit_wars.orbit_wars`).
"""

from __future__ import annotations

import math

MAX_SPEED = 6.0


def fleet_speed(ships: int) -> float:
    """Return the per-turn fleet speed for a given ship count.

    Mirrors the official formula:
        speed = 1.0 + (MAX_SPEED - 1) * (log(ships)/log(1000))^1.5
    Bottoms at 1.0 (a single ship); tops at MAX_SPEED at >=1000 ships.
    """
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def straight_line_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def turns_to_arrive(
    src_x: float, src_y: float, tgt_x: float, tgt_y: float, ships: int
) -> int:
    """Integer turns for a fleet of `ships` to traverse the straight line."""
    distance = straight_line_distance(src_x, src_y, tgt_x, tgt_y)
    if distance <= 0.0:
        return 1
    speed = fleet_speed(max(1, ships))
    return max(1, int(math.ceil(distance / speed)))
