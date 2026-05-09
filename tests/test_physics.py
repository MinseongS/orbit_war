"""Tests for fleet trajectory physics."""

import math

import pytest

from orbit_war.sim.physics import (
    fleet_speed,
    straight_line_distance,
    turns_to_arrive,
)


@pytest.mark.parametrize(
    "ships,expected_speed",
    [
        (1, 1.0),
        (1000, 6.0),  # max_speed
    ],
)
def test_fleet_speed_endpoints(ships: int, expected_speed: float):
    assert math.isclose(fleet_speed(ships), expected_speed, abs_tol=1e-9)


def test_fleet_speed_monotonic_in_ships():
    last = 0.0
    for ships in (1, 5, 10, 50, 100, 500, 1000, 5000):
        s = fleet_speed(ships)
        assert s >= last
        last = s


def test_straight_line_distance_basic():
    assert straight_line_distance(0.0, 0.0, 3.0, 4.0) == 5.0


def test_turns_to_arrive_at_least_one():
    assert turns_to_arrive(0.0, 0.0, 0.5, 0.5, ships=10) >= 1


def test_turns_to_arrive_scales_with_distance():
    near = turns_to_arrive(0.0, 0.0, 5.0, 0.0, ships=100)
    far = turns_to_arrive(0.0, 0.0, 50.0, 0.0, ships=100)
    assert far > near


def test_fleet_speed_uses_log_curve():
    s100 = fleet_speed(100)
    assert 1.5 < s100 < 5.5
