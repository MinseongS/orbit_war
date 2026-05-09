"""Tests for orbit_war.plan_gen.aim."""

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    Planet,
)

from orbit_war.plan_gen.aim import aim_with_orbit_prediction
from orbit_war.sim.observation import GameView


def _view_with(planets: tuple[Planet, ...], angular_velocity: float = 0.05) -> GameView:
    return GameView(
        player=0,
        planets=planets,
        fleets=(),
        angular_velocity=angular_velocity,
        initial_planets=planets,
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )


def test_aim_to_static_target_matches_naive_atan2():
    """A static (far-from-sun) planet doesn't move — orbit-aware angle equals naive."""
    src = Planet(0, 0, 10.0, 50.0, 1.0, 50, 1)
    # orbital_radius = hypot(97-50, 50-50) = 47; 47 + 5 = 52 >= ROTATION_RADIUS_LIMIT=50
    static_tgt = Planet(1, -1, 97.0, 50.0, 5.0, 5, 1)  # outside ROTATION_LIMIT
    view = _view_with((src, static_tgt))
    angle, _arrival_turn = aim_with_orbit_prediction(src, static_tgt, ships=10, view=view)
    expected = math.atan2(50.0 - 50.0, 97.0 - 10.0)
    assert math.isclose(angle, expected, abs_tol=1e-9)


def test_aim_to_orbiting_target_drifts_from_naive():
    """An orbiting target makes the orbit-aware angle differ from atan2 to current pos."""
    src = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    orbiting_tgt = Planet(1, -1, CENTER + 20, CENTER, 1.0, 5, 1)  # inside ROTATION_LIMIT
    view = _view_with((src, orbiting_tgt), angular_velocity=0.05)
    naive = math.atan2(orbiting_tgt.y - src.y, orbiting_tgt.x - src.x)
    angle, arrival_turn = aim_with_orbit_prediction(src, orbiting_tgt, ships=10, view=view)
    assert arrival_turn >= 1
    # With angular_velocity 0.05 over multiple turns, the angle should drift noticeably.
    assert not math.isclose(angle, naive, abs_tol=1e-3)


def test_aim_returns_arrival_turn_at_least_one():
    src = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    tgt = Planet(1, -1, 11.0, 10.0, 1.0, 5, 1)  # very close
    view = _view_with((src, tgt))
    _, arrival_turn = aim_with_orbit_prediction(src, tgt, ships=10, view=view)
    assert arrival_turn >= 1
