"""Tests for orbit_war.plan_gen.filters."""

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_war.plan_gen.filters import filter_capturable
from orbit_war.plan_gen.step import Step
from orbit_war.sim.observation import GameView


def _view_with(planets: tuple[Planet, ...]) -> GameView:
    return GameView(
        player=0,
        planets=planets,
        fleets=(),
        angular_velocity=0.04,
        initial_planets=planets,
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )


def test_filter_drops_underpowered_attacks():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 30, 1)
    view = _view_with((me, enemy))
    steps = [
        Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=20, score=0.5),
    ]
    assert filter_capturable(steps, view) == []  # 20 < 30+1


def test_filter_keeps_winning_attacks():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 30, 1)
    view = _view_with((me, enemy))
    steps = [
        Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=31, score=0.5),
    ]
    assert len(filter_capturable(steps, view)) == 1


def test_filter_passes_friendly_reinforcements_unconditionally():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    sibling = Planet(1, 0, 90.0, 90.0, 1.0, 30, 1)
    view = _view_with((me, sibling))
    steps = [
        Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=5, score=0.5),
    ]
    assert len(filter_capturable(steps, view)) == 1


def test_filter_passes_through_step_for_unknown_target():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    view = _view_with((me,))
    steps = [
        Step(from_planet_id=0, target_planet_id=999, angle=0.0, ships=5, score=0.5),
    ]
    assert len(filter_capturable(steps, view)) == 1
