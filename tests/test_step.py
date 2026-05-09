"""Tests for the Step abstraction."""

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_war.plan_gen.step import Step, ships_needed_to_capture


def test_step_packs_to_kaggle_move_format():
    step = Step(
        from_planet_id=3,
        target_planet_id=7,
        angle=1.234,
        ships=42,
        score=0.5,
    )
    assert step.as_move() == [3, 1.234, 42]


def test_ships_needed_to_capture_includes_one_extra():
    target = Planet(0, -1, 10.0, 10.0, 1.0, 30, 2)
    assert ships_needed_to_capture(target) == 31


def test_ships_needed_to_capture_owned_returns_zero():
    target = Planet(0, 0, 10.0, 10.0, 1.0, 30, 2)
    assert ships_needed_to_capture(target, player=0) == 0


def test_step_angle_to_target_is_atan2_to_target():
    src = Planet(0, 0, 0.0, 0.0, 1.0, 50, 1)
    tgt = Planet(1, -1, 3.0, 4.0, 1.0, 5, 1)
    expected = math.atan2(4.0, 3.0)
    assert math.isclose(Step.angle_to(src, tgt), expected)


def test_step_orderable_by_score_descending():
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=10, score=0.3)
    b = Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=10, score=0.9)
    c = Step(from_planet_id=0, target_planet_id=3, angle=0.0, ships=10, score=0.6)
    descending = sorted([a, b, c], key=lambda s: -s.score)
    assert [s.target_planet_id for s in descending] == [2, 3, 1]
