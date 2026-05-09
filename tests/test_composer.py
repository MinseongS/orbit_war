"""Tests for the step composer."""

from orbit_war.plan_gen.composer import compose_plan
from orbit_war.plan_gen.step import Step


def test_compose_returns_empty_when_no_steps():
    assert compose_plan([], surplus_by_planet={0: 100}) == []


def test_compose_picks_highest_score_first():
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=10, score=0.3)
    b = Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=10, score=0.9)
    plan = compose_plan([a, b], surplus_by_planet={0: 30})
    assert plan[0].target_planet_id == 2  # b first


def test_compose_respects_surplus():
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=20, score=0.9)
    b = Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=20, score=0.8)
    plan = compose_plan([a, b], surplus_by_planet={0: 25})
    assert len(plan) == 1
    assert plan[0].target_planet_id == 1


def test_compose_partial_steps_when_truncatable():
    """If a step's `ships` exceeds remaining surplus, the composer may
    truncate to remaining surplus (>=1) and still emit."""
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=20, score=0.9)
    b = Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=20, score=0.8)
    plan = compose_plan([a, b], surplus_by_planet={0: 25}, allow_truncation=True)
    assert len(plan) == 2
    assert plan[1].ships == 5


def test_compose_unrelated_planets_dont_interact():
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=10, score=0.9)
    b = Step(from_planet_id=2, target_planet_id=3, angle=0.0, ships=10, score=0.8)
    plan = compose_plan([a, b], surplus_by_planet={0: 10, 2: 10})
    assert len(plan) == 2


def test_compose_ignores_steps_from_planets_without_surplus():
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=10, score=0.9)
    plan = compose_plan([a], surplus_by_planet={0: 0})
    assert plan == []


def test_compose_plan_calls_validator_when_provided():
    """The validator should receive the chosen plan + surplus snapshot and
    may return either the plan unchanged or an empty list."""
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=10, score=0.9)
    captured: list[list[Step]] = []

    def my_validator(plan: list[Step]) -> list[Step]:
        captured.append(list(plan))
        return []  # always reject

    result = compose_plan(
        [a], surplus_by_planet={0: 30}, validator=my_validator,
    )
    assert result == []
    assert captured == [[a]]
