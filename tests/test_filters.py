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


def test_filter_keeps_combined_partial_attacks_when_aggregate_exceeds_defender():
    """Two partial contributions of 20 each (combined 40) vs defender 30+1=31
    should both survive — the combined fleet captures even though no single
    contribution is sufficient."""
    src1 = Planet(0, 0, 5.0, 5.0, 1.0, 50, 1)
    src2 = Planet(1, 0, 95.0, 5.0, 1.0, 50, 1)
    enemy = Planet(2, 1, 50.0, 50.0, 1.0, 30, 1)
    view = _view_with((src1, src2, enemy))
    steps = [
        Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=20, score=0.5),
        Step(from_planet_id=1, target_planet_id=2, angle=0.0, ships=20, score=0.5),
    ]
    out = filter_capturable(steps, view)
    assert len(out) == 2, "combined contributions should both survive"


def test_filter_drops_all_partial_attacks_when_aggregate_below_defender():
    """Two contributions of 5 each (combined 10) vs defender 30+1=31 should
    both be dropped — even combined we cannot capture."""
    src1 = Planet(0, 0, 5.0, 5.0, 1.0, 50, 1)
    src2 = Planet(1, 0, 95.0, 5.0, 1.0, 50, 1)
    enemy = Planet(2, 1, 50.0, 50.0, 1.0, 30, 1)
    view = _view_with((src1, src2, enemy))
    steps = [
        Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=5, score=0.5),
        Step(from_planet_id=1, target_planet_id=2, angle=0.0, ships=5, score=0.5),
    ]
    out = filter_capturable(steps, view)
    assert out == []


def test_filter_solo_sufficient_attack_survives_even_with_weak_partner():
    """Regression: a single-source attack of 50 ships (vs 30+1 defender) must
    survive even when a sibling step from another source contributes only 5
    ships to the same target. The early aggregate-only rule incorrectly
    grouped the 50-ship attack with the 5-ship partial; if combined (55)
    happened to fall short, both were dropped. The short-circuit fix lets
    individually-sufficient attacks pass regardless of group fate."""
    src1 = Planet(0, 0, 5.0, 5.0, 1.0, 100, 1)
    src2 = Planet(1, 0, 95.0, 5.0, 1.0, 10, 1)
    enemy = Planet(2, 1, 50.0, 50.0, 1.0, 30, 1)
    view = _view_with((src1, src2, enemy))
    steps = [
        # Solo attack — 50 ships > 31 needed. Must pass.
        Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=50, score=0.9),
        # Weak partial — 5 ships. By itself can't capture.
        Step(from_planet_id=1, target_planet_id=2, angle=0.0, ships=5, score=0.3),
    ]
    out = filter_capturable(steps, view)
    surviving_solo = [s for s in out if s.from_planet_id == 0]
    assert len(surviving_solo) == 1, (
        "individually-sufficient solo attack must survive regardless of "
        "weaker partial contributions sharing the target"
    )
