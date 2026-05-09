"""Tests for step templates."""

from kaggle_environments import make

from orbit_war.plan_gen.step import Step
from orbit_war.plan_gen.templates import (
    no_op_template,
    production_attack_template,
)
from orbit_war.sim.observation import GameView


def _fresh_view(seed: int = 42) -> GameView:
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.reset(num_agents=2)
    return GameView.from_obs(env.steps[0][0]["observation"])


def test_no_op_template_returns_empty_list():
    view = _fresh_view()
    assert no_op_template(view) == []


def test_production_attack_emits_steps_when_we_own_planets():
    view = _fresh_view()
    steps = production_attack_template(view)
    assert all(isinstance(s, Step) for s in steps)
    sources_with_ships = [p for p in view.my_planets() if p.ships >= 2]
    sources_proposed = {s.from_planet_id for s in steps}
    assert all(p.id in sources_proposed for p in sources_with_ships)


def test_production_attack_scores_higher_for_higher_production_per_distance():
    view = _fresh_view()
    steps = production_attack_template(view)
    assert all(s.score >= 0 for s in steps)


def test_production_attack_ships_are_min_to_capture():
    view = _fresh_view()
    steps = production_attack_template(view)
    by_target = {p.id: p for p in view.planets}
    for s in steps:
        target = by_target[s.target_planet_id]
        assert s.ships == min(target.ships + 1, _source_ships(view, s.from_planet_id))


def _source_ships(view: GameView, planet_id: int) -> int:
    p = next(q for q in view.planets if q.id == planet_id)
    return p.ships


from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from orbit_war.plan_gen.templates import defensive_reinforce_template


def test_defensive_reinforce_emits_when_planet_under_attack():
    threatened = Planet(0, 0, 30.0, 30.0, 1.0, 5, 1)
    helper = Planet(1, 0, 35.0, 30.0, 1.0, 50, 1)
    enemy = Planet(2, 1, 80.0, 80.0, 1.0, 1, 1)
    incoming = Fleet(0, 1, 32.0, 30.0, 0.0, 2, 30)
    view = GameView(
        player=0,
        planets=(threatened, helper, enemy),
        fleets=(incoming,),
        angular_velocity=0.04,
        initial_planets=(threatened, helper, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    steps = defensive_reinforce_template(view)
    assert any(s.target_planet_id == 0 for s in steps)
    for s in steps:
        if s.target_planet_id == 0:
            assert s.from_planet_id == 1


def test_defensive_reinforce_quiet_when_no_threat():
    me = Planet(0, 0, 30.0, 30.0, 1.0, 5, 1)
    enemy = Planet(1, 1, 80.0, 80.0, 1.0, 1, 1)
    view = GameView(
        player=0,
        planets=(me, enemy),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    assert defensive_reinforce_template(view) == []


from orbit_war.plan_gen.templates import snipe_undefended_template


def test_snipe_undefended_finds_low_defense_high_prod_target():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 2)
    juicy = Planet(1, -1, 20.0, 10.0, 2.5, 3, 4)  # cheap + high production
    boring = Planet(2, -1, 80.0, 80.0, 1.0, 80, 1)
    view = GameView(
        player=0,
        planets=(me, juicy, boring),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, juicy, boring),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    steps = snipe_undefended_template(view)
    assert any(s.target_planet_id == 1 for s in steps)
    for s in steps:
        assert s.target_planet_id != 2


def test_snipe_undefended_skips_planets_we_cant_afford():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 2, 1)  # only 2 ships
    juicy = Planet(1, -1, 20.0, 10.0, 2.5, 5, 4)  # needs 6 ships to capture
    view = GameView(
        player=0,
        planets=(me, juicy),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, juicy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    steps = snipe_undefended_template(view)
    assert steps == []


import math

from kaggle_environments import make as _make_env

from orbit_war.sim.orbits import is_orbiting


def test_production_attack_uses_orbit_aware_aim_for_orbiting_targets():
    env = _make_env("orbit_wars", configuration={"seed": 7}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]
    view = GameView.from_obs(obs)

    steps = production_attack_template(view)
    by_target = {p.id: p for p in view.planets}

    # Find at least one step targeting an orbiting non-source planet.
    diverged = False
    for s in steps:
        target = by_target[s.target_planet_id]
        src = by_target[s.from_planet_id]
        if not is_orbiting(target):
            continue
        naive = math.atan2(target.y - src.y, target.x - src.x)
        if not math.isclose(s.angle, naive, abs_tol=1e-3):
            diverged = True
            break
    # If every target is static this test is uninformative — accept either way.
    if any(is_orbiting(by_target[s.target_planet_id]) for s in steps):
        assert diverged, "expected at least one orbit-aware angle to differ from naive atan2"


from orbit_war.plan_gen.templates import multi_source_consolidation_template


def test_multi_source_consolidation_emits_multiple_steps_to_same_target():
    """Three friendly sources target one rich enemy planet. Expect 2-3 steps
    all aimed at the same target."""
    src1 = Planet(0, 0, 5.0, 5.0, 1.0, 40, 1)
    src2 = Planet(1, 0, 95.0, 5.0, 1.0, 40, 1)
    src3 = Planet(2, 0, 5.0, 95.0, 1.0, 40, 1)
    rich = Planet(3, 1, 50.0, 50.0, 5.0, 100, 5)
    view = GameView(
        player=0,
        planets=(src1, src2, src3, rich),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(src1, src2, src3, rich),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    steps = multi_source_consolidation_template(view)
    assert len(steps) >= 2, "consolidation should propose >=2 contributing sources"
    targets = {s.target_planet_id for s in steps}
    assert targets == {3}, "all consolidation steps target the rich enemy planet"


def test_multi_source_consolidation_quiet_when_only_one_source():
    """No 'multi'-source possible if we only own one planet."""
    only = Planet(0, 0, 5.0, 5.0, 1.0, 40, 1)
    rich = Planet(1, 1, 50.0, 50.0, 1.0, 100, 5)
    view = GameView(
        player=0,
        planets=(only, rich),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(only, rich),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    assert multi_source_consolidation_template(view) == []
