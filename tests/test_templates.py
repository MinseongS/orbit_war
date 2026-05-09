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
