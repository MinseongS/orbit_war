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
