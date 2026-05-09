"""Tests for the typed observation view."""

import math

from kaggle_environments import make

from orbit_war.sim.observation import GameView


def test_gameview_parses_first_step():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]

    view = GameView.from_obs(obs)

    assert view.player == 0
    assert len(view.planets) >= 8  # at least 2 starting groups of 4
    assert all(p.id == i for i, p in enumerate(view.planets))
    assert any(p.owner == 0 for p in view.planets), "player 0 should own a home planet"
    assert any(p.owner == 1 for p in view.planets), "player 1 should own a home planet"
    assert isinstance(view.angular_velocity, float)


def test_gameview_my_planets_and_targets():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]

    view = GameView.from_obs(obs)

    assert len(view.my_planets()) >= 1
    assert all(p.owner == view.player for p in view.my_planets())
    assert all(p.owner != view.player for p in view.targets())


def test_gameview_distance_uses_euclidean():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]

    view = GameView.from_obs(obs)
    a, b = view.planets[0], view.planets[1]

    expected = math.hypot(a.x - b.x, a.y - b.y)
    assert view.distance(a, b) == expected


def test_gameview_handles_none_valued_obs_fields():
    """from_obs should not crash if obs has None for list-valued fields."""
    obs = {
        "planets": None,
        "fleets": None,
        "initial_planets": None,
        "comet_planet_ids": None,
        "player": 0,
        "angular_velocity": 0.05,
        "remainingOverageTime": None,
    }
    view = GameView.from_obs(obs)
    assert view.planets == ()
    assert view.fleets == ()
    assert view.initial_planets == ()
    assert view.comet_planet_ids == frozenset()
    assert view.remaining_overage_time == 0.0


def test_gameview_exposes_step_field():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]

    view = GameView.from_obs(obs)
    assert view.step == obs.get("step", 0)


def test_gameview_exposes_comets_metadata():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]

    view = GameView.from_obs(obs)
    assert isinstance(view.comets, tuple)
    for group in view.comets:
        assert "planet_ids" in group
