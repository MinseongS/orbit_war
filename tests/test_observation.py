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
