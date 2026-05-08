"""Tests for closed-form orbit position and the precompute table."""

import math

import pytest
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    ROTATION_RADIUS_LIMIT,
    Planet,
)

from orbit_war.sim.observation import GameView
from orbit_war.sim.orbits import (
    is_orbiting,
    planet_position_at,
    precompute_position_table,
)


def _run_game_seed(seed: int):
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run(["random", "random"])
    return env


def test_static_planet_position_unchanged_over_time():
    env = _run_game_seed(seed=7)
    obs0 = env.steps[0][0]["observation"]
    view = GameView.from_obs(obs0)
    angular_velocity = view.angular_velocity

    static = next(p for p in view.initial_planets if not is_orbiting(p))

    for turn in (0, 1, 50, 250, 499):
        x, y = planet_position_at(static, turn, angular_velocity)
        assert math.isclose(x, static.x)
        assert math.isclose(y, static.y)


def test_orbiting_planet_matches_official_sim_at_each_recorded_step():
    env = _run_game_seed(seed=7)
    obs0 = env.steps[0][0]["observation"]
    view0 = GameView.from_obs(obs0)
    angular_velocity = view0.angular_velocity
    initial_by_id = {p.id: p for p in view0.initial_planets}

    orbiting_ids = [p.id for p in view0.initial_planets if is_orbiting(p)]
    assert orbiting_ids, "seed 7 should produce at least one orbiting planet"

    # Sample a handful of turns from the recorded episode (skip turn 0 — trivial).
    sample_turns = [25, 100, 250, 400]
    for turn in sample_turns:
        recorded_obs = env.steps[turn][0]["observation"]
        recorded = {p[0]: Planet(*p) for p in recorded_obs["planets"]}
        for pid in orbiting_ids:
            initial = initial_by_id[pid]
            expected_x = recorded[pid].x
            expected_y = recorded[pid].y
            actual_x, actual_y = planet_position_at(initial, turn, angular_velocity)
            assert math.isclose(actual_x, expected_x, abs_tol=1e-9), (
                f"x drift on planet {pid} at turn {turn}"
            )
            assert math.isclose(actual_y, expected_y, abs_tol=1e-9), (
                f"y drift on planet {pid} at turn {turn}"
            )


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 2026])
def test_precompute_table_matches_per_step_function(seed: int):
    env = _run_game_seed(seed=seed)
    obs0 = env.steps[0][0]["observation"]
    view = GameView.from_obs(obs0)

    table = precompute_position_table(
        initial_planets=view.initial_planets,
        angular_velocity=view.angular_velocity,
        max_turn=500,
    )

    sample_turns = [0, 1, 17, 123, 300, 499]
    for p in view.initial_planets:
        for t in sample_turns:
            tx, ty = table[p.id][t]
            ex, ey = planet_position_at(p, t, view.angular_velocity)
            assert math.isclose(tx, ex)
            assert math.isclose(ty, ey)


def test_is_orbiting_uses_official_threshold():
    static_like = Planet(0, -1, 5, 5, 1.0, 50, 1)
    orbiting_like = Planet(0, -1, CENTER + 10, CENTER, 1.0, 50, 1)
    assert is_orbiting(orbiting_like)
    far = Planet(0, -1, CENTER + ROTATION_RADIUS_LIMIT, CENTER, 1.0, 50, 1)
    assert not is_orbiting(far)
