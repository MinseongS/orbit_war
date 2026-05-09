"""Tests for the lightweight forward simulator."""

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from orbit_war.sim.game import forward_simulate
from orbit_war.sim.observation import GameView


def test_forward_simulate_returns_gameview_with_advanced_step():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    view = GameView.from_obs(env.steps[0][0]["observation"])

    out = forward_simulate(view, actions_per_player=[[], []], n_turns=10)

    assert isinstance(out, GameView)
    assert out.step == view.step + 10


def test_forward_simulate_advances_owned_planet_garrisons_by_production():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 5, 3)  # production 3
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 5, 1)
    view = GameView(
        player=0,
        planets=(me, enemy),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )
    out = forward_simulate(view, actions_per_player=[[], []], n_turns=5)
    me_after = next(p for p in out.planets if p.id == 0)
    assert me_after.ships == 5 + 3 * 5  # 5 turns at production 3


def test_forward_simulate_neutral_planet_garrisons_do_not_grow():
    n = Planet(0, -1, 10.0, 10.0, 1.0, 5, 3)
    view = GameView(
        player=0,
        planets=(n,),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(n,),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )
    out = forward_simulate(view, actions_per_player=[[], []], n_turns=10)
    assert out.planets[0].ships == 5


def test_forward_simulate_processes_inbound_fleet_capture():
    """An inbound fleet of 100 ships landing on a 5-ship neutral should capture it."""
    me = Planet(0, 0, 10.0, 10.0, 1.0, 100, 1)
    target = Planet(1, -1, 15.0, 10.0, 1.0, 5, 1)  # moved further out so fleet clearly outside me's radius
    incoming = Fleet(0, 0, 12.0, 10.0, 0.0, 0, 100)  # starts outside me's radius (radius=1, me at x=10)
    view = GameView(
        player=0,
        planets=(me, target),
        fleets=(incoming,),
        angular_velocity=0.04,
        initial_planets=(me, target),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )
    out = forward_simulate(view, actions_per_player=[[], []], n_turns=5)
    target_after = next(p for p in out.planets if p.id == 1)
    assert target_after.owner == 0  # captured
