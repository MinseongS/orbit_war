"""Tests for orbit_war.eval.features."""

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from orbit_war.eval.features import (
    arrival_turns_to,
    effective_garrison,
    incoming_threat,
    surplus_ships,
    total_production,
    total_ships,
)
from orbit_war.sim.observation import GameView


def _fresh_view(seed: int = 42) -> GameView:
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.reset(num_agents=2)
    return GameView.from_obs(env.steps[0][0]["observation"])


def test_total_ships_sums_owned_planets_and_fleets():
    view = _fresh_view()
    expected = sum(p.ships for p in view.planets if p.owner == view.player)
    expected += sum(f.ships for f in view.fleets if f.owner == view.player)
    assert total_ships(view, view.player) == expected


def test_total_production_sums_owned_planets():
    view = _fresh_view()
    expected = sum(p.production for p in view.planets if p.owner == view.player)
    assert total_production(view, view.player) == expected


def test_surplus_ships_returns_zero_for_uncontested_planet():
    view = _fresh_view()
    surplus = surplus_ships(view, view.player)
    for p in view.my_planets():
        assert surplus[p.id] == p.ships


def test_surplus_ships_zero_when_inbound_enemy_overwhelms():
    """Construct a tiny synthetic view: my planet has 5 ships, enemy fleet
    of 100 is en route. Surplus should be 0 (we'll lose the planet)."""
    my = Planet(0, 0, 10.0, 10.0, 1.0, 5, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 1, 1)
    incoming = Fleet(0, 1, 12.0, 12.0, 0.0, 1, 100)
    view = GameView(
        player=0,
        planets=(my, enemy),
        fleets=(incoming,),
        angular_velocity=0.04,
        initial_planets=(my, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    surplus = surplus_ships(view, 0)
    assert surplus[0] == 0


def test_incoming_threat_counts_enemy_ships_in_horizon():
    my = Planet(0, 0, 50.0, 50.0, 1.0, 30, 2)
    enemy = Planet(1, 1, 90.0, 50.0, 1.0, 5, 1)
    fleet = Fleet(0, 1, 51.0, 50.0, 0.0, 1, 25)  # right next to us
    view = GameView(
        player=0,
        planets=(my, enemy),
        fleets=(fleet,),
        angular_velocity=0.04,
        initial_planets=(my, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    assert incoming_threat(view, player=0, planet_id=0, horizon=50) == 25


def test_arrival_turns_to_uses_physics():
    src = Planet(0, 0, 0.0, 0.0, 1.0, 100, 1)
    tgt = Planet(1, -1, 30.0, 0.0, 1.0, 5, 1)
    view = GameView(
        player=0,
        planets=(src, tgt),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(src, tgt),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )
    turns = arrival_turns_to(view, src_planet=src, target_planet=tgt, ships=100)
    assert turns >= 1


def test_effective_garrison_grows_with_production_for_owned_planet():
    my = Planet(0, 0, 50.0, 50.0, 1.0, 10, 3)
    view = GameView(
        player=0,
        planets=(my,),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(my,),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )
    g0 = effective_garrison(view, planet_id=0, at_turn=0)
    g5 = effective_garrison(view, planet_id=0, at_turn=5)
    assert g5 == g0 + 3 * 5


def test_effective_garrison_static_for_neutral():
    n = Planet(0, -1, 50.0, 50.0, 1.0, 10, 3)
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
    assert effective_garrison(view, planet_id=0, at_turn=0) == 10
    assert effective_garrison(view, planet_id=0, at_turn=10) == 10
