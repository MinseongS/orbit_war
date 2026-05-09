"""Tests for orbit_war.plan_gen.opponent."""

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_war.plan_gen.opponent import predict_opponent_plan
from orbit_war.sim.observation import GameView


def test_predict_opponent_plan_returns_action_list():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 50, 1)
    neutral = Planet(2, -1, 50.0, 50.0, 1.0, 5, 2)
    view = GameView(
        player=0,
        planets=(me, enemy, neutral),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, enemy, neutral),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    actions = predict_opponent_plan(view, opponent=1)
    assert isinstance(actions, list)
    for move in actions:
        assert len(move) == 3
        from_id, angle, ships = move
        assert isinstance(from_id, int)
        assert isinstance(angle, float)
        assert isinstance(ships, int)
        # The opponent should only launch from their own planets.
        assert from_id == 1


def test_predict_opponent_plan_quiet_when_opponent_has_no_planets():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    view = GameView(
        player=0,
        planets=(me,),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me,),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    assert predict_opponent_plan(view, opponent=1) == []
