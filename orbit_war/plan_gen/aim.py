"""Orbit-aware fleet aiming.

Targets that orbit the central sun move while a fleet is in flight. Naively
aiming at the target's current position causes the fleet to miss. We predict
the target's position at fleet arrival via the closed-form
`sim.orbits.planet_position_at`, then re-derive the launch angle. This is
a fixed-point iteration: arrival turn depends on distance, distance depends
on predicted position. Three iterations converge for any realistic case."""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_war.sim.observation import GameView
from orbit_war.sim.orbits import is_orbiting, planet_position_at
from orbit_war.sim.physics import turns_to_arrive

MAX_AIM_ITERATIONS = 4
AIM_CONVERGENCE_DELTA = 0.1  # board units


def aim_with_orbit_prediction(
    src: Planet,
    target: Planet,
    ships: int,
    view: GameView,
) -> tuple[float, int]:
    """Return (angle, arrival_turn) for a fleet leaving `src` toward `target`.

    For static targets, this is straight atan2. For orbiting targets, predict
    where the target will be at fleet arrival and aim there.

    Returns the angle in radians and the integer arrival turn count.
    """
    initial_target = _initial_planet(view, target.id)
    if initial_target is None or not is_orbiting(initial_target):
        # Static target: no prediction needed.
        angle = math.atan2(target.y - src.y, target.x - src.x)
        arrival = turns_to_arrive(src.x, src.y, target.x, target.y, max(1, ships))
        return angle, arrival

    # Orbiting target: iterate angle ↔ arrival_turn fixed point.
    tx, ty = target.x, target.y
    arrival = 1
    for _ in range(MAX_AIM_ITERATIONS):
        new_arrival = turns_to_arrive(src.x, src.y, tx, ty, max(1, ships))
        future_step = view.step + new_arrival
        nx, ny = planet_position_at(initial_target, future_step, view.angular_velocity)
        if abs(nx - tx) < AIM_CONVERGENCE_DELTA and abs(ny - ty) < AIM_CONVERGENCE_DELTA:
            tx, ty, arrival = nx, ny, new_arrival
            break
        tx, ty, arrival = nx, ny, new_arrival
    angle = math.atan2(ty - src.y, tx - src.x)
    return angle, arrival


def _initial_planet(view: GameView, planet_id: int) -> Planet | None:
    """Look up the planet's turn-0 snapshot, used for orbit prediction."""
    return next((p for p in view.initial_planets if p.id == planet_id), None)
