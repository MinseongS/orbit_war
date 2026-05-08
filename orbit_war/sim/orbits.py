"""Closed-form planet position prediction.

Orbit Wars planets either:
  - rotate around the central sun at a fixed angular velocity, when
    ``orbital_radius + planet_radius < ROTATION_RADIUS_LIMIT``; or
  - are static.

Both cases are expressible in closed form from the planet's initial
position and the current turn. We never iterate to predict positions —
we look them up.

Phase convention
----------------
``env.steps[N]`` contains the observation presented to agents at the
start of turn N.  The official simulator advances orbiting planets by
``angular_velocity * step`` where ``step`` is the *current* step counter
that starts at 1 and increments before the observation is broadcast.
This means:

* ``env.steps[0]`` — initial setup, no rotation applied → angle factor 0
* ``env.steps[1]`` — after step 1 processed, position = ``av * 0`` (step
  is 1 but the observation already stored shows the *pre*-step position)

Empirically confirmed: the planet position visible in ``env.steps[N]``
matches ``initial_angle + angular_velocity * (N - 1)`` for N >= 1, and
equals the initial position at N = 0.

Therefore ``planet_position_at(initial, turn, av)`` returns the position
that would appear in ``env.steps[turn]``, using:
  - angle = initial_angle + av * max(0, turn - 1)
"""

from __future__ import annotations

import math
from typing import Iterable

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    ROTATION_RADIUS_LIMIT,
    Planet,
)


def is_orbiting(planet: Planet) -> bool:
    """True iff the planet rotates around the sun."""
    dx = planet.x - CENTER
    dy = planet.y - CENTER
    orbital_radius = math.hypot(dx, dy)
    return orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT


def planet_position_at(
    initial: Planet, turn: int, angular_velocity: float
) -> tuple[float, float]:
    """Return (x, y) for a planet at the given turn, given its initial state.

    ``initial`` must be the planet's snapshot at turn 0 (i.e. an entry of
    ``obs.initial_planets``).  For static planets the initial position is
    returned unchanged.

    The returned coordinates match ``env.steps[turn][0]["observation"]``
    planet positions — i.e. the position an agent sees at the start of
    ``turn``.  At turn 0 this is the initial position; at turn N >= 1 the
    planet has been advanced by ``angular_velocity * (N - 1)`` radians.
    """
    if not is_orbiting(initial):
        return initial.x, initial.y

    dx = initial.x - CENTER
    dy = initial.y - CENTER
    orbital_radius = math.hypot(dx, dy)
    initial_angle = math.atan2(dy, dx)
    # The simulator applies rotation with angle = initial_angle + av * step,
    # where 'step' is the step counter used *inside* the tick that produces
    # the observation stored at env.steps[turn].  Empirically env.steps[N]
    # holds position av*(N-1) for N>=1 and the initial position at N=0.
    steps_elapsed = max(0, turn - 1)
    angle = initial_angle + angular_velocity * steps_elapsed
    return (
        CENTER + orbital_radius * math.cos(angle),
        CENTER + orbital_radius * math.sin(angle),
    )


def precompute_position_table(
    initial_planets: Iterable[Planet],
    angular_velocity: float,
    max_turn: int,
) -> dict[int, list[tuple[float, float]]]:
    """Return ``{planet_id: [(x, y) for turn in 0..max_turn]}`` for fast lookup.

    Built once at the start of a game; reused for every search call.
    The position at index ``t`` matches ``planet_position_at(p, t, av)``
    and is consistent with what agents see in ``env.steps[t]``.
    """
    table: dict[int, list[tuple[float, float]]] = {}
    for p in initial_planets:
        if not is_orbiting(p):
            table[p.id] = [(p.x, p.y)] * (max_turn + 1)
            continue
        dx = p.x - CENTER
        dy = p.y - CENTER
        orbital_radius = math.hypot(dx, dy)
        initial_angle = math.atan2(dy, dx)
        positions: list[tuple[float, float]] = []
        for t in range(max_turn + 1):
            steps_elapsed = max(0, t - 1)
            angle = initial_angle + angular_velocity * steps_elapsed
            positions.append(
                (
                    CENTER + orbital_radius * math.cos(angle),
                    CENTER + orbital_radius * math.sin(angle),
                )
            )
        table[p.id] = positions
    return table
