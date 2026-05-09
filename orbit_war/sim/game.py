"""Lightweight forward simulator for plan validation.

This is a *cheap, approximate* re-implementation of the official Orbit Wars
step function — enough to forward-simulate ~15 turns of a candidate plan and
score the resulting position. It is NOT a parity-exact port of the official
simulator. Specifically:

  - Comet spawn/expire is skipped (comets present at sim start persist).
  - Combat is simplified pairwise (largest army wins, surplus over defender
    flips ownership).
  - Sun avoidance: any fleet whose path segment crosses within 12 units of
    (50, 50) is destroyed. (Sun radius is 10; 12 gives a small buffer.)
  - Planet positions use the closed-form `planet_position_at` already in
    `sim.orbits`.

Use this for plan ranking and lookahead-based pruning. Do NOT use it as the
authoritative game engine — the real environment makes the final call."""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    BOARD_SIZE,
    CENTER,
    Fleet,
    Planet,
)

from orbit_war.sim.observation import GameView
from orbit_war.sim.orbits import is_orbiting, planet_position_at
from orbit_war.sim.physics import fleet_speed

SUN_BUFFER = 12.0  # actual sun radius is 10; 2-unit safety margin


def forward_simulate(
    view: GameView,
    actions_per_player: list[list[list]],
    n_turns: int,
) -> GameView:
    """Advance the game `n_turns` turns from `view`.

    `actions_per_player[i]` is the action list for player i (list of
    `[from_planet_id, angle, ships]`). Actions are processed once at the
    start, spawning fleets; subsequent turns advance physics + production
    + collisions only."""
    planets = list(view.planets)
    fleets = list(view.fleets)

    # Apply turn-0 actions first.
    fleets.extend(_spawn_fleets(planets, actions_per_player))

    for tick in range(n_turns):
        current_step = view.step + tick + 1

        # Advance fleets one tick.
        moved: list[Fleet] = []
        for f in fleets:
            speed = fleet_speed(f.ships)
            new_x = f.x + math.cos(f.angle) * speed
            new_y = f.y + math.sin(f.angle) * speed
            if _crosses_sun(f.x, f.y, new_x, new_y):
                continue  # destroyed
            if new_x < 0 or new_x > BOARD_SIZE or new_y < 0 or new_y > BOARD_SIZE:
                continue  # off-board
            moved.append(Fleet(f.id, f.owner, new_x, new_y, f.angle, f.from_planet_id, f.ships))
        fleets = moved

        # Snap planet positions for collision/production at this tick.
        positions = _snap_positions(view, current_step)
        for i, p in enumerate(planets):
            x, y = positions[p.id]
            planets[i] = Planet(p.id, p.owner, x, y, p.radius, p.ships, p.production)

        # Production for owned/enemy planets (not neutrals).
        for i, p in enumerate(planets):
            if p.owner == -1:
                continue
            planets[i] = Planet(p.id, p.owner, p.x, p.y, p.radius, p.ships + p.production, p.production)

        # Resolve fleet→planet collisions.
        survivors: list[Fleet] = []
        arrivals: dict[int, list[tuple[int, int]]] = {}
        for f in fleets:
            collided_planet = _planet_collision(f, planets)
            if collided_planet is None:
                survivors.append(f)
                continue
            arrivals.setdefault(collided_planet, []).append((f.owner, f.ships))
        fleets = survivors

        for planet_id, arrival_list in arrivals.items():
            i = next(idx for idx, p in enumerate(planets) if p.id == planet_id)
            planets[i] = _resolve_combat(planets[i], arrival_list)

    return GameView(
        player=view.player,
        planets=tuple(planets),
        fleets=tuple(fleets),
        angular_velocity=view.angular_velocity,
        initial_planets=view.initial_planets,
        comet_planet_ids=view.comet_planet_ids,
        remaining_overage_time=view.remaining_overage_time,
        step=view.step + n_turns,
        comets=view.comets,
    )


def _crosses_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - CENTER, y1 - CENTER
    a = dx * dx + dy * dy
    if a < 1e-9:
        return math.hypot(fx, fy) < SUN_BUFFER
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - SUN_BUFFER * SUN_BUFFER
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def _snap_positions(view: GameView, step: int) -> dict[int, tuple[float, float]]:
    """Compute current (x, y) for every planet at `step`."""
    by_id_initial = {p.id: p for p in view.initial_planets}
    out: dict[int, tuple[float, float]] = {}
    for p in view.planets:
        initial = by_id_initial.get(p.id)
        if initial is None or not is_orbiting(initial):
            out[p.id] = (p.x, p.y)
        else:
            out[p.id] = planet_position_at(initial, step, view.angular_velocity)
    return out


def _planet_collision(fleet: Fleet, planets: list[Planet]) -> int | None:
    for p in planets:
        if math.hypot(fleet.x - p.x, fleet.y - p.y) <= p.radius:
            return p.id
    return None


def _resolve_combat(planet: Planet, arrivals: list[tuple[int, int]]) -> Planet:
    """Simplified combat. Sum ships per owner among arrivals; the planet's
    garrison defends as the planet's own owner (or as neutral)."""
    by_owner: dict[int, int] = {}
    for owner, ships in arrivals:
        by_owner[owner] = by_owner.get(owner, 0) + ships
    # The planet itself contributes its garrison under its current owner.
    by_owner[planet.owner] = by_owner.get(planet.owner, 0) + planet.ships
    # Largest force wins; surplus = largest - 2nd largest.
    sorted_forces = sorted(by_owner.items(), key=lambda kv: -kv[1])
    if len(sorted_forces) == 1:
        winner, ships = sorted_forces[0]
        return Planet(planet.id, winner, planet.x, planet.y, planet.radius, ships, planet.production)
    (winner, top), (_runner, runner_up) = sorted_forces[0], sorted_forces[1]
    surplus = top - runner_up
    return Planet(planet.id, winner, planet.x, planet.y, planet.radius, surplus, planet.production)


def _spawn_fleets(planets: list[Planet], actions_per_player: list[list[list]]) -> list[Fleet]:
    """Spawn fleets from each player's first-turn action list. Fleet IDs are
    synthesized starting at 10_000 to avoid colliding with existing fleet IDs."""
    by_id = {p.id: p for p in planets}
    out: list[Fleet] = []
    next_fleet_id = 10_000
    for player, action_list in enumerate(actions_per_player):
        for move in action_list:
            from_id, angle, ships = int(move[0]), float(move[1]), int(move[2])
            src = by_id.get(from_id)
            if src is None or src.owner != player or ships <= 0 or ships > src.ships:
                continue
            spawn_x = src.x + math.cos(angle) * (src.radius + 0.5)
            spawn_y = src.y + math.sin(angle) * (src.radius + 0.5)
            out.append(Fleet(next_fleet_id, player, spawn_x, spawn_y, angle, from_id, ships))
            next_fleet_id += 1
    return out
