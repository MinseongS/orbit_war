"""Pure feature extractors over GameView.

Every function is a pure read-only function; no caching, no side effects.
The composer is allowed to call these many times per turn, but each is
shaped to be cheap (linear in #planets + #fleets at worst).
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from orbit_war.sim.observation import GameView
from orbit_war.sim.physics import (
    straight_line_distance,
    turns_to_arrive,
)


def total_ships(view: GameView, player: int) -> int:
    """Owned planet garrisons + own in-flight fleet ships."""
    n = sum(p.ships for p in view.planets if p.owner == player)
    n += sum(f.ships for f in view.fleets if f.owner == player)
    return n


def total_production(view: GameView, player: int) -> int:
    """Sum of `production` across planets we own."""
    return sum(p.production for p in view.planets if p.owner == player)


def incoming_threat(
    view: GameView, player: int, planet_id: int, horizon: int = 20
) -> int:
    """Ships in enemy fleets aimed at `planet_id`, arriving within `horizon` turns.

    Approximation: any enemy fleet whose straight-line ETA to the planet's
    current position is <= horizon counts. We don't try to determine the
    fleet's *intended* destination — Orbit Wars fleets fly along a fixed
    angle, and judging intent perfectly requires geometric ray-cast tests
    that belong with W3's deeper combat sim. Over-counting here is fine
    for a defensive heuristic.
    """
    target = next((p for p in view.planets if p.id == planet_id), None)
    if target is None:
        return 0
    total = 0
    for f in view.fleets:
        if f.owner == player or f.owner == -1:
            continue
        eta = turns_to_arrive(f.x, f.y, target.x, target.y, max(1, f.ships))
        if eta <= horizon:
            total += f.ships
    return total


def arrival_turns_to(
    view: GameView, src_planet: Planet, target_planet: Planet, ships: int
) -> int:
    """Straight-line arrival turns for a fleet leaving `src_planet`."""
    return turns_to_arrive(
        src_planet.x,
        src_planet.y,
        target_planet.x,
        target_planet.y,
        ships,
    )


def effective_garrison(view: GameView, planet_id: int, at_turn: int) -> int:
    """Approximate ships present on `planet_id` `at_turn` turns from now.

    Owned/enemy planets accrue production; neutrals do not.
    Does NOT account for inbound fleet arrivals (caller composes those).
    """
    p = next((q for q in view.planets if q.id == planet_id), None)
    if p is None:
        return 0
    if p.owner == -1:
        return p.ships
    return p.ships + p.production * max(0, at_turn)


def surplus_ships(view: GameView, player: int) -> dict[int, int]:
    """Per-owned-planet ships that can be spent without immediately losing it.

    Conservative rule: for each owned planet, look at the worst-case enemy
    arrival within a 30-turn horizon. If the largest single-source enemy
    fleet exceeds (current garrison + production * 5), the planet is
    threatened — surplus is 0. Otherwise surplus = current garrison.

    This is intentionally simple. Real defensive timing belongs in W3 with
    timeline-aware combat resolution. The W2 composer just needs a number
    that prevents it from emptying a planet that's about to die.
    """
    surplus: dict[int, int] = {}
    for p in view.planets:
        if p.owner != player:
            continue
        worst_arrival = 0
        for f in view.fleets:
            if f.owner == player or f.owner == -1:
                continue
            eta = turns_to_arrive(f.x, f.y, p.x, p.y, max(1, f.ships))
            if eta <= 30:
                worst_arrival = max(worst_arrival, f.ships)
        defenders = p.ships + p.production * 5  # 5-turn anticipated production
        if worst_arrival > defenders:
            surplus[p.id] = 0
        else:
            reserve = max(0, worst_arrival - p.production * 5)
            surplus[p.id] = max(0, p.ships - reserve)
    return surplus
