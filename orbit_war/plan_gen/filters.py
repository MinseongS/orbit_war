"""Composer-side filters that run between template emission and the composer.

Templates emit candidate steps freely; filters drop steps that are obviously
counter-productive (e.g. attacks that send fewer ships than the defender,
which would waste ships in transit without capturing anything)."""

from __future__ import annotations

from orbit_war.plan_gen.step import Step
from orbit_war.sim.observation import GameView


def filter_capturable(steps: list[Step], view: GameView) -> list[Step]:
    """Remove attack steps that cannot capture their target with the ships sent.

    Friendly reinforcements are passed through unconditionally."""
    planet_by_id = {p.id: p for p in view.planets}
    player = view.player
    result: list[Step] = []
    for s in steps:
        target = planet_by_id.get(s.target_planet_id)
        if target is None:
            result.append(s)
            continue
        if target.owner == player:
            result.append(s)
            continue
        needed = int(target.ships) + 1
        if s.ships >= needed:
            result.append(s)
    return result
