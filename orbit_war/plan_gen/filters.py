"""Composer-side filters that run between template emission and the composer.

Templates emit candidate steps freely; filters drop steps that cannot
plausibly capture their target.

The filter aggregates per target: multi-source consolidation steps survive
iff the combined incoming ships exceed the defender. Friendly reinforcement
and unknown-target steps pass through unconditionally."""

from __future__ import annotations

from collections import defaultdict

from orbit_war.plan_gen.step import Step
from orbit_war.sim.observation import GameView


def filter_capturable(steps: list[Step], view: GameView) -> list[Step]:
    """Aggregate per target. Drop all attack steps targeting a planet whose
    combined incoming ships are insufficient to capture (defender + 1).

    Friendly reinforcements and steps with unknown target IDs pass through.
    Within an attacking-target group, all steps survive together or are all
    dropped together."""
    planet_by_id = {p.id: p for p in view.planets}
    player = view.player

    grouped: dict[int, list[Step]] = defaultdict(list)
    pass_through: list[Step] = []
    for s in steps:
        target = planet_by_id.get(s.target_planet_id)
        if target is None or target.owner == player:
            pass_through.append(s)
            continue
        grouped[s.target_planet_id].append(s)

    result: list[Step] = list(pass_through)
    for target_id, group in grouped.items():
        target = planet_by_id[target_id]
        combined = sum(s.ships for s in group)
        needed = int(target.ships) + 1
        if combined >= needed:
            result.extend(group)
        # else: all contributions to this target are dropped
    return result
