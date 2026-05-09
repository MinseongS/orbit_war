"""Composer-side filters that run between template emission and the composer.

Templates emit candidate steps freely; filters drop steps that cannot
plausibly capture their target.

Filter rules:
  1. Friendly reinforcements and unknown-target steps pass through.
  2. A single-source attack with `ships >= defender + 1` always passes
     (this is the W3 behavior — preserved to avoid the W4-era regression
     where valid solo attacks got dragged into failing consolidation
     groups and dropped).
  3. Otherwise, multi-source consolidation steps survive iff their
     combined ships exceed the defender. Sub-threshold partial steps
     are dropped together."""

from __future__ import annotations

from collections import defaultdict

from orbit_war.plan_gen.step import Step
from orbit_war.sim.observation import GameView


def filter_capturable(steps: list[Step], view: GameView) -> list[Step]:
    """Per-target filter with single-source short-circuit.

    Each step targeting an enemy planet is evaluated as follows:
      - If the step's ships alone meet `defender + 1`, it passes.
      - Otherwise, partial steps are grouped by target. The group survives
        together iff the combined ships exceed the defender; otherwise
        all sub-threshold steps in the group are dropped.

    Friendly reinforcements and unknown-target steps pass through
    unconditionally."""
    planet_by_id = {p.id: p for p in view.planets}
    player = view.player

    grouped: dict[int, list[Step]] = defaultdict(list)
    result: list[Step] = []
    for s in steps:
        target = planet_by_id.get(s.target_planet_id)
        if target is None or target.owner == player:
            result.append(s)
            continue
        needed = int(target.ships) + 1
        if s.ships >= needed:
            # Solo attack is independently sufficient — short-circuit.
            result.append(s)
            continue
        # Partial contribution: defer to per-target aggregate decision.
        grouped[s.target_planet_id].append(s)

    for target_id, group in grouped.items():
        target = planet_by_id[target_id]
        needed = int(target.ships) + 1
        combined = sum(s.ships for s in group)
        if combined >= needed:
            result.extend(group)
        # else: the partial group cannot capture even combined; drop all
    return result
