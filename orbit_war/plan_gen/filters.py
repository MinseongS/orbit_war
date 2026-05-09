"""Composer-side filters that run between template emission and the composer.

Templates emit candidate steps freely; filters drop steps that are obviously
counter-productive (e.g. attacks that send fewer ships than the defender,
which would waste ships in transit without capturing anything).

Note on aggregation history: a per-target aggregation variant was attempted
in W4.1 to let multi-source consolidation steps survive on the combined-ship
total. Re-measurement (W4.8) showed the aggregation regressed *every* bot
vs public_tactical (v3 dropped from 64% to 32% with the aggregated filter,
recovers to 64% with this per-step filter). The aggregation was reverted.
Consolidation steps now get dropped silently, same as in W3. W5 will
redesign multi_source_consolidation_template to only emit when no single
source can solo-capture the target — making the aggregation unnecessary."""

from __future__ import annotations

from orbit_war.plan_gen.step import Step
from orbit_war.sim.observation import GameView


def filter_capturable(steps: list[Step], view: GameView) -> list[Step]:
    """Remove attack steps that cannot capture their target with the ships sent.

    Friendly reinforcements and unknown-target steps pass through
    unconditionally. This is the W3 per-step semantics; multi-source
    consolidation partial steps are silently dropped (intended; W5 will
    redesign the consolidation template to avoid emitting them at all
    when single-source attacks suffice)."""
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
