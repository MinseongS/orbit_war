"""Step composer: rank candidate steps and greedily combine under
per-source surplus constraints. The Melis "steps + greedy combine"
pattern, ported to Orbit Wars."""

from __future__ import annotations

from typing import Callable, Iterable

from orbit_war.plan_gen.step import Step


def compose_plan(
    steps: Iterable[Step],
    surplus_by_planet: dict[int, int],
    allow_truncation: bool = False,
    validator: Callable[[list[Step]], list[Step]] | None = None,
) -> list[Step]:
    """Sort `steps` by descending score and greedily commit, debiting
    `surplus_by_planet[step.from_planet_id]` each time.

    If `allow_truncation` is True, a step that exceeds remaining surplus
    is shrunk to the surplus (provided >=1 ships remain). Otherwise it
    is skipped.

    If `validator` is provided, the assembled plan is passed through it
    after greedy combine; the validator may return the same plan or any
    other list of Steps (typically `[]` to revert to no-op).

    Returns the committed steps in the order they were chosen (or the
    validator's return value, if supplied).
    """
    remaining = dict(surplus_by_planet)
    plan: list[Step] = []
    for step in sorted(steps, key=lambda s: -s.score):
        avail = remaining.get(step.from_planet_id, 0)
        if avail <= 0:
            continue
        if step.ships <= avail:
            plan.append(step)
            remaining[step.from_planet_id] = avail - step.ships
            continue
        if not allow_truncation:
            continue
        truncated_ships = avail
        if truncated_ships < 1:
            continue
        plan.append(
            Step(
                from_planet_id=step.from_planet_id,
                target_planet_id=step.target_planet_id,
                angle=step.angle,
                ships=int(truncated_ships),
                score=step.score,
            )
        )
        remaining[step.from_planet_id] = 0
    if validator is not None:
        plan = validator(plan)
    return plan
