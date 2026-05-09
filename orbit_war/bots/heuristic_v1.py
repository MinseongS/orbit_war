"""heuristic_v1: composes four step templates into a single agent.

This is our first real bot. Templates emit candidate steps; the composer
sorts by score and greedily commits under per-planet surplus. Weights here
are hand-set initial guesses; heuristic_v2 will tune them via linear
regression on self-play data."""

from __future__ import annotations

from orbit_war.eval.features import surplus_ships
from orbit_war.plan_gen.composer import compose_plan
from orbit_war.plan_gen.step import Step
from orbit_war.plan_gen.templates import (
    defensive_reinforce_template,
    no_op_template,
    production_attack_template,
    snipe_undefended_template,
)
from orbit_war.sim.observation import GameView

# Hand-set initial weights; v2 will replace these with regression-fit values.
TEMPLATE_WEIGHTS: dict[str, float] = {
    "no_op": 0.0,
    "production_attack": 1.0,
    "defensive_reinforce": 2.0,  # defending is high-priority by default
    "snipe_undefended": 1.5,     # sniping is high-EV
}


def _weighted(steps: list[Step], weight: float) -> list[Step]:
    """Apply `weight` to every step's score. Cheap immutable rewrite."""
    if weight == 1.0:
        return steps
    return [
        Step(
            from_planet_id=s.from_planet_id,
            target_planet_id=s.target_planet_id,
            angle=s.angle,
            ships=s.ships,
            score=s.score * weight,
        )
        for s in steps
    ]


def _filter_capturable(steps: list[Step], view: GameView) -> list[Step]:
    """Remove attack steps that cannot capture their target with the ships sent."""
    planet_by_id = {p.id: p for p in view.planets}
    player = view.player
    result = []
    for s in steps:
        target = planet_by_id.get(s.target_planet_id)
        if target is None:
            result.append(s)
            continue
        # Friendly reinforcement — always keep.
        if target.owner == player:
            result.append(s)
            continue
        # Attack/capture — only keep if we have enough ships.
        needed = int(target.ships) + 1
        if s.ships >= needed:
            result.append(s)
    return result


def agent(obs) -> list[list]:
    view = GameView.from_obs(obs)

    candidates: list[Step] = []
    candidates.extend(_weighted(no_op_template(view), TEMPLATE_WEIGHTS["no_op"]))
    candidates.extend(
        _weighted(
            production_attack_template(view),
            TEMPLATE_WEIGHTS["production_attack"],
        )
    )
    candidates.extend(
        _weighted(
            defensive_reinforce_template(view),
            TEMPLATE_WEIGHTS["defensive_reinforce"],
        )
    )
    candidates.extend(
        _weighted(
            snipe_undefended_template(view),
            TEMPLATE_WEIGHTS["snipe_undefended"],
        )
    )

    # Prune attack candidates that cannot capture their target — sending fewer
    # ships than the defender wastes ships without gaining anything.
    candidates = _filter_capturable(candidates, view)

    surplus = surplus_ships(view, view.player)
    plan = compose_plan(candidates, surplus, allow_truncation=False)
    return [s.as_move() for s in plan]
