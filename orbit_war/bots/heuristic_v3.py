"""heuristic_v3: W3 bot with orbit-aware aiming, two new templates,
and forward-sim plan validation.

Differences from heuristic_v1:
- Templates aim at predicted future positions (via aim_with_orbit_prediction).
- Adds multi_source_consolidation_template and comet_rush_template.
- compose_plan is called with a validator that forward-simulates 15 turns
  and reverts to no-op if the projected position evaluates worse than the
  current position."""

from __future__ import annotations

from orbit_war.eval.features import (
    surplus_ships,
    total_production,
    total_ships,
)
from orbit_war.plan_gen.composer import compose_plan
from orbit_war.plan_gen.filters import filter_capturable
from orbit_war.plan_gen.step import Step
from orbit_war.plan_gen.templates import (
    comet_rush_template,
    defensive_reinforce_template,
    multi_source_consolidation_template,
    no_op_template,
    production_attack_template,
    snipe_undefended_template,
)
from orbit_war.sim.game import forward_simulate
from orbit_war.sim.observation import GameView

TEMPLATE_WEIGHTS: dict[str, float] = {
    "no_op": 0.0,
    "production_attack": 1.0,
    "defensive_reinforce": 2.0,
    "snipe_undefended": 1.5,
    "multi_source_consolidation": 1.2,
    "comet_rush": 0.8,
}

PLAN_VALIDATION_HORIZON = 15


def _weighted(steps: list[Step], weight: float) -> list[Step]:
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


def _position_eval(view: GameView) -> float:
    me = view.player
    opp = 1 - me  # 1v1 only for W3
    return (
        total_ships(view, me) - total_ships(view, opp)
        + 5.0 * (total_production(view, me) - total_production(view, opp))
    )


def _make_validator(view: GameView):
    """Return a validator that simulates the plan forward 15 turns and
    reverts to no-op if the projected eval drops below the current eval."""
    baseline = _position_eval(view)

    def validator(plan: list[Step]) -> list[Step]:
        if not plan:
            return plan
        my_actions = [s.as_move() for s in plan]
        # Opponent: passive (no actions). This is a fixed-policy approximation
        # of opponent behavior; W4 may add adversarial best-response.
        actions_per_player = [[], []]
        actions_per_player[view.player] = my_actions
        future = forward_simulate(view, actions_per_player, n_turns=PLAN_VALIDATION_HORIZON)
        future_eval = _position_eval(future)
        if future_eval < baseline - 5.0:  # 5-ship slack
            return []  # revert to no-op
        return plan

    return validator


def agent(obs) -> list[list]:
    view = GameView.from_obs(obs)

    candidates: list[Step] = []
    candidates.extend(_weighted(no_op_template(view), TEMPLATE_WEIGHTS["no_op"]))
    candidates.extend(_weighted(production_attack_template(view), TEMPLATE_WEIGHTS["production_attack"]))
    candidates.extend(_weighted(defensive_reinforce_template(view), TEMPLATE_WEIGHTS["defensive_reinforce"]))
    candidates.extend(_weighted(snipe_undefended_template(view), TEMPLATE_WEIGHTS["snipe_undefended"]))
    candidates.extend(_weighted(multi_source_consolidation_template(view), TEMPLATE_WEIGHTS["multi_source_consolidation"]))
    candidates.extend(_weighted(comet_rush_template(view), TEMPLATE_WEIGHTS["comet_rush"]))

    candidates = filter_capturable(candidates, view)

    surplus = surplus_ships(view, view.player)
    plan = compose_plan(
        candidates,
        surplus,
        allow_truncation=False,
        validator=_make_validator(view),
    )
    return [s.as_move() for s in plan]
