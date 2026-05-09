"""heuristic_v4: W4 bot.

Differences from heuristic_v3:
- multi_source_consolidation now actually fires (filter_capturable was
  fixed in W4.1 to aggregate per target).
- Adds trade_down_strike_template for late-game grinding when ahead.
- Validator remains passive (W4.3's adversarial variant via
  predict_opponent_plan caused excessive plan vetoes — see commit notes).
  The predict_opponent_plan import is retained for W5 to retry with a
  different opponent-prediction strategy."""

from __future__ import annotations

from orbit_war.eval.features import (
    surplus_ships,
    total_production,
    total_ships,
)
from orbit_war.plan_gen.composer import compose_plan
from orbit_war.plan_gen.filters import filter_capturable
from orbit_war.plan_gen.opponent import predict_opponent_plan
from orbit_war.plan_gen.step import Step
from orbit_war.plan_gen.templates import (
    comet_rush_template,
    defensive_reinforce_template,
    multi_source_consolidation_template,
    no_op_template,
    production_attack_template,
    snipe_undefended_template,
    trade_down_strike_template,
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
    "trade_down_strike": 0.9,
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
    opp = 1 - me
    return (
        total_ships(view, me) - total_ships(view, opp)
        + 5.0 * (total_production(view, me) - total_production(view, opp))
    )


def _make_validator(view: GameView):
    """Validator that simulates the plan + a predicted opponent response,
    forwards 15 turns, and reverts to no-op if eval drops > 5.

    Note: the adversarial (predicted-opponent) variant vetoed too many good
    plans in practice, causing regression vs public_tactical and v1. We use
    the passive-opponent variant (opponent takes no actions during sim) which
    is the same approach as v3. The predict_opponent_plan import is retained
    for forward compatibility with v5's regression-weight approach."""
    baseline = _position_eval(view)

    def validator(plan: list[Step]) -> list[Step]:
        if not plan:
            return plan
        my_actions = [s.as_move() for s in plan]
        # Passive opponent: no actions during forward sim.
        # Adversarial (opp_actions = predict_opponent_plan(view, opp)) caused
        # over-vetoing — net regression vs public_tactical and v1.
        actions_per_player = [[], []]
        actions_per_player[view.player] = my_actions
        future = forward_simulate(view, actions_per_player, n_turns=PLAN_VALIDATION_HORIZON)
        future_eval = _position_eval(future)
        if future_eval < baseline - 5.0:
            return []
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
    candidates.extend(_weighted(trade_down_strike_template(view), TEMPLATE_WEIGHTS["trade_down_strike"]))

    candidates = filter_capturable(candidates, view)

    surplus = surplus_ships(view, view.player)
    plan = compose_plan(
        candidates,
        surplus,
        allow_truncation=False,
        validator=_make_validator(view),
    )
    return [s.as_move() for s in plan]
