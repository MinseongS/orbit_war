"""Predict an opponent plan for use in adversarial plan validation.

Approach: run production_attack from the opponent's perspective. This is
cheap, requires no state mutation, and gives a reasonable lower-bound
estimate of opponent activity for forward-simulation purposes.

The returned action list is in Kaggle move format `[from_id, angle, ships]`
and is suitable for direct use in `forward_simulate(actions_per_player=...)`."""

from __future__ import annotations

from dataclasses import replace

from orbit_war.eval.features import surplus_ships
from orbit_war.plan_gen.composer import compose_plan
from orbit_war.plan_gen.filters import filter_capturable
from orbit_war.plan_gen.templates import production_attack_template
from orbit_war.sim.observation import GameView


def predict_opponent_plan(view: GameView, opponent: int) -> list[list]:
    """Return a list of actions the opponent would plausibly take this turn.

    Uses production_attack from the opponent's perspective + composer.
    """
    if not any(p.owner == opponent for p in view.planets):
        return []

    opp_view = replace(view, player=opponent)

    candidates = production_attack_template(opp_view)
    candidates = filter_capturable(candidates, opp_view)
    surplus = surplus_ships(opp_view, opponent)
    plan = compose_plan(candidates, surplus)
    return [s.as_move() for s in plan]
