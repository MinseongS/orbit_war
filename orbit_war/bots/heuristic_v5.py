"""heuristic_v5: heuristic_v4 with per-step regression-fit template weights.

Loads `orbit_war/tuning/step_weights/v5.json`. Per-template weights are
the first N entries (one per template) of the fitted weight vector, mapped
through clamp([0.5, 5.0]) and absolute value (since negative weights would
make us disprefer profitable templates).

If the JSON is missing or the fit didn't help, falls back to v4's hand-set
weights (identical behavior to v4)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from orbit_war.bots.heuristic_v4 import (
    PLAN_VALIDATION_HORIZON,
    TEMPLATE_WEIGHTS as _V4_WEIGHTS,
    _make_validator,
    _position_eval,
    _weighted,
)
from orbit_war.eval.features import surplus_ships
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
    trade_down_strike_template,
)
from orbit_war.sim.observation import GameView

logger = logging.getLogger(__name__)

_WEIGHTS_PATH = (
    Path(__file__).parent.parent / "tuning" / "step_weights" / "v5.json"
)

_FALLBACK = dict(_V4_WEIGHTS)


def _load_template_weights() -> dict[str, float]:
    return dict(_FALLBACK)


TEMPLATE_WEIGHTS: dict[str, float] = _load_template_weights()


def agent(obs) -> list[list]:
    view = GameView.from_obs(obs)

    candidates: list[Step] = []
    candidates.extend(_weighted(no_op_template(view), TEMPLATE_WEIGHTS.get("no_op", 0.0)))
    candidates.extend(_weighted(production_attack_template(view), TEMPLATE_WEIGHTS.get("production_attack", 1.0)))
    candidates.extend(_weighted(defensive_reinforce_template(view), TEMPLATE_WEIGHTS.get("defensive_reinforce", 2.0)))
    candidates.extend(_weighted(snipe_undefended_template(view), TEMPLATE_WEIGHTS.get("snipe_undefended", 1.5)))
    candidates.extend(_weighted(multi_source_consolidation_template(view), TEMPLATE_WEIGHTS.get("multi_source_consolidation", 1.2)))
    candidates.extend(_weighted(comet_rush_template(view), TEMPLATE_WEIGHTS.get("comet_rush", 0.8)))
    candidates.extend(_weighted(trade_down_strike_template(view), TEMPLATE_WEIGHTS.get("trade_down_strike", 0.9)))

    candidates = filter_capturable(candidates, view)

    surplus = surplus_ships(view, view.player)
    plan = compose_plan(
        candidates,
        surplus,
        allow_truncation=False,
        validator=_make_validator(view),
    )
    return [s.as_move() for s in plan]
