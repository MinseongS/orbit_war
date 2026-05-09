"""Per-step regression data collector.

For each turn in each game, capture the steps the bot would launch from
the current view, simulate each step in isolation, label with the eval
delta. Use the (feature, delta) pairs to fit per-step weights in
`tuning/regression.py`."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
from kaggle_environments import make

from orbit_war.eval.features import total_production, total_ships
from orbit_war.plan_gen.step import Step
from orbit_war.plan_gen.templates import (
    comet_rush_template,
    defensive_reinforce_template,
    multi_source_consolidation_template,
    production_attack_template,
    snipe_undefended_template,
    trade_down_strike_template,
)
from orbit_war.sim.game import forward_simulate
from orbit_war.sim.observation import GameView

Agent = Callable[[dict], list]

# Template-class one-hot features
TEMPLATE_NAMES: tuple[str, ...] = (
    "production_attack",
    "defensive_reinforce",
    "snipe_undefended",
    "multi_source_consolidation",
    "comet_rush",
    "trade_down_strike",
)

STEP_FEATURE_NAMES: tuple[str, ...] = (
    *(f"is_{n}" for n in TEMPLATE_NAMES),
    "src_ships",
    "src_production",
    "tgt_ships",
    "tgt_production",
    "step_ships",
    "ship_lead",
    "step_normalized",
)


def _eval(view: GameView) -> float:
    me = view.player
    opp = 1 - me
    return (
        total_ships(view, me) - total_ships(view, opp)
        + 5.0 * (total_production(view, me) - total_production(view, opp))
    )


def _step_features(
    step: Step,
    template_idx: int,
    view: GameView,
) -> np.ndarray:
    by_id = {p.id: p for p in view.planets}
    src = by_id.get(step.from_planet_id)
    tgt = by_id.get(step.target_planet_id)
    src_ships = float(src.ships) if src else 0.0
    src_prod = float(src.production) if src else 0.0
    tgt_ships = float(tgt.ships) if tgt else 0.0
    tgt_prod = float(tgt.production) if tgt else 0.0
    me = view.player
    opp = 1 - me
    ship_lead = float(total_ships(view, me) - total_ships(view, opp))

    one_hot = np.zeros(len(TEMPLATE_NAMES), dtype=np.float64)
    if 0 <= template_idx < len(TEMPLATE_NAMES):
        one_hot[template_idx] = 1.0

    return np.concatenate(
        [
            one_hot,
            np.array(
                [src_ships, src_prod, tgt_ships, tgt_prod,
                 float(step.ships), ship_lead, view.step / 500.0],
                dtype=np.float64,
            ),
        ]
    )


def collect_step_dataset(
    bots: Sequence[Agent],
    seeds: Iterable[int],
    eval_horizon: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Run games between `bots`. At each turn, for each candidate step
    each template would emit, simulate that single step in isolation and
    label with the eval delta after `eval_horizon` turns.

    Returns (X, y) where X has shape (n_steps, len(STEP_FEATURE_NAMES))."""
    if len(bots) != 2:
        raise ValueError("collect_step_dataset expects exactly two bots")

    rows: list[np.ndarray] = []
    labels: list[float] = []

    template_funcs = (
        production_attack_template,
        defensive_reinforce_template,
        snipe_undefended_template,
        multi_source_consolidation_template,
        comet_rush_template,
        trade_down_strike_template,
    )

    for seed in seeds:
        env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 500}, debug=True)
        env.run(list(bots))
        for step_idx in range(0, len(env.steps), 25):  # sample every 25 turns
            for player_idx in range(2):
                obs = env.steps[step_idx][player_idx]["observation"]
                view = GameView.from_obs(obs)
                baseline_eval = _eval(view)
                for tmpl_idx, tmpl in enumerate(template_funcs):
                    candidates = tmpl(view)
                    for s in candidates:
                        actions_per_player = [[], []]
                        actions_per_player[view.player] = [s.as_move()]
                        future = forward_simulate(view, actions_per_player, n_turns=eval_horizon)
                        delta = _eval(future) - baseline_eval
                        rows.append(_step_features(s, tmpl_idx, view))
                        labels.append(delta)

    if not rows:
        empty_X = np.zeros((0, len(STEP_FEATURE_NAMES)), dtype=np.float64)
        return empty_X, np.zeros(0, dtype=np.float64)
    return np.stack(rows), np.array(labels, dtype=np.float64)
