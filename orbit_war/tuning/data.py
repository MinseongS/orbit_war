"""Self-play dataset collector for linear weight regression.

Plays games between supplied bots, samples per-turn feature vectors per
player, labels each row with the *eventual game outcome* from that
player's perspective (+1 win, 0 draw, -1 loss), and returns the resulting
dataset as numpy arrays.

Per a1k0n's Tron 2010 trick: empirical coefficient tuning on ~10k games
beat hand-tuning. This module provides the data side of that pipeline;
`tuning.regression` does the fitting."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
from kaggle_environments import make

from orbit_war.eval.features import (
    incoming_threat,
    surplus_ships,
    total_production,
    total_ships,
)
from orbit_war.sim.observation import GameView

Agent = Callable[[dict], list]

FEATURE_NAMES: tuple[str, ...] = (
    "total_ships_self",
    "total_ships_enemy",
    "ship_diff",
    "total_production_self",
    "total_production_enemy",
    "production_diff",
    "owned_planets_self",
    "owned_planets_enemy",
    "surplus_total_self",
    "incoming_threat_self",
    "step_normalized",
)


def _vectorize(view: GameView, opponent: int) -> np.ndarray:
    me = view.player
    ts_me = total_ships(view, me)
    ts_op = total_ships(view, opponent)
    tp_me = total_production(view, me)
    tp_op = total_production(view, opponent)
    own_me = sum(1 for p in view.planets if p.owner == me)
    own_op = sum(1 for p in view.planets if p.owner == opponent)
    surplus_me = sum(surplus_ships(view, me).values())
    threat_me = sum(
        incoming_threat(view, me, p.id, horizon=20)
        for p in view.planets
        if p.owner == me
    )
    return np.array(
        [
            ts_me,
            ts_op,
            ts_me - ts_op,
            tp_me,
            tp_op,
            tp_me - tp_op,
            own_me,
            own_op,
            surplus_me,
            threat_me,
            view.step / 500.0,
        ],
        dtype=np.float64,
    )


def _label(reward: float | None) -> int:
    if reward is None:
        return 0
    if reward > 0:
        return 1
    if reward < 0:
        return -1
    return 0


def collect_self_play_dataset(
    bots: Sequence[Agent],
    seeds: Iterable[int],
    sample_every: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y). Each row of X is a feature vector at one sampled turn,
    each entry of y is the *eventual* outcome (+1/0/-1) for the player whose
    perspective the row was taken from."""
    if len(bots) != 2:
        raise ValueError("collect_self_play_dataset expects exactly two bots")

    rows: list[np.ndarray] = []
    labels: list[int] = []
    # Run both orderings of each matchup to collect symmetric data and to
    # ensure sufficient rows even when one ordering ends quickly.
    bot_orderings = [list(bots), list(reversed(bots))]
    for seed in seeds:
        for ordering in bot_orderings:
            env = make(
                "orbit_wars",
                configuration={"seed": seed, "episodeSteps": 500},
                debug=True,
            )
            env.run(ordering)
            final = env.steps[-1]
            outcome_per_player = [_label(s.reward) for s in final]
            for step_idx, frame in enumerate(env.steps):
                if step_idx % sample_every != 0:
                    continue
                for player_idx in range(2):
                    obs = frame[player_idx]["observation"]
                    view = GameView.from_obs(obs)
                    opponent = 1 - player_idx
                    rows.append(_vectorize(view, opponent))
                    labels.append(outcome_per_player[player_idx])
    return np.stack(rows), np.array(labels, dtype=np.int8)
