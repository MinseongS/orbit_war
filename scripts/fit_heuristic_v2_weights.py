"""Generate self-play data and fit heuristic_v2 weights.

Run via: uv run python scripts/fit_heuristic_v2_weights.py
Writes: orbit_war/tuning/weights/v2.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbit_war.bots import (
    greedy_baseline,
    heuristic_v1,
    public_tactical,
    random_bot,
    starter_bot,
)
from orbit_war.tuning.data import FEATURE_NAMES, collect_self_play_dataset
from orbit_war.tuning.regression import fit_weights

OUT = Path(__file__).parent.parent / "orbit_war" / "tuning" / "weights" / "v2.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairings = [
        (heuristic_v1.agent, random_bot.agent),
        (heuristic_v1.agent, starter_bot.agent),
        (heuristic_v1.agent, greedy_baseline.agent),
        (heuristic_v1.agent, public_tactical.agent),
    ]
    seeds_per_pairing = list(range(50))  # 4 pairings * 50 seeds = 200 games

    all_X = []
    all_y = []
    for bot_a, bot_b in pairings:
        print(f"Collecting from {bot_a.__module__} vs {bot_b.__module__}…")
        X, y = collect_self_play_dataset([bot_a, bot_b], seeds_per_pairing, sample_every=20)
        all_X.append(X)
        all_y.append(y)
    X = np.vstack(all_X)
    y = np.hstack(all_y).astype(np.float64)
    print(f"Total dataset: {X.shape[0]} rows, {X.shape[1]} features")

    weights = fit_weights(X, y)
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "weights": weights.tolist(),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")
    print(f"Weights: {dict(zip(FEATURE_NAMES, weights.round(4)))}")


if __name__ == "__main__":
    main()
