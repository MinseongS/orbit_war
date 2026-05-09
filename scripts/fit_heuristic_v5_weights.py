"""Generate per-step data and fit heuristic_v5 weights.

Run via: uv run python scripts/fit_heuristic_v5_weights.py
Writes: orbit_war/tuning/step_weights/v5.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbit_war.bots import (
    greedy_baseline,
    heuristic_v4,
    public_tactical,
    starter_bot,
)
from orbit_war.tuning.regression import fit_weights
from orbit_war.tuning.step_data import (
    STEP_FEATURE_NAMES,
    TEMPLATE_NAMES,
    collect_step_dataset,
)

OUT = Path(__file__).parent.parent / "orbit_war" / "tuning" / "step_weights" / "v5.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairings = [
        (heuristic_v4.agent, starter_bot.agent),
        (heuristic_v4.agent, greedy_baseline.agent),
        (heuristic_v4.agent, public_tactical.agent),
    ]
    seeds_per_pairing = list(range(15))  # ~45 games

    all_X = []
    all_y = []
    for bot_a, bot_b in pairings:
        print(f"Collecting from {bot_a.__module__} vs {bot_b.__module__}…")
        X, y = collect_step_dataset([bot_a, bot_b], seeds_per_pairing, eval_horizon=10)
        all_X.append(X)
        all_y.append(y)
    X = np.vstack(all_X)
    y = np.hstack(all_y)
    print(f"Total dataset: {X.shape[0]} step samples, {X.shape[1]} features")

    weights = fit_weights(X, y)
    payload = {
        "feature_names": list(STEP_FEATURE_NAMES),
        "template_names": list(TEMPLATE_NAMES),
        "weights": weights.tolist(),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")
    print(f"Per-template weights (first {len(TEMPLATE_NAMES)} entries):")
    for name, w in zip(TEMPLATE_NAMES, weights[: len(TEMPLATE_NAMES)]):
        print(f"  {name}: {w:+.4f}")


if __name__ == "__main__":
    main()
