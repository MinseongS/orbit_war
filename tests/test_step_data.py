"""Tests for the per-step data collector."""

import numpy as np

from orbit_war.bots import heuristic_v1, random_bot
from orbit_war.tuning.step_data import (
    STEP_FEATURE_NAMES,
    collect_step_dataset,
)


def test_step_dataset_shape():
    X, y = collect_step_dataset(
        bots=[heuristic_v1.agent, random_bot.agent],
        seeds=(1, 2),
        eval_horizon=10,
    )
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(STEP_FEATURE_NAMES)
    assert X.shape[0] >= 10  # heuristic_v1 launches plenty of steps


def test_step_features_are_unique():
    assert len(set(STEP_FEATURE_NAMES)) == len(STEP_FEATURE_NAMES)
