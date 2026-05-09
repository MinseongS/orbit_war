"""Tests for the self-play data collector."""

import numpy as np

from orbit_war.bots import greedy_baseline, random_bot
from orbit_war.tuning.data import (
    FEATURE_NAMES,
    collect_self_play_dataset,
)


def test_dataset_shape_matches_collected_games():
    X, y = collect_self_play_dataset(
        bots=[greedy_baseline.agent, random_bot.agent],
        seeds=(1, 2),
        sample_every=10,  # one feature row every 10 turns
    )
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(FEATURE_NAMES)
    assert X.shape[0] >= 100


def test_outcome_label_is_in_minus_one_zero_one():
    X, y = collect_self_play_dataset(
        bots=[greedy_baseline.agent, random_bot.agent],
        seeds=(1,),
        sample_every=20,
    )
    assert set(np.unique(y)).issubset({-1, 0, 1})


def test_feature_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
