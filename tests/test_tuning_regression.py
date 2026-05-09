"""Tests for the regression weight fitter."""

import numpy as np

from orbit_war.tuning.regression import fit_weights


def test_fit_weights_recovers_true_weights_on_synthetic_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    true_w = np.array([2.0, -1.0, 0.5])
    y = X @ true_w + rng.normal(scale=0.1, size=500)

    fitted = fit_weights(X, y)
    assert fitted.shape == (3,)
    assert np.allclose(fitted, true_w, atol=0.1)


def test_fit_weights_handles_singular_designs():
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    fitted = fit_weights(X, y)
    assert fitted.shape == (2,)
    assert np.all(np.isfinite(fitted))
