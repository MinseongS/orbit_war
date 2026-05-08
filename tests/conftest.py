"""Shared fixtures for the orbit_war test suite."""

import pytest


@pytest.fixture
def fast_config() -> dict:
    """A short-episode config used by tests that don't need a 500-turn game."""
    return {"episodeSteps": 30, "actTimeout": 5, "seed": 1234}


@pytest.fixture
def seed_42_config() -> dict:
    """Standard 500-turn game on a fixed seed for parity tests."""
    return {"seed": 42}
