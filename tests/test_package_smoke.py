"""Smoke test: package imports cleanly."""

import orbit_war
import orbit_war.sim
import orbit_war.bots
import orbit_war.eval_harness


def test_package_importable():
    assert orbit_war is not None
    assert orbit_war.sim is not None
    assert orbit_war.bots is not None
    assert orbit_war.eval_harness is not None
