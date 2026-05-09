"""Submission entry point. Re-exports heuristic_v4 from our package.

heuristic_v4 is the W4 champion: same as v3 (orbit-aware fleet aiming +
6 W3 templates + 15-turn forward-sim plan validation) plus the
trade_down_strike template that fires in late game (step >= 300) when
ahead by >=20 ships. Measured 64% vs public_tactical, 100% vs
random/starter, 92% vs greedy. ~50% vs v3 in self-play (essentially
equivalent — trade_down only activates in narrow conditions)."""

from orbit_war.bots.heuristic_v4 import agent  # noqa: F401
