"""Submission entry point. Re-exports heuristic_v3 from our package.

heuristic_v3 is the W3 champion: orbit-aware fleet aiming + 6 templates
(no_op, production_attack, defensive_reinforce, snipe_undefended,
multi_source_consolidation, comet_rush) + 15-turn forward-sim plan
validation. Measured 64% vs public_tactical, 100% vs random/starter,
92% vs greedy."""

from orbit_war.bots.heuristic_v3 import agent  # noqa: F401
