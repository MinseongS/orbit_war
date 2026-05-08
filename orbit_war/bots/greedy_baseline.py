"""Production-per-distance greedy.

For each owned planet we score every non-owned planet by
    score = production / (1 + distance)
and capture the highest-scoring one we can afford with `target.ships + 1`
ships. Slightly stronger than the starter sniper because it prefers
high-production neutrals over merely-close ones."""

from __future__ import annotations

import math

from orbit_war.sim.observation import GameView


def agent(obs) -> list[list]:
    view = GameView.from_obs(obs)
    targets = view.targets()
    if not targets:
        return []

    moves: list[list] = []
    for mine in view.my_planets():
        scored = sorted(
            targets,
            key=lambda t: t.production / (1.0 + GameView.distance(mine, t)),
            reverse=True,
        )
        for t in scored:
            ships_needed = t.ships + 1
            if mine.ships >= ships_needed:
                angle = math.atan2(t.y - mine.y, t.x - mine.x)
                moves.append([int(mine.id), float(angle), int(ships_needed)])
                break
    return moves
