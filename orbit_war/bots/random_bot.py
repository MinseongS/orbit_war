"""A baseline bot that launches uniform-random fleets from each owned planet
with probability 1/3 per turn, sending half of available ships in a random
direction. Intended only as the bottom of the bot zoo."""

from __future__ import annotations

import math
import random as _random

from orbit_war.sim.observation import GameView


def agent(obs) -> list[list]:
    view = GameView.from_obs(obs)
    moves: list[list] = []
    for p in view.my_planets():
        if p.ships < 2:
            continue
        if _random.random() > 1 / 3:
            continue
        ships = p.ships // 2
        angle = _random.uniform(-math.pi, math.pi)
        moves.append([int(p.id), float(angle), int(ships)])
    return moves
