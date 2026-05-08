"""Starter-kit Nearest Planet Sniper, packaged for the zoo.

Identical strategy to `starter_kit/main.py`: each owned planet captures the
nearest non-owned planet whenever it has enough ships."""

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
        nearest = min(targets, key=lambda t: GameView.distance(mine, t))
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([int(mine.id), float(angle), int(ships_needed)])
    return moves
