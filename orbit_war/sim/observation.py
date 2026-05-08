"""Typed read-only view over a raw orbit_wars observation dict."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet


@dataclass(frozen=True)
class GameView:
    player: int
    planets: tuple[Planet, ...]
    fleets: tuple[Fleet, ...]
    angular_velocity: float
    initial_planets: tuple[Planet, ...]
    comet_planet_ids: frozenset[int]
    remaining_overage_time: float

    @staticmethod
    def from_obs(obs) -> "GameView":
        get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        planets = tuple(Planet(*p) for p in get("planets", []))
        fleets = tuple(Fleet(*f) for f in get("fleets", []))
        initial = tuple(Planet(*p) for p in get("initial_planets", []))
        return GameView(
            player=int(get("player", 0)),
            planets=planets,
            fleets=fleets,
            angular_velocity=float(get("angular_velocity", 0.0)),
            initial_planets=initial,
            comet_planet_ids=frozenset(get("comet_planet_ids", []) or []),
            remaining_overage_time=float(get("remainingOverageTime", 0.0)),
        )

    def my_planets(self) -> tuple[Planet, ...]:
        return tuple(p for p in self.planets if p.owner == self.player)

    def enemy_planets(self) -> tuple[Planet, ...]:
        return tuple(
            p for p in self.planets if p.owner != self.player and p.owner != -1
        )

    def neutral_planets(self) -> tuple[Planet, ...]:
        return tuple(p for p in self.planets if p.owner == -1)

    def targets(self) -> tuple[Planet, ...]:
        return tuple(p for p in self.planets if p.owner != self.player)

    def my_fleets(self) -> tuple[Fleet, ...]:
        return tuple(f for f in self.fleets if f.owner == self.player)

    def enemy_fleets(self) -> tuple[Fleet, ...]:
        return tuple(f for f in self.fleets if f.owner != self.player)

    @staticmethod
    def distance(a: Planet | Fleet, b: Planet | Fleet) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)
