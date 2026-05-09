"""The Step abstraction: a single ranked launch proposal.

Step templates emit `Step` instances; the composer ranks and combines them
into the final per-turn action list."""

from __future__ import annotations

import math
from dataclasses import dataclass

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


@dataclass(frozen=True)
class Step:
    """A single launch proposal, scored in isolation by its template."""

    from_planet_id: int
    target_planet_id: int
    angle: float
    ships: int
    score: float

    def as_move(self) -> list:
        """Serialise to the Kaggle action format `[from_id, angle, ships]`."""
        return [self.from_planet_id, self.angle, self.ships]

    @staticmethod
    def angle_to(src: Planet, target: Planet) -> float:
        return math.atan2(target.y - src.y, target.x - src.x)


def ships_needed_to_capture(target: Planet, player: int | None = None) -> int:
    """Minimum attacker ships to flip ownership of `target`.

    If `player` is given and already owns the target, returns 0 (no capture
    needed). Otherwise returns `target.ships + 1` (Orbit Wars combat: the
    attacker survives only the surplus over the defender)."""
    if player is not None and target.owner == player:
        return 0
    return int(target.ships) + 1
