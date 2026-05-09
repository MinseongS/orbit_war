"""Step templates: per-template generators that emit ranked launch proposals.

Each template is a pure function over GameView, returning a list of `Step`s.
Templates score steps in isolation; the composer is responsible for combining
them under per-source ship-budget constraints.

W2 ships four templates. More land in W3-W4 as we add comet timing,
multi-source consolidation, and tack/feint patterns.
"""

from __future__ import annotations

from orbit_war.plan_gen.step import Step, ships_needed_to_capture
from orbit_war.sim.observation import GameView


def no_op_template(view: GameView) -> list[Step]:
    """Sentinel: propose no action. Lets the composer rank the empty plan
    against alternatives in case 'wait and accumulate' is best."""
    return []


def production_attack_template(view: GameView) -> list[Step]:
    """Per-owned-planet, propose attacks on the best `production / (1+distance)`
    non-owned target we can afford with `target.ships + 1` ships.

    Scores: production / (1 + distance). Direct port of greedy_baseline so
    the composer always has at least one strong baseline candidate. Ships
    are capped at the source's current garrison.
    """
    targets = view.targets()
    if not targets:
        return []

    proposals: list[Step] = []
    for src in view.my_planets():
        if src.ships < 1:
            continue
        best = max(
            targets,
            key=lambda t: t.production / (1.0 + GameView.distance(src, t)),
        )
        needed = ships_needed_to_capture(best)
        ships = min(int(src.ships), needed)
        if ships < 1:
            continue
        score = best.production / (1.0 + GameView.distance(src, best))
        proposals.append(
            Step(
                from_planet_id=int(src.id),
                target_planet_id=int(best.id),
                angle=Step.angle_to(src, best),
                ships=int(ships),
                score=float(score),
            )
        )
    return proposals


from orbit_war.eval.features import incoming_threat


def defensive_reinforce_template(view: GameView) -> list[Step]:
    """For each owned planet under threat, propose a reinforcement from the
    nearest friendly planet that has surplus ships.

    Scoring favours higher threat and shorter rescue distance.
    """
    proposals: list[Step] = []
    for target in view.my_planets():
        threat = incoming_threat(view, view.player, target.id, horizon=30)
        if threat == 0:
            continue
        defender_window = target.ships + target.production * 5
        if threat <= defender_window:
            continue  # We'll survive without help.
        deficit = threat - defender_window
        helpers = [
            p
            for p in view.my_planets()
            if p.id != target.id and p.ships > 1
        ]
        if not helpers:
            continue
        nearest = min(helpers, key=lambda h: GameView.distance(h, target))
        ships = min(int(nearest.ships), deficit + 1)
        if ships < 1:
            continue
        score = deficit / (1.0 + GameView.distance(nearest, target))
        proposals.append(
            Step(
                from_planet_id=int(nearest.id),
                target_planet_id=int(target.id),
                angle=Step.angle_to(nearest, target),
                ships=int(ships),
                score=float(score),
            )
        )
    return proposals
