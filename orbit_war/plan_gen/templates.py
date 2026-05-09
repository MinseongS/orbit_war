"""Step templates: per-template generators that emit ranked launch proposals.

Each template is a pure function over GameView, returning a list of `Step`s.
Templates score steps in isolation; the composer is responsible for combining
them under per-source ship-budget constraints.

W2 ships four templates. More land in W3-W4 as we add comet timing,
multi-source consolidation, and tack/feint patterns.
"""

from __future__ import annotations

from orbit_war.eval.features import incoming_threat
from orbit_war.plan_gen.aim import aim_with_orbit_prediction
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
        score = best.production / (1.0 + GameView.distance(src, best))
        angle, _arrival = aim_with_orbit_prediction(src, best, ships, view)
        proposals.append(
            Step(
                from_planet_id=int(src.id),
                target_planet_id=int(best.id),
                angle=angle,
                ships=int(ships),
                score=float(score),
            )
        )
    return proposals


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
        angle, _arrival = aim_with_orbit_prediction(nearest, target, ships, view)
        proposals.append(
            Step(
                from_planet_id=int(nearest.id),
                target_planet_id=int(target.id),
                angle=angle,
                ships=int(ships),
                score=float(score),
            )
        )
    return proposals


SNIPE_DEFENSE_THRESHOLD = 10
SNIPE_PRODUCTION_THRESHOLD = 2


def snipe_undefended_template(view: GameView) -> list[Step]:
    """Find low-defence, high-production targets and route the closest source.

    Filters: target must be non-owned, `ships < SNIPE_DEFENSE_THRESHOLD`,
    `production >= SNIPE_PRODUCTION_THRESHOLD`. Source must be able to
    afford `target.ships + 1` ships.
    """
    candidates = [
        t
        for t in view.targets()
        if t.ships < SNIPE_DEFENSE_THRESHOLD
        and t.production >= SNIPE_PRODUCTION_THRESHOLD
    ]
    if not candidates:
        return []

    sources = [p for p in view.my_planets() if p.ships >= 2]
    if not sources:
        return []

    proposals: list[Step] = []
    for tgt in candidates:
        nearest = min(sources, key=lambda s: GameView.distance(s, tgt))
        needed = ships_needed_to_capture(tgt)
        if nearest.ships < needed:
            continue
        score = tgt.production / (needed + 1.0)
        angle, _arrival = aim_with_orbit_prediction(nearest, tgt, needed, view)
        proposals.append(
            Step(
                from_planet_id=int(nearest.id),
                target_planet_id=int(tgt.id),
                angle=angle,
                ships=int(needed),
                score=float(score),
            )
        )
    return proposals


CONSOLIDATION_MIN_TARGET_PRODUCTION = 3
CONSOLIDATION_TOP_K_SOURCES = 4


def multi_source_consolidation_template(view: GameView) -> list[Step]:
    """For each rich non-owned target, gather contributing fleets from up to
    `CONSOLIDATION_TOP_K_SOURCES` of our nearest planets that can each afford
    a partial contribution. The composer's surplus check decides which
    contributions land.

    Each source contributes ~ceil(needed / N) ships (capped at half its
    garrison) so that across N sources, the combined fleet exceeds the
    target's defender count."""
    sources = [p for p in view.my_planets() if p.ships >= 5]
    if len(sources) < 2:
        return []

    rich_targets = [
        t
        for t in view.targets()
        if t.production >= CONSOLIDATION_MIN_TARGET_PRODUCTION
    ]
    if not rich_targets:
        return []

    proposals: list[Step] = []
    for tgt in rich_targets:
        ranked_sources = sorted(sources, key=lambda s: GameView.distance(s, tgt))
        contributing = ranked_sources[:CONSOLIDATION_TOP_K_SOURCES]
        if len(contributing) < 2:
            continue
        needed = ships_needed_to_capture(tgt)
        per_source_quota = max(1, (needed + len(contributing) - 1) // len(contributing))
        for src in contributing:
            ships = min(per_source_quota + 2, src.ships // 2)
            if ships < 1:
                continue
            angle, _arrival = aim_with_orbit_prediction(src, tgt, ships, view)
            score = tgt.production / (1.0 + GameView.distance(src, tgt))
            proposals.append(
                Step(
                    from_planet_id=int(src.id),
                    target_planet_id=int(tgt.id),
                    angle=angle,
                    ships=int(ships),
                    score=float(score) * 1.2,  # consolidation bonus
                )
            )
    return proposals
