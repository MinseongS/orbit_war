"""Step templates: per-template generators that emit ranked launch proposals.

Each template is a pure function over GameView, returning a list of `Step`s.
Templates score steps in isolation; the composer is responsible for combining
them under per-source ship-budget constraints.

W2 ships four templates. More land in W3-W4 as we add comet timing,
multi-source consolidation, and tack/feint patterns.
"""

from __future__ import annotations

import math

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


COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
COMET_RUSH_PRE_WINDOW = 5  # fire in the 5 steps before each spawn
COMET_RUSH_QUADRANT_TARGETS = (
    (25.0, 25.0),
    (75.0, 25.0),
    (25.0, 75.0),
    (75.0, 75.0),
)


def _is_comet_pre_window(step: int) -> bool:
    return any(0 < spawn - step <= COMET_RUSH_PRE_WINDOW for spawn in COMET_SPAWN_STEPS)


def comet_rush_template(view: GameView) -> list[Step]:
    """In the 5 turns before each comet spawn, propose attacks from each owned
    planet aimed at the four quadrant centers where comets typically appear.

    Each step sends a small probe (10-20 ships, capped at src.ships // 4)
    so we don't drain home planets. Uses the source's own id as a placeholder
    target_planet_id (filter_capturable treats this as a self-reinforcement
    and passes it through; the composer's surplus check still applies)."""
    if not _is_comet_pre_window(view.step):
        return []

    sources = [p for p in view.my_planets() if p.ships >= 10]
    if not sources:
        return []

    proposals: list[Step] = []
    for src in sources:
        for tx, ty in COMET_RUSH_QUADRANT_TARGETS:
            ships = min(20, src.ships // 4)
            if ships < 10:
                continue
            angle = math.atan2(ty - src.y, tx - src.x)
            distance_score = 1.0 / (1.0 + math.hypot(tx - src.x, ty - src.y))
            proposals.append(
                Step(
                    from_planet_id=int(src.id),
                    target_planet_id=int(src.id),  # placeholder: comet has no stable id at launch
                    angle=float(angle),
                    ships=int(ships),
                    score=float(distance_score) * 0.8,
                )
            )
    return proposals


TRADE_DOWN_MIN_STEP = 300
TRADE_DOWN_MIN_LEAD = 20


def trade_down_strike_template(view: GameView) -> list[Step]:
    """Late-game template: when ahead, trade ships with the opponent.

    Each trade preserves absolute lead while reducing both totals — risk
    reduction when winning. Fires only when:
      - `view.step >= TRADE_DOWN_MIN_STEP` (late game), and
      - our total ships exceed opponent's by `TRADE_DOWN_MIN_LEAD` ships.

    Sources: each owned planet with ships >= 10.
    Targets: every enemy planet within 70 board units.
    Each step sends `min(target.ships + 1, source.ships // 3)` so we don't
    drain a single source on a single trade.
    """
    if view.step < TRADE_DOWN_MIN_STEP:
        return []

    me = view.player
    my_ships = sum(p.ships for p in view.planets if p.owner == me)
    enemy_ships = sum(p.ships for p in view.planets if p.owner != me and p.owner != -1)
    if my_ships - enemy_ships < TRADE_DOWN_MIN_LEAD:
        return []

    enemies = list(view.enemy_planets())
    if not enemies:
        return []

    proposals: list[Step] = []
    for src in view.my_planets():
        if src.ships < 10:
            continue
        for tgt in enemies:
            if GameView.distance(src, tgt) > 70:
                continue
            ships = min(int(tgt.ships) + 1, src.ships // 3)
            if ships < 5:
                continue
            angle, _arrival = aim_with_orbit_prediction(src, tgt, ships, view)
            score = (my_ships - enemy_ships) / (1.0 + GameView.distance(src, tgt))
            proposals.append(
                Step(
                    from_planet_id=int(src.id),
                    target_planet_id=int(tgt.id),
                    angle=angle,
                    ships=int(ships),
                    score=float(score) * 0.6,  # slight de-emphasis vs offense
                )
            )
    return proposals
