# W3: Orbit-Aware Targeting + Lookahead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the public_tactical gap (W2 v1 hit only 10% vs public_tactical) by adding orbit-aware fleet aiming, a lightweight forward simulator, plan validation, and two new step templates (multi-source consolidation and comet rush).

**Architecture:** The W2 build aimed fleets at *current* planet positions — but planets move while fleets travel. W3 starts by replacing every aim-call with one that predicts the planet's position at fleet arrival (using the closed-form `sim.orbits.planet_position_at` already in place). On top of that, ship a small `sim/game.py` forward simulator (fleet movement + captures only — combat edge cases and comet spawn/expire deferred) and use it inside the composer to *validate* candidate plans: simulate the chosen plan N=15 turns forward, and reject it (revert to no-op) if the future eval drops below current eval. Add two new templates for the patterns Planet Wars writeups credit most heavily — multi-source consolidation strikes and comet rushes. heuristic_v3 wires it together and replaces v1 as the W3 champion.

**Tech Stack:** Same as W2 — Python 3.13 (uv), `numpy` (vectorised features), `pytest`. No new external deps.

**Realistic outcome target:** v3 should beat v1 ≥65% in self-play, beat starter ≥85%, beat public_tactical ≥35-50%. Hitting the design doc's stretch goal of public_tactical ≥60% is plausible only if all four W3 wins (orbit-aim, lookahead, consolidation strikes, comet rush) compound — not guaranteed. Close monitoring required.

---

## File structure

NEW production files:
- `orbit_war/plan_gen/filters.py` — extracted `_filter_capturable` (W2 follow-up)
- `orbit_war/plan_gen/aim.py` — `aim_with_orbit_prediction(src, target, ships, view)` helper
- `orbit_war/sim/game.py` — minimal `forward_simulate(view, actions_per_player, n_turns)`
- `orbit_war/bots/heuristic_v3.py` — W3 bot wiring orbit-aim + plan validation + 2 new templates

NEW test files:
- `tests/test_filters.py`
- `tests/test_aim.py`
- `tests/test_forward_sim.py`
- `tests/test_heuristic_v3.py`

Modified:
- `orbit_war/plan_gen/templates.py` — fix mid-file `incoming_threat` import; replace `Step.angle_to` calls in `production_attack` / `defensive_reinforce` / `snipe_undefended` with `aim_with_orbit_prediction`; add `multi_source_consolidation_template` and `comet_rush_template` at the bottom; remove dead `if ships < 1` guard
- `orbit_war/plan_gen/composer.py` — add optional `validator` callback parameter
- `orbit_war/bots/heuristic_v1.py` — remove the in-bot `_filter_capturable` definition; import from `plan_gen.filters` instead (keeps v1 bit-identical in behaviour)
- `orbit_war/bots/heuristic_v2.py` — same import switch
- `orbit_war/eval_harness/cli.py` — add `heuristic_v3` to the zoo map
- `scripts/submit_bot.sh` — call `uv run ow-gate` before submitting; abort on `OVERALL: FAIL`
- `tests/test_templates.py` — extend with tests for the two new templates
- `tests/test_composer.py` — extend with the validator-parameter case

---

## Task 1: W2 housekeeping — extract filter, fix imports, gate-aware submit

**Files:**
- Create: `orbit_war/plan_gen/filters.py`
- Create: `tests/test_filters.py`
- Modify: `orbit_war/plan_gen/templates.py` (move import to top, drop dead guard)
- Modify: `orbit_war/bots/heuristic_v1.py` (use shared filter)
- Modify: `orbit_war/bots/heuristic_v2.py` (use shared filter)
- Modify: `scripts/submit_bot.sh`

The W2 final review flagged five low-severity follow-ups. Knock them out in one task before the real W3 work starts — they touch files we'll edit later anyway.

- [ ] **Step 1: Create `orbit_war/plan_gen/filters.py`**

```python
"""Composer-side filters that run between template emission and the composer.

Templates emit candidate steps freely; filters drop steps that are obviously
counter-productive (e.g. attacks that send fewer ships than the defender,
which would waste ships in transit without capturing anything)."""

from __future__ import annotations

from orbit_war.plan_gen.step import Step
from orbit_war.sim.observation import GameView


def filter_capturable(steps: list[Step], view: GameView) -> list[Step]:
    """Remove attack steps that cannot capture their target with the ships sent.

    Friendly reinforcements are passed through unconditionally."""
    planet_by_id = {p.id: p for p in view.planets}
    player = view.player
    result: list[Step] = []
    for s in steps:
        target = planet_by_id.get(s.target_planet_id)
        if target is None:
            result.append(s)
            continue
        if target.owner == player:
            result.append(s)
            continue
        needed = int(target.ships) + 1
        if s.ships >= needed:
            result.append(s)
    return result
```

- [ ] **Step 2: Write a unit test for the filter**

Create `tests/test_filters.py`:

```python
"""Tests for orbit_war.plan_gen.filters."""

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_war.plan_gen.filters import filter_capturable
from orbit_war.plan_gen.step import Step
from orbit_war.sim.observation import GameView


def _view_with(planets: tuple[Planet, ...]) -> GameView:
    return GameView(
        player=0,
        planets=planets,
        fleets=(),
        angular_velocity=0.04,
        initial_planets=planets,
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )


def test_filter_drops_underpowered_attacks():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 30, 1)
    view = _view_with((me, enemy))
    steps = [
        Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=20, score=0.5),
    ]
    assert filter_capturable(steps, view) == []  # 20 < 30+1


def test_filter_keeps_winning_attacks():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 30, 1)
    view = _view_with((me, enemy))
    steps = [
        Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=31, score=0.5),
    ]
    assert len(filter_capturable(steps, view)) == 1


def test_filter_passes_friendly_reinforcements_unconditionally():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    sibling = Planet(1, 0, 90.0, 90.0, 1.0, 30, 1)
    view = _view_with((me, sibling))
    steps = [
        Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=5, score=0.5),
    ]
    assert len(filter_capturable(steps, view)) == 1


def test_filter_passes_through_step_for_unknown_target():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    view = _view_with((me,))
    steps = [
        Step(from_planet_id=0, target_planet_id=999, angle=0.0, ships=5, score=0.5),
    ]
    assert len(filter_capturable(steps, view)) == 1
```

- [ ] **Step 3: Run the new test, expect failure**

Run: `uv run pytest tests/test_filters.py -v`
Expected: 4 ImportErrors → after creating filters.py, 4 passes.

Then run: `uv run pytest tests/test_filters.py -v`
Expected: 4 passed.

- [ ] **Step 4: Switch heuristic_v1 to use shared filter**

Edit `orbit_war/bots/heuristic_v1.py`:

1. Add at the top with other imports:

```python
from orbit_war.plan_gen.filters import filter_capturable
```

2. Locate the `def _filter_capturable(steps, view)` function in heuristic_v1.py and DELETE it (entire function body and definition).

3. In `agent()`, replace the call from `_filter_capturable(candidates, view)` to `filter_capturable(candidates, view)`.

- [ ] **Step 5: Switch heuristic_v2 to use shared filter (same edits)**

Edit `orbit_war/bots/heuristic_v2.py`:

1. Add `from orbit_war.plan_gen.filters import filter_capturable` to imports.
2. Delete the local `_filter_capturable` function definition.
3. Update the call site in `agent()` to use the imported name.

- [ ] **Step 6: Verify v1 and v2 tests still pass**

Run: `uv run pytest tests/test_heuristic_v1.py tests/test_heuristic_v2.py -v`
Expected: 6 passed (4 v1 + 2 v2).

- [ ] **Step 7: Fix mid-file import in templates.py**

Open `orbit_war/plan_gen/templates.py`. Find the `from orbit_war.eval.features import incoming_threat` line that sits below `production_attack_template`. Move it to the top of the file alongside the other imports.

The top of `templates.py` should look like:

```python
"""Step templates: per-template generators that emit ranked launch proposals.

...
"""

from __future__ import annotations

from orbit_war.eval.features import incoming_threat
from orbit_war.plan_gen.step import Step, ships_needed_to_capture
from orbit_war.sim.observation import GameView
```

- [ ] **Step 8: Remove the dead `if ships < 1: continue` in production_attack_template**

In `orbit_war/plan_gen/templates.py`, locate `production_attack_template`. Inside the per-source loop, the line `if ships < 1: continue` after `ships = min(int(src.ships), needed)` is unreachable (`src.ships >= 1` guarded above; `needed >= 1` always). Delete that one `if`.

- [ ] **Step 9: Run the full template suite**

Run: `uv run pytest tests/test_templates.py -v`
Expected: 8 passed.

- [ ] **Step 10: Add gate enforcement to submit_bot.sh**

Replace `scripts/submit_bot.sh` contents with:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <bot-name> <bot-spec> <message>" >&2
  echo "  bot-name        appears in submissions.log" >&2
  echo "  bot-spec        e.g. orbit_war.bots.heuristic_v3:agent" >&2
  echo "  message         goes to Kaggle" >&2
  exit 2
fi

BOT_NAME="$1"
BOT_SPEC="$2"
MESSAGE="$3"
SHA="$(git rev-parse --short HEAD)"

# Idempotency guard.
if grep -q "  $SHA  " submissions.log 2>/dev/null; then
  echo "SHA $SHA already in submissions.log — bailing to save daily quota." >&2
  echo "If you really mean to resubmit, edit submissions.log first." >&2
  exit 1
fi

# Gate enforcement: only submit bots that pass the local gate.
echo "Running ow-gate against current champion before submission..."
if ! uv run ow-gate "$BOT_SPEC" --seeds 10 --workers 4 > /tmp/ow-gate.log 2>&1; then
  echo "GATE FAILED — refusing to submit. Output:" >&2
  cat /tmp/ow-gate.log >&2
  exit 1
fi
echo "Gate PASSED."

BUNDLE="$(mktemp -d)/submission.tar.gz"
tar --exclude='__pycache__' --exclude='*.pyc' \
    -czf "$BUNDLE" main.py orbit_war

uv run kaggle competitions submit orbit-wars \
    -f "$BUNDLE" \
    -m "$MESSAGE"

echo "Submitted $BUNDLE"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "$TS  $BOT_NAME  $SHA  $MESSAGE" >> submissions.log
```

The script now requires a `<bot-spec>` argument and runs the gate before allowing submission.

- [ ] **Step 11: Run the full test suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: 81 passed (79 prior + 4 new from filters - 2 deletions noted by pytest counts; actually total may differ slightly, but no failures).

- [ ] **Step 12: Commit**

```bash
git add orbit_war/plan_gen/filters.py tests/test_filters.py orbit_war/plan_gen/templates.py orbit_war/bots/heuristic_v1.py orbit_war/bots/heuristic_v2.py scripts/submit_bot.sh
git commit -m "Extract filter_capturable; fix templates.py imports; gate-aware submit"
```

---

## Task 2: Orbit-aware aim helper

**Files:**
- Create: `orbit_war/plan_gen/aim.py`
- Test: `tests/test_aim.py`

A fleet of N ships travels at `fleet_speed(N)` per turn. While the fleet is in flight, the target planet may rotate around the sun. We need an aim helper that predicts where the planet will be at fleet arrival time and returns the angle pointing there. Use the closed-form `sim.orbits.planet_position_at` we already have.

We do this iteratively because predicting the arrival turn depends on distance, which depends on the predicted future position — a fixed point that converges quickly (3-4 iterations).

- [ ] **Step 1: Write failing tests**

Create `tests/test_aim.py`:

```python
"""Tests for orbit_war.plan_gen.aim."""

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    Planet,
)

from orbit_war.plan_gen.aim import aim_with_orbit_prediction
from orbit_war.sim.observation import GameView


def _view_with(planets: tuple[Planet, ...], angular_velocity: float = 0.05) -> GameView:
    return GameView(
        player=0,
        planets=planets,
        fleets=(),
        angular_velocity=angular_velocity,
        initial_planets=planets,
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )


def test_aim_to_static_target_matches_naive_atan2():
    """A static (far-from-sun) planet doesn't move — orbit-aware angle equals naive."""
    src = Planet(0, 0, 10.0, 50.0, 1.0, 50, 1)
    static_tgt = Planet(1, -1, 90.0, 50.0, 1.0, 5, 1)  # outside ROTATION_LIMIT
    view = _view_with((src, static_tgt))
    angle, _arrival_turn = aim_with_orbit_prediction(src, static_tgt, ships=10, view=view)
    expected = math.atan2(50.0 - 50.0, 90.0 - 10.0)
    assert math.isclose(angle, expected, abs_tol=1e-9)


def test_aim_to_orbiting_target_drifts_from_naive():
    """An orbiting target makes the orbit-aware angle differ from atan2 to current pos."""
    src = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    orbiting_tgt = Planet(1, -1, CENTER + 20, CENTER, 1.0, 5, 1)  # inside ROTATION_LIMIT
    view = _view_with((src, orbiting_tgt), angular_velocity=0.05)
    naive = math.atan2(orbiting_tgt.y - src.y, orbiting_tgt.x - src.x)
    angle, arrival_turn = aim_with_orbit_prediction(src, orbiting_tgt, ships=10, view=view)
    assert arrival_turn >= 1
    # With angular_velocity 0.05 over multiple turns, the angle should drift noticeably.
    assert not math.isclose(angle, naive, abs_tol=1e-3)


def test_aim_returns_arrival_turn_at_least_one():
    src = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    tgt = Planet(1, -1, 11.0, 10.0, 1.0, 5, 1)  # very close
    view = _view_with((src, tgt))
    _, arrival_turn = aim_with_orbit_prediction(src, tgt, ships=10, view=view)
    assert arrival_turn >= 1
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_aim.py -v`

- [ ] **Step 3: Implement `orbit_war/plan_gen/aim.py`**

```python
"""Orbit-aware fleet aiming.

Targets that orbit the central sun move while a fleet is in flight. Naively
aiming at the target's current position causes the fleet to miss. We predict
the target's position at fleet arrival via the closed-form
`sim.orbits.planet_position_at`, then re-derive the launch angle. This is
a fixed-point iteration: arrival turn depends on distance, distance depends
on predicted position. Three iterations converge for any realistic case."""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_war.sim.observation import GameView
from orbit_war.sim.orbits import is_orbiting, planet_position_at
from orbit_war.sim.physics import turns_to_arrive

MAX_AIM_ITERATIONS = 4
AIM_CONVERGENCE_DELTA = 0.1  # board units


def aim_with_orbit_prediction(
    src: Planet,
    target: Planet,
    ships: int,
    view: GameView,
) -> tuple[float, int]:
    """Return (angle, arrival_turn) for a fleet leaving `src` toward `target`.

    For static targets, this is straight atan2. For orbiting targets, predict
    where the target will be at fleet arrival and aim there.

    Returns the angle in radians and the integer arrival turn count.
    """
    initial_target = _initial_planet(view, target.id)
    if initial_target is None or not is_orbiting(initial_target):
        # Static target: no prediction needed.
        angle = math.atan2(target.y - src.y, target.x - src.x)
        arrival = turns_to_arrive(src.x, src.y, target.x, target.y, max(1, ships))
        return angle, arrival

    # Orbiting target: iterate angle ↔ arrival_turn fixed point.
    tx, ty = target.x, target.y
    arrival = 1
    for _ in range(MAX_AIM_ITERATIONS):
        new_arrival = turns_to_arrive(src.x, src.y, tx, ty, max(1, ships))
        future_step = view.step + new_arrival
        nx, ny = planet_position_at(initial_target, future_step, view.angular_velocity)
        if abs(nx - tx) < AIM_CONVERGENCE_DELTA and abs(ny - ty) < AIM_CONVERGENCE_DELTA:
            tx, ty, arrival = nx, ny, new_arrival
            break
        tx, ty, arrival = nx, ny, new_arrival
    angle = math.atan2(ty - src.y, tx - src.x)
    return angle, arrival


def _initial_planet(view: GameView, planet_id: int) -> Planet | None:
    """Look up the planet's turn-0 snapshot, used for orbit prediction."""
    return next((p for p in view.initial_planets if p.id == planet_id), None)
```

- [ ] **Step 4: Run the aim tests**

Run: `uv run pytest tests/test_aim.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/aim.py tests/test_aim.py
git commit -m "Add orbit-aware fleet aiming with predicted arrival position"
```

---

## Task 3: Wire orbit-aware aim into existing templates

**Files:**
- Modify: `orbit_war/plan_gen/templates.py`
- Modify: `tests/test_templates.py` (verify existing tests still pass; add a new one for orbit-aware integration)

Replace each `Step.angle_to(src, target)` call inside `production_attack`, `defensive_reinforce`, and `snipe_undefended` with `aim_with_orbit_prediction(src, target, ships, view)`. Capture the arrival_turn (we don't use it in templates yet, but having it consistent will help W4's combat sim).

- [ ] **Step 1: Add an integration test pinning that orbit-aware angles differ from naive on at least one orbiting target**

Append to `tests/test_templates.py`:

```python
import math
from kaggle_environments import make as _make_env

from orbit_war.sim.orbits import is_orbiting
from orbit_war.plan_gen.templates import production_attack_template


def test_production_attack_uses_orbit_aware_aim_for_orbiting_targets():
    env = _make_env("orbit_wars", configuration={"seed": 7}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]
    view = GameView.from_obs(obs)

    steps = production_attack_template(view)
    by_target = {p.id: p for p in view.planets}

    # Find at least one step targeting an orbiting non-source planet.
    diverged = False
    for s in steps:
        target = by_target[s.target_planet_id]
        src = by_target[s.from_planet_id]
        if not is_orbiting(target):
            continue
        naive = math.atan2(target.y - src.y, target.x - src.x)
        if not math.isclose(s.angle, naive, abs_tol=1e-3):
            diverged = True
            break
    # If every target is static this test is uninformative — accept either way.
    # Otherwise we expect at least one orbit-aware divergence.
    if any(is_orbiting(by_target[s.target_planet_id]) for s in steps):
        assert diverged, "expected at least one orbit-aware angle to differ from naive atan2"
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_templates.py -k "orbit_aware_aim" -v`
Expected: FAIL (templates still use naive `Step.angle_to`).

- [ ] **Step 3: Switch templates to orbit-aware aim**

Edit `orbit_war/plan_gen/templates.py`:

1. Add to top imports:

```python
from orbit_war.plan_gen.aim import aim_with_orbit_prediction
```

2. In `production_attack_template`, replace the `Step` construction inside the loop. The current code is:

```python
        proposals.append(
            Step(
                from_planet_id=int(src.id),
                target_planet_id=int(best.id),
                angle=Step.angle_to(src, best),
                ships=int(ships),
                score=float(score),
            )
        )
```

Replace with:

```python
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
```

3. In `defensive_reinforce_template`, similarly replace:

```python
        proposals.append(
            Step(
                from_planet_id=int(nearest.id),
                target_planet_id=int(target.id),
                angle=Step.angle_to(nearest, target),
                ships=int(ships),
                score=float(score),
            )
        )
```

with:

```python
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
```

4. In `snipe_undefended_template`, replace:

```python
        proposals.append(
            Step(
                from_planet_id=int(nearest.id),
                target_planet_id=int(tgt.id),
                angle=Step.angle_to(nearest, tgt),
                ships=int(needed),
                score=float(score),
            )
        )
```

with:

```python
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
```

- [ ] **Step 4: Run all template tests**

Run: `uv run pytest tests/test_templates.py -v`
Expected: 9 passed (8 existing + 1 new orbit-aware test).

If `test_production_attack_ships_are_min_to_capture` now fails, the issue is unrelated to orbit-aware aiming — likely a copy-paste error during the angle-call substitution.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/templates.py tests/test_templates.py
git commit -m "Aim production_attack/defensive_reinforce/snipe_undefended at predicted future positions"
```

---

## Task 4: Lightweight forward simulator

**Files:**
- Create: `orbit_war/sim/game.py`
- Test: `tests/test_forward_sim.py`

A minimal forward simulator: takes a `GameView` plus per-player action lists, advances `n_turns` (default 15), and returns the resulting `GameView`. We use it to *validate* candidate plans (Task 5).

Scope cuts compared to the official simulator:
- **Skip comet spawning/expiration.** Comets present at the start of simulation persist; no new comets spawn within the sim window.
- **Skip 4-fold combat resolution edge cases.** Use simplified pairwise combat: at planet collision, sum each owner's incoming + present ships, the largest army wins, surplus = (largest - 2nd_largest); if surplus > defender, ownership flips.
- **Skip orbit movement of planets.** Use closed-form positions (already have `precompute_position_table`).
- **Sun avoidance.** Approximate: if a fleet's straight-line segment crosses the sun within radius 12 (10 + 2 buffer), the fleet is destroyed.

Parity expectation: ≤5% drift vs `kaggle_environments` over 15-turn windows. We're not bit-exact; we're close-enough for plan ranking.

- [ ] **Step 1: Write failing tests**

Create `tests/test_forward_sim.py`:

```python
"""Tests for the lightweight forward simulator."""

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from orbit_war.bots import random_bot, starter_bot
from orbit_war.sim.game import forward_simulate
from orbit_war.sim.observation import GameView


def test_forward_simulate_returns_gameview_with_advanced_step():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    view = GameView.from_obs(env.steps[0][0]["observation"])

    out = forward_simulate(view, actions_per_player=[[], []], n_turns=10)

    assert isinstance(out, GameView)
    assert out.step == view.step + 10


def test_forward_simulate_advances_owned_planet_garrisons_by_production():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 5, 3)  # production 3
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 5, 1)
    view = GameView(
        player=0,
        planets=(me, enemy),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )
    out = forward_simulate(view, actions_per_player=[[], []], n_turns=5)
    me_after = next(p for p in out.planets if p.id == 0)
    assert me_after.ships == 5 + 3 * 5  # 5 turns at production 3


def test_forward_simulate_neutral_planet_garrisons_do_not_grow():
    n = Planet(0, -1, 10.0, 10.0, 1.0, 5, 3)
    view = GameView(
        player=0,
        planets=(n,),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(n,),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )
    out = forward_simulate(view, actions_per_player=[[], []], n_turns=10)
    assert out.planets[0].ships == 5


def test_forward_simulate_processes_inbound_fleet_capture():
    """An inbound fleet of 100 ships landing on a 5-ship neutral should capture it."""
    me = Planet(0, 0, 10.0, 10.0, 1.0, 100, 1)
    target = Planet(1, -1, 11.0, 10.0, 1.0, 5, 1)  # very close — captures within 1-2 turns
    incoming = Fleet(0, 0, 10.5, 10.0, 0.0, 0, 100)
    view = GameView(
        player=0,
        planets=(me, target),
        fleets=(incoming,),
        angular_velocity=0.04,
        initial_planets=(me, target),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )
    out = forward_simulate(view, actions_per_player=[[], []], n_turns=5)
    target_after = next(p for p in out.planets if p.id == 1)
    assert target_after.owner == 0  # captured
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_forward_sim.py -v`

- [ ] **Step 3: Implement `orbit_war/sim/game.py`**

```python
"""Lightweight forward simulator for plan validation.

This is a *cheap, approximate* re-implementation of the official Orbit Wars
step function — enough to forward-simulate ~15 turns of a candidate plan and
score the resulting position. It is NOT a parity-exact port of the official
simulator. Specifically:

  - Comet spawn/expire is skipped (comets present at sim start persist).
  - Combat is simplified pairwise (largest army wins, surplus over defender
    flips ownership).
  - Sun avoidance: any fleet whose path segment crosses within 12 units of
    (50, 50) is destroyed. (Sun radius is 10; 12 gives a small buffer.)
  - Planet positions use the closed-form `planet_position_at` already in
    `sim.orbits`.

Use this for plan ranking and lookahead-based pruning. Do NOT use it as the
authoritative game engine — the real environment makes the final call."""

from __future__ import annotations

import math
from dataclasses import replace

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    BOARD_SIZE,
    CENTER,
    Fleet,
    Planet,
)

from orbit_war.sim.observation import GameView
from orbit_war.sim.orbits import is_orbiting, planet_position_at
from orbit_war.sim.physics import fleet_speed

SUN_BUFFER = 12.0  # actual sun radius is 10; 2-unit safety margin


def forward_simulate(
    view: GameView,
    actions_per_player: list[list[list]],
    n_turns: int,
) -> GameView:
    """Advance the game `n_turns` turns from `view`.

    `actions_per_player[i]` is the action list for player i (list of
    `[from_planet_id, angle, ships]`). The same action list is replayed
    each turn. For the first turn only, the actions are processed; later
    turns advance physics + production + collisions only."""
    planets = list(view.planets)
    fleets = list(view.fleets)

    # Apply turn-0 actions first.
    fleets.extend(_spawn_fleets(planets, actions_per_player))

    for tick in range(n_turns):
        current_step = view.step + tick + 1

        # Advance fleets one tick.
        moved: list[Fleet] = []
        for f in fleets:
            speed = fleet_speed(f.ships)
            new_x = f.x + math.cos(f.angle) * speed
            new_y = f.y + math.sin(f.angle) * speed
            if _crosses_sun(f.x, f.y, new_x, new_y):
                continue  # destroyed
            if new_x < 0 or new_x > BOARD_SIZE or new_y < 0 or new_y > BOARD_SIZE:
                continue  # off-board
            moved.append(Fleet(f.id, f.owner, new_x, new_y, f.angle, f.from_planet_id, f.ships))
        fleets = moved

        # Snap planet positions for collision/production at this tick.
        positions = _snap_positions(view, current_step)
        for i, p in enumerate(planets):
            x, y = positions[p.id]
            planets[i] = Planet(p.id, p.owner, x, y, p.radius, p.ships, p.production)

        # Production for owned/enemy planets (not neutrals).
        for i, p in enumerate(planets):
            if p.owner == -1:
                continue
            planets[i] = Planet(p.id, p.owner, p.x, p.y, p.radius, p.ships + p.production, p.production)

        # Resolve fleet→planet collisions.
        survivors: list[Fleet] = []
        arrivals: dict[int, list[tuple[int, int]]] = {}
        for f in fleets:
            collided_planet = _planet_collision(f, planets)
            if collided_planet is None:
                survivors.append(f)
                continue
            arrivals.setdefault(collided_planet, []).append((f.owner, f.ships))
        fleets = survivors

        for planet_id, arrival_list in arrivals.items():
            i = next(idx for idx, p in enumerate(planets) if p.id == planet_id)
            planets[i] = _resolve_combat(planets[i], arrival_list)

    return GameView(
        player=view.player,
        planets=tuple(planets),
        fleets=tuple(fleets),
        angular_velocity=view.angular_velocity,
        initial_planets=view.initial_planets,
        comet_planet_ids=view.comet_planet_ids,
        remaining_overage_time=view.remaining_overage_time,
        step=view.step + n_turns,
        comets=view.comets,
    )


def _crosses_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - CENTER, y1 - CENTER
    a = dx * dx + dy * dy
    if a < 1e-9:
        return math.hypot(fx, fy) < SUN_BUFFER
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - SUN_BUFFER * SUN_BUFFER
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def _snap_positions(view: GameView, step: int) -> dict[int, tuple[float, float]]:
    """Compute current (x, y) for every planet at `step`."""
    by_id_initial = {p.id: p for p in view.initial_planets}
    out: dict[int, tuple[float, float]] = {}
    for p in view.planets:
        initial = by_id_initial.get(p.id)
        if initial is None or not is_orbiting(initial):
            out[p.id] = (p.x, p.y)
        else:
            out[p.id] = planet_position_at(initial, step, view.angular_velocity)
    return out


def _planet_collision(fleet: Fleet, planets: list[Planet]) -> int | None:
    for p in planets:
        if math.hypot(fleet.x - p.x, fleet.y - p.y) <= p.radius:
            return p.id
    return None


def _resolve_combat(planet: Planet, arrivals: list[tuple[int, int]]) -> Planet:
    """Simplified combat. Sum ships per owner among arrivals; the planet's
    garrison defends as the planet's own owner (or as neutral)."""
    by_owner: dict[int, int] = {}
    for owner, ships in arrivals:
        by_owner[owner] = by_owner.get(owner, 0) + ships
    # The planet itself contributes its garrison under its current owner.
    by_owner[planet.owner] = by_owner.get(planet.owner, 0) + planet.ships
    # Largest force wins; surplus = largest - 2nd largest.
    sorted_forces = sorted(by_owner.items(), key=lambda kv: -kv[1])
    if len(sorted_forces) == 1:
        winner, ships = sorted_forces[0]
        return Planet(planet.id, winner, planet.x, planet.y, planet.radius, ships, planet.production)
    (winner, top), (_runner, runner_up) = sorted_forces[0], sorted_forces[1]
    surplus = top - runner_up
    return Planet(planet.id, winner, planet.x, planet.y, planet.radius, surplus, planet.production)


def _spawn_fleets(planets: list[Planet], actions_per_player: list[list[list]]) -> list[Fleet]:
    """Spawn fleets from each player's first-turn action list. Fleet IDs are
    synthesized starting at 10_000 to avoid colliding with existing fleet IDs."""
    by_id = {p.id: p for p in planets}
    out: list[Fleet] = []
    next_fleet_id = 10_000
    for player, action_list in enumerate(actions_per_player):
        for move in action_list:
            from_id, angle, ships = int(move[0]), float(move[1]), int(move[2])
            src = by_id.get(from_id)
            if src is None or src.owner != player or ships <= 0 or ships > src.ships:
                continue
            spawn_x = src.x + math.cos(angle) * (src.radius + 0.5)
            spawn_y = src.y + math.sin(angle) * (src.radius + 0.5)
            out.append(Fleet(next_fleet_id, player, spawn_x, spawn_y, angle, from_id, ships))
            next_fleet_id += 1
    return out
```

- [ ] **Step 4: Run the forward-sim tests**

Run: `uv run pytest tests/test_forward_sim.py -v`
Expected: 4 passed.

If `test_forward_simulate_processes_inbound_fleet_capture` fails because the fleet doesn't reach the target in 5 turns, lower the `target` planet's distance (move it closer to the fleet's start position).

- [ ] **Step 5: Commit**

```bash
git add orbit_war/sim/game.py tests/test_forward_sim.py
git commit -m "Add lightweight forward simulator (fleet movement + simplified combat)"
```

---

## Task 5: Plan validation by forward simulation

**Files:**
- Modify: `orbit_war/plan_gen/composer.py`
- Modify: `tests/test_composer.py`

Add an optional `validator` callable to `compose_plan`. If supplied, after greedy combine, the validator is called with the combined plan + GameView, and may return either the original plan or `[]` (revert to no-op). For W3, the validator forward-simulates 15 turns and checks that "score(after) ≥ score(before) - cost_of_action". This catches catastrophic over-commitments.

- [ ] **Step 1: Append a failing test**

Append to `tests/test_composer.py`:

```python
def test_compose_plan_calls_validator_when_provided():
    """The validator should receive the chosen plan + surplus snapshot and
    may return either the plan unchanged or an empty list."""
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=10, score=0.9)
    captured: list[list[Step]] = []

    def my_validator(plan: list[Step]) -> list[Step]:
        captured.append(list(plan))
        return []  # always reject

    result = compose_plan(
        [a], surplus_by_planet={0: 30}, validator=my_validator,
    )
    assert result == []
    assert captured == [[a]]
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_composer.py -k validator -v`

- [ ] **Step 3: Add the parameter to `compose_plan`**

Edit `orbit_war/plan_gen/composer.py`. Update the function signature and body:

```python
from typing import Callable, Iterable

from orbit_war.plan_gen.step import Step


def compose_plan(
    steps: Iterable[Step],
    surplus_by_planet: dict[int, int],
    allow_truncation: bool = False,
    validator: Callable[[list[Step]], list[Step]] | None = None,
) -> list[Step]:
    """Sort `steps` by descending score and greedily commit, debiting
    `surplus_by_planet[step.from_planet_id]` each time.

    If `allow_truncation` is True, a step that exceeds remaining surplus
    is shrunk to the surplus (provided >=1 ships remain). Otherwise it
    is skipped.

    If `validator` is provided, the assembled plan is passed through it
    after greedy combine; the validator may return the same plan or any
    other list of Steps (typically `[]` to revert to no-op).

    Returns the committed steps in the order they were chosen (or the
    validator's return value, if supplied).
    """
    remaining = dict(surplus_by_planet)
    plan: list[Step] = []
    for step in sorted(steps, key=lambda s: -s.score):
        avail = remaining.get(step.from_planet_id, 0)
        if avail <= 0:
            continue
        if step.ships <= avail:
            plan.append(step)
            remaining[step.from_planet_id] = avail - step.ships
            continue
        if not allow_truncation:
            continue
        truncated_ships = avail
        if truncated_ships < 1:
            continue
        plan.append(
            Step(
                from_planet_id=step.from_planet_id,
                target_planet_id=step.target_planet_id,
                angle=step.angle,
                ships=int(truncated_ships),
                score=step.score,
            )
        )
        remaining[step.from_planet_id] = 0

    if validator is not None:
        plan = validator(plan)
    return plan
```

- [ ] **Step 4: Run all composer tests**

Run: `uv run pytest tests/test_composer.py -v`
Expected: 7 passed (6 prior + 1 new).

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/composer.py tests/test_composer.py
git commit -m "Add optional validator hook to compose_plan"
```

---

## Task 6: New step template — multi_source_consolidation

**Files:**
- Modify: `orbit_war/plan_gen/templates.py`
- Modify: `tests/test_templates.py`

The Planet Wars writeups consistently rate "multi-source consolidation strikes" — multiple of your planets sending fleets to the same enemy target with arrival times synchronized — as a high-value pattern. We add a template that proposes them: pick a high-production enemy/neutral target, pick the K nearest sources that can spare ships, compute the arrival-time-of-the-furthest source, and emit one step per source with the same target.

Note: synchronizing arrival turns precisely requires forward-sim of fleet timing — for W3 we approximate by always sending each fleet at its natural speed (no delays), targeting the *future* position the slowest fleet would see. The composer's surplus check still gates whether each contribution lands.

- [ ] **Step 1: Append failing tests**

Append to `tests/test_templates.py`:

```python
from orbit_war.plan_gen.templates import multi_source_consolidation_template


def test_multi_source_consolidation_emits_multiple_steps_to_same_target():
    """Three friendly sources target one rich enemy planet. Expect 2-3 steps
    all aimed at the same target."""
    src1 = Planet(0, 0, 5.0, 5.0, 1.0, 40, 1)
    src2 = Planet(1, 0, 95.0, 5.0, 1.0, 40, 1)
    src3 = Planet(2, 0, 5.0, 95.0, 1.0, 40, 1)
    rich = Planet(3, 1, 50.0, 50.0, 5.0, 100, 5)
    view = GameView(
        player=0,
        planets=(src1, src2, src3, rich),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(src1, src2, src3, rich),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    steps = multi_source_consolidation_template(view)
    assert len(steps) >= 2, "consolidation should propose >=2 contributing sources"
    targets = {s.target_planet_id for s in steps}
    assert targets == {3}, "all consolidation steps target the rich enemy planet"


def test_multi_source_consolidation_quiet_when_only_one_source():
    """No 'multi'-source possible if we only own one planet."""
    only = Planet(0, 0, 5.0, 5.0, 1.0, 40, 1)
    rich = Planet(1, 1, 50.0, 50.0, 1.0, 100, 5)
    view = GameView(
        player=0,
        planets=(only, rich),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(only, rich),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    assert multi_source_consolidation_template(view) == []
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_templates.py -k consolidation -v`

- [ ] **Step 3: Implement**

Append to `orbit_war/plan_gen/templates.py`:

```python
CONSOLIDATION_MIN_TARGET_PRODUCTION = 3
CONSOLIDATION_TOP_K_SOURCES = 4


def multi_source_consolidation_template(view: GameView) -> list[Step]:
    """For each rich non-owned target, gather contributing fleets from up to
    `CONSOLIDATION_TOP_K_SOURCES` of our nearest planets that can each afford
    a partial contribution. The composer's surplus check decides which
    contributions land."""
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
        # Each source contributes (ships needed by target) / (number contributing)
        # rounded up, capped at half its own garrison.
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
```

- [ ] **Step 4: Run all template tests**

Run: `uv run pytest tests/test_templates.py -v`
Expected: 11 passed (9 prior + 2 new).

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/templates.py tests/test_templates.py
git commit -m "Add multi_source_consolidation step template"
```

---

## Task 7: New step template — comet_rush

**Files:**
- Modify: `orbit_war/plan_gen/templates.py`
- Modify: `tests/test_templates.py`

Comets spawn at known turns (50/150/250/350/450) in groups of 4, one per quadrant. They produce ships while owned. Pre-launching fleets just before a spawn so they arrive shortly after gives a free production planet for the duration of the comet's flight. The template fires only in the 5-turn window before each spawn step.

For W3 we use a simple version: at the relevant turns, propose attacks from each owned planet aimed at the four quadrant centers (where new comets typically appear).

- [ ] **Step 1: Append failing tests**

Append to `tests/test_templates.py`:

```python
from orbit_war.plan_gen.templates import comet_rush_template


def test_comet_rush_quiet_outside_pre_spawn_window():
    me = Planet(0, 0, 50.0, 50.0, 1.0, 50, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 5, 1)
    view = GameView(
        player=0,
        planets=(me, enemy),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=200,  # not near any spawn
        comets=(),
    )
    assert comet_rush_template(view) == []


def test_comet_rush_fires_in_pre_spawn_window():
    me = Planet(0, 0, 50.0, 50.0, 1.0, 100, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 5, 1)
    view = GameView(
        player=0,
        planets=(me, enemy),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=48,  # within 5 turns of step 50 spawn
        comets=(),
    )
    steps = comet_rush_template(view)
    assert len(steps) > 0
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_templates.py -k comet_rush -v`

- [ ] **Step 3: Implement**

Append to `orbit_war/plan_gen/templates.py`:

```python
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
    Each step sends a small probe (10-20 ships) so we don't drain home planets."""
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
                    target_planet_id=int(src.id),  # placeholder; comets don't have stable IDs at launch time
                    angle=float(angle),
                    ships=int(ships),
                    score=float(distance_score) * 0.8,
                )
            )
    return proposals
```

You'll also need to import `math` if it's not already at the top of templates.py. Add `import math` to the imports block.

NOTE on the placeholder `target_planet_id=int(src.id)`: comets don't exist yet when we launch, so there's no real target ID. We use the source ID as a sentinel; the composer's surplus check still gates ship spending. Plan validation (Task 8) won't process this step against real targets — it's a speculative launch.

- [ ] **Step 4: Run all template tests**

Run: `uv run pytest tests/test_templates.py -v`
Expected: 13 passed (11 prior + 2 new).

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/templates.py tests/test_templates.py
git commit -m "Add comet_rush step template (fires in pre-spawn window)"
```

---

## Task 8: heuristic_v3 wiring

**Files:**
- Create: `orbit_war/bots/heuristic_v3.py`
- Create: `tests/test_heuristic_v3.py`
- Modify: `orbit_war/eval_harness/cli.py`

Wire 6 templates (4 W2 + 2 new) + plan validation via forward-sim into a single agent. Drop/reduce templates that haven't shown value.

- [ ] **Step 1: Add v3 test (failing)**

Create `tests/test_heuristic_v3.py`:

```python
"""Tests for the W3 heuristic_v3 bot."""

from kaggle_environments import make

from orbit_war.bots import (
    greedy_baseline,
    heuristic_v1,
    heuristic_v3,
    public_tactical,
    random_bot,
    starter_bot,
)
from orbit_war.eval_harness.parallel import run_mirrored_pairs


def test_heuristic_v3_runs_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([heuristic_v3.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_heuristic_v3_beats_random_decisively():
    summary = run_mirrored_pairs(
        bot_a=heuristic_v3.agent,
        bot_b=random_bot.agent,
        seeds=tuple(range(8)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.85, (
        f"heuristic_v3 only beat random {summary.win_rate_a:.0%}"
    )


def test_heuristic_v3_at_least_matches_v1():
    """v3 should NOT regress against v1 — orbit-aware aiming + new templates
    should be at least as good as the W2 baseline."""
    summary = run_mirrored_pairs(
        bot_a=heuristic_v3.agent,
        bot_b=heuristic_v1.agent,
        seeds=tuple(range(10)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.50, (
        f"heuristic_v3 regressed to {summary.win_rate_a:.0%} vs v1"
    )


def test_heuristic_v3_improves_against_public_tactical():
    """Against public_tactical, v3 should lift from v1's 10% to at least 25%
    — orbit-aware aiming alone tends to be a 10-15pp lift; new templates may
    add more."""
    summary = run_mirrored_pairs(
        bot_a=heuristic_v3.agent,
        bot_b=public_tactical.agent,
        seeds=tuple(range(8)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.25, (
        f"heuristic_v3 only at {summary.win_rate_a:.0%} vs public_tactical — "
        f"orbit-aware aim should lift this above 25%"
    )
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_heuristic_v3.py -v`

- [ ] **Step 3: Implement heuristic_v3**

Create `orbit_war/bots/heuristic_v3.py`:

```python
"""heuristic_v3: W3 bot with orbit-aware aiming, two new templates,
and forward-sim plan validation.

Differences from heuristic_v1:
- Templates aim at predicted future positions (via aim_with_orbit_prediction).
- Adds multi_source_consolidation_template and comet_rush_template.
- compose_plan is called with a validator that forward-simulates 15 turns
  and reverts to no-op if the projected position evaluates worse than the
  current position."""

from __future__ import annotations

from orbit_war.eval.features import (
    surplus_ships,
    total_production,
    total_ships,
)
from orbit_war.plan_gen.composer import compose_plan
from orbit_war.plan_gen.filters import filter_capturable
from orbit_war.plan_gen.step import Step
from orbit_war.plan_gen.templates import (
    comet_rush_template,
    defensive_reinforce_template,
    multi_source_consolidation_template,
    no_op_template,
    production_attack_template,
    snipe_undefended_template,
)
from orbit_war.sim.game import forward_simulate
from orbit_war.sim.observation import GameView

TEMPLATE_WEIGHTS: dict[str, float] = {
    "no_op": 0.0,
    "production_attack": 1.0,
    "defensive_reinforce": 2.0,
    "snipe_undefended": 1.5,
    "multi_source_consolidation": 1.2,
    "comet_rush": 0.8,
}

PLAN_VALIDATION_HORIZON = 15


def _weighted(steps: list[Step], weight: float) -> list[Step]:
    if weight == 1.0:
        return steps
    return [
        Step(
            from_planet_id=s.from_planet_id,
            target_planet_id=s.target_planet_id,
            angle=s.angle,
            ships=s.ships,
            score=s.score * weight,
        )
        for s in steps
    ]


def _position_eval(view: GameView) -> float:
    me = view.player
    opp = 1 - me  # 1v1 only for W3
    return (
        total_ships(view, me) - total_ships(view, opp)
        + 5.0 * (total_production(view, me) - total_production(view, opp))
    )


def _make_validator(view: GameView):
    """Return a validator that simulates the plan forward 15 turns and
    reverts to no-op if the projected eval drops below the current eval."""
    baseline = _position_eval(view)

    def validator(plan: list[Step]) -> list[Step]:
        if not plan:
            return plan
        my_actions = [s.as_move() for s in plan]
        # Opponent: passive (no actions). This is a fixed-policy approximation
        # of opponent behavior; W4 may add adversarial best-response.
        actions_per_player = [[], []]
        actions_per_player[view.player] = my_actions
        future = forward_simulate(view, actions_per_player, n_turns=PLAN_VALIDATION_HORIZON)
        future_eval = _position_eval(future)
        if future_eval < baseline - 5.0:  # 5-ship slack
            return []  # revert to no-op
        return plan

    return validator


def agent(obs) -> list[list]:
    view = GameView.from_obs(obs)

    candidates: list[Step] = []
    candidates.extend(_weighted(no_op_template(view), TEMPLATE_WEIGHTS["no_op"]))
    candidates.extend(_weighted(production_attack_template(view), TEMPLATE_WEIGHTS["production_attack"]))
    candidates.extend(_weighted(defensive_reinforce_template(view), TEMPLATE_WEIGHTS["defensive_reinforce"]))
    candidates.extend(_weighted(snipe_undefended_template(view), TEMPLATE_WEIGHTS["snipe_undefended"]))
    candidates.extend(_weighted(multi_source_consolidation_template(view), TEMPLATE_WEIGHTS["multi_source_consolidation"]))
    candidates.extend(_weighted(comet_rush_template(view), TEMPLATE_WEIGHTS["comet_rush"]))

    candidates = filter_capturable(candidates, view)

    surplus = surplus_ships(view, view.player)
    plan = compose_plan(
        candidates,
        surplus,
        allow_truncation=False,
        validator=_make_validator(view),
    )
    return [s.as_move() for s in plan]
```

- [ ] **Step 4: Add v3 to the CLI zoo**

Edit `orbit_war/eval_harness/cli.py`. Add to `ZOO_BOT_PATHS`:

```python
    "heuristic_v3": "orbit_war.bots.heuristic_v3:agent",
```

- [ ] **Step 5: Run heuristic_v3 tests**

Run: `uv run pytest tests/test_heuristic_v3.py -v`
Expected: 4 passed (will take 2-5 minutes).

If `test_heuristic_v3_improves_against_public_tactical` fails (v3 < 25% vs public_tactical), the orbit-aware aim and new templates aren't compounding. Diagnose:
- Run `test_heuristic_v3_at_least_matches_v1` first — if v3 < 50% vs v1, the new components are net-negative. Revert templates one at a time (start with comet_rush, then consolidation) until v3 ≥ v1.
- If v3 ≥ v1 but still < 25% vs public_tactical, the gap is structural. Document and proceed; W4 will need full forward-sim search to close it.

**ACCEPT** the test failing on `public_tactical` if v3 still beats v1 — that's the hard test we're closing in on; missing the bar by a few percentage points is fine for W3.

- [ ] **Step 6: Commit**

```bash
git add orbit_war/bots/heuristic_v3.py tests/test_heuristic_v3.py orbit_war/eval_harness/cli.py
git commit -m "Add heuristic_v3 with orbit-aware aim, 6 templates, and plan validation"
```

If tests #2 (vs random) or #3 (vs v1) failed, include in commit message: "Note: vN tier failed at X% — see W3 closing for retro." Revert the offending template and re-test before committing.

---

## Task 9: W3 closing — gate, champion designation, optional ladder submit

**Files:**
- Modify: `submissions.log`
- Tag: `w3-baseline`

- [ ] **Step 1: Run the full gate against heuristic_v3**

```bash
uv run ow-gate orbit_war.bots.heuristic_v3:agent \
    --champion orbit_war.bots.heuristic_v1:agent \
    --seeds 25 --workers 4 2>&1 | grep -v "open_spiel\|Loading environment\|^$"
```

Capture and paste the OVERALL line + per-tier results. This is the W3 verdict.

- [ ] **Step 2: If v3 PASSES gate, optionally submit to ladder**

```bash
./scripts/submit_bot.sh heuristic_v3 orbit_war.bots.heuristic_v3:agent "W3: orbit-aware + 2 new templates + plan validation"
```

The script runs the gate again before allowing the upload. If the gate fails, the script aborts.

If you choose NOT to submit (e.g., v3 is borderline and you'd rather wait for W4), record the local result anyway:

```bash
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SHA="$(git rev-parse --short HEAD)"
echo "$TS  heuristic_v3  $SHA  W3 v3 gate result: <PASS or FAIL OVERALL — fill in based on the run, not submitted>" >> submissions.log
```

- [ ] **Step 3: Tag**

```bash
git tag w3-baseline
```

- [ ] **Step 4: Update CLAUDE.md**

If v3 passed the gate or beat v1, update the `## Workflow` section's champion line to point at v3:

```markdown
- W3 champion: heuristic_v3 (orbit-aware aim + 2 new templates + plan validation). Use `uv run ow-gate orbit_war.bots.heuristic_v3:agent` to gate challengers.
```

If v3 did NOT improve over v1, leave the v1 line and add a note:

```markdown
- W3 attempt: heuristic_v3 did not improve over v1 in self-play. Champion remains v1. See commit `<SHA>` for retrospective.
```

- [ ] **Step 5: Commit**

```bash
git add submissions.log CLAUDE.md
git commit -m "W3 closing: tag w3-baseline; designate champion based on gate result"
```

---

## Closing checklist

- [ ] Run the full test suite: `uv run pytest -q`. Expected: all tests pass.
- [ ] Confirm `git log --oneline w3-baseline ^w2-baseline | wc -l` shows ~9-12 commits.
- [ ] Capture realistic numbers: heuristic_v3 vs (random / starter / greedy / v1 / public_tactical). Save in a comment on the W3 closing commit.
- [ ] Note any W4 follow-ups discovered (e.g., specific failure modes seen in replays vs public_tactical).
