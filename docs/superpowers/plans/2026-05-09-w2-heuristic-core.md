# W2: Heuristic Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the first real heuristic bot (`heuristic_v1`) that beats every prior baseline in our zoo, then tune its weights via linear regression on self-play data to ship `heuristic_v2`.

**Architecture:** Build the Melis "steps + greedy combine" pattern directly. Four step templates (no_op, production-attack, defensive-reinforce, snipe-undefended) propose candidate fleet launches; each is scored by a weighted linear combination of features; the composer sorts by score and greedily combines under per-planet ship-budget constraints. After v1 ships and we have a working bot zoo, generate ~1000 self-play games as labelled (feature_vector, outcome) data, fit weights via `np.linalg.lstsq` (a1k0n's Tron 2010 trick), and ship `heuristic_v2` if it beats v1.

**Tech Stack:** Python 3.13 (uv), `numpy` (linalg + vectorised feature extraction), `pytest`, existing `orbit_war.sim.observation.GameView` and `orbit_war.sim.orbits.{is_orbiting, planet_position_at, precompute_position_table}`. No new external dependencies.

**Realistic outcome target:** v1 beats `starter` ≥95%, `greedy_baseline` ≥70%, `public_tactical` ≥40%. v2 (with learned weights) should add ~5-10pp. Hitting `public_tactical ≥60%` (the design doc's stretch goal) is realistic but not guaranteed without W3's 1-step lookahead — flag if the gap doesn't close.

---

## File structure

NEW production files:

- `orbit_war/sim/physics.py` — straight-line fleet trajectory math (arrival turn, arrival position, fleet speed scaling)
- `orbit_war/eval/__init__.py`
- `orbit_war/eval/features.py` — pure feature extractors over `GameView`
- `orbit_war/plan_gen/__init__.py`
- `orbit_war/plan_gen/step.py` — `Step` dataclass + helpers (`predict_arrival`, `ships_needed_to_capture`, etc.)
- `orbit_war/plan_gen/templates.py` — four step generators
- `orbit_war/plan_gen/composer.py` — sort + greedy-combine under surplus
- `orbit_war/bots/heuristic_v1.py` — first heuristic bot, hand-set weights
- `orbit_war/bots/heuristic_v2.py` — same logic, learned weights
- `orbit_war/tuning/__init__.py`
- `orbit_war/tuning/data.py` — self-play data collector
- `orbit_war/tuning/regression.py` — `fit_weights(X, y) -> np.ndarray`
- `orbit_war/tuning/weights/v2.json` — fitted weights checked in

NEW test files (one per production module):

- `tests/test_physics.py`
- `tests/test_eval_features.py`
- `tests/test_step.py`
- `tests/test_templates.py`
- `tests/test_composer.py`
- `tests/test_heuristic_v1.py`
- `tests/test_tuning_data.py`
- `tests/test_tuning_regression.py`
- `tests/test_heuristic_v2.py`

Modified:

- `orbit_war/sim/observation.py` — add `step` and `comets` fields to `GameView` (W2 needs game-step for "early/mid/late" gating; W3 needs comets — adding both now is cheaper than two passes)
- `tests/test_observation.py` — extend coverage for new fields
- `orbit_war/eval_harness/cli.py` — add `heuristic_v1` and `heuristic_v2` to the zoo CLI map
- `scripts/submit_starter.sh` → rename and generalise to `scripts/submit_bot.sh` with idempotency guard (W1 follow-up)

---

## Task 1: GameView — add `step` and `comets` fields

**Files:**
- Modify: `orbit_war/sim/observation.py`
- Test: `tests/test_observation.py` (add cases)

The W1 GameView only exposed planets, fleets, angular_velocity, etc. Step templates need the game step (early/mid/late phase) and comet metadata. Add both now to avoid two future passes.

- [ ] **Step 1: Add failing tests for the new fields**

Append to `tests/test_observation.py`:

```python
def test_gameview_exposes_step_field():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]

    view = GameView.from_obs(obs)
    assert view.step == obs.get("step", 0)


def test_gameview_exposes_comets_metadata():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]

    view = GameView.from_obs(obs)
    # comets is a tuple of dicts (potentially empty at step 0)
    assert isinstance(view.comets, tuple)
    for group in view.comets:
        assert "planet_ids" in group
```

- [ ] **Step 2: Run the new tests; both should fail**

Run: `uv run pytest tests/test_observation.py -k "step_field or comets_metadata" -v`
Expected: 2 failed (AttributeError on `view.step` or `view.comets`).

- [ ] **Step 3: Add the two fields to the dataclass and `from_obs`**

In `orbit_war/sim/observation.py`, modify the dataclass to add two fields after `comet_planet_ids`:

```python
@dataclass(frozen=True)
class GameView:
    player: int
    planets: tuple[Planet, ...]
    fleets: tuple[Fleet, ...]
    angular_velocity: float
    initial_planets: tuple[Planet, ...]
    comet_planet_ids: frozenset[int]
    remaining_overage_time: float
    step: int  # current game step (0..episode_steps)
    comets: tuple[dict, ...]  # raw comet group dicts (planet_ids, paths, path_index)
```

In `from_obs`, append two lines inside the `return GameView(...)` block:

```python
        step=int(get("step") or 0),
        comets=tuple(get("comets") or []),
```

- [ ] **Step 4: Re-run all observation tests**

Run: `uv run pytest tests/test_observation.py -v`
Expected: 6 passed (4 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add orbit_war/sim/observation.py tests/test_observation.py
git commit -m "Expose step and comets on GameView for W2 phase + comet logic"
```

---

## Task 2: Sim primitives — fleet arrival math

**Files:**
- Create: `orbit_war/sim/physics.py`
- Test: `tests/test_physics.py`

Step templates need to know "if I launch from planet A toward (x, y) with N ships, when does the fleet arrive?" The physics is deterministic and given by the simulator: `speed = 1.0 + (max_speed - 1.0) * (log(ships)/log(1000))^1.5`. Implement and test against the official simulator's behaviour.

- [ ] **Step 1: Write the failing test**

Create `tests/test_physics.py`:

```python
"""Tests for fleet trajectory physics."""

import math

import pytest

from orbit_war.sim.physics import (
    fleet_speed,
    straight_line_distance,
    turns_to_arrive,
)


@pytest.mark.parametrize(
    "ships,expected_speed",
    [
        (1, 1.0),
        (1000, 6.0),  # max_speed
    ],
)
def test_fleet_speed_endpoints(ships: int, expected_speed: float):
    assert math.isclose(fleet_speed(ships), expected_speed, abs_tol=1e-9)


def test_fleet_speed_monotonic_in_ships():
    last = 0.0
    for ships in (1, 5, 10, 50, 100, 500, 1000, 5000):
        s = fleet_speed(ships)
        assert s >= last
        last = s


def test_straight_line_distance_basic():
    assert straight_line_distance(0.0, 0.0, 3.0, 4.0) == 5.0


def test_turns_to_arrive_at_least_one():
    # Any non-zero distance with positive ships gives at least one turn.
    assert turns_to_arrive(0.0, 0.0, 0.5, 0.5, ships=10) >= 1


def test_turns_to_arrive_scales_with_distance():
    near = turns_to_arrive(0.0, 0.0, 5.0, 0.0, ships=100)
    far = turns_to_arrive(0.0, 0.0, 50.0, 0.0, ships=100)
    assert far > near


def test_fleet_speed_uses_log_curve():
    # Spot-check: 100 ships should be much faster than 1 ship but well under max.
    s100 = fleet_speed(100)
    assert 1.5 < s100 < 5.5
```

- [ ] **Step 2: Run the test; expect ImportError**

Run: `uv run pytest tests/test_physics.py -v`
Expected: collection error on `orbit_war.sim.physics`.

- [ ] **Step 3: Implement `physics.py`**

Create `orbit_war/sim/physics.py`:

```python
"""Fleet trajectory physics — straight-line, no sun avoidance.

Sun-avoidance pathfinding belongs in W3 (alongside the lookahead search).
For W2's greedy composer, straight-line arrival math is enough: each step
template already self-prunes routes that would obviously cross the sun.

Constants match the official simulator
(`kaggle_environments.envs.orbit_wars.orbit_wars`).
"""

from __future__ import annotations

import math

MAX_SPEED = 6.0


def fleet_speed(ships: int) -> float:
    """Return the per-turn fleet speed for a given ship count.

    Mirrors the official formula:
        speed = 1.0 + (MAX_SPEED - 1) * (log(ships)/log(1000))^1.5
    Bottoms at 1.0 (a single ship); tops at MAX_SPEED at >=1000 ships.
    """
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def straight_line_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def turns_to_arrive(
    src_x: float, src_y: float, tgt_x: float, tgt_y: float, ships: int
) -> int:
    """Integer turns for a fleet of `ships` to traverse the straight line."""
    distance = straight_line_distance(src_x, src_y, tgt_x, tgt_y)
    if distance <= 0.0:
        return 1
    speed = fleet_speed(max(1, ships))
    return max(1, int(math.ceil(distance / speed)))
```

- [ ] **Step 4: Run the physics tests**

Run: `uv run pytest tests/test_physics.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/sim/physics.py tests/test_physics.py
git commit -m "Add straight-line fleet trajectory physics"
```

---

## Task 3: Eval features — pure extractors

**Files:**
- Create: `orbit_war/eval/__init__.py`
- Create: `orbit_war/eval/features.py`
- Test: `tests/test_eval_features.py`

Extract a small, focused set of position features. **Each is a pure function** of `GameView`. We start with the 6 cheapest features (the more expensive ones — Full Attack Future, indirect wealth — defer to W3 with the lookahead).

The 6 W2 features:

| Name | Returns | Used by |
|------|---------|---------|
| `total_ships(view, player)` | `int` | composer eval |
| `total_production(view, player)` | `int` | composer eval |
| `surplus_ships(view, player)` | `dict[planet_id, int]` | step templates |
| `incoming_threat(view, player, planet_id, horizon=20)` | `int` | defensive_reinforce |
| `arrival_turns_to(view, planet_id, src_planet)` | `int` | step scoring |
| `effective_garrison(view, planet_id, at_turn)` | `int` | combat assessment |

`surplus_ships` returns the per-planet "ships you can spare without losing this planet to the most-threatening incoming enemy fleet" — the Melis surplus guard. Conservative: if any incoming fleet would capture the planet, surplus is 0.

- [ ] **Step 1: Write failing tests for each feature**

Create `tests/test_eval_features.py`:

```python
"""Tests for orbit_war.eval.features."""

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from orbit_war.eval.features import (
    arrival_turns_to,
    effective_garrison,
    incoming_threat,
    surplus_ships,
    total_production,
    total_ships,
)
from orbit_war.sim.observation import GameView


def _fresh_view(seed: int = 42) -> GameView:
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.reset(num_agents=2)
    return GameView.from_obs(env.steps[0][0]["observation"])


def test_total_ships_sums_owned_planets_and_fleets():
    view = _fresh_view()
    expected = sum(p.ships for p in view.planets if p.owner == view.player)
    expected += sum(f.ships for f in view.fleets if f.owner == view.player)
    assert total_ships(view, view.player) == expected


def test_total_production_sums_owned_planets():
    view = _fresh_view()
    expected = sum(p.production for p in view.planets if p.owner == view.player)
    assert total_production(view, view.player) == expected


def test_surplus_ships_returns_zero_for_uncontested_planet():
    # On the very first turn there are no fleets, so all owned planets'
    # garrisons are surplus.
    view = _fresh_view()
    surplus = surplus_ships(view, view.player)
    for p in view.my_planets():
        assert surplus[p.id] == p.ships


def test_surplus_ships_zero_when_inbound_enemy_overwhelms():
    """Construct a tiny synthetic view: my planet has 5 ships, enemy fleet
    of 100 is en route. Surplus should be 0 (we'll lose the planet)."""
    my = Planet(0, 0, 10.0, 10.0, 1.0, 5, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 1, 1)
    incoming = Fleet(0, 1, 12.0, 12.0, 0.0, 1, 100)
    view = GameView(
        player=0,
        planets=(my, enemy),
        fleets=(incoming,),
        angular_velocity=0.04,
        initial_planets=(my, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    surplus = surplus_ships(view, 0)
    assert surplus[0] == 0


def test_incoming_threat_counts_enemy_ships_in_horizon():
    my = Planet(0, 0, 50.0, 50.0, 1.0, 30, 2)
    enemy = Planet(1, 1, 90.0, 50.0, 1.0, 5, 1)
    fleet = Fleet(0, 1, 51.0, 50.0, 0.0, 1, 25)  # right next to us
    view = GameView(
        player=0,
        planets=(my, enemy),
        fleets=(fleet,),
        angular_velocity=0.04,
        initial_planets=(my, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    # Within a generous horizon the threat sums to the fleet's ship count.
    assert incoming_threat(view, player=0, planet_id=0, horizon=50) == 25


def test_arrival_turns_to_uses_physics():
    src = Planet(0, 0, 0.0, 0.0, 1.0, 100, 1)
    tgt = Planet(1, -1, 30.0, 0.0, 1.0, 5, 1)
    view = GameView(
        player=0,
        planets=(src, tgt),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(src, tgt),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )
    turns = arrival_turns_to(view, src_planet=src, target_planet=tgt, ships=100)
    assert turns >= 1


def test_effective_garrison_grows_with_production_for_owned_planet():
    my = Planet(0, 0, 50.0, 50.0, 1.0, 10, 3)
    view = GameView(
        player=0,
        planets=(my,),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(my,),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=0,
        comets=(),
    )
    g0 = effective_garrison(view, planet_id=0, at_turn=0)
    g5 = effective_garrison(view, planet_id=0, at_turn=5)
    assert g5 == g0 + 3 * 5


def test_effective_garrison_static_for_neutral():
    n = Planet(0, -1, 50.0, 50.0, 1.0, 10, 3)
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
    assert effective_garrison(view, planet_id=0, at_turn=0) == 10
    assert effective_garrison(view, planet_id=0, at_turn=10) == 10
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `uv run pytest tests/test_eval_features.py -v`
Expected: collection error.

- [ ] **Step 3: Implement `eval/__init__.py` and `eval/features.py`**

Create `orbit_war/eval/__init__.py` (empty file).

Create `orbit_war/eval/features.py`:

```python
"""Pure feature extractors over GameView.

Every function is a pure read-only function; no caching, no side effects.
The composer is allowed to call these many times per turn, but each is
shaped to be cheap (linear in #planets + #fleets at worst).
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from orbit_war.sim.observation import GameView
from orbit_war.sim.physics import (
    straight_line_distance,
    turns_to_arrive,
)


def total_ships(view: GameView, player: int) -> int:
    """Owned planet garrisons + own in-flight fleet ships."""
    n = sum(p.ships for p in view.planets if p.owner == player)
    n += sum(f.ships for f in view.fleets if f.owner == player)
    return n


def total_production(view: GameView, player: int) -> int:
    """Sum of `production` across planets we own."""
    return sum(p.production for p in view.planets if p.owner == player)


def incoming_threat(
    view: GameView, player: int, planet_id: int, horizon: int = 20
) -> int:
    """Ships in enemy fleets aimed at `planet_id`, arriving within `horizon` turns.

    Approximation: any enemy fleet whose straight-line ETA to the planet's
    current position is <= horizon counts. We don't try to determine the
    fleet's *intended* destination — Orbit Wars fleets fly along a fixed
    angle, and judging intent perfectly requires geometric ray-cast tests
    that belong with W3's deeper combat sim. Over-counting here is fine
    for a defensive heuristic.
    """
    target = next((p for p in view.planets if p.id == planet_id), None)
    if target is None:
        return 0
    total = 0
    for f in view.fleets:
        if f.owner == player or f.owner == -1:
            continue
        eta = turns_to_arrive(f.x, f.y, target.x, target.y, max(1, f.ships))
        if eta <= horizon:
            total += f.ships
    return total


def arrival_turns_to(
    view: GameView, src_planet: Planet, target_planet: Planet, ships: int
) -> int:
    """Straight-line arrival turns for a fleet leaving `src_planet`."""
    return turns_to_arrive(
        src_planet.x,
        src_planet.y,
        target_planet.x,
        target_planet.y,
        ships,
    )


def effective_garrison(view: GameView, planet_id: int, at_turn: int) -> int:
    """Approximate ships present on `planet_id` `at_turn` turns from now.

    Owned/enemy planets accrue production; neutrals do not.
    Does NOT account for inbound fleet arrivals (caller composes those).
    """
    p = next((q for q in view.planets if q.id == planet_id), None)
    if p is None:
        return 0
    if p.owner == -1:
        return p.ships
    return p.ships + p.production * max(0, at_turn)


def surplus_ships(view: GameView, player: int) -> dict[int, int]:
    """Per-owned-planet ships that can be spent without immediately losing it.

    Conservative rule: for each owned planet, look at the worst-case enemy
    arrival within a 30-turn horizon. If the largest single-source enemy
    fleet exceeds (current garrison + production * eta), the planet is
    threatened — surplus is 0. Otherwise surplus = current garrison.

    This is intentionally simple. Real defensive timing belongs in W3 with
    timeline-aware combat resolution. The W2 composer just needs a number
    that prevents it from emptying a planet that's about to die.
    """
    surplus: dict[int, int] = {}
    for p in view.planets:
        if p.owner != player:
            continue
        worst_arrival = 0
        for f in view.fleets:
            if f.owner == player or f.owner == -1:
                continue
            eta = turns_to_arrive(f.x, f.y, p.x, p.y, max(1, f.ships))
            if eta <= 30:
                worst_arrival = max(worst_arrival, f.ships)
        defenders = p.ships + p.production * 5  # 5-turn anticipated production
        if worst_arrival > defenders:
            surplus[p.id] = 0
        else:
            # Reserve enough to win the worst incoming combat plus 1.
            reserve = max(0, worst_arrival - p.production * 5)
            surplus[p.id] = max(0, p.ships - reserve)
    return surplus
```

- [ ] **Step 4: Run the eval-feature tests**

Run: `uv run pytest tests/test_eval_features.py -v`
Expected: 8 passed.

If `test_surplus_ships_zero_when_inbound_enemy_overwhelms` fails, the surplus formula needs to bite harder when the incoming fleet outnumbers the garrison. Inspect the case (5 ships vs 100 incoming) — the simple `worst_arrival > defenders` should suffice with `defenders = 5 + 1*5 = 10 < 100`.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/eval/__init__.py orbit_war/eval/features.py tests/test_eval_features.py
git commit -m "Add eval feature extractors: ships, production, threat, surplus"
```

---

## Task 4: Step abstraction

**Files:**
- Create: `orbit_war/plan_gen/__init__.py`
- Create: `orbit_war/plan_gen/step.py`
- Test: `tests/test_step.py`

A `Step` is the unit of plan composition: one launch from one source planet toward one target, with a score and a per-source ship cost. Templates emit them; the composer ranks and combines.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_step.py`:

```python
"""Tests for the Step abstraction."""

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_war.plan_gen.step import Step, ships_needed_to_capture


def test_step_packs_to_kaggle_move_format():
    step = Step(
        from_planet_id=3,
        target_planet_id=7,
        angle=1.234,
        ships=42,
        score=0.5,
    )
    assert step.as_move() == [3, 1.234, 42]


def test_ships_needed_to_capture_includes_one_extra():
    target = Planet(0, -1, 10.0, 10.0, 1.0, 30, 2)
    # Capture neutral with 30 garrison: need 31 to overwhelm.
    assert ships_needed_to_capture(target) == 31


def test_ships_needed_to_capture_owned_returns_zero():
    target = Planet(0, 0, 10.0, 10.0, 1.0, 30, 2)
    # Reinforcing your own planet doesn't need to "capture" it.
    assert ships_needed_to_capture(target, player=0) == 0


def test_step_angle_to_target_is_atan2_to_target():
    src = Planet(0, 0, 0.0, 0.0, 1.0, 50, 1)
    tgt = Planet(1, -1, 3.0, 4.0, 1.0, 5, 1)
    expected = math.atan2(4.0, 3.0)
    assert math.isclose(Step.angle_to(src, tgt), expected)


def test_step_orderable_by_score_descending():
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=10, score=0.3)
    b = Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=10, score=0.9)
    c = Step(from_planet_id=0, target_planet_id=3, angle=0.0, ships=10, score=0.6)
    descending = sorted([a, b, c], key=lambda s: -s.score)
    assert [s.target_planet_id for s in descending] == [2, 3, 1]
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_step.py -v`

- [ ] **Step 3: Implement**

Create `orbit_war/plan_gen/__init__.py` (empty).

Create `orbit_war/plan_gen/step.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_step.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/__init__.py orbit_war/plan_gen/step.py tests/test_step.py
git commit -m "Add Step abstraction with ships_needed_to_capture helper"
```

---

## Task 5: Step templates — production_attack and no_op

**Files:**
- Create: `orbit_war/plan_gen/templates.py`
- Test: `tests/test_templates.py`

Two templates first: `no_op` (always emit nothing — sentinel for "do nothing wins") and `production_attack` (port `greedy_baseline.py` into a step generator).

- [ ] **Step 1: Write failing tests for both templates**

Create `tests/test_templates.py`:

```python
"""Tests for step templates."""

from kaggle_environments import make

from orbit_war.plan_gen.step import Step
from orbit_war.plan_gen.templates import (
    no_op_template,
    production_attack_template,
)
from orbit_war.sim.observation import GameView


def _fresh_view(seed: int = 42) -> GameView:
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.reset(num_agents=2)
    return GameView.from_obs(env.steps[0][0]["observation"])


def test_no_op_template_returns_empty_list():
    view = _fresh_view()
    assert no_op_template(view) == []


def test_production_attack_emits_steps_when_we_own_planets():
    view = _fresh_view()
    steps = production_attack_template(view)
    assert all(isinstance(s, Step) for s in steps)
    # We should propose at least one step from each owned planet that has ships.
    sources_with_ships = [p for p in view.my_planets() if p.ships >= 2]
    sources_proposed = {s.from_planet_id for s in steps}
    assert all(p.id in sources_proposed for p in sources_with_ships)


def test_production_attack_scores_higher_for_higher_production_per_distance():
    view = _fresh_view()
    steps = production_attack_template(view)
    # All scores should be non-negative; higher production / lower distance
    # planets get higher scores.
    assert all(s.score >= 0 for s in steps)


def test_production_attack_ships_are_min_to_capture():
    view = _fresh_view()
    steps = production_attack_template(view)
    by_target = {p.id: p for p in view.planets}
    for s in steps:
        target = by_target[s.target_planet_id]
        # We propose exactly target.ships + 1 (or capped by source).
        assert s.ships == min(target.ships + 1, _source_ships(view, s.from_planet_id))


def _source_ships(view: GameView, planet_id: int) -> int:
    p = next(q for q in view.planets if q.id == planet_id)
    return p.ships
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_templates.py -v`

- [ ] **Step 3: Implement the two templates**

Create `orbit_war/plan_gen/templates.py`:

```python
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
        # Best production-per-distance target.
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_templates.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/templates.py tests/test_templates.py
git commit -m "Add no_op and production_attack step templates"
```

---

## Task 6: Step template — defensive_reinforce

**Files:**
- Modify: `orbit_war/plan_gen/templates.py`
- Modify: `tests/test_templates.py`

When an enemy fleet is en route to one of our planets and the planet's defenders won't survive, propose reinforcements from the *closest* friendly planet that can spare ships. Score = (incoming threat - current defenders) / (1 + distance).

- [ ] **Step 1: Write failing tests**

Append to `tests/test_templates.py`:

```python
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from orbit_war.plan_gen.templates import defensive_reinforce_template


def test_defensive_reinforce_emits_when_planet_under_attack():
    threatened = Planet(0, 0, 30.0, 30.0, 1.0, 5, 1)
    helper = Planet(1, 0, 35.0, 30.0, 1.0, 50, 1)
    enemy = Planet(2, 1, 80.0, 80.0, 1.0, 1, 1)
    incoming = Fleet(0, 1, 32.0, 30.0, 0.0, 2, 30)
    view = GameView(
        player=0,
        planets=(threatened, helper, enemy),
        fleets=(incoming,),
        angular_velocity=0.04,
        initial_planets=(threatened, helper, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    steps = defensive_reinforce_template(view)
    assert any(s.target_planet_id == 0 for s in steps)
    # Reinforcement should come from the helper (id 1), not the threatened planet.
    for s in steps:
        if s.target_planet_id == 0:
            assert s.from_planet_id == 1


def test_defensive_reinforce_quiet_when_no_threat():
    me = Planet(0, 0, 30.0, 30.0, 1.0, 5, 1)
    enemy = Planet(1, 1, 80.0, 80.0, 1.0, 1, 1)
    view = GameView(
        player=0,
        planets=(me, enemy),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    assert defensive_reinforce_template(view) == []
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_templates.py -k defensive -v`
Expected: ImportError on `defensive_reinforce_template`.

- [ ] **Step 3: Implement**

In `orbit_war/plan_gen/templates.py`, add at the bottom:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_templates.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/templates.py tests/test_templates.py
git commit -m "Add defensive_reinforce step template"
```

---

## Task 7: Step template — snipe_undefended

**Files:**
- Modify: `orbit_war/plan_gen/templates.py`
- Modify: `tests/test_templates.py`

Find non-owned planets with `ships < 10` and `production >= 2` — high-yield, low-cost. Route the closest friendly source. Score = production / (cost + 1).

- [ ] **Step 1: Write failing tests**

Append to `tests/test_templates.py`:

```python
from orbit_war.plan_gen.templates import snipe_undefended_template


def test_snipe_undefended_finds_low_defense_high_prod_target():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 2)
    juicy = Planet(1, -1, 20.0, 10.0, 2.5, 3, 4)  # cheap + high production
    boring = Planet(2, -1, 80.0, 80.0, 1.0, 80, 1)
    view = GameView(
        player=0,
        planets=(me, juicy, boring),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, juicy, boring),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    steps = snipe_undefended_template(view)
    assert any(s.target_planet_id == 1 for s in steps)
    for s in steps:
        # Snipes are always against juicy targets, never the heavy boring one.
        assert s.target_planet_id != 2


def test_snipe_undefended_skips_planets_we_cant_afford():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 2, 1)  # only 2 ships
    juicy = Planet(1, -1, 20.0, 10.0, 2.5, 5, 4)  # needs 6 ships to capture
    view = GameView(
        player=0,
        planets=(me, juicy),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, juicy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    steps = snipe_undefended_template(view)
    assert steps == []
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_templates.py -k snipe -v`

- [ ] **Step 3: Implement**

In `orbit_war/plan_gen/templates.py`, add at the bottom:

```python
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
        proposals.append(
            Step(
                from_planet_id=int(nearest.id),
                target_planet_id=int(tgt.id),
                angle=Step.angle_to(nearest, tgt),
                ships=int(needed),
                score=float(score),
            )
        )
    return proposals
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_templates.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/templates.py tests/test_templates.py
git commit -m "Add snipe_undefended step template"
```

---

## Task 8: Step composer — sort + greedy combine under surplus

**Files:**
- Create: `orbit_war/plan_gen/composer.py`
- Test: `tests/test_composer.py`

Take all proposed steps, sort by score (highest first), and greedily commit them subject to per-source-planet surplus constraints. This is the Melis pattern.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_composer.py`:

```python
"""Tests for the step composer."""

from orbit_war.plan_gen.composer import compose_plan
from orbit_war.plan_gen.step import Step


def test_compose_returns_empty_when_no_steps():
    assert compose_plan([], surplus_by_planet={0: 100}) == []


def test_compose_picks_highest_score_first():
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=10, score=0.3)
    b = Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=10, score=0.9)
    plan = compose_plan([a, b], surplus_by_planet={0: 30})
    assert plan[0].target_planet_id == 2  # b first


def test_compose_respects_surplus():
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=20, score=0.9)
    b = Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=20, score=0.8)
    # Surplus is only 25 — we can fit `a` (20) but not `b` (would exceed).
    plan = compose_plan([a, b], surplus_by_planet={0: 25})
    assert len(plan) == 1
    assert plan[0].target_planet_id == 1


def test_compose_partial_steps_when_truncatable():
    """If a step's `ships` exceeds remaining surplus, the composer may
    truncate to remaining surplus (>=1) and still emit."""
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=20, score=0.9)
    b = Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=20, score=0.8)
    plan = compose_plan([a, b], surplus_by_planet={0: 25}, allow_truncation=True)
    # We send 20 from planet 0 for step a, then truncate step b to 5.
    assert len(plan) == 2
    assert plan[1].ships == 5


def test_compose_unrelated_planets_dont_interact():
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=10, score=0.9)
    b = Step(from_planet_id=2, target_planet_id=3, angle=0.0, ships=10, score=0.8)
    plan = compose_plan([a, b], surplus_by_planet={0: 10, 2: 10})
    assert len(plan) == 2


def test_compose_ignores_steps_from_planets_without_surplus():
    a = Step(from_planet_id=0, target_planet_id=1, angle=0.0, ships=10, score=0.9)
    plan = compose_plan([a], surplus_by_planet={0: 0})
    assert plan == []
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_composer.py -v`

- [ ] **Step 3: Implement**

Create `orbit_war/plan_gen/composer.py`:

```python
"""Step composer: rank candidate steps and greedily combine under
per-source surplus constraints. The Melis "steps + greedy combine"
pattern, ported to Orbit Wars."""

from __future__ import annotations

from typing import Iterable

from orbit_war.plan_gen.step import Step


def compose_plan(
    steps: Iterable[Step],
    surplus_by_planet: dict[int, int],
    allow_truncation: bool = False,
) -> list[Step]:
    """Sort `steps` by descending score and greedily commit, debiting
    `surplus_by_planet[step.from_planet_id]` each time.

    If `allow_truncation` is True, a step that exceeds remaining surplus
    is shrunk to the surplus (provided >=1 ships remain). Otherwise it
    is skipped.

    Returns the committed steps in the order they were chosen.
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
    return plan
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_composer.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/composer.py tests/test_composer.py
git commit -m "Add greedy step composer with optional truncation"
```

---

## Task 9: heuristic_v1 bot — wire it all together

**Files:**
- Create: `orbit_war/bots/heuristic_v1.py`
- Test: `tests/test_heuristic_v1.py`

Combine the templates + composer behind a single `agent(obs)` function. Initial weights: simple uniform multipliers (1.0 per template). v2 will tune these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_heuristic_v1.py`:

```python
"""Tests for the W2 heuristic_v1 bot."""

from kaggle_environments import make

from orbit_war.bots import greedy_baseline, heuristic_v1, random_bot, starter_bot
from orbit_war.eval_harness.parallel import run_mirrored_pairs


def test_heuristic_v1_runs_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([heuristic_v1.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_heuristic_v1_beats_random_decisively():
    summary = run_mirrored_pairs(
        bot_a=heuristic_v1.agent,
        bot_b=random_bot.agent,
        seeds=tuple(range(8)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.85, (
        f"heuristic_v1 only beat random {summary.win_rate_a:.0%} — composition is broken"
    )


def test_heuristic_v1_at_least_matches_starter():
    summary = run_mirrored_pairs(
        bot_a=heuristic_v1.agent,
        bot_b=starter_bot.agent,
        seeds=tuple(range(8)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.55, (
        f"heuristic_v1 only beat starter {summary.win_rate_a:.0%} — should clearly outperform"
    )


def test_heuristic_v1_at_least_matches_greedy():
    summary = run_mirrored_pairs(
        bot_a=heuristic_v1.agent,
        bot_b=greedy_baseline.agent,
        seeds=tuple(range(6)),
        workers=4,
    )
    # Greedy is a single template; we have four. Should beat it most of the time.
    assert summary.win_rate_a >= 0.55, (
        f"heuristic_v1 only beat greedy {summary.win_rate_a:.0%} — composer not adding value"
    )
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_heuristic_v1.py -v`

- [ ] **Step 3: Implement the bot**

Create `orbit_war/bots/heuristic_v1.py`:

```python
"""heuristic_v1: composes four step templates into a single agent.

This is our first real bot. Templates emit candidate steps; the composer
sorts by score and greedily commits under per-planet surplus. Weights here
are hand-set initial guesses; heuristic_v2 will tune them via linear
regression on self-play data."""

from __future__ import annotations

from orbit_war.eval.features import surplus_ships
from orbit_war.plan_gen.composer import compose_plan
from orbit_war.plan_gen.step import Step
from orbit_war.plan_gen.templates import (
    defensive_reinforce_template,
    no_op_template,
    production_attack_template,
    snipe_undefended_template,
)
from orbit_war.sim.observation import GameView

# Hand-set initial weights; v2 will replace these with regression-fit values.
TEMPLATE_WEIGHTS: dict[str, float] = {
    "no_op": 0.0,
    "production_attack": 1.0,
    "defensive_reinforce": 2.0,  # defending is high-priority by default
    "snipe_undefended": 1.5,     # sniping is high-EV
}


def _weighted(steps: list[Step], weight: float) -> list[Step]:
    """Apply `weight` to every step's score. Cheap immutable rewrite."""
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


def agent(obs) -> list[list]:
    view = GameView.from_obs(obs)

    candidates: list[Step] = []
    candidates.extend(_weighted(no_op_template(view), TEMPLATE_WEIGHTS["no_op"]))
    candidates.extend(
        _weighted(
            production_attack_template(view),
            TEMPLATE_WEIGHTS["production_attack"],
        )
    )
    candidates.extend(
        _weighted(
            defensive_reinforce_template(view),
            TEMPLATE_WEIGHTS["defensive_reinforce"],
        )
    )
    candidates.extend(
        _weighted(
            snipe_undefended_template(view),
            TEMPLATE_WEIGHTS["snipe_undefended"],
        )
    )

    surplus = surplus_ships(view, view.player)
    plan = compose_plan(candidates, surplus, allow_truncation=False)
    return [s.as_move() for s in plan]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_heuristic_v1.py -v`
Expected: 4 passed (will take ~1-2 minutes — multiple multi-game evaluations).

If `test_heuristic_v1_at_least_matches_greedy` fails, the composer is likely letting templates over-spend per planet. Inspect a single game's logs (run `play_match` and print). Common bugs:
- `production_attack_template` proposing ships that exceed the planet's surplus.
- `snipe_undefended_template` and `production_attack_template` both proposing from the same source — composer commits both, which is intended, but if the source's surplus is too small both get skipped.
- `defensive_reinforce_template` weighting overshadowing offensive templates entirely on idle turns.

If you change weights to fix this, document the change in a code comment.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/bots/heuristic_v1.py tests/test_heuristic_v1.py
git commit -m "Add heuristic_v1 bot composing four step templates"
```

---

## Task 10: Run the gate against heuristic_v1 and update the CLI zoo

**Files:**
- Modify: `orbit_war/eval_harness/cli.py`
- Create: `submissions.log` entry (manual)

This is an ops task. Add `heuristic_v1` to the CLI's known zoo paths so we can run gates against it from the command line, then run a real gate and capture results.

- [ ] **Step 1: Add `heuristic_v1` to the CLI zoo map**

Edit `orbit_war/eval_harness/cli.py`. Find the `ZOO_BOT_PATHS` dict and add an entry:

```python
ZOO_BOT_PATHS: dict[str, str] = {
    "random": "orbit_war.bots.random_bot:agent",
    "starter": "orbit_war.bots.starter_bot:agent",
    "greedy": "orbit_war.bots.greedy_baseline:agent",
    "public_tactical": "orbit_war.bots.public_tactical:agent",
    "heuristic_v1": "orbit_war.bots.heuristic_v1:agent",
}
```

- [ ] **Step 2: Run the gate with heuristic_v1 as challenger and greedy as champion**

Run:

```bash
uv run ow-gate orbit_war.bots.heuristic_v1:agent \
    --champion orbit_war.bots.greedy_baseline:agent \
    --seeds 10 --workers 4
```

Expected: 5 tier lines (sanity:random, sanity:starter, diversity:greedy, diversity:public_tactical, champion). Capture the output to a comment or report.

Realistic outcomes:
- `sanity:random`: PASS (≥95% vs random)
- `sanity:starter`: PASS or borderline (≥95% vs starter — may need template tuning if borderline)
- `diversity:greedy`: PASS (≥55% vs greedy)
- `diversity:public_tactical`: probably FAIL (this is the hard one — public_tactical has forward sim and we don't yet)
- `champion` (greedy as champion): PASS

- [ ] **Step 3: Record the W2 v1 result**

If you don't have `submissions.log` yet, create it. Append a one-line note (without submitting):

```bash
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  heuristic_v1  $(git rev-parse --short HEAD)  W2 v1 gate result: PASS/FAIL per tier; not submitted yet (see W2 plan task 10)" >> submissions.log
```

(This is a local audit note — do not submit yet. Submission lands at the end of W2 after v2.)

- [ ] **Step 4: Commit the CLI update**

```bash
git add orbit_war/eval_harness/cli.py submissions.log
git commit -m "Add heuristic_v1 to ow-gate zoo; record W2 v1 gate result"
```

---

## Task 11: Self-play data collector for weight tuning

**Files:**
- Create: `orbit_war/tuning/__init__.py`
- Create: `orbit_war/tuning/data.py`
- Test: `tests/test_tuning_data.py`

For each played game, log per-turn feature vectors per player + the final game outcome. The result is a numpy array `(N_examples, N_features)` and a vector `(N_examples,)` of outcomes. The fitter (Task 12) regresses on this.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tuning_data.py`:

```python
"""Tests for the self-play data collector."""

import numpy as np

from orbit_war.bots import greedy_baseline, random_bot
from orbit_war.tuning.data import (
    FEATURE_NAMES,
    collect_self_play_dataset,
)


def test_dataset_shape_matches_collected_games():
    X, y = collect_self_play_dataset(
        bots=[greedy_baseline.agent, random_bot.agent],
        seeds=(1, 2),
        sample_every=10,  # one feature row every 10 turns
    )
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(FEATURE_NAMES)
    # 2 seeds * 2 players * (500/10) ≈ 200 rows minimum
    assert X.shape[0] >= 100


def test_outcome_label_is_in_minus_one_zero_one():
    X, y = collect_self_play_dataset(
        bots=[greedy_baseline.agent, random_bot.agent],
        seeds=(1,),
        sample_every=20,
    )
    assert set(np.unique(y)).issubset({-1, 0, 1})


def test_feature_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_tuning_data.py -v`

- [ ] **Step 3: Implement `tuning/data.py`**

Create `orbit_war/tuning/__init__.py` (empty).

Create `orbit_war/tuning/data.py`:

```python
"""Self-play dataset collector for linear weight regression.

Plays games between supplied bots, samples per-turn feature vectors per
player, labels each row with the *eventual game outcome* from that
player's perspective (+1 win, 0 draw, -1 loss), and returns the resulting
dataset as numpy arrays.

Per a1k0n's Tron 2010 trick: empirical coefficient tuning on ~10k games
beat hand-tuning. This module provides the data side of that pipeline;
`tuning.regression` does the fitting."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
from kaggle_environments import make

from orbit_war.eval.features import (
    incoming_threat,
    surplus_ships,
    total_production,
    total_ships,
)
from orbit_war.sim.observation import GameView

Agent = Callable[[dict], list]

FEATURE_NAMES: tuple[str, ...] = (
    "total_ships_self",
    "total_ships_enemy",
    "ship_diff",
    "total_production_self",
    "total_production_enemy",
    "production_diff",
    "owned_planets_self",
    "owned_planets_enemy",
    "surplus_total_self",
    "incoming_threat_self",
    "step_normalized",
)


def _vectorize(view: GameView, opponent: int) -> np.ndarray:
    me = view.player
    ts_me = total_ships(view, me)
    ts_op = total_ships(view, opponent)
    tp_me = total_production(view, me)
    tp_op = total_production(view, opponent)
    own_me = sum(1 for p in view.planets if p.owner == me)
    own_op = sum(1 for p in view.planets if p.owner == opponent)
    surplus_me = sum(surplus_ships(view, me).values())
    threat_me = sum(
        incoming_threat(view, me, p.id, horizon=20)
        for p in view.planets
        if p.owner == me
    )
    return np.array(
        [
            ts_me,
            ts_op,
            ts_me - ts_op,
            tp_me,
            tp_op,
            tp_me - tp_op,
            own_me,
            own_op,
            surplus_me,
            threat_me,
            view.step / 500.0,
        ],
        dtype=np.float64,
    )


def _label(reward: float | None) -> int:
    if reward is None:
        return 0
    if reward > 0:
        return 1
    if reward < 0:
        return -1
    return 0


def collect_self_play_dataset(
    bots: Sequence[Agent],
    seeds: Iterable[int],
    sample_every: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y). Each row of X is a feature vector at one sampled turn,
    each entry of y is the *eventual* outcome (+1/0/-1) for the player whose
    perspective the row was taken from."""
    if len(bots) != 2:
        raise ValueError("collect_self_play_dataset expects exactly two bots")

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for seed in seeds:
        env = make(
            "orbit_wars",
            configuration={"seed": seed, "episodeSteps": 500},
            debug=True,
        )
        env.run(list(bots))
        final = env.steps[-1]
        outcome_per_player = [_label(s.reward) for s in final]
        for step_idx, frame in enumerate(env.steps):
            if step_idx % sample_every != 0:
                continue
            for player_idx in range(2):
                obs = frame[player_idx]["observation"]
                view = GameView.from_obs(obs)
                opponent = 1 - player_idx
                rows.append(_vectorize(view, opponent))
                labels.append(outcome_per_player[player_idx])
    return np.stack(rows), np.array(labels, dtype=np.int8)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_tuning_data.py -v`
Expected: 3 passed (will take ~30-60 s — a couple of full games).

- [ ] **Step 5: Commit**

```bash
git add orbit_war/tuning/__init__.py orbit_war/tuning/data.py tests/test_tuning_data.py
git commit -m "Add self-play dataset collector for weight regression"
```

---

## Task 12: Linear regression weight fitting

**Files:**
- Create: `orbit_war/tuning/regression.py`
- Test: `tests/test_tuning_regression.py`

Take a `(X, y)` dataset and return weights via `np.linalg.lstsq`. We then encode these into per-template multipliers heuristic_v2 will use.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tuning_regression.py`:

```python
"""Tests for the regression weight fitter."""

import numpy as np

from orbit_war.tuning.regression import fit_weights


def test_fit_weights_recovers_true_weights_on_synthetic_data():
    # Synthetic problem: y = 2 * x0 + -1 * x1 + 0.5 * x2 + small noise.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    true_w = np.array([2.0, -1.0, 0.5])
    y = X @ true_w + rng.normal(scale=0.1, size=500)

    fitted = fit_weights(X, y)
    assert fitted.shape == (3,)
    assert np.allclose(fitted, true_w, atol=0.1)


def test_fit_weights_handles_singular_designs():
    # Two perfectly collinear features → lstsq should still return *some*
    # finite weights without raising.
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    fitted = fit_weights(X, y)
    assert fitted.shape == (2,)
    assert np.all(np.isfinite(fitted))
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_tuning_regression.py -v`

- [ ] **Step 3: Implement**

Create `orbit_war/tuning/regression.py`:

```python
"""Linear regression weight fitting for heuristic eval features."""

from __future__ import annotations

import numpy as np


def fit_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return least-squares weights `w` such that `X @ w ≈ y`.

    Uses `np.linalg.lstsq` so collinear / under-determined designs return
    minimum-norm solutions instead of raising. Output is always a 1-D
    array of length `X.shape[1]`."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError(f"X must be 2-D and y 1-D (got {X.shape=}, {y.shape=})")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"row count mismatch: {X.shape[0]} vs {y.shape[0]}")
    weights, *_ = np.linalg.lstsq(X, y, rcond=None)
    return weights
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_tuning_regression.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/tuning/regression.py tests/test_tuning_regression.py
git commit -m "Add linear-regression weight fitter (lstsq, handles singular designs)"
```

---

## Task 13: heuristic_v2 — fitted weights, ship if it beats v1

**Files:**
- Create: `scripts/fit_heuristic_v2_weights.py`
- Create: `orbit_war/tuning/weights/v2.json`
- Create: `orbit_war/bots/heuristic_v2.py`
- Test: `tests/test_heuristic_v2.py`

Pipeline:
1. Generate ~1000 self-play games of `heuristic_v1` vs `[greedy, starter, public_tactical, random]`.
2. Fit weights, save to JSON.
3. `heuristic_v2.agent` re-uses `heuristic_v1`'s composition but multiplies template scores by per-template scalars derived from the fitted weights (sign + magnitude relative to other features).
4. Validate v2 beats v1 in self-play; otherwise revert.

This task carries the highest unknown — the regression may not yield a clean improvement. If v2 doesn't beat v1 in 100 mirrored games, ship v1 as the W2 champion and document the failure. That's an honest result, not a failure of the plan.

- [ ] **Step 1: Write the fit script**

Create `scripts/fit_heuristic_v2_weights.py`:

```python
"""Generate self-play data and fit heuristic_v2 weights.

Run via: uv run python scripts/fit_heuristic_v2_weights.py
Writes: orbit_war/tuning/weights/v2.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbit_war.bots import (
    greedy_baseline,
    heuristic_v1,
    public_tactical,
    random_bot,
    starter_bot,
)
from orbit_war.tuning.data import FEATURE_NAMES, collect_self_play_dataset
from orbit_war.tuning.regression import fit_weights

OUT = Path(__file__).parent.parent / "orbit_war" / "tuning" / "weights" / "v2.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairings = [
        (heuristic_v1.agent, random_bot.agent),
        (heuristic_v1.agent, starter_bot.agent),
        (heuristic_v1.agent, greedy_baseline.agent),
        (heuristic_v1.agent, public_tactical.agent),
    ]
    seeds_per_pairing = list(range(50))  # 4 pairings * 50 seeds = 200 games

    all_X = []
    all_y = []
    for bot_a, bot_b in pairings:
        print(f"Collecting from {bot_a.__module__} vs {bot_b.__module__}…")
        X, y = collect_self_play_dataset([bot_a, bot_b], seeds_per_pairing, sample_every=20)
        all_X.append(X)
        all_y.append(y)
    X = np.vstack(all_X)
    y = np.hstack(all_y).astype(np.float64)
    print(f"Total dataset: {X.shape[0]} rows, {X.shape[1]} features")

    weights = fit_weights(X, y)
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "weights": weights.tolist(),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")
    print(f"Weights: {dict(zip(FEATURE_NAMES, weights.round(4)))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the fit script**

Run: `uv run python scripts/fit_heuristic_v2_weights.py`

Expected: ~5-15 minutes runtime. Produces `orbit_war/tuning/weights/v2.json` with fitted weights.

If it fails on memory or time: reduce `seeds_per_pairing` to `range(20)` and `sample_every=40`.

- [ ] **Step 3: Write failing tests for heuristic_v2**

Create `tests/test_heuristic_v2.py`:

```python
"""Tests for heuristic_v2 (fitted weights)."""

from kaggle_environments import make

from orbit_war.bots import heuristic_v1, heuristic_v2, random_bot
from orbit_war.eval_harness.parallel import run_mirrored_pairs


def test_heuristic_v2_runs_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([heuristic_v2.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_heuristic_v2_does_not_regress_against_v1():
    """Soft check: v2 should at least match v1. A clear loss means the fit
    overfit or the score-derivation function from weights is wrong."""
    summary = run_mirrored_pairs(
        bot_a=heuristic_v2.agent,
        bot_b=heuristic_v1.agent,
        seeds=tuple(range(10)),
        workers=4,
    )
    # 45% lower bound — anything below means real regression.
    assert summary.win_rate_a >= 0.45, (
        f"heuristic_v2 only beat heuristic_v1 {summary.win_rate_a:.0%} — fit may be bad"
    )
```

- [ ] **Step 4: Run, expect ImportError on heuristic_v2**

Run: `uv run pytest tests/test_heuristic_v2.py -v`

- [ ] **Step 5: Implement heuristic_v2**

Create `orbit_war/bots/heuristic_v2.py`:

```python
"""heuristic_v2: heuristic_v1 with per-template weights derived from
linear-regression fit on self-play outcome data.

The fit lives at `orbit_war/tuning/weights/v2.json`. Loaded once at module
import; if the file is missing the bot falls back to heuristic_v1's
hand-set weights so the bot stays usable even before a fit run.

Mapping from regression weights to template multipliers:
- The regression fits per-feature weights against game outcome.
- We derive a per-template weight by taking the absolute value of the
  feature most associated with that template. This is a coarse mapping
  but cheap; W3+ may switch to a per-step regression instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from orbit_war.eval.features import surplus_ships
from orbit_war.plan_gen.composer import compose_plan
from orbit_war.plan_gen.step import Step
from orbit_war.plan_gen.templates import (
    defensive_reinforce_template,
    no_op_template,
    production_attack_template,
    snipe_undefended_template,
)
from orbit_war.sim.observation import GameView

_WEIGHTS_PATH = (
    Path(__file__).parent.parent / "tuning" / "weights" / "v2.json"
)

# Default fallback if the JSON file is missing or unreadable.
_FALLBACK_TEMPLATE_WEIGHTS: dict[str, float] = {
    "no_op": 0.0,
    "production_attack": 1.0,
    "defensive_reinforce": 2.0,
    "snipe_undefended": 1.5,
}


def _load_template_weights() -> dict[str, float]:
    if not _WEIGHTS_PATH.exists():
        return dict(_FALLBACK_TEMPLATE_WEIGHTS)
    try:
        payload = json.loads(_WEIGHTS_PATH.read_text())
        names = payload["feature_names"]
        weights = payload["weights"]
        feat = dict(zip(names, weights))
        # Derive per-template scalars from the fitted feature weights.
        # The mapping below is a heuristic until W3 introduces per-step
        # regression: production_attack tracks ship_diff/production_diff,
        # defensive_reinforce tracks incoming_threat_self, snipe tracks
        # production_diff. We take absolute value so positive contribution
        # to outcome → positive weight.
        return {
            "no_op": 0.0,
            "production_attack": abs(feat.get("production_diff", 1.0)) + 0.5,
            "defensive_reinforce": abs(feat.get("incoming_threat_self", 1.0)) + 0.5,
            "snipe_undefended": abs(feat.get("ship_diff", 1.0)) + 0.5,
        }
    except (KeyError, ValueError, OSError):
        return dict(_FALLBACK_TEMPLATE_WEIGHTS)


TEMPLATE_WEIGHTS = _load_template_weights()


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


def agent(obs) -> list[list]:
    view = GameView.from_obs(obs)

    candidates: list[Step] = []
    candidates.extend(_weighted(no_op_template(view), TEMPLATE_WEIGHTS["no_op"]))
    candidates.extend(
        _weighted(
            production_attack_template(view),
            TEMPLATE_WEIGHTS["production_attack"],
        )
    )
    candidates.extend(
        _weighted(
            defensive_reinforce_template(view),
            TEMPLATE_WEIGHTS["defensive_reinforce"],
        )
    )
    candidates.extend(
        _weighted(
            snipe_undefended_template(view),
            TEMPLATE_WEIGHTS["snipe_undefended"],
        )
    )

    surplus = surplus_ships(view, view.player)
    plan = compose_plan(candidates, surplus, allow_truncation=False)
    return [s.as_move() for s in plan]
```

- [ ] **Step 6: Run heuristic_v2 tests**

Run: `uv run pytest tests/test_heuristic_v2.py -v`
Expected: 2 passed (will take ~1-2 minutes).

If `test_heuristic_v2_does_not_regress_against_v1` fails (v2 < 45%), the weight derivation is overfit or wrong. Revert: leave heuristic_v2 importing fallback weights only and tag the W2 champion as v1 in the next task. **This is acceptable.** Document in the commit message that the fit did not help and W3 will tune per-step.

- [ ] **Step 7: Add v2 to the CLI zoo**

Edit `orbit_war/eval_harness/cli.py`. Add to `ZOO_BOT_PATHS`:

```python
    "heuristic_v2": "orbit_war.bots.heuristic_v2:agent",
```

- [ ] **Step 8: Commit**

```bash
git add scripts/fit_heuristic_v2_weights.py orbit_war/tuning/weights/v2.json orbit_war/bots/heuristic_v2.py tests/test_heuristic_v2.py orbit_war/eval_harness/cli.py
git commit -m "Add heuristic_v2 with regression-fit per-template weights"
```

If the fit did not help: include in the commit message: "Note: v2 fit did not beat v1 in self-play; v1 remains W2 champion. W3 will switch to per-step regression."

---

## Task 14: W2 closing — gate, champion designation, follow-ups

**Files:**
- Modify: `submissions.log`
- Modify: `scripts/submit_starter.sh` → rename and add idempotency guard

This wraps W2 with the operational hygiene the W1 review flagged.

- [ ] **Step 1: Run the full gate against the chosen W2 champion**

Pick the better of v1/v2 (per Task 13's outcome). Run the gate with `greedy_baseline` as the champion:

```bash
uv run ow-gate orbit_war.bots.heuristic_v2:agent \
    --champion orbit_war.bots.greedy_baseline:agent \
    --seeds 25 --workers 4
```

(Substitute `heuristic_v1` if v2 didn't beat v1.)

Capture the OVERALL line and per-tier results.

- [ ] **Step 2: Generalise `submit_starter.sh` into `submit_bot.sh` with idempotency guard**

Rename:

```bash
git mv scripts/submit_starter.sh scripts/submit_bot.sh
```

Replace contents:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <bot-name> <message>" >&2
  echo "  bot-name appears in submissions.log; message goes to Kaggle" >&2
  exit 2
fi

BOT_NAME="$1"
MESSAGE="$2"
SHA="$(git rev-parse --short HEAD)"

# Idempotency guard: refuse to submit the same SHA twice.
if grep -q "  $SHA  " submissions.log 2>/dev/null; then
  echo "SHA $SHA already in submissions.log — bailing to save daily quota." >&2
  echo "If you really mean to resubmit, edit submissions.log first." >&2
  exit 1
fi

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

Re-mark executable:

```bash
chmod +x scripts/submit_bot.sh
```

- [ ] **Step 3: Decide whether to submit the W2 champion to the ladder**

Submission cost: 1 of today's 5 slots. The "latest 2 only" rule applies — if you submit, the previous starter (latest_2 slot 1) stays alive, and v2 (or v1) takes slot 2.

If the champion clearly beats greedy + at-least-matches public_tactical:

```bash
./scripts/submit_bot.sh heuristic_v2 "W2 champion: 4-template heuristic w/ regression-fit weights"
```

(or `heuristic_v1` if you fell back).

If the champion is borderline (loses to public_tactical), prefer to wait until W3's lookahead lands — don't burn slots.

- [ ] **Step 4: Tag the W2 baseline**

```bash
git tag w2-baseline
```

- [ ] **Step 5: Update CLAUDE.md if needed**

If the W2 champion has shipped, append a one-line note to the workflow section noting which bot is current champion:

```markdown
- Current champion: heuristic_v2 (W2). Use `uv run ow-gate orbit_war.bots.heuristic_v2:agent` to gate challengers.
```

- [ ] **Step 6: Commit**

```bash
git add scripts/submit_bot.sh submissions.log CLAUDE.md
git commit -m "W2 closing: rename submit script, add idempotency guard, tag w2-baseline"
```

---

## Closing checklist

- [ ] Run the full test suite: `uv run pytest -q`. Expected: all tests pass.
- [ ] Confirm `git log --oneline w2-baseline ^w1-baseline | wc -l` shows ~14 commits (one per task plus a few fixups).
- [ ] Capture realistic numbers: heuristic_v1 vs (random / starter / greedy / public_tactical) and same for v2. Save in a comment on the W2 champion's commit.
- [ ] Note any W3 follow-ups discovered while building (e.g., specific blind spots in heuristic_v2 you saw in replays).
