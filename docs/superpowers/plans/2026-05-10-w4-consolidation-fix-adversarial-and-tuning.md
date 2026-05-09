# W4: Consolidation Fix, Adversarial Validator, and Per-Step Tuning

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land four W4 wins on top of W3: (1) fix the dead `multi_source_consolidation` template by making `filter_capturable` aggregate per target before deciding; (2) replace the passive-opponent assumption in heuristic_v3's validator with a predicted opponent plan; (3) add a trade-down strike template that fires when ahead in late game; (4) attempt per-step regression to learn template weights (v2's per-game regression failed; per-step is the design's next-best path).

**Architecture:** The W3 final review identified that consolidation steps were silently filtered out because the per-step `ships >= target.ships + 1` rule never accounted for combined contributions. We change `filter_capturable` to compute combined ships per target across all steps; partial contributions to a target survive iff their *combined* ships ≥ defender + 1. The adversarial validator runs our own templates from the opponent's perspective, then forward-simulates both action sets together — fixing the "passive opponent" overestimate that biases v3's plan acceptance. The trade-down template fires only in late game (step ≥ 300) when ahead, attacking enemy planets at unfavourable trades to grind absolute scores in our favour. Per-step regression labels each emitted step with its forward-sim eval delta and fits weights against per-step features (template type, source/target production, distance bucket, lead/deficit at launch).

**Tech Stack:** Same as W3. No new external deps.

**Realistic outcome target:** v4 (consolidation + adversarial + trade-down) should beat v3 ≥55% over 50 mirrored games and beat v1 ≥55% (the W3 gate failure). v5 (with per-step regression weights, if the fit succeeds) is a stretch — could fall back to v4's hand-set weights as v2 did.

---

## File structure

NEW production files:
- `orbit_war/bots/heuristic_v4.py` — bot with all W4 wins wired in
- `orbit_war/bots/heuristic_v5.py` — bot with per-step regression weights (or fallback to v4 weights)
- `orbit_war/tuning/step_data.py` — per-step data collector (sibling to existing `tuning/data.py`)
- `orbit_war/tuning/step_weights/v5.json` — fitted per-template weights

NEW test files:
- `tests/test_heuristic_v4.py`
- `tests/test_heuristic_v5.py`
- `tests/test_step_data.py`

Modified:
- `orbit_war/plan_gen/filters.py` — `filter_capturable` aggregates per target
- `orbit_war/plan_gen/templates.py` — add `trade_down_strike_template`
- `orbit_war/sim/game.py` — add a comment documenting the tie-break direction in `_resolve_combat`
- `orbit_war/eval_harness/cli.py` — register heuristic_v4 and heuristic_v5
- `scripts/submit_bot.sh` — bump gate seeds from 10 to 25
- `tests/test_filters.py` — extend with the per-target aggregation tests
- `tests/test_templates.py` — extend with trade-down strike tests

---

## Task 1: Fix `filter_capturable` to aggregate per target

**Files:**
- Modify: `orbit_war/plan_gen/filters.py`
- Modify: `tests/test_filters.py`

The W3 review found that `filter_capturable` rejects each step in isolation. Multi-source consolidation steps are individually below `target.ships + 1`, so all are dropped. The fix is to aggregate combined ships per target across all steps that target it; if combined ≥ defender + 1, all contributing steps survive; otherwise, all are dropped. Friendly reinforcement and unknown-target steps still pass through unconditionally.

- [ ] **Step 1: Append two failing tests to `tests/test_filters.py`**

```python
def test_filter_keeps_combined_partial_attacks_when_aggregate_exceeds_defender():
    """Two partial contributions of 20 each (combined 40) vs defender 30+1=31
    should both survive — the combined fleet captures even though no single
    contribution is sufficient."""
    src1 = Planet(0, 0, 5.0, 5.0, 1.0, 50, 1)
    src2 = Planet(1, 0, 95.0, 5.0, 1.0, 50, 1)
    enemy = Planet(2, 1, 50.0, 50.0, 1.0, 30, 1)
    view = _view_with((src1, src2, enemy))
    steps = [
        Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=20, score=0.5),
        Step(from_planet_id=1, target_planet_id=2, angle=0.0, ships=20, score=0.5),
    ]
    out = filter_capturable(steps, view)
    assert len(out) == 2, "combined contributions should both survive"


def test_filter_drops_all_partial_attacks_when_aggregate_below_defender():
    """Two contributions of 5 each (combined 10) vs defender 30+1=31 should
    both be dropped — even combined we cannot capture."""
    src1 = Planet(0, 0, 5.0, 5.0, 1.0, 50, 1)
    src2 = Planet(1, 0, 95.0, 5.0, 1.0, 50, 1)
    enemy = Planet(2, 1, 50.0, 50.0, 1.0, 30, 1)
    view = _view_with((src1, src2, enemy))
    steps = [
        Step(from_planet_id=0, target_planet_id=2, angle=0.0, ships=5, score=0.5),
        Step(from_planet_id=1, target_planet_id=2, angle=0.0, ships=5, score=0.5),
    ]
    out = filter_capturable(steps, view)
    assert out == []
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_filters.py -k "combined or aggregate" -v`
Expected: 2 failures (current filter rejects per-step).

- [ ] **Step 3: Replace `filter_capturable` with per-target aggregation version**

Replace the body of `orbit_war/plan_gen/filters.py` with:

```python
"""Composer-side filters that run between template emission and the composer.

Templates emit candidate steps freely; filters drop steps that cannot
plausibly capture their target.

The filter aggregates per target: multi-source consolidation steps survive
iff the combined incoming ships exceed the defender. Friendly reinforcement
and unknown-target steps pass through unconditionally."""

from __future__ import annotations

from collections import defaultdict

from orbit_war.plan_gen.step import Step
from orbit_war.sim.observation import GameView


def filter_capturable(steps: list[Step], view: GameView) -> list[Step]:
    """Aggregate per target. Drop all attack steps targeting a planet whose
    combined incoming ships are insufficient to capture (defender + 1).

    Friendly reinforcements and steps with unknown target IDs pass through.
    Within an attacking-target group, all steps survive together or are all
    dropped together."""
    planet_by_id = {p.id: p for p in view.planets}
    player = view.player

    grouped: dict[int, list[Step]] = defaultdict(list)
    pass_through: list[Step] = []
    for s in steps:
        target = planet_by_id.get(s.target_planet_id)
        if target is None or target.owner == player:
            pass_through.append(s)
            continue
        grouped[s.target_planet_id].append(s)

    result: list[Step] = list(pass_through)
    for target_id, group in grouped.items():
        target = planet_by_id[target_id]
        combined = sum(s.ships for s in group)
        needed = int(target.ships) + 1
        if combined >= needed:
            result.extend(group)
        # else: all contributions to this target are dropped
    return result
```

- [ ] **Step 4: Run all filter tests**

Run: `uv run pytest tests/test_filters.py -v`
Expected: 6 passed (4 prior + 2 new).

- [ ] **Step 5: Run all template + bot tests to confirm no regression**

Run: `uv run pytest tests/test_templates.py tests/test_heuristic_v1.py tests/test_heuristic_v2.py tests/test_heuristic_v3.py -v`
Expected: all pass. The aggregation change preserves all single-step semantics — only the multi-step case behavior changes.

If `test_heuristic_v3_at_least_matches_v1` shows v3 stronger than before — good, that's the consolidation finally firing. If it shows v3 weaker, the fix may have an unintended side effect; diagnose before continuing.

- [ ] **Step 6: Commit**

```bash
git add orbit_war/plan_gen/filters.py tests/test_filters.py
git commit -m "filter_capturable now aggregates per target so consolidation steps survive"
```

---

## Task 2: Bump submit gate seeds + document combat tie

**Files:**
- Modify: `scripts/submit_bot.sh`
- Modify: `orbit_war/sim/game.py`

Two tiny housekeeping changes from the W3 review.

- [ ] **Step 1: Edit `scripts/submit_bot.sh`**

Find the line that currently reads:

```bash
if ! uv run ow-gate "$BOT_SPEC" --seeds 10 --workers 4 > /tmp/ow-gate.log 2>&1; then
```

Replace `10` with `25`:

```bash
if ! uv run ow-gate "$BOT_SPEC" --seeds 25 --workers 4 > /tmp/ow-gate.log 2>&1; then
```

10 seeds gives ±16pp standard error at the 55% champion threshold. 25 seeds gives ±10pp — still wide but less likely to false-pass a borderline bot.

- [ ] **Step 2: Add a tie-break comment to `_resolve_combat` in `orbit_war/sim/game.py`**

Find the `_resolve_combat` function. Just above the `sorted_forces = sorted(...)` line, add:

```python
    # Known divergence from official sim: when two owners arrive with equal
    # ship counts the result depends on dict insertion order (arrivals are
    # inserted before the garrison contribution), so a tie resolves toward
    # the attacker. The official simulator may handle ties differently.
    # Acceptable within this simulator's documented ~5% drift budget.
```

- [ ] **Step 3: Run the full suite to confirm no surprises**

Run: `uv run pytest -q`
Expected: 102 passed (100 prior + 2 from Task 1's new filter tests).

- [ ] **Step 4: Commit**

```bash
git add scripts/submit_bot.sh orbit_war/sim/game.py
git commit -m "Bump gate seeds to 25; document combat tie convention"
```

---

## Task 3: Adversarial validator — predict opponent plan instead of passive

**Files:**
- Create: `orbit_war/plan_gen/opponent.py` — predict an opponent plan via our own templates
- Test: `tests/test_opponent.py`

The W3 validator assumed `actions_per_player[opp] = []`. This makes the simulated future too rosy and biases the validator toward acceptance. We replace the assumption with a *predicted* opponent plan, generated by running production_attack from the opponent's perspective — a cheap, no-state-mutation way to estimate "what would they do this turn?".

- [ ] **Step 1: Write failing tests**

Create `tests/test_opponent.py`:

```python
"""Tests for orbit_war.plan_gen.opponent."""

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbit_war.plan_gen.opponent import predict_opponent_plan
from orbit_war.sim.observation import GameView


def test_predict_opponent_plan_returns_action_list():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 50, 1)
    neutral = Planet(2, -1, 50.0, 50.0, 1.0, 5, 2)
    view = GameView(
        player=0,
        planets=(me, enemy, neutral),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, enemy, neutral),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    actions = predict_opponent_plan(view, opponent=1)
    assert isinstance(actions, list)
    for move in actions:
        assert len(move) == 3
        from_id, angle, ships = move
        assert isinstance(from_id, int)
        assert isinstance(angle, float)
        assert isinstance(ships, int)
        # The opponent should only launch from their own planets.
        assert from_id == 1


def test_predict_opponent_plan_quiet_when_opponent_has_no_planets():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 50, 1)
    view = GameView(
        player=0,
        planets=(me,),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me,),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=10,
        comets=(),
    )
    assert predict_opponent_plan(view, opponent=1) == []
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_opponent.py -v`

- [ ] **Step 3: Implement `orbit_war/plan_gen/opponent.py`**

```python
"""Predict an opponent plan for use in adversarial plan validation.

Approach: run production_attack from the opponent's perspective. This is
cheap, requires no state mutation, and gives a reasonable lower-bound
estimate of opponent activity for forward-simulation purposes.

The returned action list is in Kaggle move format `[from_id, angle, ships]`
and is suitable for direct use in `forward_simulate(actions_per_player=...)`."""

from __future__ import annotations

from dataclasses import replace

from orbit_war.eval.features import surplus_ships
from orbit_war.plan_gen.composer import compose_plan
from orbit_war.plan_gen.filters import filter_capturable
from orbit_war.plan_gen.templates import production_attack_template
from orbit_war.sim.observation import GameView


def predict_opponent_plan(view: GameView, opponent: int) -> list[list]:
    """Return a list of actions the opponent would plausibly take this turn.

    Uses production_attack from the opponent's perspective + composer.
    """
    if not any(p.owner == opponent for p in view.planets):
        return []

    opp_view = replace(view, player=opponent)

    candidates = production_attack_template(opp_view)
    candidates = filter_capturable(candidates, opp_view)
    surplus = surplus_ships(opp_view, opponent)
    plan = compose_plan(candidates, surplus)
    return [s.as_move() for s in plan]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_opponent.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/opponent.py tests/test_opponent.py
git commit -m "Add predict_opponent_plan helper for adversarial plan validation"
```

---

## Task 4: New template — trade_down_strike

**Files:**
- Modify: `orbit_war/plan_gen/templates.py`
- Modify: `tests/test_templates.py`

When ahead in the late game, deliberately exchange ships with the opponent at unfavourable rates. Each trade preserves our absolute lead while reducing both totals — the way Planet Wars satirists call "risk reduction when ahead." Fires only when (a) `view.step >= 300`, (b) our total ships > opponent's by margin > 20.

- [ ] **Step 1: Append failing tests**

Append to `tests/test_templates.py`:

```python
from orbit_war.plan_gen.templates import trade_down_strike_template


def test_trade_down_quiet_when_not_ahead():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 30, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 100, 1)
    view = GameView(
        player=0,
        planets=(me, enemy),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=350,
        comets=(),
    )
    assert trade_down_strike_template(view) == []


def test_trade_down_quiet_in_early_game():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 200, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 30, 1)
    view = GameView(
        player=0,
        planets=(me, enemy),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=100,  # too early
        comets=(),
    )
    assert trade_down_strike_template(view) == []


def test_trade_down_fires_in_late_game_when_ahead():
    me = Planet(0, 0, 10.0, 10.0, 1.0, 200, 1)
    enemy = Planet(1, 1, 90.0, 90.0, 1.0, 30, 1)
    view = GameView(
        player=0,
        planets=(me, enemy),
        fleets=(),
        angular_velocity=0.04,
        initial_planets=(me, enemy),
        comet_planet_ids=frozenset(),
        remaining_overage_time=0.0,
        step=400,
        comets=(),
    )
    steps = trade_down_strike_template(view)
    assert len(steps) > 0
    assert all(s.target_planet_id == 1 for s in steps)
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_templates.py -k "trade_down" -v`

- [ ] **Step 3: Implement**

Append to `orbit_war/plan_gen/templates.py`:

```python
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
```

- [ ] **Step 4: Run all template tests**

Run: `uv run pytest tests/test_templates.py -v`
Expected: 16 passed (13 prior + 3 new).

- [ ] **Step 5: Commit**

```bash
git add orbit_war/plan_gen/templates.py tests/test_templates.py
git commit -m "Add trade_down_strike template (late-game ship exchange when ahead)"
```

---

## Task 5: heuristic_v4 wiring

**Files:**
- Create: `orbit_war/bots/heuristic_v4.py`
- Test: `tests/test_heuristic_v4.py`
- Modify: `orbit_war/eval_harness/cli.py`

Wire 7 templates (6 from W3 + trade_down_strike) + the adversarial validator (predicted opponent plan instead of passive) + the now-working consolidation filter.

- [ ] **Step 1: Write failing tests**

Create `tests/test_heuristic_v4.py`:

```python
"""Tests for the W4 heuristic_v4 bot."""

from kaggle_environments import make

from orbit_war.bots import (
    heuristic_v1,
    heuristic_v3,
    heuristic_v4,
    public_tactical,
    random_bot,
)
from orbit_war.eval_harness.parallel import run_mirrored_pairs


def test_heuristic_v4_runs_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([heuristic_v4.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_heuristic_v4_beats_random_decisively():
    summary = run_mirrored_pairs(
        bot_a=heuristic_v4.agent,
        bot_b=random_bot.agent,
        seeds=tuple(range(8)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.85, (
        f"heuristic_v4 only beat random {summary.win_rate_a:.0%}"
    )


def test_heuristic_v4_at_least_matches_v3():
    """v4 should NOT regress against v3 — consolidation actually firing
    + adversarial validator should be neutral-or-positive."""
    summary = run_mirrored_pairs(
        bot_a=heuristic_v4.agent,
        bot_b=heuristic_v3.agent,
        seeds=tuple(range(10)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.45, (
        f"heuristic_v4 only at {summary.win_rate_a:.0%} vs v3 — regression"
    )


def test_heuristic_v4_beats_v1():
    """The W3 champion gate failure was 46% vs v1. With consolidation working
    and an adversarial validator, v4 should clearly clear ≥55%."""
    summary = run_mirrored_pairs(
        bot_a=heuristic_v4.agent,
        bot_b=heuristic_v1.agent,
        seeds=tuple(range(15)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.55, (
        f"heuristic_v4 only at {summary.win_rate_a:.0%} vs v1 — gap not closed"
    )


def test_heuristic_v4_holds_against_public_tactical():
    """v3 hit 64% vs public_tactical. v4 should hold or improve."""
    summary = run_mirrored_pairs(
        bot_a=heuristic_v4.agent,
        bot_b=public_tactical.agent,
        seeds=tuple(range(8)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.50, (
        f"heuristic_v4 only at {summary.win_rate_a:.0%} vs public_tactical — regression"
    )
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_heuristic_v4.py -v`

- [ ] **Step 3: Implement heuristic_v4**

Create `orbit_war/bots/heuristic_v4.py`:

```python
"""heuristic_v4: W4 bot.

Differences from heuristic_v3:
- multi_source_consolidation now actually fires (filter_capturable was
  fixed in W4.1 to aggregate per target).
- Validator uses a *predicted* opponent plan via predict_opponent_plan
  instead of assuming the opponent is passive.
- Adds trade_down_strike_template for late-game grinding when ahead."""

from __future__ import annotations

from orbit_war.eval.features import (
    surplus_ships,
    total_production,
    total_ships,
)
from orbit_war.plan_gen.composer import compose_plan
from orbit_war.plan_gen.filters import filter_capturable
from orbit_war.plan_gen.opponent import predict_opponent_plan
from orbit_war.plan_gen.step import Step
from orbit_war.plan_gen.templates import (
    comet_rush_template,
    defensive_reinforce_template,
    multi_source_consolidation_template,
    no_op_template,
    production_attack_template,
    snipe_undefended_template,
    trade_down_strike_template,
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
    "trade_down_strike": 0.9,
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
    opp = 1 - me
    return (
        total_ships(view, me) - total_ships(view, opp)
        + 5.0 * (total_production(view, me) - total_production(view, opp))
    )


def _make_validator(view: GameView):
    """Validator that simulates the plan + a predicted opponent response,
    forwards 15 turns, and reverts to no-op if eval drops > 5."""
    baseline = _position_eval(view)
    me = view.player
    opp = 1 - me
    opp_actions = predict_opponent_plan(view, opp)

    def validator(plan: list[Step]) -> list[Step]:
        if not plan:
            return plan
        my_actions = [s.as_move() for s in plan]
        actions_per_player = [[], []]
        actions_per_player[me] = my_actions
        actions_per_player[opp] = opp_actions
        future = forward_simulate(view, actions_per_player, n_turns=PLAN_VALIDATION_HORIZON)
        future_eval = _position_eval(future)
        if future_eval < baseline - 5.0:
            return []
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
    candidates.extend(_weighted(trade_down_strike_template(view), TEMPLATE_WEIGHTS["trade_down_strike"]))

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

- [ ] **Step 4: Add v4 to the CLI zoo**

Edit `orbit_war/eval_harness/cli.py`. Add to `ZOO_BOT_PATHS`:

```python
    "heuristic_v4": "orbit_war.bots.heuristic_v4:agent",
```

- [ ] **Step 5: Run heuristic_v4 tests**

Run: `uv run pytest tests/test_heuristic_v4.py -v`
Expected: 5 passed (will take 3-6 minutes).

If `test_heuristic_v4_beats_v1` fails (v4 < 55% vs v1):
- Diagnose by removing one component at a time. Comment out trade_down_strike first; rerun. Then comet_rush. Then check if v4 beats v3 (test_heuristic_v4_at_least_matches_v3 should still pass).
- If v4 ≥ v3 but still < 55% vs v1, the v3-vs-v1 gap may be deeper than W4 expected; document and proceed (W4 is still net-positive).

If `test_heuristic_v4_holds_against_public_tactical` fails (v4 < 50%): the adversarial validator may be too aggressive at vetoing. Lower the validator slack from `5.0` to a higher number (e.g., `15.0`) and rerun.

- [ ] **Step 6: Commit**

```bash
git add orbit_war/bots/heuristic_v4.py tests/test_heuristic_v4.py orbit_war/eval_harness/cli.py
git commit -m "Add heuristic_v4 with consolidation fix, adversarial validator, trade_down"
```

If you needed to disable a component, change the commit message to: "Add heuristic_v4 with <enabled list>; <component> reduced winrate (see commit notes)".

---

## Task 6: Per-step data collector

**Files:**
- Create: `orbit_war/tuning/step_data.py`
- Test: `tests/test_step_data.py`

For each step our bot launches, label it with the eval delta from forward simulation: `delta = eval(state_after_15_turns) - eval(state_now)`. The features are the step's template (one-hot), source planet's production/ships, target's production/ships, distance bucket, our current lead/deficit. Train a regression on `(features → delta)` to learn which step types tend to be net-positive in which contexts.

This is a different shape from W2's per-game regression: instead of asking "what makes a winning game state?", we ask "what makes a step a good move?" — a sharper, more learnable signal.

- [ ] **Step 1: Write failing tests**

Create `tests/test_step_data.py`:

```python
"""Tests for the per-step data collector."""

import numpy as np

from orbit_war.bots import heuristic_v1, random_bot
from orbit_war.tuning.step_data import (
    STEP_FEATURE_NAMES,
    collect_step_dataset,
)


def test_step_dataset_shape():
    X, y = collect_step_dataset(
        bots=[heuristic_v1.agent, random_bot.agent],
        seeds=(1, 2),
        eval_horizon=10,
    )
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(STEP_FEATURE_NAMES)
    assert X.shape[0] >= 10  # heuristic_v1 launches plenty of steps


def test_step_features_are_unique():
    assert len(set(STEP_FEATURE_NAMES)) == len(STEP_FEATURE_NAMES)
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_step_data.py -v`

- [ ] **Step 3: Implement `orbit_war/tuning/step_data.py`**

```python
"""Per-step regression data collector.

For each turn in each game, capture the steps the bot would launch from
the current view, simulate each step in isolation, label with the eval
delta. Use the (feature, delta) pairs to fit per-step weights in
`tuning/regression.py`."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
from kaggle_environments import make

from orbit_war.eval.features import total_production, total_ships
from orbit_war.plan_gen.step import Step
from orbit_war.plan_gen.templates import (
    comet_rush_template,
    defensive_reinforce_template,
    multi_source_consolidation_template,
    production_attack_template,
    snipe_undefended_template,
    trade_down_strike_template,
)
from orbit_war.sim.game import forward_simulate
from orbit_war.sim.observation import GameView

Agent = Callable[[dict], list]

# Template-class one-hot features
TEMPLATE_NAMES: tuple[str, ...] = (
    "production_attack",
    "defensive_reinforce",
    "snipe_undefended",
    "multi_source_consolidation",
    "comet_rush",
    "trade_down_strike",
)

STEP_FEATURE_NAMES: tuple[str, ...] = (
    *(f"is_{n}" for n in TEMPLATE_NAMES),
    "src_ships",
    "src_production",
    "tgt_ships",
    "tgt_production",
    "step_ships",
    "ship_lead",
    "step_normalized",
)


def _eval(view: GameView) -> float:
    me = view.player
    opp = 1 - me
    return (
        total_ships(view, me) - total_ships(view, opp)
        + 5.0 * (total_production(view, me) - total_production(view, opp))
    )


def _step_features(
    step: Step,
    template_idx: int,
    view: GameView,
) -> np.ndarray:
    by_id = {p.id: p for p in view.planets}
    src = by_id.get(step.from_planet_id)
    tgt = by_id.get(step.target_planet_id)
    src_ships = float(src.ships) if src else 0.0
    src_prod = float(src.production) if src else 0.0
    tgt_ships = float(tgt.ships) if tgt else 0.0
    tgt_prod = float(tgt.production) if tgt else 0.0
    me = view.player
    opp = 1 - me
    ship_lead = float(total_ships(view, me) - total_ships(view, opp))

    one_hot = np.zeros(len(TEMPLATE_NAMES), dtype=np.float64)
    if 0 <= template_idx < len(TEMPLATE_NAMES):
        one_hot[template_idx] = 1.0

    return np.concatenate(
        [
            one_hot,
            np.array(
                [src_ships, src_prod, tgt_ships, tgt_prod,
                 float(step.ships), ship_lead, view.step / 500.0],
                dtype=np.float64,
            ),
        ]
    )


def collect_step_dataset(
    bots: Sequence[Agent],
    seeds: Iterable[int],
    eval_horizon: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Run games between `bots`. At each turn, for each candidate step
    each template would emit, simulate that single step in isolation and
    label with the eval delta after `eval_horizon` turns.

    Returns (X, y) where X has shape (n_steps, len(STEP_FEATURE_NAMES))."""
    if len(bots) != 2:
        raise ValueError("collect_step_dataset expects exactly two bots")

    rows: list[np.ndarray] = []
    labels: list[float] = []

    template_funcs = (
        production_attack_template,
        defensive_reinforce_template,
        snipe_undefended_template,
        multi_source_consolidation_template,
        comet_rush_template,
        trade_down_strike_template,
    )

    for seed in seeds:
        env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 500}, debug=True)
        env.run(list(bots))
        for step_idx in range(0, len(env.steps), 25):  # sample every 25 turns
            for player_idx in range(2):
                obs = env.steps[step_idx][player_idx]["observation"]
                view = GameView.from_obs(obs)
                baseline_eval = _eval(view)
                for tmpl_idx, tmpl in enumerate(template_funcs):
                    candidates = tmpl(view)
                    for s in candidates:
                        actions_per_player = [[], []]
                        actions_per_player[view.player] = [s.as_move()]
                        future = forward_simulate(view, actions_per_player, n_turns=eval_horizon)
                        delta = _eval(future) - baseline_eval
                        rows.append(_step_features(s, tmpl_idx, view))
                        labels.append(delta)

    if not rows:
        empty_X = np.zeros((0, len(STEP_FEATURE_NAMES)), dtype=np.float64)
        return empty_X, np.zeros(0, dtype=np.float64)
    return np.stack(rows), np.array(labels, dtype=np.float64)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_step_data.py -v`
Expected: 2 passed (will take ~30-60 s — runs 2 games and simulates many steps).

- [ ] **Step 5: Commit**

```bash
git add orbit_war/tuning/step_data.py tests/test_step_data.py
git commit -m "Add per-step regression data collector"
```

---

## Task 7: heuristic_v5 with per-step regression weights

**Files:**
- Create: `scripts/fit_heuristic_v5_weights.py`
- Create: `orbit_war/tuning/step_weights/v5.json`
- Create: `orbit_war/bots/heuristic_v5.py`
- Test: `tests/test_heuristic_v5.py`
- Modify: `orbit_war/eval_harness/cli.py`

Generate ~200 self-play games of `heuristic_v4` vs the zoo, collect step data, fit weights with `np.linalg.lstsq`, ship as v5. As in W2.13, if v5 doesn't clearly beat v4, fall back to v4's hand-set weights and document the failure.

- [ ] **Step 1: Write the fit script**

Create `scripts/fit_heuristic_v5_weights.py`:

```python
"""Generate per-step data and fit heuristic_v5 weights.

Run via: uv run python scripts/fit_heuristic_v5_weights.py
Writes: orbit_war/tuning/step_weights/v5.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbit_war.bots import (
    greedy_baseline,
    heuristic_v4,
    public_tactical,
    starter_bot,
)
from orbit_war.tuning.regression import fit_weights
from orbit_war.tuning.step_data import (
    STEP_FEATURE_NAMES,
    TEMPLATE_NAMES,
    collect_step_dataset,
)

OUT = Path(__file__).parent.parent / "orbit_war" / "tuning" / "step_weights" / "v5.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairings = [
        (heuristic_v4.agent, starter_bot.agent),
        (heuristic_v4.agent, greedy_baseline.agent),
        (heuristic_v4.agent, public_tactical.agent),
    ]
    seeds_per_pairing = list(range(15))  # ~45 games

    all_X = []
    all_y = []
    for bot_a, bot_b in pairings:
        print(f"Collecting from {bot_a.__module__} vs {bot_b.__module__}…")
        X, y = collect_step_dataset([bot_a, bot_b], seeds_per_pairing, eval_horizon=10)
        all_X.append(X)
        all_y.append(y)
    X = np.vstack(all_X)
    y = np.hstack(all_y)
    print(f"Total dataset: {X.shape[0]} step samples, {X.shape[1]} features")

    weights = fit_weights(X, y)
    payload = {
        "feature_names": list(STEP_FEATURE_NAMES),
        "template_names": list(TEMPLATE_NAMES),
        "weights": weights.tolist(),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")
    print(f"Per-template weights (first {len(TEMPLATE_NAMES)} entries):")
    for name, w in zip(TEMPLATE_NAMES, weights[: len(TEMPLATE_NAMES)]):
        print(f"  {name}: {w:+.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the fit script**

Run: `uv run python scripts/fit_heuristic_v5_weights.py`

Expected runtime: 15-30 minutes. Produces `orbit_war/tuning/step_weights/v5.json`.

If it errors or runs >45 minutes: reduce `seeds_per_pairing` to `range(8)` and re-run. Document the reduction.

- [ ] **Step 3: Write failing tests for heuristic_v5**

Create `tests/test_heuristic_v5.py`:

```python
"""Tests for heuristic_v5 (per-step regression weights)."""

from kaggle_environments import make

from orbit_war.bots import heuristic_v4, heuristic_v5, random_bot
from orbit_war.eval_harness.parallel import run_mirrored_pairs


def test_heuristic_v5_runs_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([heuristic_v5.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_heuristic_v5_does_not_regress_against_v4():
    """Soft check: v5 should at least match v4."""
    summary = run_mirrored_pairs(
        bot_a=heuristic_v5.agent,
        bot_b=heuristic_v4.agent,
        seeds=tuple(range(10)),
        workers=4,
    )
    assert summary.win_rate_a >= 0.45, (
        f"heuristic_v5 only beat heuristic_v4 {summary.win_rate_a:.0%} — fit may be bad"
    )
```

- [ ] **Step 4: Run, expect ImportError on heuristic_v5**

Run: `uv run pytest tests/test_heuristic_v5.py -v`

- [ ] **Step 5: Implement heuristic_v5**

Create `orbit_war/bots/heuristic_v5.py`:

```python
"""heuristic_v5: heuristic_v4 with per-step regression-fit template weights.

Loads `orbit_war/tuning/step_weights/v5.json`. Per-template weights are
the first N entries (one per template) of the fitted weight vector, mapped
through clamp([0.5, 5.0]) and absolute value (since negative weights would
make us disprefer profitable templates).

If the JSON is missing or the fit didn't help, falls back to v4's hand-set
weights (identical behavior to v4)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from orbit_war.bots.heuristic_v4 import (
    PLAN_VALIDATION_HORIZON,
    TEMPLATE_WEIGHTS as _V4_WEIGHTS,
    _make_validator,
    _position_eval,
    _weighted,
)
from orbit_war.eval.features import surplus_ships
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
    trade_down_strike_template,
)
from orbit_war.sim.observation import GameView

logger = logging.getLogger(__name__)

_WEIGHTS_PATH = (
    Path(__file__).parent.parent / "tuning" / "step_weights" / "v5.json"
)

_FALLBACK = dict(_V4_WEIGHTS)


def _load_template_weights() -> dict[str, float]:
    if not _WEIGHTS_PATH.exists():
        return dict(_FALLBACK)
    try:
        payload = json.loads(_WEIGHTS_PATH.read_text())
        template_names = payload["template_names"]
        weights = payload["weights"]
        n_templates = len(template_names)
        per_template = weights[:n_templates]

        def _clamp(v: float, lo: float = 0.5, hi: float = 5.0) -> float:
            return max(lo, min(hi, abs(v)))

        result = {"no_op": 0.0}
        for name, w in zip(template_names, per_template):
            result[name] = _clamp(w * 10)  # scale up since deltas are small
        return result
    except (KeyError, ValueError, OSError) as exc:
        logger.warning("heuristic_v5: failed to load %s (%s); falling back", _WEIGHTS_PATH, exc)
        return dict(_FALLBACK)


TEMPLATE_WEIGHTS: dict[str, float] = _load_template_weights()


def agent(obs) -> list[list]:
    view = GameView.from_obs(obs)

    candidates: list[Step] = []
    candidates.extend(_weighted(no_op_template(view), TEMPLATE_WEIGHTS.get("no_op", 0.0)))
    candidates.extend(_weighted(production_attack_template(view), TEMPLATE_WEIGHTS.get("production_attack", 1.0)))
    candidates.extend(_weighted(defensive_reinforce_template(view), TEMPLATE_WEIGHTS.get("defensive_reinforce", 2.0)))
    candidates.extend(_weighted(snipe_undefended_template(view), TEMPLATE_WEIGHTS.get("snipe_undefended", 1.5)))
    candidates.extend(_weighted(multi_source_consolidation_template(view), TEMPLATE_WEIGHTS.get("multi_source_consolidation", 1.2)))
    candidates.extend(_weighted(comet_rush_template(view), TEMPLATE_WEIGHTS.get("comet_rush", 0.8)))
    candidates.extend(_weighted(trade_down_strike_template(view), TEMPLATE_WEIGHTS.get("trade_down_strike", 0.9)))

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

- [ ] **Step 6: Add v5 to the CLI zoo**

Edit `orbit_war/eval_harness/cli.py`. Add to `ZOO_BOT_PATHS`:

```python
    "heuristic_v5": "orbit_war.bots.heuristic_v5:agent",
```

- [ ] **Step 7: Run heuristic_v5 tests**

Run: `uv run pytest tests/test_heuristic_v5.py -v`
Expected: 2 passed (1-3 minutes).

If `test_heuristic_v5_does_not_regress_against_v4` fails: the per-step regression overfit. Edit `_load_template_weights` to always return `_FALLBACK` (i.e., behave identically to v4). Document in the commit message: "Note: per-step regression did not improve over v4; v5 ships with v4 weights as fallback. W5 may try other tuning strategies (CMA-ES, gradient-free optimization)."

- [ ] **Step 8: Commit**

```bash
git add scripts/fit_heuristic_v5_weights.py orbit_war/tuning/step_weights/v5.json orbit_war/bots/heuristic_v5.py tests/test_heuristic_v5.py orbit_war/eval_harness/cli.py
git commit -m "<see below>"
```

Commit message:
- If v5 beat v4: `"Add heuristic_v5 with per-step regression-fit template weights"`
- If v5 fell back: `"Add heuristic_v5 scaffold (per-step fit did not improve over v4; W5 will try other tuning)"`

---

## Task 8: W4 closing — gate, champion designation, optional ladder submit

**Files:**
- Modify: `submissions.log`
- Modify: `CLAUDE.md`
- Tag: `w4-baseline`

- [ ] **Step 1: Pick the W4 champion**

Whichever of v4/v5 won more in self-play tests. Default: v4 if v5 fell back.

Set the variable `W4_CHAMPION` for use in subsequent steps:
- If v5 beat v4: `W4_CHAMPION=heuristic_v5`
- If v5 fell back: `W4_CHAMPION=heuristic_v4`

- [ ] **Step 2: Run the full gate against the W4 champion**

```bash
uv run ow-gate orbit_war.bots.${W4_CHAMPION}:agent \
    --champion orbit_war.bots.heuristic_v3:agent \
    --seeds 25 --workers 4 2>&1 | grep -v "open_spiel\|Loading environment\|^$"
```

(Substitute `${W4_CHAMPION}` with the actual bot name.)

Capture the OVERALL line + per-tier results. PASTE THEM IN YOUR REPORT.

- [ ] **Step 3: Submission decision**

If OVERALL is PASS or the champion tier (vs v3) is ≥60%, submit:

```bash
./scripts/submit_bot.sh ${W4_CHAMPION} orbit_war.bots.${W4_CHAMPION}:agent "W4: consolidation fix + adversarial validator + trade_down + (optional v5 weights)"
```

The script runs the gate again before submitting. If it fails the internal gate (which uses greedy as default champion), it aborts.

If you choose NOT to submit, append a one-line audit record:

```bash
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SHA="$(git rev-parse --short HEAD)"
echo "$TS  ${W4_CHAMPION}  $SHA  W4 result: <fill in based on the run>; not submitted" >> submissions.log
```

- [ ] **Step 4: Tag**

```bash
git tag w4-baseline
```

- [ ] **Step 5: Update CLAUDE.md**

Find the workflow line:

```
- W3 champion: heuristic_v3 (orbit-aware aim + 6 templates + 15-turn plan validation). Use `uv run ow-gate orbit_war.bots.heuristic_v3:agent` to gate challengers.
```

Replace with:

```
- W4 champion: <heuristic_v4|v5> (consolidation fix + adversarial validator + trade_down + per-step weights if v5 won). Use `uv run ow-gate orbit_war.bots.<W4_CHAMPION>:agent` to gate challengers.
```

(Fill in actual champion name.)

- [ ] **Step 6: Run the full test suite once more**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add submissions.log CLAUDE.md
git commit -m "W4 closing: tag w4-baseline; designate <champion> as W4 champion"
```

---

## Closing checklist

- [ ] Run the full test suite: `uv run pytest -q`. Expected: all tests pass.
- [ ] `git log --oneline w4-baseline ^w3-baseline | wc -l` shows ~9-12 commits.
- [ ] Capture realistic numbers: champion vs (random / starter / greedy / v1 / v3 / public_tactical). Save in a comment on the W4 closing commit.
- [ ] Note any W5 follow-ups (e.g., "v5 fit failed because X — try CMA-ES").
