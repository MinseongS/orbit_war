# W1: Infrastructure + Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the foundation for the Orbit Wars bot — a tested observation parser, closed-form orbit prediction, a 3-bot zoo, parallel self-play match runner, stratified evaluation gate, and the starter bot submitted to the Kaggle ladder.

**Architecture:** Use `kaggle_environments` directly as the simulator (it's 812 lines of pure Python, no native deps — re-implementation deferred to a later week if profiling demands it). Build a thin `orbit_war` package on top: typed observation views, an analytical orbit-position function (parity-tested against the official simulator), a bot zoo (random/starter/greedy), and a multiprocessing match harness with Wilson confidence intervals and a stratified submission gate. End the week by submitting the starter bot to start collecting ladder data.

**Tech Stack:** Python 3.13, `kaggle-environments>=1.28.0`, `numpy`, `pytest`, `multiprocessing`, `kaggle` CLI. Package management via `uv`.

---

## File structure

Files created (all under repo root `/Users/minseong/project/orbit_war/`):

- `orbit_war/__init__.py` — package marker
- `orbit_war/sim/__init__.py`
- `orbit_war/sim/observation.py` — typed views (`GameView`) over the raw obs dict
- `orbit_war/sim/orbits.py` — closed-form `planet_position_at(planet, turn, angular_velocity)` + precompute table builder
- `orbit_war/bots/__init__.py`
- `orbit_war/bots/random_bot.py` — uniform random legal-action bot
- `orbit_war/bots/starter_bot.py` — Nearest Planet Sniper logic from `starter_kit/main.py`, wrapped for our package
- `orbit_war/bots/greedy_baseline.py` — production-per-distance greedy
- `orbit_war/bots/public_tactical.py` — translation of the public `sigmaborov/orbit-wars-2026-tactical-heuristic` notebook
- `orbit_war/eval_harness/__init__.py`
- `orbit_war/eval_harness/match.py` — `play_match(bot_a, bot_b, seed) -> MatchResult`
- `orbit_war/eval_harness/parallel.py` — `run_pairs(bot_a, bot_b, seeds, workers) -> list[MatchResult]`
- `orbit_war/eval_harness/stats.py` — Wilson CI, two-proportion z-test sample-size calculator, mirrored-pair aggregator
- `orbit_war/eval_harness/gate.py` — stratified `gate(challenger, zoo) -> GateReport`
- `tests/__init__.py`
- `tests/conftest.py` — shared fixtures (seeded simulator, fast-game config)
- `tests/test_orbits.py`
- `tests/test_observation.py`
- `tests/test_match.py`
- `tests/test_stats.py`
- `tests/test_gate.py`
- `scripts/pull_public_tactical.sh` — downloads the public notebook via `kaggle kernels pull`
- `scripts/submit_starter.sh` — first ladder submission of the starter
- `submissions.log` — append-only submission record

Files modified:

- `pyproject.toml` — declare package layout (`[tool.hatch.build.targets.wheel] packages = ["orbit_war"]`) and pytest config
- `CLAUDE.md` — add `Run tests:` line

---

## Task 1: Package skeleton + pytest wiring

**Files:**
- Create: `orbit_war/__init__.py`
- Create: `orbit_war/sim/__init__.py`
- Create: `orbit_war/bots/__init__.py`
- Create: `orbit_war/eval_harness/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `pyproject.toml`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p orbit_war/sim orbit_war/bots orbit_war/eval_harness tests
touch orbit_war/__init__.py orbit_war/sim/__init__.py orbit_war/bots/__init__.py orbit_war/eval_harness/__init__.py tests/__init__.py
```

- [ ] **Step 2: Append package + pytest config to `pyproject.toml`**

Append at end of file:

```toml

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["orbit_war"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
filterwarnings = [
    "ignore::DeprecationWarning:kaggle_environments.*",
]
```

- [ ] **Step 3: Add minimal conftest with a fast-game config fixture**

Create `tests/conftest.py`:

```python
"""Shared fixtures for the orbit_war test suite."""

import pytest


@pytest.fixture
def fast_config() -> dict:
    """A short-episode config used by tests that don't need a 500-turn game."""
    return {"episodeSteps": 30, "actTimeout": 5, "seed": 1234}


@pytest.fixture
def seed_42_config() -> dict:
    """Standard 500-turn game on a fixed seed for parity tests."""
    return {"seed": 42}
```

- [ ] **Step 4: Add a smoke test that imports the empty package**

Create `tests/test_package_smoke.py`:

```python
"""Smoke test: package imports cleanly."""

import orbit_war
import orbit_war.sim
import orbit_war.bots
import orbit_war.eval_harness


def test_package_importable():
    assert orbit_war is not None
    assert orbit_war.sim is not None
    assert orbit_war.bots is not None
    assert orbit_war.eval_harness is not None
```

- [ ] **Step 5: Run the smoke test to verify**

Run: `uv run pytest tests/test_package_smoke.py -v`
Expected: `1 passed`.

- [ ] **Step 6: Add a `Run tests:` line to CLAUDE.md**

Edit the workflow section of `CLAUDE.md` and add a third bullet at the bottom of the workflow list:

```markdown
- Run tests with `uv run pytest -q`.
```

- [ ] **Step 7: Commit**

```bash
git add orbit_war tests pyproject.toml CLAUDE.md
git commit -m "Scaffold orbit_war package and pytest harness"
```

---

## Task 2: Typed observation parser

**Files:**
- Create: `orbit_war/sim/observation.py`
- Test: `tests/test_observation.py`

The official simulator emits `obs` dicts with raw lists for `planets`, `fleets`, `comets`. Wrap them in a typed view that exposes named-tuple access plus a few common queries we'll reuse everywhere.

- [ ] **Step 1: Write the failing test**

Create `tests/test_observation.py`:

```python
"""Tests for the typed observation view."""

import math

from kaggle_environments import make

from orbit_war.sim.observation import GameView


def test_gameview_parses_first_step():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]

    view = GameView.from_obs(obs)

    assert view.player == 0
    assert len(view.planets) >= 8  # at least 2 starting groups of 4
    assert all(p.id == i for i, p in enumerate(view.planets))
    assert any(p.owner == 0 for p in view.planets), "player 0 should own a home planet"
    assert any(p.owner == 1 for p in view.planets), "player 1 should own a home planet"
    assert isinstance(view.angular_velocity, float)


def test_gameview_my_planets_and_targets():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]

    view = GameView.from_obs(obs)

    assert len(view.my_planets()) >= 1
    assert all(p.owner == view.player for p in view.my_planets())
    assert all(p.owner != view.player for p in view.targets())


def test_gameview_distance_uses_euclidean():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    obs = env.steps[0][0]["observation"]

    view = GameView.from_obs(obs)
    a, b = view.planets[0], view.planets[1]

    expected = math.hypot(a.x - b.x, a.y - b.y)
    assert view.distance(a, b) == expected
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/test_observation.py -v`
Expected: ImportError on `orbit_war.sim.observation`.

- [ ] **Step 3: Implement `GameView`**

Create `orbit_war/sim/observation.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_observation.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/sim/observation.py tests/test_observation.py
git commit -m "Add typed GameView over orbit_wars observations"
```

---

## Task 3: Closed-form orbit position with parity test

**Files:**
- Create: `orbit_war/sim/orbits.py`
- Test: `tests/test_orbits.py`

A planet either rotates around `(50, 50)` at the game's `angular_velocity` (if `orbital_radius + planet_radius < 50`), or is static. Express both cases as a closed form keyed off the **initial** planet position and the current turn. This is the foundation for plan-arrival prediction in later weeks.

- [ ] **Step 1: Write a failing test for static planets**

Create `tests/test_orbits.py`:

```python
"""Tests for closed-form orbit position and the precompute table."""

import math

import pytest
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    ROTATION_RADIUS_LIMIT,
    Planet,
)

from orbit_war.sim.observation import GameView
from orbit_war.sim.orbits import (
    is_orbiting,
    planet_position_at,
    precompute_position_table,
)


def _run_game_seed(seed: int):
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run(["random", "random"])
    return env


def test_static_planet_position_unchanged_over_time():
    env = _run_game_seed(seed=7)
    obs0 = env.steps[0][0]["observation"]
    view = GameView.from_obs(obs0)
    angular_velocity = view.angular_velocity

    static = next(p for p in view.initial_planets if not is_orbiting(p))

    for turn in (0, 1, 50, 250, 499):
        x, y = planet_position_at(static, turn, angular_velocity)
        assert math.isclose(x, static.x)
        assert math.isclose(y, static.y)


def test_orbiting_planet_matches_official_sim_at_each_recorded_step():
    env = _run_game_seed(seed=7)
    obs0 = env.steps[0][0]["observation"]
    view0 = GameView.from_obs(obs0)
    angular_velocity = view0.angular_velocity
    initial_by_id = {p.id: p for p in view0.initial_planets}

    orbiting_ids = [p.id for p in view0.initial_planets if is_orbiting(p)]
    assert orbiting_ids, "seed 7 should produce at least one orbiting planet"

    # Sample a handful of turns from the recorded episode (skip turn 0 — trivial).
    sample_turns = [25, 100, 250, 400]
    for turn in sample_turns:
        recorded_obs = env.steps[turn][0]["observation"]
        recorded = {p[0]: Planet(*p) for p in recorded_obs["planets"]}
        for pid in orbiting_ids:
            initial = initial_by_id[pid]
            expected_x = recorded[pid].x
            expected_y = recorded[pid].y
            actual_x, actual_y = planet_position_at(initial, turn, angular_velocity)
            assert math.isclose(actual_x, expected_x, abs_tol=1e-9), (
                f"x drift on planet {pid} at turn {turn}"
            )
            assert math.isclose(actual_y, expected_y, abs_tol=1e-9), (
                f"y drift on planet {pid} at turn {turn}"
            )


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 2026])
def test_precompute_table_matches_per_step_function(seed: int):
    env = _run_game_seed(seed=seed)
    obs0 = env.steps[0][0]["observation"]
    view = GameView.from_obs(obs0)

    table = precompute_position_table(
        initial_planets=view.initial_planets,
        angular_velocity=view.angular_velocity,
        max_turn=500,
    )

    sample_turns = [0, 1, 17, 123, 300, 499]
    for p in view.initial_planets:
        for t in sample_turns:
            tx, ty = table[p.id][t]
            ex, ey = planet_position_at(p, t, view.angular_velocity)
            assert math.isclose(tx, ex)
            assert math.isclose(ty, ey)


def test_is_orbiting_uses_official_threshold():
    static_like = Planet(0, -1, 5, 5, 1.0, 50, 1)
    orbiting_like = Planet(0, -1, CENTER + 10, CENTER, 1.0, 50, 1)
    assert is_orbiting(orbiting_like)
    far = Planet(0, -1, CENTER + ROTATION_RADIUS_LIMIT, CENTER, 1.0, 50, 1)
    assert not is_orbiting(far)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/test_orbits.py -v`
Expected: ImportError on `orbit_war.sim.orbits`.

- [ ] **Step 3: Implement `is_orbiting`, `planet_position_at`, and `precompute_position_table`**

Create `orbit_war/sim/orbits.py`:

```python
"""Closed-form planet position prediction.

Orbit Wars planets either:
  - rotate around the central sun at a fixed angular velocity, when
    `orbital_radius + planet_radius < ROTATION_RADIUS_LIMIT`; or
  - are static.

Both cases are expressible in closed form from the planet's initial
position and the current turn. We never iterate to predict positions —
we look them up.
"""

from __future__ import annotations

import math
from typing import Iterable

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    ROTATION_RADIUS_LIMIT,
    Planet,
)


def is_orbiting(planet: Planet) -> bool:
    """True iff the planet rotates around the sun."""
    dx = planet.x - CENTER
    dy = planet.y - CENTER
    orbital_radius = math.hypot(dx, dy)
    return orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT


def planet_position_at(
    initial: Planet, turn: int, angular_velocity: float
) -> tuple[float, float]:
    """Return (x, y) for a planet at the given turn, given its initial state.

    `initial` must be the planet's snapshot at turn 0 (i.e. an entry of
    `obs.initial_planets`). For static planets the initial position is
    returned unchanged.
    """
    if not is_orbiting(initial):
        return initial.x, initial.y

    dx = initial.x - CENTER
    dy = initial.y - CENTER
    orbital_radius = math.hypot(dx, dy)
    initial_angle = math.atan2(dy, dx)
    angle = initial_angle + angular_velocity * turn
    return (
        CENTER + orbital_radius * math.cos(angle),
        CENTER + orbital_radius * math.sin(angle),
    )


def precompute_position_table(
    initial_planets: Iterable[Planet],
    angular_velocity: float,
    max_turn: int,
) -> dict[int, list[tuple[float, float]]]:
    """Return {planet_id: [(x, y) for turn in 0..max_turn]} for fast lookup.

    Built once at the start of a game; reused for every search call.
    """
    table: dict[int, list[tuple[float, float]]] = {}
    for p in initial_planets:
        if not is_orbiting(p):
            table[p.id] = [(p.x, p.y)] * (max_turn + 1)
            continue
        dx = p.x - CENTER
        dy = p.y - CENTER
        orbital_radius = math.hypot(dx, dy)
        initial_angle = math.atan2(dy, dx)
        positions = []
        for t in range(max_turn + 1):
            angle = initial_angle + angular_velocity * t
            positions.append(
                (
                    CENTER + orbital_radius * math.cos(angle),
                    CENTER + orbital_radius * math.sin(angle),
                )
            )
        table[p.id] = positions
    return table
```

- [ ] **Step 4: Run the orbit tests to verify they pass**

Run: `uv run pytest tests/test_orbits.py -v`
Expected: all 8 parametrized + scalar tests pass.

If any seed fails on the parity check, treat the failure as a real bug — investigate before proceeding. The official simulator's planet rotation is the source of truth.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/sim/orbits.py tests/test_orbits.py
git commit -m "Add closed-form planet position with parity vs official sim"
```

---

## Task 4: Bot zoo — random, starter, greedy

**Files:**
- Create: `orbit_war/bots/random_bot.py`
- Create: `orbit_war/bots/starter_bot.py`
- Create: `orbit_war/bots/greedy_baseline.py`
- Test: `tests/test_bots.py`

Each zoo bot exports a callable `agent(obs)` that returns a list of moves. We keep them dependency-light so they can run in worker processes.

- [ ] **Step 1: Write a failing test that exercises all three bots**

Create `tests/test_bots.py`:

```python
"""Smoke + property tests for the bot zoo."""

from kaggle_environments import make

from orbit_war.bots import greedy_baseline, random_bot, starter_bot


def _first_obs():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    return env.steps[0][0]["observation"]


def test_random_bot_returns_list():
    obs = _first_obs()
    moves = random_bot.agent(obs)
    assert isinstance(moves, list)
    for move in moves:
        assert len(move) == 3
        from_id, angle, ships = move
        assert isinstance(from_id, int)
        assert isinstance(angle, float)
        assert isinstance(ships, int)


def test_starter_bot_runs_a_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([starter_bot.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_greedy_baseline_runs_a_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([greedy_baseline.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_greedy_beats_random_on_a_few_seeds():
    """Property: a sane greedy bot should beat a uniform-random bot
    in a clear majority of games. If this fails, greedy has a bug."""
    wins = 0
    for seed in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        env = make("orbit_wars", configuration={"seed": seed}, debug=True)
        env.run([greedy_baseline.agent, random_bot.agent])
        rewards = [s.reward for s in env.steps[-1]]
        if rewards[0] > rewards[1]:
            wins += 1
    assert wins >= 8, f"greedy only beat random {wins}/10 seeds"
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/test_bots.py -v`
Expected: ImportError on `orbit_war.bots.random_bot` and friends.

- [ ] **Step 3: Implement `random_bot`**

Create `orbit_war/bots/random_bot.py`:

```python
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
```

- [ ] **Step 4: Implement `starter_bot`**

Create `orbit_war/bots/starter_bot.py` (Nearest Planet Sniper, packaged):

```python
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
```

- [ ] **Step 5: Implement `greedy_baseline`**

Create `orbit_war/bots/greedy_baseline.py`:

```python
"""Production-per-distance greedy.

For each owned planet we score every non-owned planet by
    score = production / (1 + distance)
and capture the highest-scoring one we can afford with `target.ships + 1`
ships. Slightly stronger than the starter sniper because it prefers
high-production neutrals over merely-close ones."""

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
        scored = sorted(
            targets,
            key=lambda t: t.production / (1.0 + GameView.distance(mine, t)),
            reverse=True,
        )
        for t in scored:
            ships_needed = t.ships + 1
            if mine.ships >= ships_needed:
                angle = math.atan2(t.y - mine.y, t.x - mine.x)
                moves.append([int(mine.id), float(angle), int(ships_needed)])
                break
    return moves
```

- [ ] **Step 6: Run the bot tests**

Run: `uv run pytest tests/test_bots.py -v`
Expected: 4 passed (the property test runs 10 games, takes ~10–30 s).

If `test_greedy_beats_random_on_a_few_seeds` fails, debug the greedy bot before proceeding — it's the lowest expected-skill member of the zoo and failing here indicates a real bug.

- [ ] **Step 7: Commit**

```bash
git add orbit_war/bots/random_bot.py orbit_war/bots/starter_bot.py orbit_war/bots/greedy_baseline.py tests/test_bots.py
git commit -m "Add random/starter/greedy bot zoo with smoke + property tests"
```

---

## Task 5: Single-game match runner

**Files:**
- Create: `orbit_war/eval_harness/match.py`
- Test: `tests/test_match.py`

A `MatchResult` carries everything we'll later aggregate: who won, score margin, max turn time, error status.

- [ ] **Step 1: Write the failing test**

Create `tests/test_match.py`:

```python
"""Tests for the single-game match runner."""

from orbit_war.bots import greedy_baseline, random_bot, starter_bot
from orbit_war.eval_harness.match import MatchResult, play_match


def test_play_match_returns_result_with_correct_seed():
    result = play_match(starter_bot.agent, random_bot.agent, seed=42)
    assert isinstance(result, MatchResult)
    assert result.seed == 42
    assert result.winner in (0, 1, None)


def test_play_match_records_score_margin():
    result = play_match(greedy_baseline.agent, random_bot.agent, seed=1)
    assert result.score_a + result.score_b >= 0
    assert result.score_margin == result.score_a - result.score_b


def test_play_match_detects_errors_per_side():
    def crashing_agent(obs):
        raise RuntimeError("boom")

    result = play_match(crashing_agent, random_bot.agent, seed=1)
    assert result.error_a is True
    assert result.error_b is False
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `uv run pytest tests/test_match.py -v`
Expected: ImportError on `orbit_war.eval_harness.match`.

- [ ] **Step 3: Implement `play_match`**

Create `orbit_war/eval_harness/match.py`:

```python
"""Run a single Orbit Wars game between two callable bots and report the result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kaggle_environments import make

Agent = Callable[[dict], list]


@dataclass(frozen=True)
class MatchResult:
    seed: int
    winner: int | None  # 0, 1, or None on draw
    score_a: float
    score_b: float
    error_a: bool
    error_b: bool
    turns_played: int

    @property
    def score_margin(self) -> float:
        return self.score_a - self.score_b


def play_match(
    bot_a: Agent,
    bot_b: Agent,
    seed: int,
    episode_steps: int = 500,
    act_timeout: int = 1,
) -> MatchResult:
    """Run a single 1v1 episode with bot_a as player 0 and bot_b as player 1."""
    env = make(
        "orbit_wars",
        configuration={
            "seed": seed,
            "episodeSteps": episode_steps,
            "actTimeout": act_timeout,
        },
        debug=True,
    )
    env.run([bot_a, bot_b])
    final = env.steps[-1]

    score_a = float(final[0].reward) if final[0].reward is not None else 0.0
    score_b = float(final[1].reward) if final[1].reward is not None else 0.0

    error_a = final[0].status == "ERROR"
    error_b = final[1].status == "ERROR"

    if error_a and not error_b:
        winner: int | None = 1
    elif error_b and not error_a:
        winner = 0
    elif score_a > score_b:
        winner = 0
    elif score_b > score_a:
        winner = 1
    else:
        winner = None

    return MatchResult(
        seed=seed,
        winner=winner,
        score_a=score_a,
        score_b=score_b,
        error_a=error_a,
        error_b=error_b,
        turns_played=len(env.steps),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_match.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/eval_harness/match.py tests/test_match.py
git commit -m "Add single-game match runner with error and score tracking"
```

---

## Task 6: Stats helpers — Wilson CI and sample-size calculator

**Files:**
- Create: `orbit_war/eval_harness/stats.py`
- Test: `tests/test_stats.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stats.py`:

```python
"""Tests for Wilson CI and sample-size helpers."""

import math

from orbit_war.eval_harness.stats import (
    samples_needed_for_two_proportion,
    wilson_lower_bound,
    wilson_upper_bound,
)


def test_wilson_lower_bound_zero_wins():
    # 0 wins out of 10 — lower bound must be < 0.31 (rule of three roughly).
    assert wilson_lower_bound(wins=0, n=10) == 0.0
    assert wilson_upper_bound(wins=0, n=10) > 0.0
    assert wilson_upper_bound(wins=0, n=10) < 0.31


def test_wilson_lower_bound_centered():
    # 50 wins out of 100 — lower bound should be a touch under 0.5.
    lo = wilson_lower_bound(wins=50, n=100)
    hi = wilson_upper_bound(wins=50, n=100)
    assert 0.39 < lo < 0.5
    assert 0.5 < hi < 0.61


def test_wilson_lower_bound_high_confidence():
    # 95 wins out of 100 — lower bound must clearly exceed 0.5.
    assert wilson_lower_bound(wins=95, n=100) > 0.85


def test_samples_needed_decreases_with_larger_effect():
    n_small = samples_needed_for_two_proportion(p1=0.53, p2=0.50, alpha=0.05, power=0.80)
    n_med = samples_needed_for_two_proportion(p1=0.60, p2=0.50, alpha=0.05, power=0.80)
    n_large = samples_needed_for_two_proportion(p1=0.70, p2=0.50, alpha=0.05, power=0.80)
    assert n_small > n_med > n_large
    # Sanity bounds
    assert 800 < n_small < 5000
    assert 100 < n_med < 800
    assert 30 < n_large < 200


def test_sample_size_handles_equal_proportions():
    n = samples_needed_for_two_proportion(p1=0.5, p2=0.5, alpha=0.05, power=0.80)
    assert math.isinf(n)
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `uv run pytest tests/test_stats.py -v`
Expected: ImportError on `orbit_war.eval_harness.stats`.

- [ ] **Step 3: Implement the stats helpers**

Create `orbit_war/eval_harness/stats.py`:

```python
"""Confidence intervals and sample-size calculations for win-rate comparisons.

We use:
  - Wilson score interval (better than normal-approximation, especially
    near 0 and 1) for the per-comparison CI.
  - Two-proportion z-test for sample-size targets.
"""

from __future__ import annotations

import math

# 95% two-sided z, 80% power one-sided z
Z_95 = 1.959963984540054
Z_80 = 0.8416212335729143


def _wilson(wins: int, n: int, z: float, lower: bool) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return centre - margin if lower else centre + margin


def wilson_lower_bound(wins: int, n: int, z: float = Z_95) -> float:
    """Lower bound of the Wilson score interval at confidence z (default 95%)."""
    return max(0.0, _wilson(wins, n, z, lower=True))


def wilson_upper_bound(wins: int, n: int, z: float = Z_95) -> float:
    """Upper bound of the Wilson score interval at confidence z."""
    return min(1.0, _wilson(wins, n, z, lower=False))


def samples_needed_for_two_proportion(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Return N (per group) needed to detect p1 vs p2 in a two-proportion z-test.

    Returns +inf if the proportions are equal.
    """
    if p1 == p2:
        return math.inf

    z_alpha = Z_95 if alpha == 0.05 else _z_two_sided(alpha)
    z_power = Z_80 if power == 0.80 else _z_one_sided(power)

    p_bar = (p1 + p2) / 2.0
    numerator = (
        z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_power * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    denominator = (p1 - p2) ** 2
    return math.ceil(numerator / denominator)


def _z_two_sided(alpha: float) -> float:
    # Erf-based inverse normal for the (1 - alpha/2) quantile.
    return math.sqrt(2.0) * _inv_erf(1.0 - alpha)


def _z_one_sided(power: float) -> float:
    return math.sqrt(2.0) * _inv_erf(2.0 * power - 1.0)


def _inv_erf(y: float) -> float:
    # Winitzki approximation, accurate to ~4e-3.
    a = 0.147
    ln = math.log(1.0 - y * y)
    first = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(
        math.sqrt(math.sqrt(first * first - ln / a) - first), y
    )
```

- [ ] **Step 4: Run the stats tests**

Run: `uv run pytest tests/test_stats.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/eval_harness/stats.py tests/test_stats.py
git commit -m "Add Wilson CI and two-proportion sample-size calculator"
```

---

## Task 7: Parallel mirrored-pair match runner

**Files:**
- Create: `orbit_war/eval_harness/parallel.py`
- Test: `tests/test_parallel.py`

For each seed we run the pair *both ways* (bot_a as player 0 and as player 1) so board-side bias cancels. The aggregator returns symmetric win statistics from bot_a's perspective.

- [ ] **Step 1: Write the failing test**

Create `tests/test_parallel.py`:

```python
"""Tests for the parallel mirrored-pair runner."""

from orbit_war.bots import greedy_baseline, random_bot
from orbit_war.eval_harness.parallel import PairSummary, run_mirrored_pairs


def test_run_mirrored_pairs_against_random():
    summary = run_mirrored_pairs(
        bot_a=greedy_baseline.agent,
        bot_b=random_bot.agent,
        seeds=(1, 2, 3, 4, 5),
        workers=2,
    )
    assert isinstance(summary, PairSummary)
    assert summary.games_played == 10  # 5 seeds * 2 sides
    assert 0.0 <= summary.win_rate_a <= 1.0
    assert summary.win_rate_a > 0.7  # greedy clearly beats random


def test_pair_summary_breaks_out_errors_per_bot():
    def crashing(obs):
        raise RuntimeError("boom")

    summary = run_mirrored_pairs(
        bot_a=crashing,
        bot_b=random_bot.agent,
        seeds=(1, 2),
        workers=2,
    )
    assert summary.error_rate_a == 1.0
    assert summary.error_rate_b == 0.0
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `uv run pytest tests/test_parallel.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `run_mirrored_pairs`**

Create `orbit_war/eval_harness/parallel.py`:

```python
"""Parallel mirrored-pair self-play runner.

Each seed plays *both* bots on each side; results are aggregated from the
perspective of bot_a. Uses `multiprocessing` so we scale across all CPU cores
on the host."""

from __future__ import annotations

import multiprocessing as mp
import pickle
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from orbit_war.eval_harness.match import MatchResult, play_match
from orbit_war.eval_harness.stats import wilson_lower_bound, wilson_upper_bound

Agent = Callable[[dict], list]


@dataclass(frozen=True)
class PairSummary:
    games_played: int
    wins_a: int
    wins_b: int
    draws: int
    error_rate_a: float
    error_rate_b: float
    avg_score_margin_a: float
    win_rate_lower_a: float
    win_rate_upper_a: float

    @property
    def win_rate_a(self) -> float:
        denom = self.wins_a + self.wins_b
        if denom == 0:
            return 0.5
        return self.wins_a / denom


# Workers cannot pickle local functions, so we route through module-level helpers.
def _worker(args: tuple) -> MatchResult:
    bot_a_pkl, bot_b_pkl, seed, side = args
    bot_a = pickle.loads(bot_a_pkl)
    bot_b = pickle.loads(bot_b_pkl)
    if side == 0:
        return play_match(bot_a, bot_b, seed=seed)
    flipped = play_match(bot_b, bot_a, seed=seed)
    # Re-frame from bot_a's perspective.
    return MatchResult(
        seed=seed,
        winner=(None if flipped.winner is None else 1 - flipped.winner),
        score_a=flipped.score_b,
        score_b=flipped.score_a,
        error_a=flipped.error_b,
        error_b=flipped.error_a,
        turns_played=flipped.turns_played,
    )


def run_mirrored_pairs(
    bot_a: Agent,
    bot_b: Agent,
    seeds: Iterable[int],
    workers: int = 4,
) -> PairSummary:
    """Run each seed twice — once with bot_a as player 0, once as player 1."""
    seed_list = list(seeds)
    bot_a_pkl = pickle.dumps(bot_a)
    bot_b_pkl = pickle.dumps(bot_b)
    jobs = [
        (bot_a_pkl, bot_b_pkl, seed, side)
        for seed in seed_list
        for side in (0, 1)
    ]

    if workers <= 1 or len(jobs) <= 1:
        results: Sequence[MatchResult] = [_worker(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            results = pool.map(_worker, jobs)

    return _summarize(results)


def _summarize(results: Sequence[MatchResult]) -> PairSummary:
    wins_a = sum(1 for r in results if r.winner == 0)
    wins_b = sum(1 for r in results if r.winner == 1)
    draws = sum(1 for r in results if r.winner is None)
    err_a = sum(1 for r in results if r.error_a)
    err_b = sum(1 for r in results if r.error_b)
    margin = sum(r.score_margin for r in results) / max(1, len(results))

    decisive = wins_a + wins_b
    n = max(1, len(results))
    return PairSummary(
        games_played=len(results),
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        error_rate_a=err_a / n,
        error_rate_b=err_b / n,
        avg_score_margin_a=margin,
        win_rate_lower_a=wilson_lower_bound(wins_a, max(1, decisive)),
        win_rate_upper_a=wilson_upper_bound(wins_a, max(1, decisive)),
    )
```

- [ ] **Step 4: Run the parallel tests**

Run: `uv run pytest tests/test_parallel.py -v`
Expected: 2 passed (will take ~30–60 s on multiprocessing startup + 10 games).

If pickling the agent functions fails (it can on some Python setups when bots reference closures), switch the call sites to pass module-paths instead of callables — but the lambdas above are top-level functions, so this should not occur.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/eval_harness/parallel.py tests/test_parallel.py
git commit -m "Add multiprocessing mirrored-pair match runner with summary"
```

---

## Task 8: Stratified evaluation gate

**Files:**
- Create: `orbit_war/eval_harness/gate.py`
- Test: `tests/test_gate.py`

The gate is the contract between local self-play and ladder submission. It runs the challenger against tiered opponents and *fails closed*: any tier that fails blocks submission.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gate.py`:

```python
"""Tests for the stratified submission gate."""

from orbit_war.bots import greedy_baseline, random_bot, starter_bot
from orbit_war.eval_harness.gate import GateReport, GateTier, evaluate_gate


def test_gate_passes_for_a_clearly_strong_challenger():
    # Greedy crushes random; we use a small-N config so the test runs in seconds.
    report = evaluate_gate(
        challenger=greedy_baseline.agent,
        sanity_pool={"random": random_bot.agent},
        diversity_pool={},
        champion=random_bot.agent,
        sanity_min_win_rate=0.80,
        diversity_min_win_rate=0.55,
        champion_min_win_rate=0.55,
        seeds_per_pool=(1, 2, 3),
    )
    assert isinstance(report, GateReport)
    assert report.passed is True
    sanity_results = [t for t in report.tiers if t.name == "sanity:random"]
    assert sanity_results and sanity_results[0].passed


def test_gate_fails_when_challenger_is_weaker_than_champion():
    # Random vs greedy: random will lose decisively. Champion tier must fail.
    report = evaluate_gate(
        challenger=random_bot.agent,
        sanity_pool={},
        diversity_pool={},
        champion=greedy_baseline.agent,
        sanity_min_win_rate=0.80,
        diversity_min_win_rate=0.55,
        champion_min_win_rate=0.55,
        seeds_per_pool=(1, 2, 3),
    )
    assert report.passed is False
    failed = [t for t in report.tiers if not t.passed]
    assert any(t.name == "champion" for t in failed)


def test_gate_marks_sanity_failure_explicitly():
    # Starter vs starter: rough parity, sanity demand at 80% should fail.
    report = evaluate_gate(
        challenger=starter_bot.agent,
        sanity_pool={"twin": starter_bot.agent},
        diversity_pool={},
        champion=starter_bot.agent,
        sanity_min_win_rate=0.80,
        diversity_min_win_rate=0.55,
        champion_min_win_rate=0.55,
        seeds_per_pool=(1, 2),
    )
    sanity = next(t for t in report.tiers if t.name == "sanity:twin")
    assert sanity.passed is False
    assert report.passed is False
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `uv run pytest tests/test_gate.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `evaluate_gate`**

Create `orbit_war/eval_harness/gate.py`:

```python
"""Stratified evaluation gate.

A challenger must clear all configured tiers (sanity, diversity, champion)
before it may be submitted to the ladder. The gate fails closed: any tier
that does not meet its minimum win rate blocks the submission as a whole."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

from orbit_war.eval_harness.parallel import PairSummary, run_mirrored_pairs

Agent = Callable[[dict], list]


@dataclass(frozen=True)
class GateTier:
    name: str
    passed: bool
    summary: PairSummary
    required_win_rate: float


@dataclass(frozen=True)
class GateReport:
    passed: bool
    tiers: tuple[GateTier, ...]


def evaluate_gate(
    challenger: Agent,
    sanity_pool: Mapping[str, Agent],
    diversity_pool: Mapping[str, Agent],
    champion: Agent,
    sanity_min_win_rate: float = 0.95,
    diversity_min_win_rate: float = 0.55,
    champion_min_win_rate: float = 0.55,
    seeds_per_pool: Iterable[int] = tuple(range(20)),
    workers: int = 4,
) -> GateReport:
    """Run the challenger through three tiers and report pass/fail per tier."""
    tier_results: list[GateTier] = []
    seeds = tuple(seeds_per_pool)

    for name, opponent in sanity_pool.items():
        summary = run_mirrored_pairs(challenger, opponent, seeds, workers=workers)
        tier_results.append(
            GateTier(
                name=f"sanity:{name}",
                passed=summary.win_rate_a >= sanity_min_win_rate,
                summary=summary,
                required_win_rate=sanity_min_win_rate,
            )
        )

    for name, opponent in diversity_pool.items():
        summary = run_mirrored_pairs(challenger, opponent, seeds, workers=workers)
        tier_results.append(
            GateTier(
                name=f"diversity:{name}",
                passed=summary.win_rate_a >= diversity_min_win_rate,
                summary=summary,
                required_win_rate=diversity_min_win_rate,
            )
        )

    champ_summary = run_mirrored_pairs(challenger, champion, seeds, workers=workers)
    tier_results.append(
        GateTier(
            name="champion",
            passed=champ_summary.win_rate_a >= champion_min_win_rate,
            summary=champ_summary,
            required_win_rate=champion_min_win_rate,
        )
    )

    overall = all(t.passed for t in tier_results)
    return GateReport(passed=overall, tiers=tuple(tier_results))
```

- [ ] **Step 4: Run the gate tests**

Run: `uv run pytest tests/test_gate.py -v`
Expected: 3 passed (will take ~1–3 minutes total — these run real games).

- [ ] **Step 5: Commit**

```bash
git add orbit_war/eval_harness/gate.py tests/test_gate.py
git commit -m "Add stratified evaluation gate with sanity/diversity/champion tiers"
```

---

## Task 9: Pull and translate the public Tactical Heuristic notebook

**Files:**
- Create: `scripts/pull_public_tactical.sh`
- Create: `orbit_war/bots/public_tactical.py` (manual translation step)
- Test: `tests/test_public_tactical.py`

The `sigmaborov/orbit-wars-2026-tactical-heuristic` Kaggle notebook is the de-facto floor every public ladder bot is beating. We replicate it as a zoo opponent so we can measure ourselves against the public baseline.

- [ ] **Step 1: Write the puller script**

Create `scripts/pull_public_tactical.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

DEST="${DEST:-./vendor/public_tactical}"
mkdir -p "$DEST"
uv run kaggle kernels pull sigmaborov/orbit-wars-2026-tactical-heuristic -p "$DEST" -m
echo "Notebook pulled to $DEST"
ls -la "$DEST"
```

Make it executable:

```bash
chmod +x scripts/pull_public_tactical.sh
```

- [ ] **Step 2: Pull the notebook**

Run: `./scripts/pull_public_tactical.sh`
Expected: a notebook file (`.ipynb` or `.py`) lands in `vendor/public_tactical/`.

If `kaggle kernels pull` errors with `403` or `404`, the kernel slug may have changed or you may need to accept the kernel's terms once via the website. Open https://www.kaggle.com/code/sigmaborov/orbit-wars-2026-tactical-heuristic in a browser, click "Copy & Edit" or accept terms, then retry the script.

- [ ] **Step 3: Read the pulled notebook and translate the agent function**

Open the pulled file and locate the `def agent(...)` function. Copy its logic into `orbit_war/bots/public_tactical.py`, replacing any in-notebook helper imports with the equivalents in our package (`from orbit_war.sim.observation import GameView` if appropriate).

Create `orbit_war/bots/public_tactical.py` with the translated body:

```python
"""Translation of the public Kaggle notebook
sigmaborov/orbit-wars-2026-tactical-heuristic.

This is the public-ladder floor: most baseline submissions on the leaderboard
are at or near this strategy. We keep it in the bot zoo as our floor opponent.

NOTE: this file is a manual translation of the upstream notebook fetched
via scripts/pull_public_tactical.sh. Re-run the puller and diff if you want
to refresh against an updated upstream."""

from __future__ import annotations

# --- BEGIN: replace this block with the translated agent body ---
def agent(obs):
    raise NotImplementedError(
        "Translate the pulled notebook's agent function into this body. "
        "See scripts/pull_public_tactical.sh and vendor/public_tactical/."
    )
# --- END ---
```

After translation, the body must be a real function — remove the `NotImplementedError`. **Do not commit the placeholder version.** If you cannot pull the notebook (private/removed), document the failure in this docstring and pause to ask whether to substitute a different public bot.

- [ ] **Step 4: Write a smoke test for the translated bot**

Create `tests/test_public_tactical.py`:

```python
"""Smoke tests for the translated public Tactical Heuristic bot."""

from kaggle_environments import make

from orbit_war.bots import public_tactical, random_bot
from orbit_war.eval_harness.parallel import run_mirrored_pairs


def test_public_tactical_runs_without_errors():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([public_tactical.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_public_tactical_beats_random_majority():
    summary = run_mirrored_pairs(
        bot_a=public_tactical.agent,
        bot_b=random_bot.agent,
        seeds=tuple(range(10)),
        workers=2,
    )
    # Public floor must comfortably beat random.
    assert summary.win_rate_a >= 0.85, f"public_tactical only won {summary.win_rate_a:.0%} vs random"
```

- [ ] **Step 5: Run the smoke test**

Run: `uv run pytest tests/test_public_tactical.py -v`
Expected: 2 passed. If `test_public_tactical_beats_random_majority` fails, the translation is off — diff against the pulled notebook before continuing.

- [ ] **Step 6: Add `vendor/` to `.gitignore`**

Append to `.gitignore`:

```
vendor/
```

- [ ] **Step 7: Commit**

```bash
git add scripts/pull_public_tactical.sh orbit_war/bots/public_tactical.py tests/test_public_tactical.py .gitignore
git commit -m "Add public Tactical Heuristic translation as zoo floor opponent"
```

---

## Task 10: Submit the starter bot to the ladder

**Files:**
- Create: `scripts/submit_starter.sh`
- Modify: `main.py`
- Create: `submissions.log`

This is an ops task with no test suite. The goal is to put a bot on the ladder so TrueSkill data starts accumulating.

- [ ] **Step 1: Replace root `main.py` with a thin wrapper around our packaged starter**

Replace the contents of `/Users/minseong/project/orbit_war/main.py` with:

```python
"""Submission entry point. Re-exports the starter agent from our package.

When we submit a bundle to Kaggle, this is the file the platform invokes."""

from orbit_war.bots.starter_bot import agent  # noqa: F401
```

- [ ] **Step 2: Sanity-run the new entry point locally**

Run:

```bash
uv run python -c "
from kaggle_environments import make
from main import agent
env = make('orbit_wars', configuration={'seed': 42}, debug=True)
env.run([agent, 'random'])
print([(i, s.reward, s.status) for i, s in enumerate(env.steps[-1])])
"
```

Expected: a single line of three-tuples ending with both statuses `DONE`.

- [ ] **Step 3: Bundle main.py + the orbit_war package as a tar.gz**

Create `scripts/submit_starter.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUNDLE="$(mktemp -d)/submission.tar.gz"
tar --exclude='__pycache__' --exclude='*.pyc' \
    -czf "$BUNDLE" main.py orbit_war

uv run kaggle competitions submit orbit-wars \
    -f "$BUNDLE" \
    -m "W1 baseline: starter Nearest Planet Sniper, packaged"

echo "Submitted $BUNDLE"

# Append a one-line audit record.
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SHA="$(git rev-parse --short HEAD)"
echo "$TS  starter  $SHA  W1 baseline submission" >> submissions.log
```

Make it executable:

```bash
chmod +x scripts/submit_starter.sh
```

- [ ] **Step 4: Run the submission**

Run: `./scripts/submit_starter.sh`
Expected: Kaggle CLI confirms upload; `submissions.log` gains a line.

If the CLI prints a `403` Forbidden, you have not accepted the competition rules — open https://www.kaggle.com/competitions/orbit-wars/rules and click the join button, then retry.

- [ ] **Step 5: Verify the submission shows up**

Run: `uv run kaggle competitions submissions orbit-wars`
Expected: a row showing the new submission, status `pending` or `running`. The validation episode against itself takes ~5 minutes.

- [ ] **Step 6: Commit the submission scaffolding**

```bash
git add main.py scripts/submit_starter.sh submissions.log
git commit -m "Submit packaged starter bot to ladder; record submission log"
```

---

## Task 11: Wire up an end-to-end gate run as a CLI

**Files:**
- Create: `orbit_war/eval_harness/cli.py`
- Modify: `pyproject.toml` (add a `[project.scripts]` entry)
- Test: manual smoke test

Provide a single command that runs the full stratified gate against the W1 zoo so we don't manually compose calls each time.

- [ ] **Step 1: Implement the CLI**

Create `orbit_war/eval_harness/cli.py`:

```python
"""CLI: run the stratified gate against the W1 zoo and print a report."""

from __future__ import annotations

import argparse
import importlib
from typing import Callable

from orbit_war.eval_harness.gate import evaluate_gate

ZOO_BOT_PATHS: dict[str, str] = {
    "random": "orbit_war.bots.random_bot:agent",
    "starter": "orbit_war.bots.starter_bot:agent",
    "greedy": "orbit_war.bots.greedy_baseline:agent",
    "public_tactical": "orbit_war.bots.public_tactical:agent",
}


def _load(spec: str) -> Callable:
    module_path, attr = spec.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run W1 stratified evaluation gate.")
    ap.add_argument("challenger", help="dotted path:attr of the challenger agent")
    ap.add_argument("--champion", default="orbit_war.bots.greedy_baseline:agent")
    ap.add_argument("--seeds", type=int, default=20, help="seeds per pool")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    challenger = _load(args.challenger)
    champion = _load(args.champion)

    sanity = {
        "random": _load(ZOO_BOT_PATHS["random"]),
        "starter": _load(ZOO_BOT_PATHS["starter"]),
    }
    diversity = {
        "greedy": _load(ZOO_BOT_PATHS["greedy"]),
        "public_tactical": _load(ZOO_BOT_PATHS["public_tactical"]),
    }

    report = evaluate_gate(
        challenger=challenger,
        sanity_pool=sanity,
        diversity_pool=diversity,
        champion=champion,
        seeds_per_pool=tuple(range(args.seeds)),
        workers=args.workers,
    )

    for tier in report.tiers:
        print(
            f"{tier.name:30s} "
            f"win_rate={tier.summary.win_rate_a:.3f} "
            f"games={tier.summary.games_played} "
            f"req={tier.required_win_rate:.2f} "
            f"{'PASS' if tier.passed else 'FAIL'}"
        )
    print(f"\nOVERALL: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add a `[project.scripts]` block to `pyproject.toml`**

Insert below the `[tool.pytest.ini_options]` block:

```toml

[project.scripts]
ow-gate = "orbit_war.eval_harness.cli:main"
```

- [ ] **Step 3: Sync the venv to install the script**

Run: `uv sync`
Expected: success; `ow-gate` is now available via `uv run ow-gate`.

- [ ] **Step 4: Smoke-run the CLI with greedy as challenger and random as champion**

Run:

```bash
uv run ow-gate orbit_war.bots.greedy_baseline:agent \
    --champion orbit_war.bots.random_bot:agent \
    --seeds 5 --workers 2
```

Expected: 4 tier lines printed, all `PASS`, final line `OVERALL: PASS`. Exit code 0.

- [ ] **Step 5: Commit**

```bash
git add orbit_war/eval_harness/cli.py pyproject.toml
git commit -m "Add ow-gate CLI for end-to-end stratified gate runs"
```

---

## Closing checklist (run after Task 11)

- [ ] Run the full test suite: `uv run pytest -q`. Expected: all tests pass.
- [ ] Confirm `submissions.log` has the W1 starter line.
- [ ] Confirm `uv run kaggle competitions submissions orbit-wars` shows the starter as `running` or `complete`.
- [ ] Tag the W1 baseline: `git tag w1-baseline && git log --oneline | head -15` for visibility.
- [ ] Mark W1 done in the design doc's iteration plan and prepare a one-paragraph retro to inform W2 brainstorming.
