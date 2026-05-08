# Orbit Wars Bot — Design

**Author:** minseong (solo)
**Date:** 2026-05-09
**Submission deadline:** 2026-06-23 23:59 UTC (45 days from today)
**Final leaderboard:** ~2026-07-08

## Goal

Compete in the Kaggle *Orbit Wars* simulation competition (2,318+ teams).

- **Primary target:** Kaggle gold medal (~Top 1%, ~23 teams)
- **Stretch target:** Top 10 ($5,000 prize)
- **Floor:** Silver medal (~Top 5%, ~115 teams)

Total time budget: ~120-140 hours over 45 days. Capping below the available 180h to prevent burnout and preserve a final freeze week.

## Game characteristics that drive the design

- **Bot vs bot ladder**, TrueSkill-style rating. Submissions play continuous episodes against similarly-rated opponents.
- **Hard 1-second per-turn timeout.** Constrains heavy ML inference but gives plenty of room for Python-level search if data is well-shaped.
- **Deterministic, fully observable simulator.** Every rule (orbit angle, fleet speed, combat resolution, comet trajectory) is closed-form and known in advance. *Forward simulation is exact*, which makes search dramatically more powerful than pattern-matching ML.
- **Continuous action space** (any planet × any angle × any int ship count). Cannot enumerate exhaustively. Good plans must be *generated* analytically, not searched flat.
- **500-turn games**, 5-10 symmetric planet groups, 4-fold board symmetry. Self-play matches are highly informative because of symmetry.
- **Latest-2-only submission tracking** (not best-2). Operational discipline matters as much as the bot itself: a bad late submission can permanently displace the best surviving bot.

## Approach: Heuristic core + targeted lookahead search

Rejected alternatives:
- *Pure heuristics* — caps at silver/edge-of-gold without a differentiator.
- *RL/IL* — no GPU, 4hr/day, 45 days. Iteration cycle too slow vs heuristic+search for the time budget.

Chosen: **a strong analytical heuristic core + a 1-step lookahead simulator + a portfolio/beam search over candidate launch plans.**

Core idea: each turn, generate 20-100 candidate "launch plans" (a launch plan = a list of `[from_planet, angle, ships]` triples), forward-simulate each plan against a fixed model of the opponent for `N` turns (N ≈ 30-50), score the resulting position, and execute the best plan. Refine with iterative best response: re-run with opponent's predicted best counter to your top plan.

This pattern won Halite, Lux AI, and the original Planet Wars precursors. It maps cleanly onto the engineering profile: "system that runs many fast simulations and picks the best" instead of "model that learns to play."

## Architecture

Five components, isolated and testable independently.

### 1. `sim/` — Forward simulator

A pure-Python re-implementation of the official game step function, optimized for speed and exact bitwise agreement with `kaggle-environments`.

- `Game` dataclass: planets, fleets, comets, turn, RNG seed.
- `step(game, action_per_player) -> game'`: deterministic transition, matching the rule order in `README.md`.
- `legal_moves(game, player) -> generator`: enumerate atoms (single launches), not full plans.
- Exposes orbit prediction in closed form: `planet_pos_at(planet_id, turn) -> (x, y)`. No iterative integration.
- **Tested for parity** against `kaggle_environments.envs.orbit_wars` on 100 random seeded games. Test suite is the primary correctness gate.

### 2. `eval/` — Position evaluator

Scalar value function `evaluate(game, player) -> float` used by the search to score leaf states. Not learned — handcrafted, parameterized.

Components (weighted):
- Total ships (planets + fleets)
- Production (sum of `production` of owned planets)
- Threat exposure (incoming enemy fleet ships within `T` turns)
- Frontier control (count of planets within reach of contested zones)
- Comet anticipation (expected ships from upcoming comet spawns)

Weights are constants in code, tuned by self-play sweep (CMA-ES or simple coordinate descent over a small grid).

### 3. `plan_gen/` — Launch plan generator

Given the current state, propose a portfolio of candidate plans. This is the *creative* part — diversity here is the difference between gold and silver.

Plan templates:
- **No-op** (always include — baseline)
- **Capture nearest-target with min ships** (the starter kit strategy, used as floor)
- **Production-weighted greedy** — for each owned planet, target the best `production / distance` non-owned planet
- **Defensive reinforce** — when an incoming enemy fleet is detected, send exactly enough to hold
- **Orbit-aware intercept** — predict where an orbiting planet will be at fleet arrival time, aim there
- **Snipe undefended high-prod** — if any non-owned planet has `ships < δ` and `production ≥ 3`, route ships from nearest owner
- **Comet rush** — at known spawn turns (50, 150, 250, 350, 450), pre-launch fleets toward predicted comet positions
- **Multi-source consolidation** — gather ships from multiple home planets to hit one target with overwhelming force at the same arrival turn

Generator output: a list of (plan, prior_score). The prior is used to prune before full simulation.

### 4. `search/` — Plan selector

Picks the best plan within the 1-second per-turn budget.

- **Phase 1 — Generate:** call `plan_gen` for ourselves and for opponent. Take top-K by prior (K ≈ 30).
- **Phase 2 — Simulate:** for each (our_plan, opp_plan) pair, simulate `N=30-50` turns with both plans repeating their first action and then defaulting to a fixed cheap policy. Score with `eval`.
- **Phase 3 — Solve:** treat the K×K result as a payoff matrix; pick our plan maximizing the worst-case opponent response (minimax over the matrix).
- **Phase 4 — Iterate (if budget):** add an iterative best-response refinement pass.
- **Time budget guard:** every phase checks elapsed time and returns the best plan found so far if budget is exhausted. Never time out the turn.

### 5. `harness/` — Local self-play and submission ops

Ops infrastructure. Most projects underinvest here; we will not.

- `play_match(bot_a, bot_b, n_games, seed) -> stats` — parallel self-play across CPU cores using `multiprocessing`.
- `compare(challenger, champion, n_games)` — Wilson confidence interval on win rate. **Submission gate: challenger must beat champion at p < 0.05 over ≥100 games.**
- `replay_view` — render a finished game step-by-step for human review (text grid first; HTML later if needed).
- `bench` — measure plans-per-second under the 1s budget, flag regressions.
- `submit.sh` — wraps `kaggle competitions submit`, requires the challenger to have passed `compare`, and refuses to submit otherwise. Records the submission ID and rationale to `submissions.log`.

## Data flow

```
obs (from kaggle env)
  └→ parse → Game
       └→ plan_gen.candidates(Game, me)   ──┐
       └→ plan_gen.candidates(Game, opp) ──┤
                                            ├→ search.solve → best_plan
                                            │     ↑ uses sim.step + eval
                                            └→ harness.bench (offline only)
  ← best_plan formatted as [[from_id, angle, ships], …]
```

## Iteration plan (45 days)

| Week | Calendar | Focus | Time | Deliverable |
|------|----------|-------|------|-------------|
| 1 | 05/09–05/15 | Infra + baseline | ~15h | `sim/` parity test passes; starter `main.py` submitted to ladder; `harness.compare` works |
| 2 | 05/16–05/22 | Heuristic core | ~25h | `eval/` + 3-4 plan templates beat starter ≥70% in self-play |
| 3 | 05/23–05/29 | Search v1 | ~25h | 1-step lookahead live; bot beats prior bot ≥60% |
| 4 | 05/30–06/05 | Plan diversity | ~20h | All 8 plan templates, comet rush, opponent modeling |
| 5 | 06/06–06/12 | Tuning + opt | ~20h | Profile, vectorize hot path with numpy if needed; weight tuning sweep |
| 6 | 06/13–06/20 | Meta + freeze prep | ~15h | Replay analysis vs top-100 bots, targeted patches |
| 7 | 06/20–06/23 | **FREEZE** | ~5h | No new code. Final 2 submissions are last week's validated bots. Self-play to reduce σ. |

Total: ~125h. Buffer for life events.

## Submission discipline

The "latest-2-only" rule is a sharp edge. Operating procedure:

1. **No submission without local validation.** A challenger must beat the current champion at p < 0.05 over ≥100 self-play games before submission.
2. **Daily quota is a ceiling, not a target.** Most days submit 0-1, not 5.
3. **Champion always preserved in latest-2.** Never submit two challengers in a row that haven't both passed validation. If challenger fails on ladder, the champion is one of the two remaining slots.
4. **Last 3 days are frozen.** No new code. Only resubmit (or don't) the bots already proven on the ladder.
5. **Submission log.** Every submission gets a one-line entry: ID, code commit SHA, validation result, ladder result. Used to debug regressions.

## Testing strategy

- **Unit tests** for `sim.step` parity vs `kaggle_environments` on 100 seeded games (must match exactly).
- **Property tests** for orbit prediction (`planet_pos_at(p, t) == kaggle_env_pos(p, t)`).
- **Behavior tests** at the bot level: regression suite of "scenarios" — hand-crafted boards where the correct move is known and the bot must produce it.
- **Self-play regression**: every commit on `main` triggers `compare(HEAD, last_tagged_champion, 50_games)` locally. If win rate drops below 50%, investigate before pushing.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Simulator parity drift (silent disagreement with official env) | High | Test suite enforced from week 1; never use my sim without parity passing |
| 1s timeout exceeded on real ladder hardware | Medium | Bench at 0.5s budget locally; submission gate rejects bots that exceed 0.7s in any of 100 games |
| Premature optimization wastes week 5 | Medium | Profile first, optimize only the proven hot path |
| Final-week impulse submission destroys champion | Medium | Hard freeze rule + submission log + this design doc as the contract |
| Top-tier teams have IL/RL bots we can't match | Low-medium | Acceptable. Floor is silver; gold is realistic with disciplined search |
| Solo burnout | Medium | 120-140h cap with explicit weekly hours, not "all free time" |

## Out of scope

- Reinforcement learning training (no GPU, no time).
- Imitation learning from top-bot replays (Kaggle restricts replay access; weak signal anyway).
- Custom UI / visualizer beyond text replay.
- 4-player FFA specialization (initially target 1v1 only; add FFA late if time permits).
- C/Cython extension. Pure Python + numpy is enough if the hot loop is shaped right.

## Stack

- Python 3.13 (uv-managed)
- `kaggle-environments>=1.28.0` — official simulator (oracle for parity testing)
- `numpy` — vectorization for plan-batch simulation
- `pytest` — test runner
- `kaggle` CLI — submission and ladder queries
- Local hardware: M-series Macbook with large RAM. Self-play runs across all CPU cores via `multiprocessing`.
