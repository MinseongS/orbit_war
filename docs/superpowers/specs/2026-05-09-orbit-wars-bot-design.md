# Orbit Wars Bot — Design

**Author:** minseong (solo)
**Date:** 2026-05-09 (rev 2 — added evaluation rigor and lessons from prior competitions)
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
- **Hard 1-second per-turn timeout.**
- **Deterministic, fully observable simulator.** Every rule (orbit angle, fleet speed, combat resolution, comet trajectory) is closed-form. *Forward simulation is exact.* In particular, planet positions for all 500 turns are computable at game start — exploit this aggressively.
- **Continuous action space** (any planet × any angle × any int ship count). Cannot enumerate exhaustively. Plans must be *generated* analytically.
- **500-turn games**, 5-10 symmetric planet groups, 4-fold board symmetry.
- **Latest-2-only submission tracking** (not best-2). Operational discipline matters as much as the bot itself.

## Approach: Heuristic core + 1-ply plan-portfolio search

Rejected alternatives:
- *Pure heuristics* — caps at silver/edge-of-gold without a search differentiator.
- *Pure RL/IL* — empirically fails in this format. Halite IV's winner used a **rule + DL hybrid**, not pure ML; Lux S2's winner was **search-based with deep simulation**, not RL. With 7 weeks and no GPU, RL is not the right primary path.
- *Deep search (≥2-ply)* — Planet Wars 2010 winner Gábor Melis explicitly reports: *"Two-ply fell way short of the expectations and performed worse than the one ply bot."* Deeper search amplifies eval errors. Default depth is 1-ply; deeper only in opening.

**Chosen:** an analytical heuristic core + 1-ply plan-portfolio search using Melis's "steps + greedy combine under surplus constraint" pattern (the Planet Wars 2010 winning architecture).

Each turn:
1. Generate ~50-200 candidate atomic *steps* (a single launch from one planet aimed at one target).
2. Score each step with the eval function.
3. Sort descending; greedily combine while feasibility (ships available, defenses preserved) holds.
4. In the opening (first ~3 planet captures), upgrade to 4-ply alpha-beta to win neutral races (per Melis).

Search engine = Melis's pattern. Eval = synthesis of features that recurred across Planet Wars / Halite I-IV / Lux S2 winners. Differentiation from competitors comes from *eval quality and step diversity*, not search depth.

## Architecture

Six components, isolated and testable independently.

### 1. `sim/` — Forward simulator

Pure-Python reimplementation of the official game step function, designed for speed and exact bitwise agreement with `kaggle-environments`.

- `Game` dataclass: planets, fleets, comets, turn, RNG seed.
- `step(game, action_per_player) -> game'`: deterministic transition, matching the rule order in `starter_kit/README.md`.
- **Precomputed orbit table** (`positions[planet_id][turn]`): generated once at game start, used as O(1) lookup throughout. Avoids re-computing `cos/sin` inside hot loops. This alone is expected to be a 10-100x speedup vs naive recompute.
- `legal_steps(game, player) -> generator`: enumerate atomic single-launches, not full plans.
- **Spatial pruning helper** (per Halite II): for combat-relevant filtering, use interaction radius `2 * max_speed * Δt + sun_radius + planet_radii`. Never iterate the whole map for pairwise interactions.
- **Tested for parity** against `kaggle_environments.envs.orbit_wars` on 100 random seeded games. Test suite is the primary correctness gate.

### 2. `eval/` — Position evaluator

Scalar value function `evaluate(game, player) -> float`. Handcrafted, parameterized, tuned by self-play sweep.

Features (synthesized from prior winners):

- **Total ships** (planets + fleets) — baseline.
- **Production** (sum of `production` of owned planets).
- **Time-denominated yield** (Halite III, teccles): every prospective gain is expressed as "turns until equivalent ship-yield." Comparable units make portfolio scoring possible.
- **Production-weighted reachability** (Planet Wars, Halite II): planet score ∝ growth × controllability ÷ time-to-arrival.
- **Indirect wealth** (Planet Wars, _iouri_): planet adjacent to high-growth planets is worth more — encourages forward bases.
- **Surplus / constrained departure** (Melis): ships available *without* invalidating already-promised defenses. Tunable `min_turns_to_depart` parameter; per Melis, suppressing rock-paper-scissors volatility was more valuable than locally optimal moves.
- **Threat exposure** (incoming enemy fleet ships within T turns).
- **Reaction-time delta** (Planet Wars): for each enemy planet, `enemy_reaction_time - my_reaction_time` flags out-on-a-limb capturable enemies.
- **Numeric-superiority gate before any fight** (Halite II, shummie): "don't fight battles you can't win." Hardcoded vetoes, not just penalties.
- **Trade-down bonus when ahead, conservation when behind** (Planet Wars): explicit asymmetry.
- **Comet anticipation**: expected ships from upcoming spawns at turns 50/150/250/350/450.
- **Full Attack Future invariant** (Melis): cheap end-of-turn check — *"if for all my planets, opponent cannot take that planet sending all ships, then opponent cannot take any planet."* Use as a "safe position" booster.

Weights are constants in code, tuned by **linear regression on self-play outcomes** (per a1k0n's Tron 2010 trick: 11,691 games gave him K1≈0.055, K2≈0.194). Cheap, transferable; better than CMA-ES for ~10 weights.

### 3. `plan_gen/` — Step generator

Given the current state, propose a portfolio of candidate **atomic steps** (single launches). Diversity here is the difference between gold and silver.

Step templates (synthesizing winning patterns):

- **No-op** (always include — baseline).
- **Capture nearest-target with min ships** (starter kit floor).
- **Production/distance greedy** (Planet Wars universal): for each owned planet, target best `production / distance`.
- **Defensive reinforce**: when an incoming enemy fleet is detected, send exactly enough to hold.
- **Orbit-aware intercept**: predict where an orbiting planet will be at fleet arrival, aim there.
- **Snipe undefended high-prod**: if a non-owned planet has `ships < δ` and `production ≥ 3`, route from nearest owner.
- **Comet rush**: at known spawn turns, pre-launch fleets toward predicted comet positions.
- **Multi-source consolidation strike** (Planet Wars): gather ships from multiple home planets to hit one target with overwhelming force at the same arrival turn.
- **Tack/feint** (Planet Wars top-3 satirist analysis): hit one point to draw forces, exploit thin defense elsewhere.
- **Trade-down strike**: when ahead, deliberately exchange to lock in advantage.

Output: a list of (step, prior_score). Prior is used to truncate to top-K before full simulation.

### 4. `search/` — Plan composer

The Melis "steps + greedy combine" engine.

- **Default mode (mid-game): 1-ply.**
  1. Generate atomic steps for me. Sort descending by isolated eval delta.
  2. Greedy-combine in score order under surplus constraint.
  3. Optionally simulate the combined plan N=10-30 turns forward against a fixed opponent policy; if eval drops, prune.
  4. Return the combined plan as the action.
- **Opening mode (first ~3 captures): 4-ply alpha-beta** to win neutral races (Melis's specific recipe).
- **Time budget guard**: every phase checks elapsed time. Always returns a legal action by 0.7s (mid-game) / 0.9s (opening). Never times out.
- **Dynamic horizon** (Melis): when simulating forward to score a plan, extend lookahead beyond fixed N to *"the three earliest breakeven turns of safe-to-take neutrals."*

Why not MCTS/UCT? Per Melis: *"one ship more or less can make all the difference"* — the value function is jagged in fleet count. UCT averages over noise it shouldn't.

### 5. `eval_harness/` — Local self-play evaluation

This is the engineering differentiator. Most teams underinvest here.

#### Bot zoo

A maintained pool of historical and benchmark agents:

```
zoo/
├── starter.py         (Nearest Planet Sniper from starter_kit)
├── random.py
├── greedy_baseline.py (production/distance greedy, no search)
├── public_tactical.py (clone of the public Tactical Heuristic notebook)
├── champion_v01.py
├── champion_v02.py
├── champion_v03.py    (rolling — keep last 3)
└── style/
    ├── rusher.py      (always all-in early)
    ├── turtler.py     (defensive only)
    └── snipe_meta.py  (specialized variant we lost to on the ladder, manually crafted)
```

The `style/` subfolder grows over time as we discover failure modes on the public ladder.

#### Match protocol

For each `(challenger, opponent)`:
- **Fixed seeds:** N seeds, both bots play each side per seed (mirror) → 2N games per pair.
- **Parallel:** `multiprocessing.Pool` across all CPU cores. Target ≥1000 games / 10 minutes on M-series.
- **Stratified output:** win rate, score margin (mean & p10), p99 turn time, error rate per opponent class.

#### Stratified submission gate

A challenger must pass *all* tiers to qualify for ladder submission:

| Tier | Opponents | Required | Sample N | Rationale |
|------|-----------|----------|----------|-----------|
| Sanity | random, starter | win rate ≥ 95% | 100 each | regression on weak opponents = brittle |
| Diversity | 3 historical champions | win rate ≥ 55% each | 200 each | rock-paper-scissors avoidance |
| Champion | current champion | win rate ≥ 55%, p<0.05 | 200-1000 (Wilson CI) | the headline test |
| Style robustness | every `style/` bot | win rate ≥ 50% each | 100 each | defends against known meta variations |
| Timing | every opponent | p99 turn ≤ 0.7s, max ≤ 0.9s | all of the above | safety margin under Kaggle VM speed |

Failure on *any* tier → reject. The `submit.sh` wrapper enforces this; no manual override path.

#### Sample size calculator

Two-proportion z-test for the Champion tier:

| Detect | Required N (mirrored pairs) |
|--------|-----------------------------|
| 70% vs 50% | ~25 pairs (50 games) |
| 60% vs 50% | ~100 pairs (200 games) |
| 53% vs 50% | ~500 pairs (1000 games) |

Coarse changes use small N; fine tuning uses 1000+.

#### Public-ladder validation

Self-play *cannot* detect blind spots that both bots share (Melis's documented failure mode: his `MIN-TURN-TO-DEPART=1` won self-play but lost on the public ladder). Mitigation:

- After each ladder submission, pull 10 representative episodes via `kaggle competitions episodes <id>` and `kaggle competitions replay <eid>`.
- For every loss, identify the failure mode in 2-3 sentences. If a pattern emerges across ≥3 losses, codify the opposing strategy as a new `style/` zoo bot and re-tune.
- **Treat public ladder as ground truth, self-play as gradient.**

### 6. `submit/` — Submission ops

Wraps `kaggle competitions submit`, requires `eval_harness` gate to pass, records every submission.

- `submit.sh <bot_path> --message "<reason>"`:
  1. Run `eval_harness.gate` (full stratified evaluation).
  2. If pass: submit, record `submissions.log` line with ID, commit SHA, gate stats, message.
  3. If fail: print which tier failed and abort. No `--force` flag.
- `kaggle competitions submissions orbit-wars` polled daily; status dropped into `submissions.log`.
- Replays for losses auto-downloaded for analysis.

## Data flow

```
obs (kaggle env)
  └→ parse → Game (with precomputed orbit table reused)
       └→ plan_gen.candidates(Game, me)  (steps, not full plans)
            └→ search.compose
                  uses sim.step + eval (score each step in isolation)
                  greedy combine under surplus
                  return final action
  ← action [[from_id, angle, ships], …]
```

## Iteration plan (45 days)

| Week | Calendar | Focus | Time | Deliverable |
|------|----------|-------|------|-------------|
| 1 | 05/09–05/15 | Infra + baseline | ~15h | `sim/` parity test passes; orbit table precomputed; starter `main.py` submitted to ladder; `eval_harness.gate` works against zoo of 3 bots; public Tactical Heuristic notebook reproduced as `public_tactical.py` |
| 2 | 05/16–05/22 | Heuristic core | ~25h | All eval features above implemented; weights linearly regressed on 5,000 self-play games; 4 step templates beat starter ≥95% and beat `public_tactical` ≥60% |
| 3 | 05/23–05/29 | Search v1 | ~25h | 1-ply plan-portfolio composer live; opening 4-ply mode; bot beats prior champion ≥60% over 200 mirrored games |
| 4 | 05/30–06/05 | Step diversity | ~20h | All 10 step templates including consolidation strike, comet rush, tack/feint; first style-zoo bots added based on ladder losses |
| 5 | 06/06–06/12 | Tuning + opt | ~20h | Profile, vectorize hot path with numpy if needed; second weight regression with expanded feature set |
| 6 | 06/13–06/20 | Meta + freeze prep | ~15h | Replay analysis vs top-100 bots, targeted patches, *hold final innovations* (per erdman's Halite I tactic) |
| 7 | 06/20–06/23 | **FREEZE** | ~5h | No new code. Final 2 submissions are the validated bots from week 6. Self-play to reduce TrueSkill σ. |

Total: ~125h. Buffer for life events.

## Submission discipline

The "latest-2-only" rule is a sharp edge. Operating procedure:

1. **No submission without local validation.** A challenger must pass the full `eval_harness.gate` (all 5 tiers above).
2. **Daily quota is a ceiling, not a target.** Most days submit 0-1, not 5.
3. **Champion always preserved in latest-2.** Never submit two challengers in a row that haven't both passed validation independently.
4. **Don't react-patch in the final week** (teccles's named regret from Halite III: *"chasing a competitor mid-competition led to ad-hoc fixes rather than principled design"*). The decision to freeze is binding.
5. **Hold final innovations until the closing window** (erdman, Halite I): *"good ideas were easy to replicate among top bots."* The last meaningful new feature ships in week 6.
6. **Last 3 days are frozen.** No new code. Only resubmit (or don't) bots already validated on the ladder.
7. **TrueSkill needs ≥50 games for stable read.** Don't conclude from 20 ladder matches.
8. **Submission log.** Every submission gets a one-line entry: ID, commit SHA, gate stats, ladder result. Used to debug regressions and blocks reactive patches by making the prior bot's quality concrete.

## Testing strategy

- **Unit tests**: `sim.step` parity vs `kaggle_environments` on 100 seeded games — exact match required.
- **Property tests**: orbit prediction (`positions[p][t]` matches the official simulator's planet position at turn `t`).
- **Behavior tests**: regression suite of "scenarios" — hand-crafted boards where the correct action is known and the bot must produce it (e.g., "incoming enemy fleet of 30 ships, must reinforce with 31").
- **Self-play regression**: every commit on `main` triggers `eval_harness.gate(HEAD, last_tagged_champion, 200_games)` locally. If the Champion tier fails, the commit is reverted before push.

## Lessons from prior competitions

Synthesized from research across:
- Planet Wars 2010 (Melis postmortem: https://quotenil.com/Planet-Wars-Post-Mortem.html ; satirist.org meta-analysis)
- Tron 2010 (a1k0n: https://www.a1k0n.net/2010/03/04/google-ai-postmortem.html)
- Halite I (erdman, mzotkiew)
- Halite II (https://shummie.github.io/Halite-2-Shummie/)
- Halite III (https://github.com/teccles-halite/halite3-bot)
- Halite IV (Recursion / Tom Van de Wiele rule+DL hybrid; Kha Vo IL writeup)
- Lux AI S1 (Toad Brigade — deep RL)
- Lux AI S2 (https://github.com/ryandy/Lux-S2-public — search with `FUTURE_LEN=20`)

### Search and depth
- Melis: *"Two-ply fell way short of the expectations and performed worse than the one ply bot."* — 1-ply with strong eval is the default.
- Melis: 4-ply alpha-beta in opening (until ~3 captures) won neutral races where 1-ply tied.
- a1k0n: *"a deep minimax search using a flawed evaluation heuristic is self-deluded about what its opponent is actually going to do."*
- Skip MCTS/UCT — fleet-count outcomes are jagged; one-ship deltas matter.

### Eval features
- Time-denominated scoring (Halite III): everything as turns-to-equivalent-yield.
- Surplus / feasibility check before any plan (Melis).
- Full Attack Future invariant for safe-position detection (Melis).
- Indirect wealth: planets near high-growth ones are worth more (_iouri_).
- Reaction-time delta for vulnerability detection (Melis).
- Numeric-superiority gate before fights — never fight a losing battle (shummie).
- Trade-down when ahead, conserve when behind (Planet Wars satirist).
- *"Anything you can do to make your evaluation function smarter will result in improved play in the long run."* (a1k0n)

### Plan generation
- Melis's "steps + greedy combine under surplus" is the Planet Wars winning architecture — directly applicable to Orbit Wars fleet launches.
- Plan templates that consistently appeared: production-greedy, defensive reinforce, multi-source consolidation, tack/feint, sniping undefended high-prod.

### Time-budget tricks
- Precompute deterministic state at game start (orbit positions for 500 turns).
- Spatial pruning by interaction radius (Halite II's `2*max_speed + ...`).
- Iterative collision resolution in 2-5 passes sorted by stable ID (Halite II).
- Linear regression on self-play outcomes for weight tuning (a1k0n: 11,691 games → empirical coefficients).

### Self-play gotchas
- Self-play overfits to your own quirks (Melis's `MIN-TURN-TO-DEPART=1` bug). Validate against public bots, not just self-play.
- Don't blindly fix bugs that self-play likes — they may suppress rock-paper-scissors volatility.
- TrueSkill ladder is noisy in first ~50 games; don't conclude from 20.

### When ML beat heuristics
- Halite IV winner (Recursion): **rule + DL hybrid**, not pure ML.
- Lux S1 winner: deep RL with self-play, but only with massive compute and IL bootstrap.
- Lux S2 winner: actually **search-based with deep simulation** (`FUTURE_LEN=20`), contradicting common claims.
- Bar for RL in Orbit Wars: IL bootstrap from a strong heuristic + tens of millions of self-play frames + GPU. Out of scope.

### Operational tactics
- erdman (Halite I): *"good ideas were easily replicated among top bots"* — held final innovations for closing window.
- teccles (Halite III) named regret: *"chasing a competitor mid-competition led to ad-hoc fixes rather than principled design."* Final-week reactive patching is the #1 named killer.
- 4-player tiebreaker meta (Halite II): in survival-rank formats, "hide in corner" mode beats fighting. Check Orbit Wars 4-player tiebreaker before designing endgame.

### Current Orbit Wars meta (May 2026)
- Two public notebooks already exist on Kaggle: `sigmaborov/orbit-wars-2026-tactical-heuristic` (the de-facto floor everyone is beating) and `kashiwaba/orbit-wars-reinforcement-learning-tutorial`.
- **Action:** clone both into `eval_harness/zoo/` in week 1; the Tactical Heuristic is our public-ladder baseline opponent.
- Forum chatter mentions RL but reportedly no one has shipped trained models yet (per a public mirror; verify in-app).
- Action space described by community as "huge but very prune-able" — confirms our analytical step-generation approach.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Simulator parity drift (silent disagreement with official env) | High | Test suite enforced from week 1; never use my sim without parity passing |
| 1s timeout exceeded on real ladder hardware | Medium | Bench at 0.7s budget locally; submission gate rejects bots that exceed 0.9s in any of 100 games |
| Premature optimization wastes week 5 | Medium | Profile first, optimize only the proven hot path |
| Final-week impulse submission destroys champion | High | Hard freeze rule + submission log + this design doc as the contract |
| Self-play overfits to shared blind spots | Medium-High | Public-ladder validation loop + style-zoo bots derived from ladder losses |
| Top-tier teams have IL/RL bots we can't match | Low-medium | Acceptable. Floor is silver; gold is realistic with disciplined search per Lux S2 evidence |
| Solo burnout | Medium | 120-140h cap with explicit weekly hours, not "all free time" |
| Reactive patching in week 6-7 | Medium | Pre-decided freeze date; no reactive features after week 6 |

## Out of scope

- Reinforcement learning training (no GPU, no time, ML didn't win Lux S2 anyway).
- Imitation learning from top-bot replays (Kaggle replay access is restricted; weak signal vs IL bootstrap of self-play).
- Custom UI / visualizer beyond text replay.
- 4-player FFA specialization (target 1v1 first; add FFA late if time permits, with attention to tiebreaker meta).
- C/Cython extension. Pure Python + numpy, with orbit precomputation + spatial pruning, should be enough.
- Explicit collusion / NAP strategies in 4-player (likely violates rules; expect adversaries to test).

## Stack

- Python 3.13 (uv-managed)
- `kaggle-environments>=1.28.0` — official simulator (oracle for parity testing)
- `numpy` — vectorization for plan-batch simulation
- `pytest` — test runner
- `kaggle` CLI — submission and ladder queries
- Local hardware: M-series Macbook with large RAM. Self-play across all CPU cores via `multiprocessing`.
