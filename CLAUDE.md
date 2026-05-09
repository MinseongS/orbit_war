# Orbit Wars

Kaggle simulation competition. Bot-vs-bot 1v1 / 4p FFA on a 100x100 continuous board with orbiting planets and comets.

- **Submission deadline:** 2026-06-23 (UTC 23:59)
- **Final leaderboard:** ~2026-07-08
- **Daily submission limit:** 5 bots/day, latest 2 tracked on ladder

## Project layout

- `starter_kit/` — official game spec and submission guide (auto-imported below)
- `download.py` — refresh the starter kit via Kaggle API into `./starter_kit/`
- Package management: **uv** (`uv add`, `uv run`, `pyproject.toml`)

## Workflow

- Run things with `uv run …` (never bare `python`).
- Keep agent code submission-ready: a `main.py` at the project root with an `agent(obs)` function (or a `tar.gz` bundle).
- Test locally with `kaggle_environments.make("orbit_wars")` before every submission — daily quota is precious.
- Run tests with `uv run pytest -q`.
- W4 champion: heuristic_v4 (7 templates incl. trade_down_strike, passive validator). Tied with v3 in champion tier (47.5% over 50 games — noise). Filter aggregation reverted (commit `b65f7e9`) after it regressed both v3 and v4 vs public_tactical from 64% to 32%. v5 fell back to v4 weights (per-step regression failed). Use `uv run ow-gate orbit_war.bots.heuristic_v4:agent` to gate challengers.
- W5 priorities: redesign multi_source_consolidation to only emit when no solo capture (currently dead code); retry adversarial validator with looser slack; try CMA-ES tuning (per-game and per-step regression both failed).

---

## Game rules (auto-loaded from starter kit)

@starter_kit/README.md

---

## Submission guide (auto-loaded from starter kit)

@starter_kit/agents.md
