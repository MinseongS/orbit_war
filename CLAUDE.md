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

---

## Game rules (auto-loaded from starter kit)

@starter_kit/README.md

---

## Submission guide (auto-loaded from starter kit)

@starter_kit/agents.md
