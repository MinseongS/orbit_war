#!/usr/bin/env bash
set -euo pipefail

DEST="${DEST:-./vendor/public_tactical}"
mkdir -p "$DEST"
uv run kaggle kernels pull sigmaborov/orbit-wars-2026-tactical-heuristic -p "$DEST" -m
echo "Notebook pulled to $DEST"
ls -la "$DEST"
