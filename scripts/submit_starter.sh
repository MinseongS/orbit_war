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
