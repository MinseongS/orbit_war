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
if ! uv run ow-gate "$BOT_SPEC" --seeds 25 --workers 4 > /tmp/ow-gate.log 2>&1; then
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
