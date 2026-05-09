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
