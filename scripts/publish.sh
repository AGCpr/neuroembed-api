#!/usr/bin/env bash
# publish.sh - push NeuroEmbed API to GitHub.
# Requires: gh authenticated (`gh auth status` shows logged in).
# Usage:    ./scripts/publish.sh <OWNER>
# Example:  ./scripts/publish.sh agc

set -euo pipefail
OWNER="${1:?usage: publish.sh <github-owner>}"

# Sanity checks
gh auth status >/dev/null
git -C "$(dirname "$0")/.." rev-parse --git-dir >/dev/null

REPO="neuroembed-api"
DESCRIPTION="Hosted REVE EEG foundation-model inference API"
cd "$(dirname "$0")/.."

echo "==> Creating $OWNER/$REPO (public) ..."
gh repo create "$OWNER/$REPO" --public \
    --description "$DESCRIPTION" \
    --source . \
    --remote origin \
    --push

echo "==> Setting repo topics ..."
gh repo edit "$OWNER/$REPO" --add-topic eeg --add-topic neuroscience \
    --add-topic foundation-model --add-topic bci --add-topic fastapi

echo "==> Creating v0.1.0 release ..."
gh release create v0.1.0 \
    --title "NeuroEmbed API v0.1.0" \
    --notes-file RELEASE.md \
    --target main

echo "==> Done. Visit https://github.com/$OWNER/$REPO"
