#!/usr/bin/env bash
# character_animatrem — one-liner bootstrap for a fresh GPU box.
#   curl -fsSL https://raw.githubusercontent.com/adbrasi/character_animatrem/main/bootstrap.sh | bash
# Clones the repo (if needed), loads .env, and launches the trainer.
set -euo pipefail

REPO_URL="https://github.com/adbrasi/character_animatrem"
TARGET="${ANIMATREM_DIR:-character_animatrem}"

# If we're not already inside the repo, clone it and enter.
if [ ! -f "animatrem.py" ]; then
  if [ ! -d "$TARGET/.git" ]; then
    echo "▸ Clonando $REPO_URL ..."
    git clone "$REPO_URL" "$TARGET"
  fi
  cd "$TARGET"
fi

# Load .env (OPENROUTER_API_KEY, HF_TOKEN, overrides) if present.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

exec python3 animatrem.py "$@"
