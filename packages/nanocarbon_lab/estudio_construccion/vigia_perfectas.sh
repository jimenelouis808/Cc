#!/usr/bin/env bash
# barrido_perfectas.py under the watchdog. TRACKED background task; leaves
# by itself after 110 min ("renovar") and is started again.
cd "$(dirname "$0")"
exec ./vigia.sh perfectas "^[^ ]*python[0-9.]* barrido_perfectas.py" \
  "test -f barrido_perfectas.done" \
  "OMP_NUM_THREADS=1 uv run python barrido_perfectas.py && touch barrido_perfectas.done"
