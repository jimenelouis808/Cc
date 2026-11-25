#!/usr/bin/env bash
# magnetismo_k8.py under the watchdog: done when magnetismo_k8.json exists.
# Run as a TRACKED background task; it leaves by itself after 110 min
# ("renovar") and must then be started again. The worker checkpoints every
# 10 SCF iterations, so a relaunch loses at most one chunk.
cd "$(dirname "$0")"
exec ./vigia.sh k8 "^[^ ]*python[0-9.]* magnetismo_k8.py" \
  "test -f magnetismo_k8.json" \
  "OMP_NUM_THREADS=4 uv run python magnetismo_k8.py"
