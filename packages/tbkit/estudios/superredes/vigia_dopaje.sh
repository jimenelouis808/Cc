#!/usr/bin/env bash
# dopaje_banda_plana.py under the watchdog: done at 3 structures x 2 charges x
# 3 starts. TRACKED background task; leaves by itself after 110 min.
cd "$(dirname "$0")"
exec ./vigia.sh dopaje "^[^ ]*python[0-9.]* dopaje_banda_plana.py" \
  "python3 -c 'import json,sys; sys.exit(len(json.load(open(\"dopaje_banda_plana.json\")))<18)'" \
  "OMP_NUM_THREADS=1 uv run python dopaje_banda_plana.py"
