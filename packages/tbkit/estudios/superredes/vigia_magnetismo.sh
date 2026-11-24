#!/usr/bin/env bash
# magnetismo.py under the watchdog: done when all 4 structures x 3 guesses
# are in magnetismo.json. Run as a TRACKED background task.
cd "$(dirname "$0")"
exec ./vigia.sh magnetismo "^[^ ]*python[0-9.]* magnetismo.py" \
  "python3 -c 'import json,sys; sys.exit(len(json.load(open(\"magnetismo.json\")))<12)'" \
  "OMP_NUM_THREADS=1 uv run python magnetismo.py"
