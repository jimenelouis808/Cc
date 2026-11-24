#!/usr/bin/env bash
# magnetismo_U.py under the watchdog: done at 3 structures x 3 U in
# magnetismo_U.json. Run as a TRACKED background task.
cd "$(dirname "$0")"
exec ./vigia.sh magnetismo_U "^[^ ]*python[0-9.]* magnetismo_U.py" \
  "python3 -c 'import json,sys; sys.exit(len(json.load(open(\"magnetismo_U.json\")))<9)'" \
  "OMP_NUM_THREADS=1 uv run python magnetismo_U.py"
