#!/usr/bin/env bash
# c_bandas.py under the watchdog: done when the 3 structures are in
# resultados_c_bandas.json. Run as a TRACKED background task.
cd "$(dirname "$0")"
exec ./vigia.sh c_bandas "^[^ ]*python[0-9.]* c_bandas.py" \
  "python3 -c 'import json,sys; sys.exit(len(json.load(open(\"resultados_c_bandas.json\")))<3)'" \
  "OMP_NUM_THREADS=3 uv run python c_bandas.py"
