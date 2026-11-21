#!/usr/bin/env bash
# Start every CO2 isotherm that is not finished and not already running.
# Safe to call any number of times; a rerun skips finished points.
set -uo pipefail
cd "$(dirname "$0")"
for name in super-cube super-tetrahedron rendija-6.6; do
  if pgrep -f "isoterma_co2.py corre $name" >/dev/null; then
    echo "$name: corriendo"; continue
  fi
  n=$(python3 -c "import json,sys
try: print(len(json.load(open('isotermas_co2/$name.json'))['puntos']))
except Exception: print(0)")
  if [ "$n" -ge 7 ]; then echo "$name: terminado"; continue; fi
  mkdir -p isotermas_co2
  OMP_NUM_THREADS=1 nohup uv run python isoterma_co2.py corre "$name" \
    >> "isotermas_co2/$name.log" 2>&1 &
  echo "$name: lanzado ($n/7)"
done
