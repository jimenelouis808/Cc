#!/usr/bin/env bash
# Start every cage isotherm that is not finished and not already running.
# Safe to call any number of times: a running one is left alone, and a
# relaunched one skips the points it already has. This is what the
# hourly check calls after a container restart.
set -uo pipefail
cd "$(dirname "$0")/.."
for name in super-cube super-tetrahedron; do
  folder="validacion_rendijas/isotermas/$name"
  if pgrep -f "isoterma_jaulas.py corre $name" >/dev/null; then
    echo "$name: corriendo"
    continue
  fi
  done_state=$(uv run python -c "from carbonforge.adsorption.campaign import status; print(status('$folder')['state'])" 2>/dev/null)
  if [ "$done_state" = "terminado" ]; then
    echo "$name: terminado"
    continue
  fi
  OMP_NUM_THREADS=1 nohup uv run python validacion_rendijas/isoterma_jaulas.py corre "$name" \
    >> "$folder/corrida.log" 2>&1 &
  echo "$name: lanzado ($done_state)"
done
