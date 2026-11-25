#!/usr/bin/env bash
# The 1 MPa CO2 column of the gallery, four structures at a time; finished
# pairs are skipped, so it is safe to rerun.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
uv run python galeria.py lista 2>/dev/null | grep ' co2_1mpa$' | \
  xargs -P 4 -L 1 sh -c 'uv run python galeria.py corre "$0" "$1" >> "galeria/$0__$1.log" 2>&1'
