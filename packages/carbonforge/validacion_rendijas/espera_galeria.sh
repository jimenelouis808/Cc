#!/usr/bin/env bash
# All (structure, gas) snapshots of galeria.py, four at a time. TRACKED
# background task; safe to rerun (finished pairs are skipped).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
mkdir -p galeria
uv run python galeria.py lista 2>/dev/null | \
  xargs -P 4 -L 1 sh -c 'uv run python galeria.py corre "$0" "$1" >> "galeria/$0__$1.log" 2>&1'
echo "todo terminado"
