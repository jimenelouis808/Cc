#!/usr/bin/env bash
# The quantum-sieving sweep under the watchdog. Done when every
# (structure, temperature) row is in tamizado.txt.
cd "$(dirname "$0")"
total=$(OMP_NUM_THREADS=1 uv run python -c "import tamizado; print(2*len(list(tamizado.structures())))" 2>/dev/null)
exec ./vigia.sh tamizado "python tamizado.py" \
  "[ \$(wc -l < tamizado.txt) -ge $total ]" \
  "OMP_NUM_THREADS=1 nice -n 5 uv run python tamizado.py 500000"
