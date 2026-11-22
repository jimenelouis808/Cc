#!/usr/bin/env bash
# CO2/N2 selectivity of every structure, three at a time. Run as a
# TRACKED background task; safe to rerun (finished structures are skipped).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
mkdir -p selectividad
run () { uv run python selectividad.py corre "$1" >> "selectividad/$1.log" 2>&1; }
( run rendija-6.6; run funcional-N5; run funcional-NH2 ) &
( run super-cube; run funcional-pristina ) &
( run super-tetrahedron; run funcional-OH; run funcional-COOH ) &
wait
echo "todo terminado"
