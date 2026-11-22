#!/usr/bin/env bash
# Humid flue gas on the pristine, N 5 % and COOH slits at 30 and 80 % RH.
# TRACKED background task; safe to rerun.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
mkdir -p humedad
run () { uv run python humedad.py corre "$1" "$2" >> "humedad/$1-hr$2.log" 2>&1; }
( run pristina 30; run pristina 80 ) &
( run N5 30; run N5 80 ) &
( run COOH 30; run COOH 80 ) &
wait
echo "todo terminado"
