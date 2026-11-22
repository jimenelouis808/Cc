#!/usr/bin/env bash
# Snapshots for the report. TRACKED background task; safe to rerun.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
mkdir -p instantaneas
run () { uv run python instantaneas.py corre "$1" >> "instantaneas/$1.log" 2>&1; }
( run humeda-n5 ) &
( run h2-rendija; run co2-rendija ) &
( run mezcla-n5; run co2-schwarzita-d; run mezcla-super-cube ) &
wait
uv run python instantaneas.py dibuja > instantaneas/dibuja.log 2>&1
echo "todo terminado"
