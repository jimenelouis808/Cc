#!/usr/bin/env bash
# CO2 Henry and CO2/N2 selectivity of the crystalline superstructures.
# TRACKED background task; safe to rerun.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
mkdir -p selectividad
uv run python henry_cristalinas.py > /dev/null 2>&1 &
run () { uv run python selectividad.py corre "$1" >> "selectividad/$1.log" 2>&1; }
( run cristalina-knee-schwarzite-diamond; run cristalina-knee-schwarzite-primitive ) &
( run cristalina-knee-super-graphene_k22; run cristalina-knee-super-square_k12_n7; run cristalina-knee-schwarzite-gyroid ) &
wait
echo "todo terminado"
