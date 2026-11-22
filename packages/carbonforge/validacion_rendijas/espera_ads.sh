#!/usr/bin/env bash
# The functionalised-slit CO2 study again on the adsorption force field
# (published, neutral group charges): Henry for OH, COOH and NH2 and their
# isotherms. Run as a TRACKED background task; safe to rerun.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 CAMPO=adsorption
mkdir -p isotermas_co2
run () { uv run python funcional_co2.py corre "$1" >> "isotermas_co2/funcional-$1-ads.log" 2>&1; }
uv run python funcional_co2.py henry > /dev/null 2>&1 &
( run COOH ) &
( run OH; run NH2 ) &
wait
echo "todo terminado"
