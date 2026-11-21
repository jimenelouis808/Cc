#!/usr/bin/env bash
# Run the functionalised-slit CO2 study to the end: the Henry sweep,
# then the five isotherms, two at a time. Run as a TRACKED background
# task of the session (see espera_co2.sh for why); safe to rerun, as
# every finished row or point is skipped.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
uv run python funcional_co2.py henry > /dev/null 2>&1 &
henry=$!
run () { uv run python funcional_co2.py corre "$1" >> "isotermas_co2/funcional-$1.log" 2>&1; }
mkdir -p isotermas_co2
( run pristina; run OH; run NH2 ) &
( run N5; run COOH ) &
wait $henry
( true ) &
wait
echo "todo terminado"
