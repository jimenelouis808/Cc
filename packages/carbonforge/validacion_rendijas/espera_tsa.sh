#!/usr/bin/env bash
# Desorption isotherms at 373 K (0.015 and 0.1 MPa) for a temperature
# swing, adsorption field. TRACKED background task; safe to rerun.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 CAMPO=adsorption TEMPERATURA=373
run () { uv run python funcional_co2.py corre "$1" >> "isotermas_co2/funcional-$1-ads-373K.log" 2>&1; }
( run pristina; run OH ) &
( run N5; run NH2 ) &
# N5 at 298 K again on the adsorption field: its first isotherm used the
# representative N charges, and the swing needs both ends on one field.
( run COOH; TEMPERATURA=298 uv run python funcional_co2.py corre N5 >> isotermas_co2/funcional-N5-ads.log 2>&1 ) &
wait
echo "todo terminado"
