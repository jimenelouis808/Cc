"""The N 5 % slit again with graphitic-N charges from DFT
(``dft_cargas/cargas_n_grafitico_r6.json``, IAO, C36NH15 flake): N -0.017,
its three C neighbours +0.089. The rest of the charge (+0.25 e per N) is
the electron graphitic N donates to the pi system, and it is spread over
the C_sp2 sheet, as the neutralisation already does.

Henry (2 seeds x 120k) and the CO2/N2 mixture point of selectividad.py.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np

from carbonforge.adsorption.co2 import henry_co2
from carbonforge.adsorption.mixture import run_mixture_gcmc

import selectividad as sel
from cargas_literatura import literature_wall
from funcional_co2 import HERE, TEMPERATURE, slit

DFT = json.loads((HERE / "dft_cargas" / "cargas_n_grafitico_r6.json")
                 .read_text())
TABLE = {"N_graph": DFT["N"]["iao"],
         "C_N": float(np.mean(DFT["C_N"]["iao"]))}

if __name__ == "__main__":
    atoms = slit("N5")
    wall, net = literature_wall(atoms, TABLE)
    out = HERE / "n_dft.json"
    record = json.loads(out.read_text()) if out.exists() else {}
    if "henry" not in record:
        k, q = zip(*[henry_co2(atoms, TEMPERATURE, samples=120_000, seed=s,
                               wall=wall) for s in range(2)])
        record["cargas"] = TABLE
        record["carga_neta_por_celda"] = net
        record["henry"] = {"K_H": float(np.mean(k)), "rango": [min(k), max(k)],
                           "q_st": float(np.mean(q))}
        out.write_text(json.dumps(record, indent=1))
        print(record["henry"], flush=True)
    if "mezcla" not in record:
        started = time.time()
        r = run_mixture_gcmc(atoms, sel.TEMPERATURE, sel.PRESSURE,
                             sel.FRACTIONS, equilibration=sel.EQUILIBRATION,
                             production=sel.PRODUCTION, seed=0, wall=wall)
        record["mezcla"] = asdict(r) | {"segundos": time.time() - started}
        out.write_text(json.dumps(record, indent=1))
        print(f"S = {r.selectivity:.1f} ± {r.selectivity_error:.1f}, "
              f"CO2 {r.mol_per_kg[0]:.3f} mol/kg", flush=True)
