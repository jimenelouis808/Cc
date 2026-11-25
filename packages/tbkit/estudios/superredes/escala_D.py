"""Schwarz D gap against arm length (π model): k=18 cells with 5, 7, 9
rows (nanocarbon_lab, estudio_construccion/escala_D.py), on a 2x2x2
Gamma-centred mesh (the gap of the 5-row cell is direct near Gamma).

    python escala_D.py   -> escala_D.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ase.io import read

from tbkit.analysis import bands
from tbkit.hamiltonian import System
from tbkit.kpoints import mesh
from tbkit.params import pi_model

HERE = Path(__file__).resolve().parent
OUT = HERE / "escala_D.json"


def main() -> None:
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    model = pi_model()
    for rows in (5, 7, 9):
        key = f"D_k18_n{rows}"
        if key in out:
            continue
        t0 = time.time()
        atoms = read(str(HERE / "estructuras" / f"{key}.extxyz"))
        system = System.build(atoms, model)
        k, _ = mesh(system, 2)
        e = bands(system, k)[0]
        n = len(atoms) // 2                       # half filling, one π electron per C
        vbm, cbm = float(e[:, n - 1].max()), float(e[:, n].min())
        out[key] = {"rows": rows, "atoms": len(atoms), "cell_A": float(atoms.cell.lengths()[0]),
                    "gap_eV": cbm - vbm, "direct_gap_gamma_eV": float(e[0, n] - e[0, n - 1]),
                    "seconds": time.time() - t0}
        OUT.write_text(json.dumps(out, indent=1))
        print(key, out[key], flush=True)


if __name__ == "__main__":
    main()
