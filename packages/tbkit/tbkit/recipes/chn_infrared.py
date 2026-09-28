"""GPAW IR references (frequencies, intensities, Born charges, dipoles) for C/H/N.

Run (needs GPAW)::

    python -m tbkit.recipes.chn_infrared OUT.json --workers 4

Molecules at their GPAW-relaxed geometries (``parameters/references/
gpaw_chn.json``), PBE, LCAO dzp (the settings of the fit), ``ase.vibrations.
Infrared`` (forces and dipoles by ±0.01 Å); the mass-weighted eigenvectors
are stored so that TB dipoles can be tested on GPAW's own modes. Validation
data for
:mod:`tbkit.infrared`; nothing is fitted to them.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MOLECULES = ("CH4", "NH3", "HCN", "C2H4", "C6H6", "C5H5N", "H3CNH2", "CH3CN")


def _one(name):
    import os
    import tempfile

    import numpy as np
    from ase.vibrations import Infrared

    from tbkit.params import PARAMETER_DIR
    from tbkit.references import GPAW_DEFAULTS, _n_bands, gpaw_calculator, load_references

    refs, _ = load_references(PARAMETER_DIR / "references" / "gpaw_chn.json")
    atoms = next(r for r in refs if r.label == f"{name}/eq").atoms.copy()
    atoms.pbc = False
    atoms.center(vacuum=GPAW_DEFAULTS["vacuum"])
    atoms.calc = gpaw_calculator(GPAW_DEFAULTS, _n_bands(atoms, GPAW_DEFAULTS))
    dipole = np.array(atoms.get_dipole_moment())
    with tempfile.TemporaryDirectory() as directory:
        ir = Infrared(atoms, name=os.path.join(directory, "ir"), delta=0.01)
        ir.run()
        ir.summary(log=os.devnull)
        energies = ir.get_energies()
        intensities = ir.intensities          # (D/Å)²/amu
        vectors = np.asarray(ir.modes)        # rows: mass-weighted eigenvectors e
    freq = np.where(np.abs(energies.imag) > np.abs(energies.real), -np.abs(energies.imag),
                    np.abs(energies.real)) / 1.239841984e-4
    return {"group": name, "dipole_e_angstrom": dipole.tolist(),
            "frequencies_cm1": [round(float(f), 2) for f in freq],
            "intensities_km_mol": [round(float(i) * 42.2561, 4) for i in intensities],
            "eigenvectors": np.round(vectors.reshape(len(vectors), -1, 3), 7).tolist(),
            "symbols": atoms.get_chemical_symbols()}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    from tbkit.references import GPAW_DEFAULTS, gpaw_settings_record

    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    todo = [m for m in MOLECULES if not (parts / f"{m}.json").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(_one, m): m for m in todo}
        for future in as_completed(futures):
            result = future.result()
            (parts / f"{result['group']}.json").write_text(json.dumps(result))
            print(f"{result['group']}: listo", flush=True)
    entries = [json.loads((parts / f"{m}.json").read_text()) for m in MOLECULES]
    args.out.write_text(json.dumps({"settings": gpaw_settings_record(dict(GPAW_DEFAULTS)),
                                    "infrared": entries}, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    print(f"{len(entries)} moléculas en {args.out}")


if __name__ == "__main__":
    main()
