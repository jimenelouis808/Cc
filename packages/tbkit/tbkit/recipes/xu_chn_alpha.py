"""Extra atomic polarizabilities of ``xu_chn``, fitted to GPAW (FD) tensors.

A minimal s+p basis cannot polarise into the diffuse and polarisation
functions a real atom has: even with the intra-atomic s-p dipoles, benzene's
out-of-plane α is a quarter of the measured one. The missing part is given to
each atom as a polarizable dipole, ``extra_polarizability`` (Å³ per element),
screened together with the TB charges and dipoles (:mod:`tbkit.dipoles`).
Three numbers (H, C, N), fitted here to the full α tensors (anisotropy
included, relative errors) of the training molecules of
:mod:`tbkit.recipes.chn_polarizability`; nothing else of the model changes.

Run::

    python -m tbkit.recipes.xu_chn_alpha ALPHA_REFS.json xu_chn.json OUT.json

Checked outside the fit: the test molecules, C60 against experiment and the
ε∞ of diamond (a crystal: the extra polarizability is added per cell without
local fields, as the crystal response has no screening yet).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..hamiltonian import System
from ..optics import polarizability_linear_response
from ..params import model_from_dict, read_parameter_file
from ..references import load_references

ELEMENTS = ("H", "C", "N")
_UPPER = np.triu_indices(3)

#: Experiment (Å³): C60 from T. M. Miller, CRC Handbook (76.5 ± 8); diamond
#: ε∞ = 5.7 (optical, long-wavelength limit).
C60_ALPHA_EXPERIMENT = 76.5
DIAMOND_EPS_EXPERIMENT = 5.7


def with_extra(model, values) -> object:
    return dataclasses.replace(model, extra_polarizability=dict(zip(ELEMENTS, map(float, values),
                                                                    strict=True)))


def model_tensor(model, atoms) -> np.ndarray:
    return polarizability_linear_response(System.build(atoms, model))


def fit(model, entries, geometries):
    train = [e for e in entries if e["role"] == "train"]

    def residuals(x):
        trial = with_extra(model, x)
        out = []
        for entry in train:
            reference = np.array(entry["alpha"])
            scale = np.trace(reference) / 3
            difference = model_tensor(trial, geometries[entry["group"]]) - reference
            out.append(difference[_UPPER] / scale)
        return np.concatenate(out)

    x0 = np.array([0.3, 0.8, 0.6])
    return least_squares(residuals, x0, bounds=(0.0, 10.0))


def report(model, entries, geometries) -> dict:
    out = {}
    for entry in entries:
        reference = np.array(entry["alpha"])
        tb = model_tensor(model, geometries[entry["group"]])
        out[entry["group"]] = {
            "role": entry["role"],
            "gpaw": [float(v) for v in np.sort(np.linalg.eigvalsh(reference))],
            "tb": [float(v) for v in np.sort(np.linalg.eigvalsh(tb))],
            "mean_error_percent": float(100 * (np.trace(tb) / np.trace(reference) - 1))}
    return out


def c60_alpha(model) -> float:
    from ase.build import molecule
    from ase.optimize import BFGS

    from ..calculator import TBCalculator

    c60 = molecule("C60")
    c60.calc = TBCalculator(model, kT=0.01)
    BFGS(c60, logfile=None).run(fmax=0.01)
    return float(np.trace(model_tensor(model, c60)) / 3)


def diamond_eps(model) -> float:
    from ase.build import bulk

    from ..optics import dielectric_constant

    carbon = dataclasses.replace(model, orbitals={"C": model.orbitals["C"]})
    eps = dielectric_constant(System.build(bulk("C", "diamond", a=3.567), carbon), kmesh=8)
    return float(eps[0, 0])


def run(alpha_refs: Path, parameters: Path, out: Path, verbose: bool = True) -> dict:
    data = json.loads(Path(alpha_refs).read_text(encoding="utf-8"))
    entries = data["polarizabilities"]
    base_data = read_parameter_file(parameters)
    model = model_from_dict(base_data)
    structures, _ = load_references(Path(parameters).parent / "references" / "gpaw_chn.json")
    geometries = {s.group: s.atoms for s in structures if s.label.endswith("/eq")}
    before = report(model, entries, geometries)
    result = fit(model, entries, geometries)
    fitted = with_extra(model, result.x)
    after = report(fitted, entries, geometries)
    checks = {"c60_alpha": c60_alpha(fitted), "c60_alpha_experiment": C60_ALPHA_EXPERIMENT,
              "c60_alpha_without_extra": c60_alpha(model),
              "diamond_eps": diamond_eps(fitted), "diamond_eps_experiment":
              DIAMOND_EPS_EXPERIMENT, "diamond_eps_without_extra": diamond_eps(model)}
    if verbose:
        print("α extra (Å³):", dict(zip(ELEMENTS, np.round(result.x, 4), strict=True)))
        for name in after:
            b, a = before[name], after[name]
            print(f"  {name:10s} {a['role']:5s} GPAW {np.round(a['gpaw'], 2)}  "
                  f"TB {np.round(b['tb'], 2)} → {np.round(a['tb'], 2)}  "
                  f"(media {b['mean_error_percent']:+.0f} % → {a['mean_error_percent']:+.0f} %)")
        print("  C60:", round(checks["c60_alpha_without_extra"], 1), "→",
              round(checks["c60_alpha"], 1), "Å³ (exp.", C60_ALPHA_EXPERIMENT, ")")
        print("  diamante ε∞:", round(checks["diamond_eps_without_extra"], 2), "→",
              round(checks["diamond_eps"], 2), "(exp.", DIAMOND_EPS_EXPERIMENT, ")")
    source = (f"ajustada a tensores α de GPAW ({data['settings'].get('xc')}, FD, h = "
              f"{data['settings'].get('h')} Å); receta tbkit.recipes.xu_chn_alpha")
    output = dict(base_data)
    output["extra_polarizability"] = {
        el: {"value": round(float(v), 5), "unit": "Å^3", "source": source,
             "description": "polarizabilidad atómica que la base mínima no tiene; solo "
                            "respuesta óptica, apantallada con cargas y dipolos"}
        for el, v in zip(ELEMENTS, result.x, strict=True)}
    output["alpha_fit"] = {
        "references": Path(alpha_refs).name,
        "references_sha256": hashlib.sha256(Path(alpha_refs).read_bytes()).hexdigest(),
        "optimiser": str(result.message), "molecules": after, "checks": checks}
    Path(out).write_text(json.dumps(output, indent=1, ensure_ascii=False), encoding="utf-8")
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("alpha_refs", type=Path)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    run(args.alpha_refs, args.parameters, args.out)


if __name__ == "__main__":
    main()
