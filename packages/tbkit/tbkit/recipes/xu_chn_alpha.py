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


def with_extra(model, values, elements=ELEMENTS, fixed: dict | None = None) -> object:
    extra = dict(fixed or {})
    extra.update(zip(elements, map(float, values), strict=True))
    return dataclasses.replace(model, extra_polarizability=extra)


def model_tensor(model, atoms) -> np.ndarray:
    return polarizability_linear_response(System.build(atoms, model))


def fit(model, entries, geometries, elements=ELEMENTS, fixed: dict | None = None):
    train = [e for e in entries if e["role"] == "train"]

    def residuals(x):
        trial = with_extra(model, x, elements, fixed)
        out = []
        for entry in train:
            reference = np.array(entry["alpha"])
            scale = np.trace(reference) / 3
            difference = model_tensor(trial, geometries[entry["group"]]) - reference
            out.append(difference[_UPPER] / scale)
        return np.concatenate(out)

    x0 = np.array([{"H": 0.3, "C": 0.8, "N": 0.6}.get(el, 0.6) for el in elements])
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


def run(alpha_refs, parameters: Path, out: Path, verbose: bool = True,
        elements=ELEMENTS, fixed_from: Path | None = None, geometry_refs=None) -> dict:
    """Fit the extra polarizabilities of ``elements`` (the others fixed).

    ``alpha_refs`` may be one file or several; ``fixed_from`` a parameter file
    whose ``extra_polarizability`` supplies the elements not fitted;
    ``geometry_refs`` the reference files holding the relaxed geometries
    (default: ``gpaw_chn.json`` next to the parameters).
    """
    alpha_refs = [Path(alpha_refs)] if isinstance(alpha_refs, (str, Path)) else \
        [Path(p) for p in alpha_refs]
    entries, settings = [], {}
    for path in alpha_refs:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries += data["polarizabilities"]
        settings = data["settings"]
    base_data = read_parameter_file(parameters)
    model = model_from_dict(base_data)
    fixed = {}
    if fixed_from is not None:
        fixed = {el: float(v["value"] if isinstance(v, dict) else v) for el, v in
                 read_parameter_file(fixed_from).get("extra_polarizability", {}).items()
                 if el not in elements}
    if geometry_refs is None:
        geometry_refs = [Path(parameters).parent / "references" / "gpaw_chn.json"]
    geometries = {}
    for path in geometry_refs:
        structures, _ = load_references(path)
        geometries.update({s.group: s.atoms for s in structures if s.label.endswith("/eq")})
    start = with_extra(model, [0.0] * len(elements), elements, fixed)
    before = report(start, entries, geometries)
    result = fit(model, entries, geometries, elements, fixed)
    fitted = with_extra(model, result.x, elements, fixed)
    after = report(fitted, entries, geometries)
    checks = {}
    if "C" in elements:
        checks = {"c60_alpha": c60_alpha(fitted), "c60_alpha_experiment": C60_ALPHA_EXPERIMENT,
                  "c60_alpha_without_extra": c60_alpha(model),
                  "diamond_eps": diamond_eps(fitted), "diamond_eps_experiment":
                  DIAMOND_EPS_EXPERIMENT, "diamond_eps_without_extra": diamond_eps(model)}
    if verbose:
        print("α extra (Å³):", dict(zip(elements, np.round(result.x, 4), strict=True)),
              "fijos:", fixed)
        for name in after:
            b, a = before[name], after[name]
            print(f"  {name:10s} {a['role']:5s} GPAW {np.round(a['gpaw'], 2)}  "
                  f"TB {np.round(b['tb'], 2)} → {np.round(a['tb'], 2)}  "
                  f"(media {b['mean_error_percent']:+.0f} % → {a['mean_error_percent']:+.0f} %)")
        if checks:
            print("  C60:", round(checks["c60_alpha_without_extra"], 1), "→",
                  round(checks["c60_alpha"], 1), "Å³ (exp.", C60_ALPHA_EXPERIMENT, ")")
            print("  diamante ε∞:", round(checks["diamond_eps_without_extra"], 2), "→",
                  round(checks["diamond_eps"], 2), "(exp.", DIAMOND_EPS_EXPERIMENT, ")")
    source = (f"ajustada a tensores α de GPAW ({settings.get('xc')}, FD, h = "
              f"{settings.get('h')} Å); receta tbkit.recipes.xu_chn_alpha")
    output = dict(base_data)
    extra = {el: {"value": round(float(v), 5), "unit": "Å^3", "source": source,
                  "description": "polarizabilidad atómica que la base mínima no tiene; solo "
                                 "respuesta óptica, apantallada con cargas y dipolos"}
             for el, v in zip(elements, result.x, strict=True)}
    if fixed_from is not None:
        for el, v in read_parameter_file(fixed_from).get("extra_polarizability", {}).items():
            if el not in elements:
                extra[el] = v if isinstance(v, dict) else {"value": v, "unit": "Å^3"}
                extra[el] = dict(extra[el], source=f"de {Path(fixed_from).name} (no reajustada)")
    output["extra_polarizability"] = extra
    shas = [hashlib.sha256(p.read_bytes()).hexdigest() for p in alpha_refs]
    output["alpha_fit"] = {
        "references": alpha_refs[0].name if len(alpha_refs) == 1 else [p.name for p in alpha_refs],
        "references_sha256": shas[0] if len(shas) == 1 else shas,
        "fitted_elements": list(elements),
        "optimiser": str(result.message), "molecules": after, "checks": checks}
    Path(out).write_text(json.dumps(output, indent=1, ensure_ascii=False), encoding="utf-8")
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("alpha_refs", type=Path, nargs="+",
                        help="polarizabilidades GPAW de referencia (chn_polarizability)")
    parser.add_argument("parameters", type=Path,
                        help="conjunto al que se añaden las polarizabilidades extra")
    parser.add_argument("out", type=Path,
                        help="archivo de parámetros que se escribe")
    parser.add_argument("--fit", nargs="+", default=list(ELEMENTS),
                        help="elementos cuya α extra se ajusta (por defecto H C N)")
    parser.add_argument("--fixed-from", type=Path, default=None,
                        help="archivo de parámetros con la α extra de los demás elementos")
    parser.add_argument("--geometries", type=Path, nargs="*", default=None,
                        help="archivos de referencias con las geometrías relajadas")
    args = parser.parse_args(argv)
    run(args.alpha_refs, args.parameters, args.out, elements=tuple(args.fit),
        fixed_from=args.fixed_from, geometry_refs=args.geometries)


if __name__ == "__main__":
    main()
