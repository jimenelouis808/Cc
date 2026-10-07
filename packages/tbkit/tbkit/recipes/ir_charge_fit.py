"""Fit and judge the IR charge corrections (tbkit.charge_model) against GPAW Born charges.

Run from packages/tbkit after ``recipes/born_references`` (no GPAW needed here)::

    python -m tbkit.recipes.ir_charge_fit chn       # xu_chn: C/H/N references only
    python -m tbkit.recipes.ir_charge_fit chno      # xu_chno: all references

For the TB set it computes the model's own Born charges at every reference geometry
(cached), then fits three corrections — bond flux (class-IV charges, 2 parameters
per element pair), environment flux (linear SOAP, machine learned) and both together
— and judges each by numbers it was not fitted to:

1. Born charges, leaving one structure out at a time (RMS, e);
2. IR intensities on the TB modes of each structure at its reference geometry, with
   the held-out correction (median |log10 I/I_GPAW| and the fraction within a
   factor 2, over the modes GPAW gives more than 1 km/mol);
3. the coil: GPAW's IR along the xu_chn modes of ``recipes/doped_gpaw`` against the
   corrected TB IR along the same modes (the cut-outs of the coil are in the
   training data, the periodic coil itself is not).

It writes ``out/ir_charge_fit/<set>/`` (models, report) and
``validation/ir_charge_models_<set>.json``. Adopting a model (a parameter file
the IR uses by default) is a separate decision, taken on these numbers: it must
improve 1, 2 and 3, not just the training error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms

REFS = Path("out/born_references")
WORK = Path("out/ir_charge_fit")
SETS = {"chn": ("xu_chn", {"C", "H", "N"}), "chno": ("xu_chno", {"C", "H", "N", "O"})}
KT = 0.01


def rows_for(set_name: str) -> list[dict]:
    from ..infrared import born_charges
    from ..params import load_parameters

    model_name, elements = SETS[set_name]
    model = load_parameters(model_name)
    cache = WORK / set_name / "tb_born"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(REFS.glob("*.json")):
        data = json.loads(path.read_text())
        if not set(data["symbols"]) <= elements:
            continue
        atoms = Atoms(data["symbols"], positions=data["positions_A"], cell=data["cell_A"])
        file = cache / f"{data['name']}.npy"
        if not file.exists():
            np.save(file, born_charges(atoms, model, kT=KT))
        rows.append({"atoms": atoms, "group": data["name"], "z_ref": np.array(data["born_e"]),
                     "z_tb": np.load(file)})
    return rows


def _modes(atoms: Atoms, model, cache: Path):
    from ..raman import internal_modes, phonons_for

    if cache.exists():
        data = np.load(cache)
        return data["f"], data["L"]
    warnings: list[str] = []
    f, L = phonons_for(atoms, model, 12, KT, 0.005, None, warnings)
    keep = internal_modes(atoms, f, warnings)
    np.savez(cache, f=f[keep], L=L[keep])
    return f[keep], L[keep]


def _intensity(z, L):
    from ..infrared import DEBYE_PER_EA, KM_PER_MOL

    d = np.einsum("aij,mai->mj", z, L)
    return np.sum(d ** 2, axis=1) * DEBYE_PER_EA ** 2 * KM_PER_MOL


def ir_scores(rows, set_name, corrections: dict) -> dict:
    """Median |log10(I/I_GPAW)| and fraction within ×2 on modes with I_GPAW > 1 km/mol,
    TB alone and with each correction (held out: θ fitted without that structure)."""
    from ..params import load_parameters

    model = load_parameters(SETS[set_name][0])
    folder = WORK / set_name / "modes"
    folder.mkdir(parents=True, exist_ok=True)
    logs = {k: [] for k in ["tb", *corrections]}
    per = {}
    for row in rows:
        f, L = _modes(row["atoms"], model, folder / f"{row['group']}.npz")
        ok = f > 300
        if not ok.any():
            continue
        ref = _intensity(row["z_ref"], L[ok])
        strong = ref > 1.0
        if not strong.any():
            continue
        values = {"tb": _intensity(row["z_tb"], L[ok])}
        for name, (mdl, cv_theta) in corrections.items():
            dz = np.tensordot(cv_theta[row["group"]], mdl.design(row["atoms"])[0], axes=1)
            values[name] = _intensity(row["z_tb"] + dz, L[ok])
        per[row["group"]] = {k: float(np.median(np.abs(np.log10(np.maximum(v[strong], 1e-6)
                                                                / ref[strong]))))
                             for k, v in values.items()}
        for k, v in values.items():
            logs[k] += list(np.log10(np.maximum(v[strong], 1e-6) / ref[strong]))
    return {"modes": len(logs["tb"]),
            "median_abs_log10": {k: float(np.median(np.abs(v))) for k, v in logs.items()},
            "within_factor_2": {k: float(np.mean(np.abs(v) < np.log10(2))) for k, v in logs.items()},
            "per_structure": per}


def coil_scores(corrections: dict) -> dict:
    """GPAW IR along the xu_chn coil modes (recipes/doped_gpaw) against TB ± corrections."""
    from ase.io import read

    from . import doped_gpaw as g
    from . import doped_raman as tb

    report = g.WORK / "report.json"
    if not report.exists():
        return {"note": "doped_gpaw sin terminar"}
    data = json.loads(report.read_text())
    logs = {k: [] for k in ["tb", *corrections]}
    rows = []
    for name in ("pristine", "N", "amine"):
        entry = data.get(name, {})
        if not entry.get("modes") or not tb.born_ready(name):
            continue
        atoms = read(tb.WORK / name / "relaxed.extxyz")
        z = tb.born(name)
        vectors = np.load(g.folder(name) / "vectors.npy")
        picks = json.loads((g.folder(name) / "picks.json").read_text())
        index = {p["mode"]: k for k, p in enumerate(picks)}
        dzs = {k: np.tensordot(m.theta, m.design(atoms)[0], axes=1)
               for k, (m, _) in corrections.items()}
        for mode in entry["modes"]:
            L = vectors[index[mode["mode"]]][None]
            ref = mode["pbe_ir_km_mol"]
            values = {"tb": float(_intensity(np.nan_to_num(z), L)[0])}
            for k, dz in dzs.items():
                values[k] = float(_intensity(np.nan_to_num(z) + dz, L)[0])
            rows.append({"structure": name, "tb_cm1": mode["tb_cm1"], "gpaw_km_mol": ref,
                         **{f"{k}_km_mol": v for k, v in values.items()}})
            if ref > 1.0:
                for k, v in values.items():
                    logs[k].append(np.log10(max(v, 1e-6) / ref))
    if not rows:
        return {"note": "sin modos de la coil"}
    return {"modes_over_1_km_mol": len(logs["tb"]),
            "median_abs_log10": {k: float(np.median(np.abs(v))) for k, v in logs.items()},
            "within_factor_2": {k: float(np.mean(np.abs(v) < np.log10(2))) for k, v in logs.items()},
            "modes": rows}


def main(argv=None) -> None:
    from .. import charge_model as cm

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("set", choices=tuple(SETS))
    parser.add_argument("--no-ml", action="store_true", help="sin el modelo SOAP")
    args = parser.parse_args(argv)
    rows = rows_for(args.set)
    elements = sorted(set().union(*(set(r["atoms"].get_chemical_symbols()) for r in rows)))
    candidates = {"bond_flux": cm.BondFlux.for_elements(elements)}
    if not args.no_ml:
        env = cm.EnvironmentFlux(tuple(elements))
        candidates["environment_flux"] = env
        candidates["both"] = cm.Combined([cm.BondFlux.for_elements(elements),
                                          cm.EnvironmentFlux(tuple(elements))])
    out = WORK / args.set
    out.mkdir(parents=True, exist_ok=True)
    fits, corrections = {}, {}
    for name, model in candidates.items():
        record = cm.fit(rows, model, keep_cv=True)
        corrections[name] = (model, record.pop("cv_theta"))
        fits[name] = record
        cm.save(model, out / f"{name}.json")
        print(f"{name:17s} Born RMS: TB {record['rms_tb_e']:.4f}  entrenamiento "
              f"{record['rms_train_e']:.4f}  validación cruzada {record['rms_cv_e']:.4f} e "
              f"({record['n_params']} parámetros)")
    ir = ir_scores(rows, args.set, corrections)
    coil = coil_scores(corrections) if args.set == "chn" else {"note": "solo para chn"}
    report = {"set": SETS[args.set][0], "structures": [r["group"] for r in rows],
              "reference": "GPAW LCAO dzp PBE Born charges (recipes/born_references)",
              "born_fits": fits, "ir_on_tb_modes_heldout": ir, "coil_gpaw_modes": coil}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    Path(f"validation/ir_charge_models_{args.set}.json").write_text(
        json.dumps({k: v for k, v in report.items()}, indent=1, ensure_ascii=False))
    print("IR (modos > 1 km/mol, validación cruzada): mediana |log10 I/I_GPAW| ",
          {k: round(v, 3) for k, v in ir["median_abs_log10"].items()},
          " dentro de ×2 ", {k: round(v, 2) for k, v in ir["within_factor_2"].items()})
    if "median_abs_log10" in coil:
        print("coil (modos GPAW): mediana |log10| ",
              {k: round(v, 3) for k, v in coil["median_abs_log10"].items()},
              " dentro de ×2 ", {k: round(v, 2) for k, v in coil["within_factor_2"].items()})


if __name__ == "__main__":
    main()
