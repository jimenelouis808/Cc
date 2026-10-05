"""Fit and test a Δ (linear SOAP) correction on top of a TB set.

Run (needs dscribe; TB only, minutes)::

    python -m tbkit.recipes.delta_fit tang_carbon gpaw_carbon_env.json WORKDIR \\
        --held-out amorphous_3.2 stone_wales

1. TB energy and forces of every reference structure (cached per structure).
2. The ridge strength is chosen by leave-one-group-out cross-validation on the
   training groups only (force RMSE), so the held-out groups never steer it.
3. Final fit on all training groups; errors (force RMSE, energy per atom) of
   TB alone and TB + Δ for every group, held-out ones marked.
4. The model is written to WORKDIR/delta.json with its settings; nothing is
   installed in ``parameters/`` (a correction is used deliberately, not by
   default).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RIDGES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def tb_rows(model_name: str, references: Path, workdir: Path, kT: float = 0.1) -> list[dict]:
    from ..calculator import TBCalculator
    from ..params import load_parameters
    from ..references import load_references

    model = load_parameters(model_name)
    refs, _ = load_references(references)
    cache = workdir / "tb"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, ref in enumerate(refs):
        path = cache / f"{k:03d}.npz"
        if not path.exists():
            atoms = ref.atoms.copy()
            kpts = tuple(ref.extra["kpts"]) if atoms.pbc.any() else None
            atoms.calc = TBCalculator(model, kpts=kpts, kT=kT)
            np.savez(path, energy=atoms.get_potential_energy(), forces=atoms.get_forces())
        tb = np.load(path)
        rows.append({"atoms": ref.atoms, "group": ref.group, "label": ref.label,
                     "dE": float(ref.energy - tb["energy"]),
                     "dF": np.asarray(ref.forces) - tb["forces"],
                     "F_dft": np.asarray(ref.forces)})
    return rows


def errors(rows, model=None) -> dict:
    """Per group: force RMSE and mean |energy error| per atom after removing, per group,
    nothing (energies share one reference per element through the biases)."""
    out = {}
    for row in rows:
        residual_f = row["dF"]
        residual_e = row["dE"]
        if model is not None:
            e, f = model.energy_forces(row["atoms"], row.get("soap"))
            residual_f = residual_f - f
            residual_e = residual_e - e
        g = out.setdefault(row["group"], {"f2": [], "e": []})
        g["f2"].append(np.mean(residual_f ** 2))
        g["e"].append(residual_e / len(row["atoms"]))
    result = {}
    for k, v in out.items():
        e = np.array(v["e"])
        result[k] = {"force_rmse": float(np.sqrt(np.mean(v["f2"]))),
                     "energy_spread_per_atom": float(np.std(e))}
    return result


def choose_ridge(train, species, **kw) -> tuple[float, dict]:
    from ..delta import fit_delta

    groups = sorted({r["group"] for r in train})
    scores = {}
    for ridge in RIDGES:
        f2, n = 0.0, 0
        for g in groups:
            fit_rows = [r for r in train if r["group"] != g]
            test = [r for r in train if r["group"] == g]
            model = fit_delta(fit_rows, species, ridge=ridge, **kw)
            for row in test:
                if "soap" not in row:
                    row["soap"] = model.descriptor(row["atoms"], derivatives=True)
                _, f = model.energy_forces(row["atoms"], row["soap"])
                f2 += float(np.sum((row["dF"] - f) ** 2))
                n += row["dF"].size
        scores[ridge] = float(np.sqrt(f2 / n))
    return min(scores, key=scores.get), scores


def main(argv=None) -> None:
    from ..delta import fit_delta
    from ..params import PARAMETER_DIR

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("references")
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--held-out", nargs="*", default=())
    parser.add_argument("--kT", type=float, default=0.1)
    args = parser.parse_args(argv)
    args.workdir.mkdir(parents=True, exist_ok=True)
    references = Path(args.references)
    if not references.exists():
        references = PARAMETER_DIR / "references" / args.references
    rows = tb_rows(args.model, references, args.workdir, args.kT)
    species = sorted({s for r in rows for s in r["atoms"].get_chemical_symbols()})
    train = [r for r in rows if r["group"] not in args.held_out]
    ridge, scores = choose_ridge(train, species)
    model = fit_delta(train, species, ridge=ridge)
    before, after = errors(rows), errors(rows, model)
    model.info.update({"base_model": args.model, "references": references.name,
                       "held_out": list(args.held_out), "cv_force_rmse": scores, "kT": args.kT})
    model.save(args.workdir / "delta.json")
    report = {"ridge": ridge, "cv_force_rmse": {str(k): v for k, v in scores.items()},
              "groups": {g: {"held_out": g in args.held_out, "tb": before[g], "tb_delta": after[g]}
                         for g in before}}
    (args.workdir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"ridge {ridge} (CV fuerzas {scores[ridge]:.3f} eV/Å)")
    for g, v in report["groups"].items():
        tag = "FUERA" if v["held_out"] else "ajuste"
        print(f"  {g:14s} {tag:6s} F TB {v['tb']['force_rmse']:.3f} -> TB+Δ "
              f"{v['tb_delta']['force_rmse']:.3f} eV/Å")


if __name__ == "__main__":
    main()
