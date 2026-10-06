"""The environment-dependent model (Tang et al. 1996) refitted to GPAW: ``tang_carbon``.

Run (TB only, resumable; tens of minutes)::

    python -m tbkit.recipes.tang_fit WORKDIR [--install]

The published parameters (``tang1996_published.json``) do not give the paper's
diamond with any cutoff we tried: what the paper leaves out is the cutoff,
and the weakly screened on-site shift Δe makes the result hang on it. So the
electronic part is kept as published (hoppings, screening, coordination
scaling) and what it left open is chosen against GPAW PBE
(``gpaw_carbon_env.json``):

* a scale ``s`` of the Δe amplitude and the pair cutoff, on a small grid;
* for each grid point, the repulsion polynomial f(x) = Σ c_k x^k (c1..c4;
  c0 kept, it is a constant per atom) and one energy per atom μ, by linear
  least squares on GPAW energies (per atom, all structures share μ) and
  forces. Linear because E_rep is linear in c_k at fixed φ.

The amorphous carbon at 3.2 g/cm³ and the Stone-Wales defect are held out;
the grid point is chosen on the training error only. Each grid point is a
file in WORKDIR (resumable). ``--install`` writes ``parameters/tang_carbon.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "parameters"
HELD_OUT = ("amorphous_3.2", "stone_wales")
SCALES = (1.0, 0.5, 0.25, 0.0)
CUTS = ((3.0, 3.6), (4.2, 5.0))
FORCE_WEIGHT = 1.0          # eV/Å residuals
ENERGY_WEIGHT = 10.0        # eV/atom residuals, × this


def _model(scale: float, cut: tuple):
    from ..params import model_from_dict, read_parameter_file

    data = copy.deepcopy(read_parameter_file("tang1996_published"))
    data["environment"]["functions"]["onsite"]["a1"]["value"] *= scale
    data["environment"]["pair_cut"] = list(cut)
    return data, model_from_dict(data)


def _structure_terms(model, ref):
    """Electronic energy and forces, and the repulsion basis (k = 1..4) of one structure."""
    from ..forces import band_forces, entropy_term
    from ..hamiltonian import System
    from ..kpoints import gamma, mesh
    from ..solver import solve

    atoms = ref.atoms
    system = System.build(atoms, model)
    kpts, weights = mesh(atoms, tuple(ref.extra["kpts"])) if atoms.pbc.any() else gamma()
    solution = solve(system, kpts, weights, kT=0.1)
    e_el = solution.band_energy() - entropy_term(solution)
    f_el = band_forces(solution)
    env = system.env
    x = np.zeros(len(atoms))
    np.add.at(x, env.i, env.values["rep"])
    basis_e, basis_f = [], []
    for k in range(1, 5):
        basis_e.append(float(np.sum(x ** k)))
        basis_f.append(-env.backward({"rep": k * x[env.i] ** (k - 1)}))
    c0 = model.environment.poly[0]
    return {"n": len(atoms), "e_dft": ref.energy, "f_dft": ref.forces, "e_el": e_el + c0 * len(atoms),
            "f_el": f_el, "basis_e": np.array(basis_e), "basis_f": np.array(basis_f)}


def _solve(rows):
    """c1..c4 and μ by weighted linear least squares."""
    a, b = [], []
    for r in rows:
        a.append(np.concatenate([r["basis_e"] / r["n"], [1.0]]) * ENERGY_WEIGHT)
        b.append((r["e_dft"] - r["e_el"]) / r["n"] * ENERGY_WEIGHT)
        fa = np.concatenate([r["basis_f"].reshape(4, -1).T, np.zeros((r["f_dft"].size, 1))], axis=1)
        a.extend(fa * FORCE_WEIGHT)
        b.extend((r["f_dft"] - r["f_el"]).ravel() * FORCE_WEIGHT)
    solution, *_ = np.linalg.lstsq(np.array(a), np.array(b), rcond=None)
    return solution[:4], solution[4]


def _errors(rows, c, mu):
    out = {}
    for r in rows:
        f = r["f_el"] + np.tensordot(c, r["basis_f"], axes=1)
        e = r["e_el"] + c @ r["basis_e"] + mu * r["n"]
        g = out.setdefault(r["group"], {"f2": [], "fdft2": [], "e": []})
        g["f2"].append(np.mean((f - r["f_dft"]) ** 2))
        g["fdft2"].append(np.mean(r["f_dft"] ** 2))
        g["e"].append((e - r["e_dft"]) / r["n"])
    return {k: {"force_rmse": float(np.sqrt(np.mean(v["f2"]))),
                "force_rms_dft": float(np.sqrt(np.mean(v["fdft2"]))),
                "energy_mae_per_atom": float(np.mean(np.abs(v["e"])))} for k, v in out.items()}


def grid_point(scale: float, cut: tuple, workdir: Path) -> dict:
    from ..references import load_references

    path = Path(workdir) / f"s{scale:.2f}_c{cut[0]:.1f}-{cut[1]:.1f}.json"
    if path.exists():
        return json.loads(path.read_text())
    refs, _ = load_references(ROOT / "references" / "gpaw_carbon_env.json")
    _, model = _model(scale, cut)
    rows = []
    for ref in refs:
        row = _structure_terms(model, ref)
        row["group"] = ref.group
        rows.append(row)
    train = [r for r in rows if r["group"] not in HELD_OUT]
    c, mu = _solve(train)
    errors = _errors(rows, c, mu)
    result = {"scale": scale, "cut": list(cut), "c": c.tolist(), "mu": float(mu),
              "train_force_rmse": float(np.sqrt(np.mean([errors[g]["force_rmse"] ** 2 for g in errors
                                                         if g not in HELD_OUT]))),
              "errors": errors}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(path)
    return result


def install(best: dict) -> Path:
    data, _ = _model(best["scale"], tuple(best["cut"]))
    source = ("ajustado a GPAW PBE (gpaw_carbon_env.json, sin amorphous_3.2 ni stone_wales); "
              "receta tbkit.recipes.tang_fit")
    data["name"] = "Carbono dependiente del entorno (forma de Tang et al. 1996, ajustado a GPAW)"
    data["environment"]["functions"]["onsite"]["a1"]["source"] = (
        f"Tang et al. 1996, tabla I, × {best['scale']} ({source})")
    data["environment"]["pair_cut"] = best["cut"]
    data["environment"]["cutoff_source"] = "elegido en la rejilla de la receta tang_fit (" + source + ")"
    for k, value in enumerate(best["c"], start=1):
        data["environment"]["poly"][k] = {"value": value, "unit": "eV", "source": source}
    err = best["errors"]
    data["validity"] = (
        "solo carbono, sin SCC. Forma funcional y parte electrónica de Tang et al. 1996; repulsión, "
        "escala de Δe y corte ajustados a GPAW PBE. Error de fuerzas (RMS, eV/Å) frente a GPAW: " +
        ", ".join(f"{g} {v['force_rmse']:.2f}" for g, v in err.items()) +
        " (amorphous_3.2 y stone_wales fuera del ajuste). Energías comparables entre estructuras "
        "de carbono (una sola referencia por átomo).")
    data["fit"] = {"grid": {"scales": list(SCALES), "cuts": [list(c) for c in CUTS]},
                   "held_out": list(HELD_OUT), "mu_eV_per_atom": best["mu"], "errors": err,
                   "weights": {"force": FORCE_WEIGHT, "energy_per_atom": ENERGY_WEIGHT}}
    path = ROOT / "tang_carbon.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    args.workdir.mkdir(parents=True, exist_ok=True)
    results = []
    for cut in CUTS:
        for scale in SCALES:
            r = grid_point(scale, cut, args.workdir)
            results.append(r)
            held = {g: round(r["errors"][g]["force_rmse"], 2) for g in HELD_OUT}
            print(f"Δe×{scale:.2f} corte {cut}: F_rmse entreno {r['train_force_rmse']:.3f} eV/Å, "
                  f"fuera {held}", flush=True)
    best = min(results, key=lambda r: r["train_force_rmse"])
    print("mejor:", best["scale"], best["cut"])
    if args.install:
        print(install(best))


if __name__ == "__main__":
    main()
