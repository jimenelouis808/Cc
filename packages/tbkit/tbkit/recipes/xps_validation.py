"""XPS core-level shifts: tbkit's equivalent core (Z+1) against GPAW ΔSCF with core holes.

Run from packages/tbkit::

    python -m tbkit.recipes.xps_validation setups     # C and N 1s core-hole PAW setups
    python -m tbkit.recipes.xps_validation gpaw [S]   # ΔSCF, one file per site (resumable)
    python -m tbkit.recipes.xps_validation report     # validation/xps_tb_vs_gpaw.json and
                                                      # parameters/xps_scale.json (λ per element)

GPAW reference: E[full 1s core hole on the site (the setup carries the +1)] − E[ground], PBE, real-
space grid (FD) with zero boundary conditions: an isolated ion. Plane waves were tried first
and are wrong here: the +1 ion in a periodic box carries a spurious electrostatic energy
that depends on the box, which varies with the molecule (benzene came out -11 eV and
coronene -16 eV from CH4, against experiment's tenths of an eV); the core-hole setup is generated here (``gpaw dataset
--core-hole 1s,1``). The absolute value carries the setup's frozen-core constant, so,
as in tbkit, only shifts against a reference site of the same element are compared:
CH4 for C 1s, NH3 for N 1s. Geometries are ASE's g2 set (and an N-doped coronene built
here), the same for both codes, unrelaxed.

tbkit: :mod:`tbkit.xps` with xu_chno (C 1s and N 1s) and xu_chn (C 1s).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

from ..settings import Param, add_arguments, apply

WORK = Path("out/xps")
SETUPS = WORK / "setups"
OUT = Path("validation/xps_tb_vs_gpaw.json")
GRID_H = 0.18
VACUUM = 6.0
KT = 0.01
REFERENCE = {"C": ("CH4", 0), "N": ("NH3", 0), "O": ("H2O", 0), "Se": ("se:H2Se", 0)}
#: GPAW core-hole setup of each element: (tag, --core-hole argument)
CORE = {"C": ("1s1ch", "1s,1.0"), "N": ("1s1ch", "1s,1.0"), "O": ("1s1ch", "1s,1.0"),
        "Se": ("3d1ch", "3d,1.0")}
MOLECULES = ("CH4", "C2H6", "C2H4", "C2H2", "C6H6", "CH3OH", "H2CO", "HCOOH", "CO", "CO2",
             "HCN", "CH3CN", "NH3", "C5H5N", "C4H4NH")
FLAKES = ("coronene", "coronene_N_graphitic", "coronene_N_pyridinic")
#: O 1s: more g2 molecules and the GPAW-relaxed epoxide and diol on coronene (gpaw_chno.json)
O_MOLECULES = ("H2O", "CH3OCH3", "CH3CHO", "H2O2", "CH3COOH", "CH3NO2", "C4H4O")
O_FLAKES = ("coronene_epoxide", "coronene_diol")
#: Se 3d: the GPAW-relaxed molecules of gpaw_se_hessians.json
SE_MOLECULES = ("H2Se", "CH3SeH", "CH3SeCH3", "CH3SeSeCH3", "selenophene", "CSe2", "OCSe",
                "H2CSe", "SeO2", "DMSeO", "CH3SeO2H", "CH3SeNH2")
TB_MODELS = {"C": ("xu_chno", "xu_chn"), "N": ("xu_chno",), "O": ("xu_chno",),
             "Se": ("xu_chnose",)}
METHODS = ("hole", "z+1")


def _coronene() -> Atoms:
    from .spin_validation import _flake, _saturate

    dx, dy = np.sqrt(3) * 1.42, 1.5 * 1.42
    centres = [(0, 0, 0), (dx, 0, 0), (-dx, 0, 0), (dx / 2, dy, 0), (-dx / 2, dy, 0),
               (dx / 2, -dy, 0), (-dx / 2, -dy, 0)]
    return _saturate(_flake(centres))


def _references_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "parameters" / "references"


def geometry(name: str) -> Atoms:
    from ase.build import molecule

    if name.startswith("se:"):
        entry = json.loads((_references_dir() / "gpaw_se_hessians.json").read_text()
                           )["molecules"][name[3:]]
        atoms = Atoms(entry["symbols"], positions=entry["positions"])
        atoms.center(vacuum=VACUUM)
        return atoms
    if name in O_FLAKES:
        from ..references import load_references

        refs, _ = load_references(_references_dir() / "gpaw_chno.json")
        atoms = next(r.atoms for r in refs if r.label == f"{name}/eq").copy()
        atoms.center(vacuum=VACUUM)
        return atoms
    if name in MOLECULES or name in O_MOLECULES:
        atoms = molecule(name)
        atoms.center(vacuum=VACUUM)
        return atoms
    atoms = _coronene()
    carbons = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "C"]
    centre = atoms.positions[carbons].mean(axis=0)
    r = np.linalg.norm(atoms.positions[carbons] - centre, axis=1)
    if name == "coronene_N_graphitic":       # an atom of the inner ring
        atoms.symbols[carbons[int(np.argmin(r))]] = "N"
    elif name == "coronene_N_pyridinic":     # an edge C-H becomes a two-fold N
        hydrogens = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
        h = hydrogens[0]
        c = carbons[int(np.argmin(np.linalg.norm(atoms.positions[carbons]
                                                 - atoms.positions[h], axis=1)))]
        atoms.symbols[c] = "N"
        del atoms[h]
    atoms.center(vacuum=VACUUM)
    return atoms


def sites(name: str) -> list[int]:
    """Symmetry-distinct C and N sites (by sorted distances to the other atoms)."""
    atoms = geometry(name)
    seen, out = set(), []
    d = atoms.get_all_distances()
    symbols = atoms.get_chemical_symbols()
    keep = range(len(atoms))
    hetero = [i for i, s in enumerate(symbols) if s in ("N", "O", "Se")]
    if hetero and name.startswith("coronene"):
        # a doped flake: the heteroatoms, the carbons within two bonds of the first one and
        # the farthest carbon
        n = hetero[0]
        carbons = [i for i, s in enumerate(symbols) if s == "C"]
        reach = 2.6 if symbols[n] == "N" else 1.6      # O groups: only the bonded carbons
        keep = hetero + [i for i in carbons if d[n, i] < reach] + \
            [carbons[int(np.argmax(d[n, carbons]))]]
    if name.startswith("se:"):                          # Se 3d only (xu_chnose)
        keep = [i for i, s in enumerate(symbols) if s == "Se"]
    for i in keep:
        s = symbols[i]
        if s not in REFERENCE:
            continue
        key = (s, tuple(np.round(np.sort(d[i]), 2)))
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def systems() -> dict[str, list[int]]:
    names = MOLECULES + FLAKES + O_MOLECULES + O_FLAKES + tuple(f"se:{m}" for m in SE_MOLECULES)
    return {name: sites(name) for name in names}


# ---------------------------------------------------------------- GPAW
def make_setups() -> list[Path]:
    SETUPS.mkdir(parents=True, exist_ok=True)
    gpaw = Path(sys.executable).with_name("gpaw")
    made = []
    for element, (tag, hole) in CORE.items():
        target = SETUPS / f"{element}.{tag}.PBE"
        if not target.exists():
            subprocess.run([str(gpaw), "dataset", element, "-f", "PBE", f"--core-hole={hole}",
                            "-t", tag, "-w"], cwd=SETUPS, check=True,
                           stdout=subprocess.DEVNULL)
        made.append(target)
    return made


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def run_gpaw(name: str, site: int | None) -> Path:
    """One energy: the ground state (``site`` None) or a core hole on ``site``."""
    import gpaw
    from gpaw import GPAW

    out = WORK / name.replace(":", "_") / ("ground.json" if site is None else f"hole_{site}.json")
    if out.exists():
        return out
    if str(SETUPS.resolve()) not in gpaw.setup_paths:
        gpaw.setup_paths.insert(0, str(SETUPS.resolve()))
    atoms = geometry(name)
    # the core-hole setup already carries the +1 (GPAW prints "Charge: 1.0" for it): adding
    # charge=1 on top removed a valence electron as well (a double ion, shifts of -6 to
    # -16 eV that were the valence ionisation potentials of the molecules)
    kwargs = {} if site is None else {"setups": {site: CORE[atoms[site].symbol][0]}}
    out.parent.mkdir(parents=True, exist_ok=True)
    atoms.pbc = False                    # an isolated ion: zero boundary, no image charges
    atoms.calc = GPAW(mode="fd", h=GRID_H, xc="PBE",
                      occupations={"name": "fermi-dirac", "width": KT},
                      txt=str(out.with_suffix(".txt")), **kwargs)
    _write(out, {"system": name, "site": site, "energy_eV": float(atoms.get_potential_energy()),
                 "settings": f"GPAW PBE FD h {GRID_H} Å, contorno cero (no periódico), "
                             f"Fermi-Dirac {KT} eV, vacío {VACUUM} Å, "
                             "hueco de núcleo completo (el setup lleva la carga +1; ΔSCF)"})
    return out


def gpaw_energy(name: str, site: int) -> float | None:
    try:
        folder = WORK / name.replace(":", "_")
        ground = json.loads((folder / "ground.json").read_text())["energy_eV"]
        hole = json.loads((folder / f"hole_{site}.json").read_text())["energy_eV"]
    except FileNotFoundError:
        return None
    return hole - ground


# ---------------------------------------------------------------- report
SCALE_OUT = Path("tbkit/parameters/xps_scale.json")
#: A calibration is shipped only if its prediction correlates with GPAW at least this well
#: over every site and improves the held-out half.
MIN_CORRELATION = 0.95


def _tb_values(atoms, model, site, method):
    from ..xps import ionisation_energy, ionisation_parts

    if method == "hole":
        initial, relaxation = ionisation_parts(atoms, model, site, KT)
        return {"total": initial + relaxation, "initial": initial, "relaxation": relaxation}
    return {"total": ionisation_energy(atoms, model, site, KT, method="z+1")}


def report() -> dict:
    from ..params import load_parameters

    models = {name: load_parameters(name) for names in TB_MODELS.values() for name in names}
    references = {}
    for element, (name, site) in REFERENCE.items():
        atoms = geometry(name)
        references[element] = {"gpaw": gpaw_energy(name, site)}
        for model_name in TB_MODELS[element]:
            for method in METHODS:
                try:
                    references[element][(model_name, method)] = _tb_values(
                        atoms, models[model_name], site, method)
                except ValueError:
                    pass
    rows = []
    for name, indices in systems().items():
        atoms = geometry(name)
        for site in indices:
            element = atoms[site].symbol
            g = gpaw_energy(name, site)
            row = {"system": name, "site": site, "element": element,
                   "neighbours": sorted(atoms[j].symbol for j in range(len(atoms))
                                        if j != site and atoms.get_distance(site, j) < 1.6
                                        + (0.5 if element == "Se" else 0.0)),
                   "gpaw": None if g is None or references[element]["gpaw"] is None
                   else g - references[element]["gpaw"]}
            for model_name in TB_MODELS[element]:
                for method in METHODS:
                    key = f"{model_name}/{method}"
                    base = references[element].get((model_name, method))
                    if base is None:
                        continue
                    try:
                        values = _tb_values(atoms, models[model_name], site, method)
                        row[key] = {k: values[k] - base[k] for k in values}
                    except (RuntimeError, ValueError) as error:
                        row[key] = {"error": str(error)[:160]}
            rows.append(row)
    calibration = _calibrate(rows)
    out = {"description": "Desplazamientos de energía de ligadura de núcleo (eV) frente a CH4 "
                          "(C 1s), NH3 (N 1s), H2O (O 1s) y H2Se (Se 3d): tbkit (hueco de "
                          "núcleo en el SCC, 'hole'; núcleo equivalente, 'z+1') frente a GPAW "
                          "ΔSCF (PBE, hueco de núcleo completo, ion aislado), misma geometría.",
           "rows": rows, "calibration": calibration}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def _pairs(rows, element, key):
    out = [(r["gpaw"], r[key]) for r in rows if r["element"] == element
           and r["gpaw"] is not None and "total" in r.get(key, {})
           and (r["system"], r["site"]) != REFERENCE[element]]
    return out


def _calibrate(rows: list[dict]) -> dict:
    """Per element, set and method: the raw comparison and a linear calibration fitted on
    every other site and tested on the rest (hole: ``a·initial + b·relaxation``; z+1: ``λ``).
    Shipped (``parameters/xps_scale.json``) only where it holds."""
    import gpaw

    out, table = {}, {}
    for element, model_names in TB_MODELS.items():
        for model_name in model_names:
            for method in METHODS:
                key = f"{model_name}/{method}"
                pairs = _pairs(rows, element, key)
                if len(pairs) < 4:
                    continue
                g = np.array([p[0] for p in pairs])
                total = np.array([p[1]["total"] for p in pairs])
                if method == "hole":
                    design = np.array([[p[1]["initial"], p[1]["relaxation"]] for p in pairs])
                else:
                    design = total[:, None]
                fit, test = slice(0, None, 2), slice(1, None, 2)
                coef, *_ = np.linalg.lstsq(design[fit], g[fit], rcond=None)
                predicted = design @ coef
                entry = {"sites": len(g), "sites_fit": int(len(g[fit])),
                         "sites_test": int(len(g[test])),
                         "pearson_r_raw": float(np.corrcoef(g, total)[0, 1]),
                         "mae_raw_eV": float(np.mean(np.abs(total - g))),
                         "coefficients": [float(c) for c in coef],
                         "pearson_r_calibrated": float(np.corrcoef(g, predicted)[0, 1]),
                         "mae_test_raw_eV": float(np.mean(np.abs(total[test] - g[test]))),
                         "mae_test_calibrated_eV": float(np.mean(np.abs(predicted[test]
                                                                        - g[test])))}
                entry["shipped"] = bool(entry["pearson_r_calibrated"] >= MIN_CORRELATION
                                        and entry["mae_test_calibrated_eV"]
                                        < entry["mae_test_raw_eV"] and np.all(coef > 0))
                out[f"{element}/{key}"] = entry
                if entry["shipped"]:
                    names = ["initial", "relaxation"] if method == "hole" else ["scale"]
                    table.setdefault(method, {}).setdefault(model_name, {})[element] = {
                        n: {"value": round(float(c), 4), "unit": "1",
                            "source": f"mínimos cuadrados frente a GPAW ΔSCF en {entry['sites_fit']} "
                                      f"sitios (uno sí, uno no); prueba en los otros "
                                      f"{entry['sites_test']}: MAE {entry['mae_test_raw_eV']:.2f} → "
                                      f"{entry['mae_test_calibrated_eV']:.2f} eV (GPAW "
                                      f"{gpaw.__version__}, tbkit.recipes.xps_validation)"}
                        for n, c in zip(names, coef, strict=True)}
    SCALE_OUT.write_text(json.dumps({
        "name": "calibración de los desplazamientos XPS de tbkit frente a GPAW ΔSCF",
        "reference": "este trabajo: tbkit.recipes.xps_validation",
        "system": "tbkit.xps: método 'hole' (pesos del estado inicial y de la relajación) y "
                  "'z+1' (un factor), por conjunto y elemento",
        "validity": "ajustada en la mitad de los sitios (moléculas C/H/N/O/Se y coronenos con N "
                    "u O) y probada en la otra; solo se publica si la predicción calibrada "
                    f"correlaciona con GPAW (r ≥ {MIN_CORRELATION}) y mejora la mitad apartada",
        "units": {"coefficients": "1"}, "scales": table}, indent=1, ensure_ascii=False))
    return out


PARAMS = [
    Param("GRID_H", "espaciado de la malla real de GPAW (modo FD)", "Å", "", "convergencia"),
    Param("VACUUM", "vacío alrededor de cada molécula", "Å", "", "convergencia"),
    Param("KT", "ensanchamiento de Fermi-Dirac", "eV", "", "cálculo"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("setups", "gpaw", "report", "list"))
    parser.add_argument("system", nargs="?", help="un sistema del paso gpaw (vacío: todos)")
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=WORK)
    if args.step == "setups":
        print(make_setups())
    elif args.step == "gpaw":
        make_setups()
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        for name, indices in systems().items():
            if args.system and name != args.system:
                continue
            print(run_gpaw(name, None), flush=True)
            for site in indices:
                print(run_gpaw(name, site), flush=True)
    elif args.step == "list":
        for name, indices in systems().items():
            print(name, indices)
    else:
        print(json.dumps(report()["calibration"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
