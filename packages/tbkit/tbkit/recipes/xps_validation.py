"""XPS core-level shifts: tbkit's equivalent core (Z+1) against GPAW ΔSCF with core holes.

Run from packages/tbkit::

    python -m tbkit.recipes.xps_validation setups     # C and N 1s core-hole PAW setups
    python -m tbkit.recipes.xps_validation gpaw [S]   # ΔSCF, one file per site (resumable)
    python -m tbkit.recipes.xps_validation report     # validation/xps_tb_vs_gpaw.json

GPAW reference: E[full 1s core hole on the site, charge +1] − E[ground state], PBE, plane
waves, the molecule in a box; the core-hole setup is generated here (``gpaw dataset
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
PW_CUTOFF = 400.0
VACUUM = 6.0
KT = 0.01
REFERENCE = {"C": ("CH4", 0), "N": ("NH3", 0)}
MOLECULES = ("CH4", "C2H6", "C2H4", "C2H2", "C6H6", "CH3OH", "H2CO", "HCOOH", "CO", "CO2",
             "HCN", "CH3CN", "NH3", "C5H5N", "C4H4NH")
FLAKES = ("coronene", "coronene_N_graphitic", "coronene_N_pyridinic")
TB_MODELS = {"C": ("xu_chno", "xu_chn"), "N": ("xu_chno",)}


def _coronene() -> Atoms:
    from .spin_validation import _flake, _saturate

    dx, dy = np.sqrt(3) * 1.42, 1.5 * 1.42
    centres = [(0, 0, 0), (dx, 0, 0), (-dx, 0, 0), (dx / 2, dy, 0), (-dx / 2, dy, 0),
               (dx / 2, -dy, 0), (-dx / 2, -dy, 0)]
    return _saturate(_flake(centres))


def geometry(name: str) -> Atoms:
    from ase.build import molecule

    if name in MOLECULES:
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
    if "N" in symbols and name.startswith("coronene"):
        # a doped flake: the N, the carbons within two bonds of it and the farthest one
        n = symbols.index("N")
        carbons = [i for i, s in enumerate(symbols) if s == "C"]
        keep = [n] + [i for i in carbons if d[n, i] < 2.6] + \
            [carbons[int(np.argmax(d[n, carbons]))]]
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
    return {name: sites(name) for name in MOLECULES + FLAKES}


# ---------------------------------------------------------------- GPAW
def make_setups() -> list[Path]:
    SETUPS.mkdir(parents=True, exist_ok=True)
    gpaw = Path(sys.executable).with_name("gpaw")
    made = []
    for element in REFERENCE:
        target = SETUPS / f"{element}.1s1ch.PBE"
        if not target.exists():
            subprocess.run([str(gpaw), "dataset", element, "-f", "PBE", "--core-hole=1s,1.0",
                            "-t", "1s1ch", "-w"], cwd=SETUPS, check=True,
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
    from gpaw import GPAW, PW

    out = WORK / name / ("ground.json" if site is None else f"hole_{site}.json")
    if out.exists():
        return out
    if str(SETUPS.resolve()) not in gpaw.setup_paths:
        gpaw.setup_paths.insert(0, str(SETUPS.resolve()))
    atoms = geometry(name)
    kwargs = {} if site is None else {"setups": {site: "1s1ch"}, "charge": 1.0}
    out.parent.mkdir(parents=True, exist_ok=True)
    atoms.calc = GPAW(mode=PW(PW_CUTOFF), xc="PBE", occupations={"name": "fermi-dirac",
                                                                   "width": KT},
                      txt=str(out.with_suffix(".txt")), **kwargs)
    _write(out, {"system": name, "site": site, "energy_eV": float(atoms.get_potential_energy()),
                 "settings": f"GPAW PBE PW {PW_CUTOFF} eV, FD {KT} eV, caja {VACUUM} Å, "
                             "hueco 1s completo y carga +1 (ΔSCF)"})
    return out


def gpaw_energy(name: str, site: int) -> float | None:
    try:
        ground = json.loads((WORK / name / "ground.json").read_text())["energy_eV"]
        hole = json.loads((WORK / name / f"hole_{site}.json").read_text())["energy_eV"]
    except FileNotFoundError:
        return None
    return hole - ground


# ---------------------------------------------------------------- report
def report() -> dict:
    from ..params import load_parameters
    from ..xps import ionisation_energy

    rows = []
    references = {}
    for element, (name, site) in REFERENCE.items():
        atoms = geometry(name)
        references[element] = {"gpaw": gpaw_energy(name, site)}
        for model_name in TB_MODELS[element]:
            references[element][model_name] = ionisation_energy(
                atoms, load_parameters(model_name), site, KT)
    for name, indices in systems().items():
        atoms = geometry(name)
        for site in indices:
            element = atoms[site].symbol
            g = gpaw_energy(name, site)
            row = {"system": name, "site": site, "element": element,
                   "neighbours": sorted(atoms[j].symbol for j in range(len(atoms))
                                        if j != site and atoms.get_distance(site, j) < 1.6),
                   "gpaw": None if g is None or references[element]["gpaw"] is None
                   else g - references[element]["gpaw"]}
            for model_name in TB_MODELS[element]:
                try:
                    e = ionisation_energy(atoms, load_parameters(model_name), site, KT)
                    row[model_name] = e - references[element][model_name]
                except (RuntimeError, ValueError) as error:
                    row[model_name] = None
                    row[f"{model_name}_error"] = str(error)[:160]
            rows.append(row)
    summary = {}
    for element, models in TB_MODELS.items():
        for model_name in models:
            pairs = [(r["gpaw"], r[model_name]) for r in rows if r["element"] == element
                     and r["gpaw"] is not None and r.get(model_name) is not None
                     and (r["system"], r["site"]) != REFERENCE[element]]
            if len(pairs) < 2:
                continue
            g, t = np.array(pairs).T
            slope = float(np.sum(g * t) / np.sum(t * t))       # through the origin
            summary[f"{element}1s/{model_name}"] = {
                "sites": len(pairs), "mae_eV": float(np.mean(np.abs(t - g))),
                "max_error_eV": float(np.max(np.abs(t - g))),
                "pearson_r": float(np.corrcoef(g, t)[0, 1]),
                "scale_tb_to_gpaw": slope,
                "mae_scaled_eV": float(np.mean(np.abs(slope * t - g)))}
    out = {"description": "Desplazamientos de energía de ligadura de núcleo (eV) frente a CH4 "
                          "(C 1s) y NH3 (N 1s): tbkit con núcleo equivalente (Z+1, SCC) frente "
                          "a GPAW ΔSCF (PBE, hueco 1s completo, carga +1), misma geometría.",
           "rows": rows, "summary": summary}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


PARAMS = [
    Param("PW_CUTOFF", "energía de corte de las ondas planas de GPAW", "eV", "", "convergencia"),
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
        print(json.dumps(report()["summary"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
