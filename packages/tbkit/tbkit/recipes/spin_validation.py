"""Spin-polarised SCC against spin-polarised GPAW PBE: magnetic carbon, same geometries.

Run from packages/tbkit::

    python -m tbkit.recipes.spin_validation gpaw [SYSTEM]   # resumable, one file per state
    python -m tbkit.recipes.spin_validation report          # validation/spin_tb_vs_gpaw.json

Systems (fixed, unrelaxed geometries, the same for both codes; a_CC 1.42 Å, C-H 1.09 Å):

* ``zgnr4``, ``zgnr6``, ``zgnr8``: H-saturated zigzag ribbons. States AFM (edges
  antiparallel), FM and NM: ΔE per edge carbon, edge moment, AFM gap;
* ``triangulene`` C22H12 (Lieb S = 1): triplet (M = 2) against the broken-symmetry singlet
  (M = 0) and NM;
* ``phenalenyl`` C13H9 (S = 1/2): its moment and NM;
* ``h_graphene``: an H atom on top of a carbon of a 4×4 graphene cell (that carbon 0.35 Å
  out of plane): M = 1;
* ``O2`` and ``CH2``: triplet against singlet at fixed M.

TB: ``xu_chn`` (``xu_chno`` for O2) with ``tbkit.spin`` (charge SCC on, spin constants from the table), and the
π model with mean-field Hubbard U = |t| = 2.7 eV where it applies (carbon π systems). GPAW:
plane waves 400 eV, PBE, Fermi-Dirac 0.01 eV (fixed magnetic moment where the state says
so), initial moments from the state's pattern. Moments: Mulliken in TB, GPAW's integrated
atomic moments: totals are comparable exactly, atomic values only roughly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

from ..settings import Param, add_arguments, apply

WORK = Path("out/spin")
OUT = Path("validation/spin_tb_vs_gpaw.json")
PW_CUTOFF = 400.0
KT_GPAW = 0.01
KT_TB = 0.01
K_RIBBON = 16
K_RIBBON_TB = 48
K_SHEET = 6
VACUUM = 6.0
A_CC, C_H = 1.42, 1.09
U_PI = 2.7
#: state -> (fixed magnetisation or None, initial pattern)
STATES = {
    "zgnr4": {"AFM": (None, "antiferro"), "FM": (None, "ferro"), "NM": (0.0, "none")},
    "zgnr6": {"AFM": (None, "antiferro"), "FM": (None, "ferro"), "NM": (0.0, "none")},
    "zgnr8": {"AFM": (None, "antiferro"), "FM": (None, "ferro"), "NM": (0.0, "none")},
    "triangulene": {"T": (2.0, "ferro"), "BS": (0.0, "antiferro"), "NM": (0.0, "none")},
    "phenalenyl": {"D": (1.0, "ferro"), "NM": (None, "none")},
    "h_graphene": {"M": (None, "ferro"), "NM": (None, "none")},
    "O2": {"T": (2.0, "ferro"), "S": (0.0, "none")},
    "CH2": {"T": (2.0, "ferro"), "S": (0.0, "none")},
}


# ---------------------------------------------------------------- geometries
def _saturate(carbons: np.ndarray) -> Atoms:
    """H on every two-fold carbon, along the outward bisector."""
    d = np.linalg.norm(carbons[:, None] - carbons[None], axis=-1)
    hydrogens = []
    for i, p in enumerate(carbons):
        near = np.flatnonzero((d[i] > 0.1) & (d[i] < 1.6))
        if len(near) == 2:
            out = 2 * p - carbons[near].sum(axis=0)
            hydrogens.append(p + C_H * out / np.linalg.norm(out))
    symbols = "C" * len(carbons) + "H" * len(hydrogens)
    atoms = Atoms(symbols, positions=np.vstack([carbons, *([hydrogens] if hydrogens else [])]))
    atoms.center(vacuum=VACUUM)
    return atoms


def _flake(centres) -> np.ndarray:
    hexagon = [A_CC * np.array([np.cos(np.pi / 6 + k * np.pi / 3),
                                np.sin(np.pi / 6 + k * np.pi / 3), 0.0]) for k in range(6)]
    points = []
    for c in centres:
        for h in hexagon:
            p = np.asarray(c) + h
            if all(np.linalg.norm(p - q) > 0.1 for q in points):
                points.append(p)
    return np.array(points)


def geometry(name: str) -> Atoms:
    from ase.build import graphene_nanoribbon, molecule

    dx, dy = np.sqrt(3) * A_CC, 1.5 * A_CC
    if name.startswith("zgnr"):
        n = int(name[4:])
        atoms = graphene_nanoribbon(n, 1, type="zigzag", saturated=True, C_H=C_H, C_C=A_CC,
                                    vacuum=VACUUM)
        return atoms
    if name == "triangulene":
        return _saturate(_flake([(c * dx + r * dx / 2, r * dy, 0)
                                 for r in range(3) for c in range(3 - r)]))
    if name == "phenalenyl":
        return _saturate(_flake([(0, 0, 0), (dx, 0, 0), (dx / 2, dy, 0)]))
    if name == "h_graphene":
        from ase.build import graphene

        sheet = graphene(a=np.sqrt(3) * A_CC, size=(4, 4, 1), vacuum=VACUUM)
        top = sheet.positions[0] + [0.0, 0.0, 0.35]
        sheet.positions[0] = top
        sheet += Atoms("H", positions=[top + [0.0, 0.0, 1.13]])
        sheet.pbc = (True, True, False)
        return sheet
    atoms = molecule({"O2": "O2", "CH2": "CH2_s3B1d"}[name])
    atoms.center(vacuum=VACUUM)
    return atoms


def kmesh(name: str, tb: bool = False):
    if name.startswith("zgnr"):
        n = K_RIBBON_TB if tb else K_RIBBON
        atoms = geometry(name)
        return tuple(n if p else 1 for p in atoms.pbc)
    if name == "h_graphene":
        return (K_SHEET, K_SHEET, 1)
    return None


def pattern(atoms: Atoms, kind: str) -> np.ndarray:
    """Initial moment per atom: ``ferro`` (carbons +1), ``antiferro`` (bond-graph
    sublattices of the carbons), ``none``."""
    from ..neighbors import neighbor_list

    carbon = np.array([s != "H" for s in atoms.get_chemical_symbols()])
    if kind == "none":
        return np.zeros(len(atoms))
    if kind == "ferro":
        return carbon * 1.0
    ii, jj = neighbor_list("ij", atoms, 1.6)
    colour = np.zeros(len(atoms))
    for start in np.flatnonzero(carbon):
        if colour[start]:
            continue
        colour[start], stack = 1, [start]
        while stack:
            a = stack.pop()
            for b in jj[ii == a]:
                if carbon[b] and not colour[b]:
                    colour[b] = -colour[a]
                    stack.append(b)
    return colour


def _slug(name: str, state: str) -> Path:
    return WORK / name / f"gpaw_{state}.json"


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


# ---------------------------------------------------------------- GPAW
def run_gpaw(name: str, state: str) -> Path:
    from gpaw import GPAW, PW

    out = _slug(name, state)
    if out.exists():
        return out
    atoms = geometry(name)
    fixed, kind = STATES[name][state]
    spinpol = kind != "none" or bool(fixed)
    start = pattern(atoms, kind)
    if kind == "ferro" and fixed:
        start = start / start.sum() * fixed
    elif kind == "antiferro":
        start = 0.5 * start
        if fixed is not None:          # GPAW fixes M at the sum of the initial moments
            carbon = start != 0
            start[carbon] -= (start.sum() - fixed) / carbon.sum()
    occupations = {"name": "fermi-dirac", "width": KT_GPAW}
    if spinpol and fixed is not None:
        occupations["fixmagmom"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    kpts = kmesh(name)
    atoms.set_initial_magnetic_moments(start if spinpol else None)
    atoms.calc = GPAW(mode=PW(PW_CUTOFF), xc="PBE", kpts=kpts or (1, 1, 1),
                      spinpol=spinpol, occupations=occupations,
                      txt=str(out.with_suffix(".txt")))
    energy = float(atoms.get_potential_energy())
    calc = atoms.calc
    moments = (calc.get_magnetic_moments().tolist() if spinpol else [0.0] * len(atoms))
    total = float(calc.get_magnetic_moment()) if spinpol else 0.0
    nspin = 2 if spinpol else 1
    eps = [[calc.get_eigenvalues(kpt=k, spin=s).tolist()
            for k in range(len(calc.get_ibz_k_points()))] for s in range(nspin)]
    try:
        fermi = [float(calc.get_fermi_level())] * nspin
    except AssertionError:                    # fixed moment: one Fermi level per spin
        fermi = [float(f) for f in calc.get_fermi_levels()]
    _write(out, {"system": name, "state": state, "energy_eV": energy, "magnetization": total,
                 "moments": moments, "fermi_eV": fermi, "eigenvalues": eps,
                 "settings": f"GPAW PBE PW {PW_CUTOFF} eV, FD {KT_GPAW} eV, k {kpts}, "
                             f"M {'fija ' + str(fixed) if fixed is not None and spinpol else 'libre'}"})
    return out


def gpaw_gap(data: dict) -> float:
    """Lowest empty minus highest occupied level over both spins (each with its own Fermi
    level when the moment was fixed); 0 for a metal."""
    below, above = [], []
    for eps, ef in zip(data["eigenvalues"], data["fermi_eV"], strict=True):
        eps = np.asarray(eps)
        below.append(eps[eps <= ef].max())
        above.append(eps[eps > ef].min())
    return float(max(0.0, min(above) - max(below)))


# ---------------------------------------------------------------- TB
def model_for(name: str) -> str:
    return "xu_chno" if name == "O2" else "xu_chn"


def run_tb(name: str, state: str, model_name: str | None = None) -> dict:
    from ..hamiltonian import System
    from ..params import load_parameters
    from ..spin import spin_polarised

    atoms = geometry(name)
    fixed, kind = STATES[name][state]
    system = System.build(atoms, load_parameters(model_name or model_for(name)))
    kpts = weights = None
    if kmesh(name, tb=True):
        from ..kpoints import mesh

        kpts, weights = mesh(system, kmesh(name, tb=True))
    per_atom = pattern(atoms, kind)[system.basis.atoms] * 0.5
    result = spin_polarised(system, magnetization=fixed, guess=per_atom, kT=KT_TB,
                            kpts=kpts, weights=weights, tol=1e-8, max_iter=3000)
    return {"energy_eV": result.energy, "magnetization": result.magnetization,
            "moments": [result.moments.get(a, 0.0) for a in range(len(atoms))],
            "gap_eV": result.solution.gap(), "converged": result.converged,
            "iterations": result.iterations}


def run_pi(name: str, state: str) -> dict | None:
    """π model, mean-field Hubbard U = 2.7 eV: carbon π systems only (H ignored)."""
    from ..hamiltonian import System
    from ..hubbard import mean_field
    from ..params import pi_model

    if name in ("O2", "CH2", "h_graphene"):
        return None
    atoms = geometry(name)
    fixed, kind = STATES[name][state]
    system = System.build(atoms, pi_model())
    kpts = weights = None
    if kmesh(name, tb=True):
        from ..kpoints import mesh

        kpts, weights = mesh(system, kmesh(name, tb=True))
    guess = {"antiferro": "antiferro", "ferro": "ferro", "none": "paramagnetic"}[kind]
    if fixed is not None and kind != "none":
        from ..spin import spin_polarised

        result = spin_polarised(system, W=-U_PI / 2, charges=False, magnetization=fixed,
                                guess=guess if guess != "paramagnetic" else "none",
                                kT=KT_TB, kpts=kpts, weights=weights, tol=1e-9)
        energy, total, gap = result.energy, result.magnetization, result.solution.gap()
        moments = result.moments
    else:
        result = mean_field(system, U=U_PI, kpts=kpts, weights=weights, kT=KT_TB, guess=guess,
                            tol=1e-9)
        energy, total, gap, moments = (result.energy, result.magnetization,
                                       result.solution.gap(), result.moments)
    return {"energy_eV": float(energy), "magnetization": float(total), "gap_eV": float(gap),
            "moments": [float(moments.get(a, 0.0)) for a in range(len(atoms))]}


# ---------------------------------------------------------------- report
def _edge_atoms(atoms: Atoms) -> list[int]:
    carbons = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "C"]
    axis = int(np.argmax(np.ptp(atoms.positions[carbons], axis=0) * (~np.asarray(atoms.pbc))))
    x = atoms.positions[carbons, axis]
    return [carbons[int(np.argmin(x))], carbons[int(np.argmax(x))]]


def report() -> dict:
    out = {"description": "SCC con espín de tbkit (xu_chn; π con Hubbard U = 2,7 eV) frente a "
                          "GPAW PBE con espín, misma geometría. Energías relativas al primer "
                          "estado de cada sistema; momentos: Mulliken (TB) e integrados por "
                          "átomo (GPAW).", "systems": {}}
    for name, states in STATES.items():
        atoms = geometry(name)
        entry = {"states": {}}
        for state in states:
            path = _slug(name, state)
            if not path.exists():
                continue
            g = json.loads(path.read_text())
            entry["states"][state] = {"gpaw": {"energy_eV": g["energy_eV"],
                                               "magnetization": g["magnetization"],
                                               "gap_eV": gpaw_gap(g), "moments": g["moments"]},
                                      "tb": run_tb(name, state)}
            pi = run_pi(name, state)
            if pi is not None:
                entry["states"][state]["pi_hubbard"] = pi
        first = next(iter(states))
        if first not in entry["states"]:
            continue
        for state, row in entry["states"].items():
            for code, data in row.items():
                data["relative_eV"] = data["energy_eV"] - entry["states"][first][code]["energy_eV"]
        if name.startswith("zgnr"):
            edges = _edge_atoms(atoms)
            entry["edge_atoms"] = edges
            for row in entry["states"].values():
                for data in row.values():
                    data["edge_moments"] = [data["moments"][i] for i in edges]
        out["systems"][name] = entry
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


PARAMS = [
    Param("PW_CUTOFF", "energía de corte de las ondas planas de GPAW", "eV", "", "convergencia"),
    Param("K_RIBBON", "puntos k de GPAW a lo largo de las cintas", "", "", "convergencia"),
    Param("K_RIBBON_TB", "puntos k del TB a lo largo de las cintas", "", "", "convergencia"),
    Param("KT_TB", "ensanchamiento de Fermi-Dirac del TB", "eV", "", "cálculo"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("gpaw", "report", "list"))
    parser.add_argument("system", nargs="?", help="un sistema del paso gpaw (vacío: todos)")
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=WORK)
    if args.step == "gpaw":
        for name in ([args.system] if args.system else STATES):
            for state in STATES[name]:
                print(run_gpaw(name, state), flush=True)
    elif args.step == "list":
        for name, states in STATES.items():
            print(name, {s: _slug(name, s).exists() for s in states})
    else:
        print(json.dumps({n: {s: {c: {k: v for k, v in d.items() if k in
                                      ("relative_eV", "magnetization", "gap_eV")}
                                  for c, d in r.items()} for s, r in e["states"].items()}
                          for n, e in report()["systems"].items()}, indent=1))


if __name__ == "__main__":
    main()
