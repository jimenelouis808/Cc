"""GPAW references between graphene layers: bands and binding, for the interlayer π hopping
and the dispersion term (step 2 of the tight-binding plan).

Every TB set of the package cuts its laws at 2.6-3.3 Å and has no dispersion, so two
layers 3.35 Å apart neither couple electronically nor bind. This recipe computes what a
fix must reproduce, on graphene at the lattice constant of ``band_validation``:

* ``AB@d``, ``AA@d``: bilayers at interlayer distance d (Bernal and eclipsed);
* ``ABC@d``: rhombohedral trilayer;
* ``graphite@d``: AB graphite, c = 2d;
* ``slide@s``: the top layer of an AA bilayer at 3.35 Å moved by s·a_cc along a bond
  (s = 1 is AB, 1.5 the saddle point SP);
* ``mono``: one layer in the same cell as the slabs (the zero of every binding energy).

Each point stores the PBE energy, the D3(BJ) dispersion of that geometry computed apart
(simple-dftd3, PBE parameters: PBE+D3 is the energy reference of the fit, PBE alone
binds nothing) and, for the systems of ``BANDS``, PBE bands along Γ-M-K-Γ in the
format of ``band_validation`` so its metrics apply unchanged. Slabs share one cell
height, so the plane-wave basis is identical for a slab and its monolayer.

Steps::

    python -m tbkit.recipes.interlayer gpaw [NAME]   # one point, or every missing one
    python -m tbkit.recipes.interlayer list          # the points and which are done

Resumable: one JSON per point in ``out/interlayer``; an existing file is not recomputed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

from ..settings import Param, add_arguments, apply

WORK = Path("out/interlayer")
GRAPHENE_A = 2.467          # Å, as band_validation (PBE)
CELL_Z = 25.0               # Å, height of every slab cell
PW_CUTOFF = 500.0           # eV
KMESH = 18                  # in-plane SCF mesh n×n
KMESH_Z_GRAPHITE = 6
KT = 0.05                   # eV
NK_PATH = 90
CURVE_D = (2.8, 3.0, 3.2, 3.35, 3.5, 3.7, 4.0, 4.5, 5.5)
BANDS = ("AB@3.35", "AB@3.0", "AB@3.7", "AA@3.35", "graphite@3.35", "ABC@3.35")
POINTS = (("mono",) + BANDS
          + tuple(f"{s}@{d}" for s in ("AB", "AA") for d in CURVE_D)
          + ("graphite@3.2", "graphite@3.5", "slide@0.5", "slide@1.5"))


def _unique(names):
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


POINTS = _unique(POINTS)


def _layer(a: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell vectors (2D) and the two atoms of one layer, a bond along +x."""
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([a / 2, a * np.sqrt(3) / 2, 0.0])
    a_cc = a / np.sqrt(3)
    pos = np.array([[0.0, 0.0, 0.0], [a / 2, a_cc / 2, 0.0]])
    return np.array([a1, a2]), pos


def stack(shifts: list[float], d: float, a: float = GRAPHENE_A, height: float | None = None
          ) -> Atoms:
    """Layers d apart, layer i moved by shifts[i]·a_cc along the bond direction."""
    cell2, pos = _layer(a)
    a_cc = a / np.sqrt(3)
    bond = (pos[1] - pos[0]) / np.linalg.norm(pos[1] - pos[0])
    n = len(shifts)
    z0 = (CELL_Z - (n - 1) * d) / 2 if height is None else 0.0
    layers = [pos + s * a_cc * bond + [0.0, 0.0, z0 + i * d] for i, s in enumerate(shifts)]
    cell = np.vstack([cell2, [0.0, 0.0, CELL_Z if height is None else height]])
    atoms = Atoms("C" * (2 * n), positions=np.vstack(layers), cell=cell,
                  pbc=(True, True, height is not None))
    atoms.wrap()
    return atoms


def geometry(name: str) -> Atoms:
    if name == "mono":
        return stack([0.0], 0.0)
    kind, value = name.split("@")
    d = float(value)
    if kind == "AB":
        return stack([0.0, 1.0], d)
    if kind == "AA":
        return stack([0.0, 0.0], d)
    if kind == "ABC":
        return stack([0.0, 1.0, 2.0], d)
    if kind == "graphite":
        return stack([0.0, 1.0], d, height=2 * d)
    if kind == "slide":
        return stack([0.0, float(value)], 3.35)
    raise ValueError(f"Sistema desconocido: {name!r}.")


def kmesh(name: str) -> tuple[int, int, int]:
    return (KMESH, KMESH, KMESH_Z_GRAPHITE if name.startswith("graphite") else 1)


def kpath(atoms: Atoms):
    """Γ-M-K-Γ in the plane (kz = 0 for graphite)."""
    bp = atoms.cell.bandpath("GMKG", npoints=NK_PATH, pbc=(True, True, False))
    x, sx, labels = bp.get_linear_kpoint_axis()
    return bp.kpts, x, list(zip(sx.tolist(), labels))


def _slug(name: str) -> str:
    return name.replace("@", "_d").replace(".", "p")


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def dispersion(atoms: Atoms) -> float:
    """D3(BJ) energy with PBE parameters (eV), simple-dftd3."""
    from dftd3.interface import DispersionModel, RationalDampingParam

    bohr = 0.52917721067
    hartree = 27.211386245988
    model = DispersionModel(atoms.numbers, atoms.positions / bohr,
                            np.asarray(atoms.cell) / bohr, np.asarray(atoms.pbc))
    return float(model.get_dispersion(RationalDampingParam(method="pbe"), grad=False)["energy"]
                 * hartree)


def run_gpaw(name: str) -> Path:
    from gpaw import GPAW, PW, FermiDirac

    out = WORK / f"{_slug(name)}.json"
    if out.exists():
        return out
    atoms = geometry(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    atoms.calc = GPAW(mode=PW(PW_CUTOFF), xc="PBE", kpts=kmesh(name),
                      occupations=FermiDirac(KT), txt=str(out.with_suffix(".txt")))
    energy = float(atoms.get_potential_energy())
    data = {"system": name, "n_atoms": len(atoms), "energy_pbe_eV": energy,
            "energy_d3_eV": dispersion(atoms), "cell": np.asarray(atoms.cell).tolist(),
            "positions": atoms.positions.tolist(), "pbc": [bool(p) for p in atoms.pbc],
            "fermi_eV": float(atoms.calc.get_fermi_level()),
            "settings": f"GPAW PBE PW {PW_CUTOFF} eV, FD {KT} eV, k {kmesh(name)}; "
                        f"D3(BJ) PBE (simple-dftd3) apart"}
    if name in BANDS:
        k, _, _ = kpath(atoms)
        n_occ = len(atoms) * 2
        band = atoms.calc.fixed_density(kpts=k, symmetry="off", txt=None,
                                        nbands=n_occ + 2 * len(atoms),
                                        convergence={"bands": n_occ + len(atoms)})
        eps = np.array([band.get_eigenvalues(kpt=i) for i in range(len(k))])
        data.update({"code": "gpaw_pw", "n_occ": n_occ, "kpts": k.tolist(),
                     "bands": eps[:, : n_occ + len(atoms)].tolist()})
    _write(out, data)
    return out


def load(name: str) -> dict:
    return json.loads((WORK / f"{_slug(name)}.json").read_text())


def binding(name: str, data: dict | None = None, mono: dict | None = None) -> dict:
    """Binding per atom relative to separate monolayers (meV/atom): PBE, D3 and their sum."""
    data = data or load(name)
    mono = mono or load("mono")
    n = data["n_atoms"]
    layers = n // 2
    out = {}
    for key in ("energy_pbe_eV", "energy_d3_eV"):
        out[key.split("_")[1]] = 1000 * (data[key] - layers * mono[key]) / n
    out["pbe_d3"] = out["pbe"] + out["d3"]
    return out


PARAMS = [
    Param("PW_CUTOFF", "energía de corte de las ondas planas de GPAW", "eV", "", "convergencia"),
    Param("KMESH", "malla k del SCF en el plano (n×n)", "", "", "convergencia"),
    Param("KMESH_Z_GRAPHITE", "puntos k del grafito a lo largo de c", "", "", "convergencia"),
    Param("CELL_Z", "altura de la celda de las láminas (la misma para todas)", "Å", "",
          "convergencia"),
    Param("KT", "ensanchamiento de Fermi-Dirac", "eV", "", "cálculo"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tbkit.recipes.interlayer",
        description="Referencias GPAW entre capas de grafeno: bandas y energía de enlace.")
    parser.add_argument("step", choices=("gpaw", "list"))
    parser.add_argument("system", nargs="?", help="un punto (vacío: todos los que faltan)")
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=WORK)
    if args.step == "gpaw":
        for name in ([args.system] if args.system else POINTS):
            print(run_gpaw(name), flush=True)
    else:
        for name in POINTS:
            done = (WORK / f"{_slug(name)}.json").exists()
            print(f"{name:16s} {'hecho' if done else '—'}")


if __name__ == "__main__":
    main()
