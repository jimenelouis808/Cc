"""GPAW Born charges to fit and test IR charge corrections (tbkit.charge_model).

Run from packages/tbkit (needs GPAW; resumable: one file per structure, and per
displacement while it runs)::

    python -m tbkit.recipes.born_references fragments     # coil cut-outs (no GPAW)
    python -m tbkit.recipes.born_references list          # what will be computed
    python -m tbkit.recipes.born_references run NAME      # one structure

Structures:

* every C/H/N and C/H/N/O molecule with a GPAW geometry in
  ``parameters/references/gpaw_chn.json`` and ``gpaw_chno.json`` (44);
* three cut-outs of the 204-atom coil, at the xu_chn geometries of
  ``recipes/doped_raman``: around the pentagon atom closest to a heptagon in the pristine coil,
  around graphitic N, and around the -NH2 group, every atom within ``RADIUS`` Å and
  the cut bonds closed with H along the bond (C-H 1.09, N-H 1.01 Å). They carry the
  5-7 environments the molecules lack.

GPAW exactly as for the fit references (``references.GPAW_DEFAULTS``: LCAO dzp, PBE,
h 0.2 Å, 5 Å vacuum). Born charges Z*[a, i, j] = ∂μ_j/∂x_{a,i} by central differences
of GPAW's dipole (±``DELTA`` Å), one calculator reused through the displacements of a
structure (it restarts from the last density). Born charges are compared at the
same geometry in both methods, so the cut-outs need no relaxation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from ..settings import Param, add_arguments, apply

WORK = Path("out/born_references")
DELTA = 0.01
RADIUS = 5.0
GPAW_SETTINGS: dict = {}           # on top of references.GPAW_DEFAULTS
BOND = {"C": 1.09, "N": 1.01, "O": 0.97}


def molecules() -> dict[str, Atoms]:
    from ..params import PARAMETER_DIR
    from ..references import load_references

    out = {}
    for file in ("gpaw_chn.json", "gpaw_chno.json"):
        refs, _ = load_references(PARAMETER_DIR / "references" / file)
        for r in refs:
            if r.label.endswith("/eq"):
                out.setdefault(r.label.split("/")[0], r.atoms.copy())
    return out


def cut_out(periodic: Atoms, centre: int, radius: float | None = None) -> Atoms:
    """Atoms within ``radius`` of ``centre`` (minimum image), cut bonds closed with H.
    An atom bonded to two or more kept atoms is kept too, so no two caps crowd."""
    from ase.neighborlist import natural_cutoffs, neighbor_list

    radius = RADIUS if radius is None else radius
    i, j, D = neighbor_list("ijD", periodic, natural_cutoffs(periodic, mult=1.2))
    bonded: dict[int, list[tuple[int, np.ndarray]]] = {}
    for a, b, v in zip(i, j, D, strict=True):
        bonded.setdefault(int(a), []).append((int(b), v))
    vec = periodic.get_distances(centre, range(len(periodic)), mic=True, vector=True)
    keep = {k for k in range(len(periodic)) if np.linalg.norm(vec[k]) <= radius}
    while True:                                    # close the gaps a cap would crowd
        extra = {b for a in keep for b, _ in bonded.get(a, []) if b not in keep and
                 sum(n in keep for n, _ in bonded.get(b, [])) >= 2}
        if not extra:
            break
        keep |= extra
    keep = sorted(keep)
    origin = periodic.positions[centre]
    symbols = [periodic[k].symbol for k in keep]
    positions = [origin + vec[k] for k in keep]
    for a in keep:
        for b, v in bonded.get(a, []):
            if b not in keep:
                unit = v / np.linalg.norm(v)
                symbols.append("H")
                positions.append(origin + vec[a] + BOND.get(periodic[a].symbol, 1.09) * unit)
    out = Atoms(symbols, positions=positions)
    out.center(vacuum=5.0)
    out.info.update({"source": "coil cut-out", "centre": int(centre), "radius_A": radius})
    return out


def fragments() -> dict[str, Atoms]:
    """The three coil cut-outs (written to ``WORK/fragments``)."""
    from .. import sites
    from . import doped_raman as tb

    folder = WORK / "fragments"
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    pristine = read(tb.WORK / "pristine" / "relaxed.extxyz")
    on = sites.ring_atoms(pristine, (5, 7))
    # In this coil no atom is on both a pentagon and a heptagon: the centre is the
    # pentagon atom closest to a heptagon, so the cut-out holds both rings.
    d = pristine.get_all_distances(mic=True)[np.ix_(on[5], on[7])]
    junction = int(on[5][int(np.argmin(d.min(axis=1)))])
    amine = read(tb.WORK / "amine" / "relaxed.extxyz")
    picks = {"coil_57": (pristine, junction),
             "coil_N": (read(tb.WORK / "N" / "relaxed.extxyz"), 0),
             "coil_NH2": (amine, amine.get_chemical_symbols().index("N"))}
    for name, (atoms, centre) in picks.items():
        path = folder / f"{name}.extxyz"
        if not path.exists():
            write(path, cut_out(atoms, centre))
        out[name] = read(path)
    return out


def structures() -> dict[str, Atoms]:
    out = molecules()
    if (WORK / "fragments").exists():
        out.update({p.stem: read(p) for p in sorted((WORK / "fragments").glob("*.extxyz"))})
    return out


def run(name: str) -> Path:
    from ..progress import Progress
    from ..references import GPAW_DEFAULTS, _n_bands, gpaw_calculator

    settings = {**GPAW_DEFAULTS, **GPAW_SETTINGS}
    path = WORK / f"{name}.json"
    if path.exists():
        return path
    atoms = structures()[name].copy()
    atoms.pbc = False
    if name not in ("coil_57", "coil_N", "coil_NH2"):
        atoms.center(vacuum=settings["vacuum"])
    parts = WORK / f"{name}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    calc = gpaw_calculator(settings, _n_bands(atoms, settings))
    base = atoms.get_positions()
    bar = Progress(6 * len(atoms), f"cargas de Born GPAW {name}",
                   status=WORK / f"progreso_{name}.json",
                   done=len(list(parts.glob("*.npy"))))
    probe = atoms.copy()
    probe.calc = calc
    for a in range(len(atoms)):
        for i in range(3):
            for sign, tag in ((1, "p"), (-1, "m")):
                file = parts / f"{a:03d}_{i}_{tag}.npy"
                if file.exists():
                    continue
                pos = base.copy()
                pos[a, i] += sign * DELTA
                probe.set_positions(pos)
                np.save(file, np.array(probe.get_dipole_moment()))
                bar.step()
    z = np.zeros((len(atoms), 3, 3))
    for a in range(len(atoms)):
        for i in range(3):
            z[a, i] = (np.load(parts / f"{a:03d}_{i}_p.npy") -
                       np.load(parts / f"{a:03d}_{i}_m.npy")) / (2 * DELTA)
    probe.set_positions(base)
    dipole = np.array(probe.get_dipole_moment())
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": name, "symbols": atoms.get_chemical_symbols(),
                               "positions_A": atoms.get_positions().tolist(),
                               "cell_A": atoms.cell.array.tolist(), "dipole_eA": dipole.tolist(),
                               "born_e": z.tolist(), "delta_A": DELTA,
                               "sum_rule_residual_e": float(np.abs(z.sum(axis=0)).max())}))
    tmp.replace(path)
    return path


#: Adjustable with --ajuste NOMBRE=VALOR (tbkit recetas born_references).
PARAMS = [
    Param("GPAW_SETTINGS", "ajustes de GPAW: modo, base, funcional, malla h (Å)", "",
          "LCAO dzp PBE h 0.2: la referencia de todo tbkit; sz no sirve (es la base mínima "
          "de tbkit y reproduce sus errores)", "cálculo"),
    Param("DELTA", "desplazamiento de cada átomo para la derivada del dipolo", "Å", "",
          "convergencia"),
    Param("RADIUS", "radio de los recortes de la coil alrededor de su centro", "Å",
          "5: el recorte tiene ~45 átomos más los H de cierre", "cálculo"),
    Param("BOND", "longitud del enlace de los H de cierre de los recortes", "Å", "", "cálculo"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("fragments", "list", "run"))
    parser.add_argument("name", nargs="?",
                        help="molécula o estructura de referencia (vacío: todas)")
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.step == "fragments":
        for name, atoms in fragments().items():
            print(f"{name}: {len(atoms)} átomos {atoms.get_chemical_formula()}")
    elif args.step == "list":
        for name, atoms in sorted(structures().items(), key=lambda kv: len(kv[1])):
            done = "hecho" if (WORK / f"{name}.json").exists() else ""
            print(f"{name:16s} {len(atoms):4d} {done}")
    else:
        print(run(args.name))


if __name__ == "__main__":
    main()
