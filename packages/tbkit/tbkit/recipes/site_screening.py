"""Where does a substitutional dopant go? TB screening of inequivalent sites, top-N for DFT.

Run (resumable: every site is a file in WORKDIR)::

    python -m tbkit.recipes.site_screening STRUCTURE N WORKDIR --model chn --relax-top 5

1. The carbons are grouped into classes by their environment: the sizes of
   the rings they sit on and their sorted distances to every atom within
   ``--radius`` Å (rounded to ``--tolerance``). Atoms related by the
   structure's symmetry (a helix, a tube, a periodic sheet) fall in one class,
   so only one per class is computed.
2. One representative per class gets the dopant and a TB single point at the
   substituted geometry (``site_XXXX.json``): a cheap first ranking.
3. The ``--relax-top`` lowest are relaxed (``site_XXXX_relaxed.json``) and
   ranked again; relaxation reorders close sites (a dopant relieving curvature
   gains more where the net is strained). On the 204-atom coil one single point
   with xu_chn takes ~3 min and there are 51 classes: ~2.5 h, then ~1 h per
   relaxed site. Relaxing every class would take days.
4. The classes are ranked by energy relative to the lowest. Same stoichiometry
   in every site, so differences need no chemical potentials. The ``--top``
   lowest are written as ``top/rank_K_site_XXXX.extxyz`` for GPAW (or any DFT,
   exchanged as files), with the ring sizes of the site in the file.

The energies are the model's: check the ordering of the top sites with DFT
before trusting it (the TB sets were fitted on molecules; crystals and curved
nets are an extrapolation, see each set's ``validity``). Functional groups
need a placement (inside/outside a curved surface, orientation) and are built
by nanocarbon_lab and come in as files; this recipe covers substitution only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from .. import sites


def environment_classes(atoms: Atoms, element: str = "C", radius: float = 4.0,
                        tolerance: float = 0.05) -> dict[tuple, list[int]]:
    """Classes of equivalent ``element`` atoms, by ring membership and distance fingerprint."""
    from ase.neighborlist import neighbor_list

    rings_of: dict[int, list[int]] = {}
    for ring in sites.rings(atoms):
        for index in ring:
            rings_of.setdefault(index, []).append(len(ring))
    i, d = neighbor_list("id", atoms, radius)
    symbols = np.array(atoms.get_chemical_symbols())
    classes: dict[tuple, list[int]] = {}
    for index in np.flatnonzero(symbols == element):
        distances = np.sort(d[i == index])
        key = (tuple(sorted(rings_of.get(int(index), []))),
               tuple(np.round(distances / tolerance).astype(int)))
        classes.setdefault(key, []).append(int(index))
    return classes


def _energy(atoms: Atoms, model, relax: bool, kmesh, kT: float, fmax: float):
    from ..calculator import TBCalculator

    atoms = atoms.copy()
    atoms.calc = TBCalculator(model, kpts=kmesh, kT=kT)
    converged = None
    if relax:
        from ase.optimize import BFGS

        converged = bool(BFGS(atoms, logfile=None).run(fmax=fmax, steps=400))
    energy = float(atoms.get_potential_energy())
    atoms.calc = None
    return atoms, energy, converged


def screen(atoms: Atoms, dopant: str, model, workdir: Path, relax_top: int = 0, top: int = 3,
           kmesh=8, kT: float = 0.05, fmax: float = 0.05, radius: float = 4.0,
           tolerance: float = 0.05, max_sites: int | None = None) -> dict:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    classes = environment_classes(atoms, "C", radius, tolerance)
    ordered = sorted(classes.values(), key=lambda members: members[0])
    if max_sites is not None and len(ordered) > max_sites:
        pick = np.linspace(0, len(ordered) - 1, max_sites).round().astype(int)
        ordered = [ordered[k] for k in sorted(set(pick))]

    def computed(site, members, relax):
        name = f"site_{site:04d}" + ("_relaxed" if relax else "")
        path = workdir / f"{name}.json"
        if not path.exists():
            doped = atoms.copy()
            doped[site].symbol = dopant
            relaxed, energy, converged = _energy(doped, model, relax, kmesh, kT, fmax)
            write(workdir / f"{name}.extxyz", relaxed)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"site": site, "equivalent": members, "energy_eV": energy,
                                       "relaxed": relax, "converged": converged}, indent=1))
            tmp.replace(path)
        return json.loads(path.read_text())

    rows = sorted((computed(m[0], m, False) for m in ordered), key=lambda r: r["energy_eV"])
    if relax_top:
        relaxed = [computed(r["site"], r["equivalent"], True) for r in rows[:relax_top]]
        rows = sorted(relaxed, key=lambda r: r["energy_eV"]) + rows[relax_top:]
    rings_of: dict[int, list[int]] = {}
    for ring in sites.rings(atoms):
        for index in ring:
            rings_of.setdefault(index, []).append(len(ring))
    for group in (True, False):
        energies = [r["energy_eV"] for r in rows if r["relaxed"] == group]
        for r in rows:
            if r["relaxed"] == group:
                r["relative_eV"] = r["energy_eV"] - min(energies)
    for r in rows:
        r["rings"] = sorted(rings_of.get(r["site"], []))
        r["multiplicity"] = len(r["equivalent"])
    (workdir / "top").mkdir(exist_ok=True)
    for rank, r in enumerate(rows[:top], start=1):
        name = f"site_{r['site']:04d}" + ("_relaxed" if r["relaxed"] else "")
        structure = read(workdir / f"{name}.extxyz")
        structure.info.update({"dopant": dopant, "site": r["site"], "rings": str(r["rings"]),
                               "relative_eV": r["relative_eV"], "relaxed": r["relaxed"],
                               "model": model.name})
        write(workdir / "top" / f"rank_{rank}_{name}.extxyz", structure)
    report = {"dopant": dopant, "model": model.name, "classes": len(classes),
              "computed": len(ordered), "relaxed": min(relax_top, len(rows)),
              "note": "relative_eV is relative within relaxed and within unrelaxed sites",
              "sites": [{k: r[k] for k in ("site", "relative_eV", "relaxed", "rings",
                                           "multiplicity", "converged")} for r in rows]}
    (workdir / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> None:
    from ..gui.actions import load_model

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("structure", type=Path)
    parser.add_argument("dopant")
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--model", default="chn", help="nombre (chn, chnob, …) o archivo")
    parser.add_argument("--relax-top", type=int, default=0,
                        help="relaja los K mejores del primer cribado")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--kmesh", type=int, default=8)
    parser.add_argument("--max-sites", type=int, default=None)
    args = parser.parse_args(argv)
    report = screen(read(args.structure), args.dopant, load_model(args.model), args.workdir,
                    relax_top=args.relax_top, top=args.top, kmesh=args.kmesh, max_sites=args.max_sites)
    print(f"{report['classes']} clases de sitio, {report['computed']} calculadas")
    for r in report["sites"]:
        tag = "relajado" if r["relaxed"] else "sin relajar"
        print(f"  sitio {r['site']:4d}  ΔE {r['relative_eV']:+.3f} eV ({tag})  anillos "
              f"{r['rings']}  ×{r['multiplicity']}")


if __name__ == "__main__":
    main()
