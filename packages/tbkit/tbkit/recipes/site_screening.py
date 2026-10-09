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


def _energy(atoms: Atoms, model, relax: bool, kmesh, kT: float, fmax: float,
            max_steps: int = 400, label: str = "", status=None):
    from ..calculator import TBCalculator

    atoms = atoms.copy()
    atoms.calc = TBCalculator(model, kpts=kmesh, kT=kT)
    converged = None
    if relax:
        from ase.optimize import BFGS

        from ..progress import Progress, watch_optimizer

        optimizer = BFGS(atoms, logfile=None)
        with Progress(None, f"relajando {label}", status=status) as bar:
            watch_optimizer(optimizer, bar)
            converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
    energy = float(atoms.get_potential_energy())
    atoms.calc = None
    return atoms, energy, converged


def screen(atoms: Atoms, dopant: str, model, workdir: Path, relax_top: int = 0, top: int = 3,
           kmesh=8, kT: float = 0.05, fmax: float = 0.05, radius: float = 4.0,
           tolerance: float = 0.05, max_sites: int | None = None,
           part: tuple = (0, 1), max_steps: int = 400) -> dict:
    """``part=(r, n)``: compute only the classes ≡ r (mod n) and stop (n processes share
    the folder); the call with n = 1 then reads every file and ranks."""
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
            relaxed, energy, converged = _energy(doped, model, relax, kmesh, kT, fmax, max_steps,
                                                 name, workdir / f"progreso_{name}.json")
            write(workdir / f"{name}.extxyz", relaxed)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"site": site, "equivalent": members, "energy_eV": energy,
                                       "relaxed": relax, "converged": converged}, indent=1))
            tmp.replace(path)
        return json.loads(path.read_text())

    from ..progress import Progress

    mine = [m for k, m in enumerate(ordered) if k % part[1] == part[0]]
    left = [m for m in mine if not (workdir / f"site_{m[0]:04d}.json").exists()]
    with Progress(len(mine), f"sitios (punto simple) parte {part[0] + 1}/{part[1]}",
                  status=workdir / f"progreso_sitios_{part[0]}.json",
                  done=len(mine) - len(left)) as bar:
        for members in left:
            computed(members[0], members, False)
            bar.step(note=f"sitio {members[0]}")
    if part[1] > 1:
        done = all((workdir / f"site_{m[0]:04d}.json").exists() for m in ordered)
        if relax_top and done:                  # second pass: the relaxations, shared too
            first = sorted((computed(m[0], m, False) for m in ordered),
                           key=lambda r: r["energy_eV"])[:relax_top]
            for k, r in enumerate(first):
                if k % part[1] == part[0]:
                    computed(r["site"], r["equivalent"], True)
        return {"part": list(part)}
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
    parser.add_argument("structure", type=Path,
                        help="estructura de partida (xyz, extxyz…)")
    parser.add_argument("dopant",
                        help="elemento que sustituye al carbono (N, B, S, P…)")
    parser.add_argument("workdir", type=Path,
                        help="carpeta de trabajo (un archivo por sitio, reanudable)")
    parser.add_argument("--model", default="chn", help="nombre (chn, chnob, …) o archivo")
    parser.add_argument("--relax-top", type=int, default=0,
                        help="relaja los K mejores del primer cribado")
    parser.add_argument("--top", type=int, default=3,
                        help="cuántos de los mejores sitios se escriben para DFT (top/)")
    parser.add_argument("--kmesh", type=int, default=8,
                        help="puntos k por eje periódico")
    parser.add_argument("--max-sites", type=int, default=None,
                        help="calcular a lo más N clases de sitio, repartidas (vacío: todas)")
    parser.add_argument("--kT", type=float, default=0.05,
                        help="temperatura electrónica (eV); la coil es casi metálica")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="criterio de las relajaciones: fuerza máxima (eV/Å)")
    parser.add_argument("--max-steps", type=int, default=400,
                        help="pasos máximos de BFGS por relajación")
    parser.add_argument("--radius", type=float, default=4.0,
                        help="radio (Å) de la huella de distancias que define una clase de sitio")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="redondeo (Å) de esas distancias: mayor junta más sitios por clase")
    parser.add_argument("--part", default="0/1", help="r/n: solo las clases ≡ r (mod n)")
    args = parser.parse_args(argv)
    r, n = (int(x) for x in args.part.split("/"))
    report = screen(read(args.structure), args.dopant, load_model(args.model), args.workdir,
                    part=(r, n), relax_top=args.relax_top, top=args.top, kmesh=args.kmesh,
                    max_sites=args.max_sites, kT=args.kT, fmax=args.fmax,
                    radius=args.radius, tolerance=args.tolerance, max_steps=args.max_steps)
    if n > 1:
        print(f"parte {r}/{n} hecha")
        return
    print(f"{report['classes']} clases de sitio, {report['computed']} calculadas")
    for r in report["sites"]:
        tag = "relajado" if r["relaxed"] else "sin relajar"
        print(f"  sitio {r['site']:4d}  ΔE {r['relative_eV']:+.3f} eV ({tag})  anillos "
              f"{r['rings']}  ×{r['multiplicity']}")


if __name__ == "__main__":
    main()
