"""Active learning: put a model's spurious minima into its training data.

Run (needs GPAW)::

    python -m tbkit.recipes.active_learning xu_chnop.json gpaw_p.json OUT.json --workers 4

For every relaxed (``/eq``) molecule of the reference file the model relaxes
from GPAW's geometry. If it wanders off -- an atom moves more than
``--threshold`` Å, or pairs with H end up inside a hopping switch-off window
(:func:`tbkit.recipes.xu_family.tail_hits`) -- the path from GPAW's geometry to
the model's minimum is sampled at ``--points`` fractions and each point is
computed with GPAW (single points, the file's settings), in the molecule's
group. Refitting with these structures tells the fit that the model's
minimum is not the real one: the trimethyl borate and phosphate esters, whose
methyl H atoms collapsed onto the neighbouring O (SCC attraction with nothing
repulsive at O...H 1.6-2.4 Å), are what this was written for.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def wandering(model, reference, threshold: float = 0.3, fmax: float = 0.005) -> dict:
    """Relax ``reference`` (a ``/eq`` structure) with the model; how far it went."""
    from ase.optimize import BFGS

    from ..calculator import TBCalculator
    from .xu_family import tail_hits

    atoms = reference.atoms.copy()
    atoms.calc = TBCalculator(model)
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=800)
    displacement = np.linalg.norm(atoms.get_positions() - reference.atoms.get_positions(),
                                  axis=1)
    # Only pairs with H count: heavy pairs sitting in a switch window turned out
    # harmless (the heavy_tail experiment), H ones gave the spurious O-H modes.
    hits = [h for h in tail_hits(model, atoms)
            if "H" in [e.rstrip("0123456789") for e in h.split()[0].split("-")]]
    relaxed = atoms.copy()
    relaxed.calc = None
    return {"atoms": relaxed, "max_displacement": float(displacement.max()), "tail_hits": hits,
            "spurious": bool(displacement.max() > threshold or hits)}


def path(start, end, fractions=FRACTIONS) -> list:
    """Linear interpolation of positions between two geometries (same atom order)."""
    out = []
    for f in fractions:
        atoms = start.copy()
        atoms.set_positions((1 - f) * start.get_positions() + f * end.get_positions())
        out.append((f, atoms))
    return out


TORSIONS = (60.0, 120.0, 180.0, 240.0, 300.0)


def hydroxyl_torsions(atoms, centre: str, steps=TORSIONS) -> list:
    """The first X-O-H of ``atoms`` (X = ``centre``), with the H turned about the
    X-O axis by each of ``steps`` degrees: a rigid torsion scan. Straight-line
    paths to a wrong minimum cross compressed bonds and say nothing about the
    torsion itself; seleninic and phosphonic acids turned their hydroxyl onto
    the other O, which only a scan of the torsion shows the fit."""
    from ase.neighborlist import neighbor_list

    symbols = atoms.get_chemical_symbols()
    ii, jj = neighbor_list("ij", atoms, {("O", "H"): 1.15, ("O", centre): 2.1,
                                         (centre, "O"): 2.1, ("H", "O"): 1.15})
    for o in (i for i, s in enumerate(symbols) if s == "O"):
        neighbours = jj[ii == o]
        xs = [j for j in neighbours if symbols[j] == centre]
        hs = [j for j in neighbours if symbols[j] == "H"]
        if xs and hs:
            x, h = xs[0], hs[0]
            out = []
            for step in steps:
                turned = atoms.copy()
                turned.set_dihedral(x_other(atoms, x, o), x, o, h,
                                    atoms.get_dihedral(x_other(atoms, x, o), x, o, h) + step,
                                    indices=[h])
                out.append((step, turned))
            return out
    return []


def x_other(atoms, x: int, o: int) -> int:
    """A neighbour of ``x`` other than ``o`` (the dihedral's first atom)."""
    d = atoms.get_distances(x, range(len(atoms)))
    order = [j for j in np.argsort(d) if j not in (x, o)]
    return int(order[0])


def _single(args):
    label, group, atoms, settings = args
    from tbkit.references import ReferenceStructure, gpaw_single_point

    atoms = atoms.copy()
    atoms.pbc = False
    atoms.center(vacuum=settings["vacuum"])
    energy, forces, levels, n_occ = gpaw_single_point(atoms, settings)
    return ReferenceStructure(label, group, atoms, energy, forces, levels, n_occ,
                              "train").to_dict()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parameters", type=Path, help="el conjunto a poner a prueba")
    parser.add_argument("references", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torsions", metavar="ELEMENT", default=None,
                        help="añade barridos rígidos de la torsión X-O-H (X = ELEMENT)")
    args = parser.parse_args(argv)
    from tbkit.params import load_parameters
    from tbkit.references import GPAW_DEFAULTS, gpaw_settings_record, load_references

    model = load_parameters(str(args.parameters))
    refs, file_settings = load_references(args.references)
    settings = dict(GPAW_DEFAULTS)
    for key in ("xc", "mode", "basis", "h", "vacuum"):
        if key in file_settings:
            settings[key] = file_settings[key]
    jobs, report = [], {}
    for ref in refs:
        if not ref.label.endswith("/eq") or ref.role != "train":
            continue
        result = wandering(model, ref, args.threshold)
        report[ref.group] = {k: result[k] for k in ("max_displacement", "tail_hits", "spurious")}
        print(f"{ref.group:14s} desplazamiento máx {result['max_displacement']:.2f} Å"
              f"{'  ← espurio' if result['spurious'] else ''}", flush=True)
        if result["spurious"]:
            for f, atoms in path(ref.atoms, result["atoms"]):
                jobs.append((f"{ref.group}/al{f:.2f}", ref.group, atoms, settings))
        if args.torsions:
            for step, atoms in hydroxyl_torsions(ref.atoms, args.torsions):
                jobs.append((f"{ref.group}/tor{step:.0f}", ref.group, atoms, settings))
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not (parts / (job[0].replace("/", "_") + ".json")).exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_single, job) for job in todo]
        for future in as_completed(futures):
            entry = future.result()
            (parts / (entry["label"].replace("/", "_") + ".json")).write_text(json.dumps(entry))
            print(f"{entry['label']}: GPAW listo", flush=True)
    structures = [json.loads((parts / (job[0].replace("/", "_") + ".json")).read_text())
                  for job in jobs]
    args.out.write_text(json.dumps({"settings": gpaw_settings_record(settings),
                                    "active_learning": {"parameters": args.parameters.name,
                                                        "threshold": args.threshold,
                                                        "torsions": args.torsions,
                                                        "report": report},
                                    "structures": structures}, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    print(f"{len(structures)} estructuras nuevas en {args.out}")


if __name__ == "__main__":
    main()
