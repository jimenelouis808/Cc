"""GPAW references for nitrogen in carbon: graphitic, pyridinic, pyrrolic and amine N.

Run (needs GPAW; resumable, one file per molecule in ``OUT.parts``)::

    python -m tbkit.recipes.n_references OUT.json --workers 1

``xu_chn`` was fitted on small molecules (``recipes/chn_references``): pyridine and
pyrrole are its only aromatic N, and graphitic N — three carbon neighbours inside an
sp² network, the substitutional N of doped graphene, nanotubes and the coil — is in
neither its training nor its test set. This adds the four motifs, each in aromatic
molecules large enough to carry a π network, built here from regular polygons (no
external structure database; GPAW relaxes them):

* graphitic: cycl[3.3.3]azine (C12H9N, phenalenyl with N at its centre) for training;
  coronene with two interior N (C22H12N2, closed shell) held out;
* pyridinic: quinoline (training); acridine and coronene_N of ``chn_references`` held
  out;
* pyrrolic: indole (training); carbazole held out;
* amine on an aromatic ring: aniline (training); 1-aminonaphthalene held out.

Training molecules get the relaxation, four random displacements and two scalings,
as in ``chn_references``; held-out ones only the relaxed geometry. The GPAW settings
are ``tbkit.references.GPAW_DEFAULTS``, those of every other reference set.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

A_CC, D_CH, D_NH = 1.40, 1.09, 1.01

TRAINING = ("cyclazine", "quinoline", "indole", "aniline")
TEST = ("coronene_N2", "acridine", "carbazole", "aminonaphthalene")


# ---------------------------------------------------------------- polygons
def _polygon(centre, n: int, start_angle: float, side: float = A_CC) -> list[np.ndarray]:
    radius = side / (2 * np.sin(np.pi / n))
    return [np.asarray(centre, float) + radius * np.array([np.cos(start_angle + 2 * np.pi * k / n),
                                                          np.sin(start_angle + 2 * np.pi * k / n), 0.0])
            for k in range(n)]


def _fuse(a, b, away_from, n: int = 6, side: float = A_CC) -> list[np.ndarray]:
    """The n-gon on edge (a, b), on the side away from ``away_from``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mid = 0.5 * (a + b)
    normal = np.cross(b - a, [0, 0, 1.0])
    normal /= np.linalg.norm(normal)
    if normal @ (mid - np.asarray(away_from, float)) < 0:
        normal = -normal
    apothem = side / (2 * np.tan(np.pi / n))
    centre = mid + apothem * normal
    angle = np.arctan2(*(a - centre)[[1, 0]])
    return _polygon(centre, n, angle, side)


def _merge(points: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 0.3 for q in out):
            out.append(p)
    return out


def _cap(skeleton: list[np.ndarray], symbols: list[str], nh: tuple[int, ...] = ()) -> Atoms:
    """Two-coordinated atoms get an H outward (an N only if listed in ``nh``)."""
    atoms = Atoms(symbols, positions=skeleton)
    i, j = neighbor_list("ij", atoms, 1.6)
    out_symbols, out_positions = list(symbols), [np.asarray(p) for p in skeleton]
    for k in range(len(atoms)):
        neighbours = j[i == k]
        if len(neighbours) != 2 or (symbols[k] == "N" and k not in nh):
            continue
        outward = 2 * atoms.positions[k] - atoms.positions[neighbours].sum(axis=0)
        outward /= np.linalg.norm(outward)
        out_symbols.append("H")
        out_positions.append(atoms.positions[k] + (D_NH if symbols[k] == "N" else D_CH) * outward)
    return Atoms(out_symbols, positions=out_positions)


def _nearest(points, target) -> int:
    return int(np.argmin([np.linalg.norm(p - np.asarray(target)) for p in points]))


# ---------------------------------------------------------------- molecules
def cyclazine() -> Atoms:
    """Cycl[3.3.3]azine C12H9N: three hexagons around one N (phenalenyl's centre)."""
    points = []
    for k in range(3):
        angle = np.pi / 2 + 2 * np.pi * k / 3
        centre = A_CC * np.array([np.cos(angle), np.sin(angle), 0.0])
        points += _polygon(centre, 6, angle + np.pi)
    points = _merge(points)
    symbols = ["C"] * len(points)
    symbols[_nearest(points, (0, 0, 0))] = "N"
    return _cap(points, symbols)


def coronene_n2() -> Atoms:
    """Coronene with two interior (graphitic) N, para on the central ring: C22H12N2."""
    a1 = np.array([1.5 * A_CC, np.sqrt(3) / 2 * A_CC, 0.0])
    a2 = np.array([1.5 * A_CC, -np.sqrt(3) / 2 * A_CC, 0.0])
    points = []
    for centre in (np.zeros(3), a1, a2, a1 - a2, -a1, -a2, a2 - a1):
        points += _polygon(centre, 6, 0.0)
    points = _merge(points)
    symbols = ["C"] * len(points)
    for target in ((A_CC, 0, 0), (-A_CC, 0, 0)):
        symbols[_nearest(points, target)] = "N"
    return _cap(points, symbols)


def _acene(n_rings: int) -> list[np.ndarray]:
    points = []
    for r in range(n_rings):
        points += _polygon((r * np.sqrt(3) * A_CC, 0, 0), 6, np.pi / 2)
    return _merge(points)


def quinoline() -> Atoms:
    points = _acene(2)
    symbols = ["C"] * len(points)
    edge = [k for k, p in enumerate(points) if abs(p[1]) > 0.5 and p[0] < 0.0]
    symbols[edge[0]] = "N"                         # an α position of the first ring
    return _cap(points, symbols)


def acridine() -> Atoms:
    points = _acene(3)
    symbols = ["C"] * len(points)
    middle = np.sqrt(3) * A_CC
    symbols[_nearest(points, (middle, A_CC, 0))] = "N"
    return _cap(points, symbols)


def _five(n_benzo: int) -> tuple[list[np.ndarray], int]:
    """Pyrrole ring with benzo rings fused on its β edges: indole (1), carbazole (2)."""
    pentagon = _polygon((0, 0, 0), 5, np.pi / 2)        # vertex 0 (the N) on top
    points = list(pentagon)
    edges = [(3, 4)] if n_benzo == 1 else [(1, 2), (3, 4)]
    for a, b in edges:
        points += _fuse(pentagon[a], pentagon[b], (0, 0, 0))
    return _merge(points), 0


def indole() -> Atoms:
    points, n = _five(1)
    symbols = ["C"] * len(points)
    symbols[n] = "N"
    return _cap(points, symbols, nh=(n,))


def carbazole() -> Atoms:
    points, n = _five(2)
    symbols = ["C"] * len(points)
    symbols[n] = "N"
    return _cap(points, symbols, nh=(n,))


def _amine_on(points: list[np.ndarray], site: int) -> Atoms:
    """NH2 on a ring carbon (planar start; GPAW relaxes the pyramid)."""
    ring = _cap(points, ["C"] * len(points))
    symbols = ring.get_chemical_symbols()
    positions = ring.get_positions()
    # the H on ``site`` becomes the N
    h = next(k for k in range(len(points), len(ring))
             if np.linalg.norm(positions[k] - positions[site]) < 1.2)
    direction = (positions[h] - positions[site]) / np.linalg.norm(positions[h] - positions[site])
    n_pos = positions[site] + 1.39 * direction
    side = np.cross(direction, [0, 0, 1.0])
    hs = [n_pos + D_NH * (0.5 * direction + s * 0.866 * side) for s in (1, -1)]
    keep = [k for k in range(len(ring)) if k != h]
    return Atoms([symbols[k] for k in keep] + ["N", "H", "H"],
                 positions=[positions[k] for k in keep] + [n_pos, *hs])


def aniline() -> Atoms:
    points = _acene(1)
    return _amine_on(points, 0)


def aminonaphthalene() -> Atoms:
    points = _acene(2)
    alpha = next(k for k, p in enumerate(points) if abs(p[1]) > 0.5 and p[0] < 0.0)
    return _amine_on(points, alpha)


BUILDERS = {"cyclazine": cyclazine, "coronene_N2": coronene_n2, "quinoline": quinoline,
            "acridine": acridine, "indole": indole, "carbazole": carbazole,
            "aniline": aniline, "aminonaphthalene": aminonaphthalene}


def structure(name: str) -> Atoms:
    return BUILDERS[name]()


def _one(args):
    name, role, index, settings = args
    from tbkit.references import generate_gpaw

    kwargs = {} if role == "train" else {"n_random": 0, "scales": ()}
    items = generate_gpaw({name: structure(name)}, settings, role=role, seed=100 * index, **kwargs)
    return [item.to_dict() for item in items]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)
    from tbkit.references import GPAW_DEFAULTS, gpaw_settings_record

    settings = dict(GPAW_DEFAULTS)
    jobs = [(n, "train", 70 + k, settings) for k, n in enumerate(TRAINING)]
    jobs += [(n, "test", 90 + k, settings) for k, n in enumerate(TEST)]
    if args.only:
        jobs = [job for job in jobs if job[0] in args.only]
    jobs.sort(key=lambda job: len(structure(job[0])))
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not (parts / f"{job[0]}.json").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        for job, result in zip(todo, pool.map(_one, todo), strict=True):
            (parts / f"{job[0]}.json").write_text(json.dumps(result))
            print(f"{job[0]}: {len(result)} estructuras", flush=True)
    structures = []
    for job in jobs:
        structures += json.loads((parts / f"{job[0]}.json").read_text())
    data = {"settings": gpaw_settings_record(settings), "structures": structures}
    args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(structures)} estructuras en {args.out}")


if __name__ == "__main__":
    main()
