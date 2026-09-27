"""GPAW reference data for the C/H/N parameter set (``parameters/xu_chn.json``).

Run (needs GPAW; about an hour on four cores)::

    python -m tbkit.recipes.chn_references OUT.json --workers 4

Training molecules: small hydrocarbons and nitrogen compounds covering
single, double, triple and aromatic C-C, C-N, N-N, C-H and N-H bonds,
including pyridinic (pyridine) and pyrrolic (pyrrole) nitrogen. Each is
relaxed, then computed with four random displacements and two uniform
scalings. The test set (not used by the fit) holds other molecules and a
pyridinic-N coronene flake, at their relaxed geometries only.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.neighborlist import neighbor_list

TRAINING = ("CH4", "C2H6", "C2H4", "C2H2", "C6H6", "butadiene", "NH3", "N2H4", "N2", "HCN",
            "H3CNH2", "CH3CN", "NCCN", "CH2NHCH2", "C5H5N", "C4H4NH")
TEST = ("C3H8", "isobutene", "H2CCHCN", "CH3CH2NH2", "C3H9N", "coronene_N")
#: Molecules whose GPAW harmonic frequencies are added (``--frequencies``).
FREQUENCIES = ("CH4", "NH3", "HCN", "C6H6", "C5H5N")


def coronene_pyridinic(a_cc: float = 1.42, d_ch: float = 1.09) -> Atoms:
    """Coronene C24H12 with one edge C-H replaced by a pyridinic N (C23H11N)."""
    a1 = np.array([1.5 * a_cc, np.sqrt(3) / 2 * a_cc, 0.0])
    a2 = np.array([1.5 * a_cc, -np.sqrt(3) / 2 * a_cc, 0.0])
    centres = [np.zeros(3), a1, a2, a1 - a2, -a1, -a2, a2 - a1]
    vertices = []
    for centre in centres:
        for k in range(6):
            angle = np.pi / 3 * k
            point = centre + a_cc * np.array([np.cos(angle), np.sin(angle), 0.0])
            if not any(np.linalg.norm(point - v) < 0.1 for v in vertices):
                vertices.append(point)
    carbon = Atoms(f"C{len(vertices)}", positions=vertices)
    i, j = neighbor_list("ij", carbon, 1.6)
    counts = np.bincount(i, minlength=len(carbon))
    symbols = ["C"] * len(carbon)
    positions = list(carbon.get_positions())
    edge = [k for k in range(len(carbon)) if counts[k] == 2]
    nitrogen = edge[0]
    for k in edge:
        if k == nitrogen:
            symbols[k] = "N"
            continue
        neighbours = j[i == k]
        outward = 2 * carbon.positions[k] - carbon.positions[neighbours].sum(axis=0)
        outward /= np.linalg.norm(outward)
        symbols.append("H")
        positions.append(carbon.positions[k] + d_ch * outward)
    return Atoms(symbols, positions=positions)


def structure(name: str) -> Atoms:
    return coronene_pyridinic() if name == "coronene_N" else molecule(name)


def _one(args):
    name, role, index, settings = args
    from tbkit.references import generate_gpaw

    atoms = structure(name)
    kwargs = {} if role == "train" else {"n_random": 0, "scales": ()}
    items = generate_gpaw({name: atoms}, settings, role=role, seed=100 * index, **kwargs)
    return [item.to_dict() for item in items]


def _frequencies(args):
    name, parts, settings = args
    from tbkit.references import ReferenceStructure, gpaw_frequencies

    eq = ReferenceStructure.from_dict(json.loads((parts / f"{name}.json").read_text())[0])
    return [round(float(v), 2) for v in gpaw_frequencies(eq.atoms, settings)]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", help="solo estas moléculas")
    parser.add_argument("--frequencies", action="store_true",
                        help="añade las frecuencias armónicas GPAW de " + ", ".join(FREQUENCIES))
    args = parser.parse_args(argv)
    from tbkit.references import GPAW_DEFAULTS, gpaw_settings_record

    settings = dict(GPAW_DEFAULTS)
    jobs = [(n, "train", k, settings) for k, n in enumerate(TRAINING)]
    jobs += [(n, "test", 50 + k, settings) for k, n in enumerate(TEST)]
    if args.only:
        jobs = [job for job in jobs if job[0] in args.only]
    # Largest first, so the long ones do not finish last on their own.
    jobs.sort(key=lambda job: -len(structure(job[0])))
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not (parts / f"{job[0]}.json").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        for job, result in zip(todo, pool.map(_one, todo), strict=True):
            (parts / f"{job[0]}.json").write_text(json.dumps(result))
            print(f"{job[0]}: {len(result)} estructuras", flush=True)
    if args.frequencies:
        todo = [n for n in FREQUENCIES if not (parts / f"{n}.freq.json").exists()]
        with ProcessPoolExecutor(args.workers) as pool:
            for name, values in zip(todo, pool.map(_frequencies, [(n, parts, settings)
                                                               for n in todo]), strict=True):
                (parts / f"{name}.freq.json").write_text(json.dumps(values))
                print(f"{name}: frecuencias", flush=True)
    structures = []
    for job in jobs:
        items = json.loads((parts / f"{job[0]}.json").read_text())
        frequencies = parts / f"{job[0]}.freq.json"
        if frequencies.exists():
            items[0]["extra"]["frequencies_cm1"] = json.loads(frequencies.read_text())
        structures += items
    data = {"settings": gpaw_settings_record(settings), "structures": structures}
    args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(structures)} estructuras en {args.out}")


if __name__ == "__main__":
    main()
