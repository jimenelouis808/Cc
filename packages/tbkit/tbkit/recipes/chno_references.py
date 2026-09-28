"""GPAW reference data for oxygen, added to the C/H/N set: ``xu_chno``.

Run (needs GPAW; about an hour on four cores)::

    python -m tbkit.recipes.chno_references OUT.json --workers 4

Same protocol as :mod:`tbkit.recipes.chn_references` (PBE, LCAO dzp; each
training molecule relaxed, then four random displacements and two uniform
scalings). Training: water, methanol, formaldehyde, acetaldehyde, formic
and acetic acid, CO, CO₂, dimethyl ether, oxirane (an epoxide), furan
(aromatic O), H₂O₂ (O-O), nitromethane (N-O) and acetamide (amide), plus
cyclopropane and bicyclobutane: three-membered rings, which Xu's C-C cannot
close (the acute-angle correction of :mod:`tbkit.repulsive` is fitted to
them together with oxirane and aziridine). Test
(relaxed only): ethanol, acetone, methyl formate, glyoxal and two
graphene-oxide motifs on coronene -- a basal epoxide and a basal 1,4-diol.
The C/H/N references are not recomputed: the fit reads both files.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule

TRAINING = ("H2O", "CH3OH", "H2CO", "CH3CHO", "HCOOH", "CH3COOH", "CO", "CO2", "CH3OCH3",
            "CH2OCH2", "C4H4O", "H2O2", "CH3NO2", "CH3CONH2", "C3H6_D3h", "bicyclobutane")
TEST = ("CH3CH2OH", "CH3COCH3", "HCOOCH3", "OCHCHO", "coronene_epoxide", "coronene_diol")


def coronene(a_cc: float = 1.42, d_ch: float = 1.09) -> Atoms:
    """Coronene C24H12, flat in the xy plane, centred at the origin."""
    from .chn_references import coronene_pyridinic

    flake = coronene_pyridinic(a_cc, d_ch)
    symbols = flake.get_chemical_symbols()
    nitrogen = symbols.index("N")
    # put back the carbon and its hydrogen where the pyridinic N was
    positions = list(flake.get_positions())
    symbols[nitrogen] = "C"
    centre = np.mean([p for p, s in zip(positions, symbols, strict=True) if s == "C"], axis=0)
    outward = positions[nitrogen] - centre
    outward[2] = 0.0
    outward /= np.linalg.norm(outward)
    symbols.append("H")
    positions.append(positions[nitrogen] + d_ch * outward)
    atoms = Atoms(symbols, positions=positions)
    atoms.translate(-atoms.get_positions()[[i for i, s in enumerate(symbols) if s == "C"]]
                    .mean(axis=0))
    return atoms


def _central_ring(atoms: Atoms) -> list[int]:
    carbons = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "C"]
    distance = np.linalg.norm(atoms.positions[carbons, :2], axis=1)
    return [carbons[i] for i in np.argsort(distance)[:6]]


def coronene_epoxide() -> Atoms:
    """An O bridging two bonded carbons of coronene's central ring (above the plane)."""
    atoms = coronene()
    ring = _central_ring(atoms)
    a = ring[0]
    b = min(ring[1:], key=lambda j: np.linalg.norm(atoms.positions[j] - atoms.positions[a]))
    middle = 0.5 * (atoms.positions[a] + atoms.positions[b])
    atoms += Atoms("O", positions=[middle + [0.0, 0.0, 1.25]])
    return atoms


def coronene_diol() -> Atoms:
    """Two OH groups on para carbons of the central ring, both above the plane."""
    atoms = coronene()
    ring = _central_ring(atoms)
    a = ring[0]
    b = max(ring[1:], key=lambda j: np.linalg.norm(atoms.positions[j] - atoms.positions[a]))
    for c in (a, b):
        o = atoms.positions[c] + [0.0, 0.0, 1.43]
        h = o + [0.55, 0.0, 0.75]
        atoms += Atoms("OH", positions=[o, h])
    return atoms


def structure(name: str) -> Atoms:
    builders = {"coronene_epoxide": coronene_epoxide, "coronene_diol": coronene_diol}
    return builders[name]() if name in builders else molecule(name)


def _one(args):
    name, role, index, settings = args
    from tbkit.references import generate_gpaw

    kwargs = {} if role == "train" else {"n_random": 0, "scales": ()}
    items = generate_gpaw({name: structure(name)}, settings, role=role, seed=1000 + 100 * index,
                          **kwargs)
    return name, [item.to_dict() for item in items]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    from tbkit.references import GPAW_DEFAULTS, gpaw_settings_record

    settings = dict(GPAW_DEFAULTS)
    jobs = [(n, "train", k, settings) for k, n in enumerate(TRAINING)]
    jobs += [(n, "test", 50 + k, settings) for k, n in enumerate(TEST)]
    jobs.sort(key=lambda job: -len(structure(job[0])))
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not (parts / f"{job[0]}.json").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_one, job) for job in todo]
        for future in as_completed(futures):
            name, result = future.result()
            (parts / f"{name}.json").write_text(json.dumps(result))
            print(f"{name}: {len(result)} estructuras", flush=True)
    structures = []
    for job in jobs:
        structures += json.loads((parts / f"{job[0]}.json").read_text())
    data = {"settings": gpaw_settings_record(settings), "structures": structures}
    args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(structures)} estructuras en {args.out}")


if __name__ == "__main__":
    main()
