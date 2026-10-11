"""Carbon in many environments, against GPAW: data for an environment-dependent model.

Run (GPAW; hours in serial, resumable, one file per point)::

    PYTHON=... MODULE=tbkit.recipes.carbon_environments \\
        tbkit/recipes/run_crystals.sh WORKDIR          # GPAW single points
    python -m tbkit.recipes.carbon_environments collect WORKDIR gpaw_carbon_env.json
    python -m tbkit.recipes.carbon_environments baseline gpaw_carbon_env.json [--model xu_carbon]

Xu's model is a fit to diamond, graphite and the linear chain: a two-centre
hopping that depends on distance alone, which is what Tang, Wang, Chan and Ho
(Phys. Rev. B 53, 979 (1996)) replaced by an environment-screened one for
structures that mix coordinations. Before building that, this measures where
Xu's model actually fails: sp (chain), sp2 (graphene, C60, a vacancy and a
Stone-Wales defect in graphene), sp3 (diamond) and amorphous carbon at three
densities, each as uniform strains (bond lengths) and random displacements
(forces). Defect and amorphous geometries are made reproducibly here
(seeded, relaxed with Xu's model) and stored with the GPAW results.

GPAW settings as the other references (PBE, LCAO dzp, h 0.2 Å, Fermi-Dirac
0.1 eV); strained cells keep the unstrained grid (see crystal_validation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms

from .crystal_validation import SETTINGS, _atoms_of, _point, _write_atomic, gpaw_factory, strained

SCALES = (0.94, 0.97, 1.0, 1.03, 1.06)


def _diamond():
    from ase.build import bulk

    return bulk("C", "diamond", a=3.567, cubic=True)


def _graphene():
    from ase.build import graphene

    return graphene("C2", a=2.46, vacuum=7.5).repeat((2, 2, 1))


def _chain():
    """Four atoms along z (room for bond alternation), 1D."""
    atoms = Atoms("C4", positions=[[0, 0, 1.28 * k] for k in range(4)],
                  cell=[10.0, 10.0, 4 * 1.28], pbc=[False, False, True])
    atoms.center(axis=(0, 1))
    return atoms


def _c60():
    from ase.build import molecule

    atoms = molecule("C60")
    atoms.center(vacuum=5.0)
    return atoms


def _relaxed_with_xu(atoms, kpts):
    from ..params import xu_carbon
    from ..tasks import relax

    return relax(atoms, xu_carbon(), kmesh=kpts, kT=0.1, fmax=0.05, steps=400)[1]


def _vacancy():
    from ase.build import graphene

    atoms = graphene("C2", a=2.46, vacuum=7.5).repeat((4, 4, 1))
    del atoms[0]
    return _relaxed_with_xu(atoms, (3, 3, 1))


def _stone_wales():
    from ase.build import graphene

    atoms = graphene("C2", a=2.46, vacuum=7.5).repeat((4, 4, 1))
    d = atoms.get_distances(0, range(len(atoms)), mic=True)
    j = int(np.argsort(d)[1])
    centre = atoms.positions[0] + 0.5 * atoms.get_distance(0, j, mic=True, vector=True)
    for k in (0, j):                       # rotate the 0-j bond by 90° in the plane
        v = atoms.get_distance(0, k, mic=True, vector=True) if k else np.zeros(3)
        rel = atoms.positions[0] + v - centre
        atoms.positions[k] = centre + np.array([-rel[1], rel[0], rel[2]])
    atoms.wrap()
    return _relaxed_with_xu(atoms, (3, 3, 1))


def _amorphous(density_g_cm3: float, n: int = 64, seed: int = 3):
    """Random packing (no pair closer than 1.25 Å) relaxed with Xu's model, at Γ."""
    rng = np.random.default_rng(seed)
    side = (n * 12.011 / (density_g_cm3 * 0.6022)) ** (1 / 3)
    positions = []
    while len(positions) < n:
        p = rng.uniform(0, side, 3)
        if all(np.linalg.norm((p - q + side / 2) % side - side / 2) > 1.25 for q in positions):
            positions.append(p)
    atoms = Atoms(f"C{n}", positions=positions, cell=[side] * 3, pbc=True)
    return _relaxed_with_xu(atoms, (1, 1, 1))


#: name -> (builder, k mesh, kinds of points). "scale": uniform strains of the
#: cell (grid of the unstrained one); "rattle": random displacements, σ 0.08 Å.
SYSTEMS = {
    "diamond": (_diamond, (4, 4, 4), ("scale", "rattle")),
    "graphene": (_graphene, (6, 6, 1), ("scale", "rattle")),
    "chain": (_chain, (1, 1, 8), ("scale", "rattle")),
    "c60": (_c60, (1, 1, 1), ("rattle",)),
    "vacancy": (_vacancy, (3, 3, 1), ("rattle",)),
    "stone_wales": (_stone_wales, (3, 3, 1), ("rattle",)),
    "amorphous_2.0": (lambda: _amorphous(2.0), (1, 1, 1), ("rattle",)),
    "amorphous_2.6": (lambda: _amorphous(2.6), (1, 1, 1), ("rattle",)),
    "amorphous_3.2": (lambda: _amorphous(3.2), (1, 1, 1), ("rattle",)),
}
CRYSTALS = SYSTEMS                         # run_crystals.sh lists them by this name


def points(base: Atoms, kinds) -> list[tuple[str, Atoms]]:
    out = [("eq", base.copy())]
    if "scale" in kinds:
        out += [(f"x{s:.2f}", strained(base, s)) for s in SCALES if s != 1.0]
    if "rattle" in kinds:
        rng = np.random.default_rng(11)
        for k in range(2):
            moved = base.copy()
            moved.positions += rng.normal(0.0, 0.08, size=moved.positions.shape)
            out.append((f"rnd{k}", moved))
    return out


def run_system(name: str, workdir: Path, log=print) -> Path:
    """Every point of one system, skipping those on disk. Idempotent."""
    folder = Path(workdir) / name
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "done"
    if marker.exists():
        return marker
    builder, kpts, kinds = SYSTEMS[name]
    geometry = folder / "geometry.json"
    if geometry.exists():                  # the generated structure, fixed once made
        base = _atoms_of(json.loads(geometry.read_text()))
    else:
        base = builder()
        _write_atomic(geometry, {"symbols": base.get_chemical_symbols(),
                                 "positions": base.get_positions().tolist(),
                                 "cell": base.cell.array.tolist(),
                                 "pbc": [bool(p) for p in base.pbc]})
    make = gpaw_factory(kpts)
    for label, atoms in points(base, kinds):
        path = folder / f"{label}.json"
        if path.exists():
            continue
        atoms.calc = make(grid_of=base) if label.startswith("x") else make()
        _write_atomic(path, _point(atoms, label))
        log(f"{name}: {label} listo")
    marker.write_text("ok\n", encoding="utf-8")
    return marker


def collect(workdir: Path, out: Path) -> Path:
    import ase

    from ..references import ReferenceStructure, save_references

    structures = []
    for name, (_, kpts, kinds) in SYSTEMS.items():
        folder = Path(workdir) / name
        if not (folder / "done").exists():
            continue
        base = _atoms_of(json.loads((folder / "geometry.json").read_text()))
        for label, _ in points(base, kinds):
            p = json.loads((folder / f"{label}.json").read_text())
            structures.append(ReferenceStructure(
                f"{name}/{label}", name, _atoms_of(p), p["energy"], np.array(p["forces"]),
                np.zeros(0), 0, "train", {"kpts": list(kpts)}))
    try:
        import gpaw
        version = gpaw.__version__
    except ImportError:
        version = None
    return save_references(out, structures, dict(SETTINGS, gpaw_version=version,
                                                  ase_version=ase.__version__))


def baseline(reference: Path, model_name: str = "xu_carbon") -> dict:
    """Per system: force RMS error (and DFT scale), and energies along the strain."""
    from ..calculator import TBCalculator
    from ..params import load_parameters
    from ..references import load_references

    refs, _ = load_references(reference)
    model = load_parameters(model_name)
    out = {}
    for name in dict.fromkeys(r.group for r in refs):
        group = [r for r in refs if r.group == name]
        kpts = tuple(group[0].extra["kpts"])
        rows = []
        for r in group:
            a = r.atoms.copy()
            a.calc = TBCalculator(model, kpts=kpts if a.pbc.any() else None, kT=0.1)
            rows.append((r.label.split("/")[1], r.energy, a.get_potential_energy(),
                         r.forces, a.get_forces()))
        e_dft0 = next(e for lab, e, *_ in rows if lab == "eq")
        e_tb0 = next(e for lab, _, e, *_ in rows if lab == "eq")
        n = len(group[0].atoms)
        err = np.concatenate([(f_tb - f_dft).ravel() for *_, f_dft, f_tb in rows])
        scale = np.concatenate([f_dft.ravel() for *_, f_dft, _ in rows])
        out[name] = {
            "atoms": n,
            "force_rmse": float(np.sqrt(np.mean(err ** 2))),
            "force_rms_dft": float(np.sqrt(np.mean(scale ** 2))),
            "dE_per_atom": {lab: [round((e_d - e_dft0) / n, 4), round((e_t - e_tb0) / n, 4)]
                            for lab, e_d, e_t, *_ in rows if lab != "eq"}}
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("gpaw")
    g.add_argument("workdir", type=Path,
                   help="carpeta de los cálculos GPAW (uno por sistema, reanudable)")
    g.add_argument("--systems", nargs="*", default=list(SYSTEMS),
                   help="sistemas a calcular (por omisión, todos)")
    c = sub.add_parser("collect")
    c.add_argument("workdir", type=Path,
                   help="carpeta de los cálculos GPAW (uno por sistema, reanudable)")
    c.add_argument("out", type=Path,
                   help="archivo JSON de referencias que se escribe")
    b = sub.add_parser("baseline")
    b.add_argument("reference", type=Path,
                   help="archivo de referencias reunido (paso collect)")
    b.add_argument("--model", default="xu_carbon",
                   help="conjunto TB con que se compara")
    args = parser.parse_args(argv)
    if args.stage == "gpaw":
        for name in args.systems:
            run_system(name, args.workdir)
    elif args.stage == "collect":
        print(collect(args.workdir, args.out))
    else:
        for name, r in baseline(args.reference, args.model).items():
            print(f"{name:14s} F_rmse {r['force_rmse']:.3f} eV/Å (DFT {r['force_rms_dft']:.2f})  "
                  f"ΔE/átomo [DFT, TB]: {r['dE_per_atom']}")


if __name__ == "__main__":
    main()
