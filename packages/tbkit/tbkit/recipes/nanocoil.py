"""Nanocoil phase 2: the periodic 204-atom knee coil with TB (Xu, Tang) and GPAW prepared.

Run (each stage resumable: every result is a file in WORKDIR)::

    python -m tbkit.recipes.nanocoil tb WORKDIR --model tang_carbon   # relax, topology, gap, Γ modes
    python -m tbkit.recipes.nanocoil tb WORKDIR --model xu_carbon
    python -m tbkit.recipes.nanocoil gpaw-k WORKDIR                   # k convergence (GPAW)
    python -m tbkit.recipes.nanocoil gpaw-relax WORKDIR               # PBE relaxation (GPAW)

Input: ``recipes/data/coil204_knee.extxyz``, built by nanocarbon_lab
(``build_knee_periodic_coil(coil_radius=7.75, pitch=15.0, sides_per_turn=6,
circumference=6, relax=True)``, preset "Nanocoil (knees, periodic, smallest)":
{5: 12, 6: 78, 7: 12}, periodic along z, Lz = 15 Å) and exchanged as a file
(the packages do not import each other). See docs/PLAN_NANOCOIL_RAMAN.md.

Checks that decide whether a result is kept (docs/PLAN, phase 2):
* topology: the bond graph (C-C < 1.85 Å) after relaxation must equal the
  input's; any bond made or broken is a TOPOLOGY_WARNING, never corrected;
* Γ modes: the three acoustic ones near zero, imaginary modes counted;
* Raman (non-resonant) only if the relaxed coil has a gap; a gapless coil
  is recorded as such (its Raman is resonant, a different calculation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

INPUT = Path(__file__).with_name("data") / "coil204_knee.extxyz"
KMESH = (1, 1, 4)
KT = 0.05
BOND = 1.85


def coil() -> Atoms:  # noqa: F821
    return read(INPUT)


def bond_graph(atoms) -> set:
    from ase.neighborlist import neighbor_list

    i, j = neighbor_list("ij", atoms, BOND)
    return {(int(a), int(b)) for a, b in zip(i, j, strict=True) if a < b}


def rings(atoms, largest: int = 7) -> list[tuple]:
    """Every simple cycle of at most ``largest`` atoms. In an sp2 net whose faces
    are 5-7 rings these are exactly the faces: two faces sharing a bond already
    make a cycle of at least 8."""
    edges = bond_graph(atoms)
    adjacency = {k: [] for k in range(len(atoms))}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    found = set()

    def walk(path):
        u = path[-1]
        for v in adjacency[u]:
            if v == path[0] and len(path) >= 3:
                found.add(tuple(sorted(path)))
            elif v > path[0] and v not in path and len(path) < largest:
                walk(path + [v])

    for start in range(len(atoms)):
        walk([start])
    return sorted(found)


def ring_census(atoms) -> dict:
    counts = {}
    for ring in rings(atoms):
        counts[len(ring)] = counts.get(len(ring), 0) + 1
    return dict(sorted(counts.items()))


def ring_atoms(atoms, sizes=(5, 7)) -> dict:
    out = {s: set() for s in sizes}
    for ring in rings(atoms):
        if len(ring) in out:
            out[len(ring)].update(ring)
    return {s: sorted(v) for s, v in out.items()}


def _write(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


def _relax_resumable(atoms, calc, folder: Path, fmax: float, steps: int = 600):
    from ase.io import Trajectory
    from ase.optimize import BFGS

    traj = folder / "relax.traj"
    if traj.exists() and traj.stat().st_size > 0:
        try:
            atoms.positions = read(traj, index=-1).positions
        except Exception:                           # noqa: BLE001 - cut mid-write
            traj.unlink()
    atoms.calc = calc
    with Trajectory(str(traj), "a", atoms) as t:
        opt = BFGS(atoms, restart=str(folder / "relax_bfgs.json"), logfile=str(folder / "relax.log"))
        opt.attach(t.write, interval=1)
        converged = bool(opt.run(fmax=fmax, steps=steps))
    return atoms, converged


def stage_tb(workdir: Path, model_name: str) -> dict:
    from ase.vibrations import Vibrations

    from ..calculator import TBCalculator
    from ..params import load_parameters

    folder = Path(workdir) / f"tb_{model_name}"
    folder.mkdir(parents=True, exist_ok=True)
    model = load_parameters(model_name)
    start = coil()
    relaxed_path = folder / "relaxed.extxyz"
    if relaxed_path.exists():
        atoms = read(relaxed_path)
        converged = json.loads((folder / "relax.json").read_text())["converged"]
    else:
        atoms, converged = _relax_resumable(start.copy(), TBCalculator(model, kpts=KMESH, kT=KT),
                                            folder, fmax=0.02)
        atoms.calc = None
        write(relaxed_path, atoms)
        _write(folder / "relax.json", {"converged": converged})
    made = bond_graph(atoms) - bond_graph(start)
    broken = bond_graph(start) - bond_graph(atoms)
    calc = TBCalculator(model, kpts=KMESH, kT=KT)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    gap = float(calc.last_solution.gap())
    bonds = np.array([atoms.get_distance(a, b, mic=True) for a, b in bond_graph(atoms)])
    report = {"model": model_name, "atoms": len(atoms), "relax_converged": converged,
              "energy_eV": energy, "gap_eV": gap,
              "bond_min_max": [float(bonds.min()), float(bonds.max())],
              "ring_census": ring_census(atoms),
              "topology": "OK" if not made and not broken else "TOPOLOGY_WARNING",
              "bonds_made": sorted(made), "bonds_broken": sorted(broken)}
    _write(folder / "report.json", report)
    # Γ modes: ase.vibrations caches every displacement in its directory (resumable)
    vib = Vibrations(atoms, name=str(folder / "vib"), delta=0.01)
    vib.run()
    data = vib.get_vibrations()
    energies, modes = data.get_energies_and_modes(all_atoms=True)
    from ase.units import invcm

    freq = np.where(np.abs(energies.imag) > np.abs(energies.real),
                    -np.abs(energies.imag), np.abs(energies.real)) / invcm
    rings = ring_atoms(atoms)
    on5, on7 = np.array(rings[5]), np.array(rings[7])
    weight = np.sum(np.abs(modes) ** 2, axis=2)                 # (modes, atoms)
    weight /= weight.sum(axis=1, keepdims=True)
    participation = 1 / np.sum(weight ** 2, axis=1) / len(atoms)
    table = [{"frequency_cm1": float(f), "participation": float(p),
              "on_pentagons": float(w[on5].sum()), "on_heptagons": float(w[on7].sum())}
             for f, p, w in zip(freq, participation, weight, strict=True)]
    np.savez(folder / "modes.npz", frequencies=freq, modes=modes)
    report.update({"modes": len(freq), "imaginary_below_-20": int(np.sum(freq < -20)),
                   "lowest_three": [float(v) for v in np.sort(np.abs(freq))[:3]],
                   "ring_atoms": {5: len(on5), 7: len(on7)},
                   "share_of_atoms_on_5_and_7": [len(on5) / len(atoms), len(on7) / len(atoms)],
                   "raman": ("non-resonant applicable" if gap > 0.1 else
                             "gapless: non-resonant Raman does not apply (resonant only)")})
    _write(folder / "report.json", report)
    _write(folder / "modes.json", table)
    return report


def _trimmed(atoms, vacuum: float):
    """Transverse cell = coil extent + 2 vacuum (z untouched)."""
    out = atoms.copy()
    xy = out.positions[:, :2]
    span = xy.max(axis=0) - xy.min(axis=0)
    cell = out.cell.array.copy()
    cell[0, 0], cell[1, 1] = span + 2 * vacuum
    out.set_cell(cell, scale_atoms=False)
    out.positions[:, :2] += -xy.min(axis=0) + vacuum
    return out


def _gpaw(kpts):
    from gpaw import GPAW, FermiDirac

    return GPAW(mode="lcao", basis="dzp", xc="PBE", h=0.2, kpts=kpts, symmetry="off",
                occupations=FermiDirac(KT), txt=None)


def stage_gpaw_k(workdir: Path, vacuum: float = 6.0) -> dict:
    """Single points at the Tang-relaxed geometry: Γ, 1x1x2, 1x1x4 (energy, forces)."""
    folder = Path(workdir) / "gpaw_k"
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(workdir) / "tb_tang_carbon" / "relaxed.extxyz"
    base = _trimmed(read(source) if source.exists() else coil(), vacuum)
    write(folder / "geometry.extxyz", base)
    out = {}
    for kz in (1, 2, 4):
        path = folder / f"k{kz}.json"
        if not path.exists():
            atoms = base.copy()
            atoms.calc = _gpaw((1, 1, kz))
            _write(path, {"kz": kz, "energy": float(atoms.get_potential_energy()),
                          "forces": atoms.get_forces().tolist()})
        out[kz] = json.loads(path.read_text())
    e4 = out[4]["energy"]
    f4 = np.array(out[4]["forces"])
    summary = {k: {"dE_per_atom_vs_k4": (v["energy"] - e4) / len(base),
                   "dF_rms_vs_k4": float(np.sqrt(np.mean((np.array(v["forces"]) - f4) ** 2)))}
               for k, v in out.items()}
    _write(folder / "summary.json", summary)
    return summary


def stage_gpaw_relax(workdir: Path, kz: int = 2, vacuum: float = 6.0) -> dict:
    folder = Path(workdir) / "gpaw_relax"
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(workdir) / "tb_tang_carbon" / "relaxed.extxyz"
    atoms = _trimmed(read(source) if source.exists() else coil(), vacuum)
    atoms, converged = _relax_resumable(atoms, _gpaw((1, 1, kz)), folder, fmax=0.05, steps=300)
    atoms.calc = None
    write(folder / "relaxed.extxyz", atoms)
    report = {"converged": converged, "kz": kz, "vacuum": vacuum,
              "topology": "OK" if bond_graph(atoms) == bond_graph(coil()) else "TOPOLOGY_WARNING"}
    _write(folder / "report.json", report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("tb", "gpaw-k", "gpaw-relax"))
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--model", default="tang_carbon")
    parser.add_argument("--kz", type=int, default=2)
    args = parser.parse_args(argv)
    if args.stage == "tb":
        print(json.dumps({k: v for k, v in stage_tb(args.workdir, args.model).items()
                          if k not in ("bonds_made", "bonds_broken")}, indent=1))
    elif args.stage == "gpaw-k":
        print(json.dumps(stage_gpaw_k(args.workdir), indent=1))
    else:
        print(json.dumps(stage_gpaw_relax(args.workdir, args.kz), indent=1))


if __name__ == "__main__":
    main()
