"""``tbkit``: tight binding from the command line.

    tbkit levels  cinta.xyz                       # niveles, gap, cargas
    tbkit levels  cinta.xyz --model sp3 --scc     # sp3 de carbono, cargas autoconsistentes
    tbkit bands   grafeno.xyz --path GKMG -o bandas.csv
    tbkit dos     tubo.xyz --kmesh 60 --pdos element -o dos.csv
    tbkit hubbard zgnr.xyz --U 2.7 --kmesh 48 --m-energy m.csv
    tbkit hubbard flake.xyz --field 0 0.5 11 -o campo.csv
    tbkit orbital benceno.xyz --band homo -o homo.cube
    tbkit gpaw-levels calc/gpaw.txt

Structures: any file ASE reads (extxyz from carbonforge or nanocarbon_lab
keeps the cell and periodicity). Models: ``--model pi`` (default), ``sp3``
(Xu carbon), or ``--skf DIR --orbitals "C=s,px,py,pz H=s"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _model(args):
    from .params import pi_model, xu_carbon
    from .skf import load_skf_set

    if args.skf:
        orbitals = {}
        for item in args.orbitals.split():
            element, names = item.split("=")
            orbitals[element] = tuple(names.split(","))
        return load_skf_set(args.skf, orbitals)
    if args.model == "sp3":
        return xu_carbon()
    return pi_model(t=args.t)


def _system(args):
    from ase.io import read

    from .hamiltonian import System

    atoms = read(args.structure)
    return System.build(atoms, _model(args))


def _kpoints(system, n):
    from .kpoints import gamma, mesh

    return mesh(system.atoms, n) if system.periodic else gamma()


def _write_table(path: Optional[str], header: list[str], columns: list[np.ndarray]) -> None:
    rows = np.column_stack(columns)
    if path is None:
        print("\t".join(header))
        for row in rows[:: max(1, len(rows) // 40)]:
            print("\t".join(f"{v:.6g}" for v in row))
        return
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="")
    print(f"→ {path}")


def cmd_levels(args) -> int:
    from .analysis import atomic_charges, bond_orders
    from .scc import self_consistent
    from .solver import solve

    system = _system(args)
    if args.scc:
        result = self_consistent(system, charge=args.charge)
        solution, charges = result.solution, result.charges
        print(result.summary())
    else:
        solution = solve(system, *_kpoints(system, args.kmesh), charge=args.charge)
        charges = atomic_charges(solution)
    homo, lumo = solution.homo_lumo()
    print(f"Modelo: {system.model.name}; {system.basis.size} orbitales, "
          f"{solution.electrons:g} electrones")
    print(f"HOMO {homo:.4f} eV, LUMO {lumo:.4f} eV, gap {solution.gap():.4f} eV, "
          f"E_F {solution.fermi:.4f} eV")
    if not system.periodic:
        levels = np.sort(solution.energies[0, 0])
        print("Niveles (eV):", " ".join(f"{e:.3f}" for e in levels))
    top = sorted(charges.items(), key=lambda kv: -abs(kv[1]))[:8]
    print("Cargas (e, + = pierde electrones):",
          ", ".join(f"{system.atoms[a].symbol}{a} {q:+.3f}" for a, q in top))
    if not system.periodic and args.bonds:
        for (a, b), value in sorted(bond_orders(solution).items(), key=lambda kv: -kv[1])[:12]:
            print(f"  enlace {a}-{b}: {value:.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "model": system.model.name, "homo": homo, "lumo": lumo, "gap": solution.gap(),
            "fermi": solution.fermi, "charges": {str(a): q for a, q in charges.items()},
            "levels": np.sort(solution.energies.ravel()).tolist()}, indent=1))
        print(f"→ {args.json}")
    return 0


def cmd_bands(args) -> int:
    from .analysis import bands
    from .kpoints import band_path

    system = _system(args)
    if not system.periodic:
        print("Las bandas necesitan una estructura periódica (pbc en el archivo).")
        return 1
    path = band_path(system.atoms, args.path, args.npoints)
    energies = bands(system, path)[0]
    x, _, _ = path.get_linear_kpoint_axis()
    _write_table(args.out, ["k"] + [f"banda{i}" for i in range(energies.shape[1])],
                 [x] + [energies[:, i] for i in range(energies.shape[1])])
    return 0


def cmd_dos(args) -> int:
    from .analysis import dos, pdos
    from .solver import solve

    system = _system(args)
    solution = solve(system, *_kpoints(system, args.kmesh))
    grid = np.linspace(solution.energies.min() - 1, solution.energies.max() + 1, args.points)
    grid, total = dos(solution, grid, args.sigma)
    columns, header = [grid - solution.fermi, total], ["E-E_F", "DOS"]
    if args.pdos:
        _, parts = pdos(solution, args.pdos, grid, args.sigma)
        for label, values in parts.items():
            header.append(label)
            columns.append(values)
    print(f"E_F = {solution.fermi:.4f} eV")
    _write_table(args.out, header, columns)
    return 0


def cmd_hubbard(args) -> int:
    from .hubbard import (
        magnetization_vs_doping,
        magnetization_vs_energy,
        magnetization_vs_field,
        mean_field,
    )

    system = _system(args)
    k, w = _kpoints(system, args.kmesh)
    options = dict(U=args.U, kpts=k, weights=w, kT=args.kT)
    if args.field:
        start, stop, n = args.field
        h, m, chi, _ = magnetization_vs_field(system, np.linspace(start, stop, int(n)),
                                              guess=args.guess, **options)
        _write_table(args.out, ["h_eV", "M_muB", "chi_muB_per_eV"], [h, m, chi])
        return 0
    if args.doping:
        start, stop, n = args.doping
        x, m, _ = magnetization_vs_doping(system, charges=np.linspace(start, stop, int(n)),
                                          guess=args.guess, **options)
        _write_table(args.out, ["carga_e", "M_muB"], [x, m])
        return 0
    result = mean_field(system, guess=args.guess, **options)
    print(result.summary())
    moments = result.moments
    top = sorted(moments.items(), key=lambda kv: -abs(kv[1]))[:10]
    print("Momentos locales (μB):",
          ", ".join(f"{system.atoms[a].symbol}{a} {m:+.3f}" for a, m in top))
    if args.m_energy:
        e, m, dm = magnetization_vs_energy(result, sigma=args.sigma)
        _write_table(args.m_energy, ["E-E_F", "m_muB", "dm_dE"], [e, m, dm])
    return 0 if result.converged else 2


def cmd_orbital(args) -> int:
    from .analysis import orbital_on_grid, write_cube
    from .solver import solve

    system = _system(args)
    if system.periodic:
        print("Orbitales en rejilla: solo sistemas finitos.")
        return 1
    solution = solve(system)
    occupied = int(round(solution.electrons / 2))
    band = {"homo": occupied - 1, "lumo": occupied}.get(args.band.lower()) \
        if not args.band.lstrip("-").isdigit() else int(args.band)
    origin, steps, values = orbital_on_grid(solution, band, spacing=args.spacing)
    energy = solution.energies[0, 0, band]
    write_cube(args.out, system.atoms, origin, steps, values,
               comment=f"tbkit banda {band} ({energy:.4f} eV)")
    print(f"Banda {band}: {energy:.4f} eV → {args.out}")
    return 0


def cmd_gpaw_levels(args) -> int:
    from .fit import read_gpaw_eigenvalues

    levels = read_gpaw_eigenvalues(args.path)
    for spin in range(levels.energies.shape[0]):
        n = levels.n_occupied(spin)
        e = levels.energies[spin]
        print(f"espín {spin}: {len(e)} niveles, {n} ocupados, HOMO {e[n - 1]:.4f}, "
              f"LUMO {e[n]:.4f} eV")
    if levels.fermi is not None:
        print(f"E_F {levels.fermi:.4f} eV")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tbkit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def structure_command(name, help_text, func):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("structure")
        p.add_argument("--model", choices=("pi", "sp3"), default="pi")
        p.add_argument("--t", type=float, default=-2.7, help="Hopping π, eV (modelo pi).")
        p.add_argument("--skf", default=None, help="Carpeta con archivos A-B.skf de DFTB.")
        p.add_argument("--orbitals", default="C=s,px,py,pz H=s",
                       help="Base con --skf, p. ej. \"C=s,px,py,pz H=s\".")
        p.add_argument("--kmesh", type=int, default=24, help="Puntos k por eje periódico.")
        p.set_defaults(func=func)
        return p

    lv = structure_command("levels", "Niveles, gap y cargas.", cmd_levels)
    lv.add_argument("--charge", type=float, default=0.0)
    lv.add_argument("--scc", action="store_true", help="Cargas autoconsistentes (finitos).")
    lv.add_argument("--bonds", action="store_true", help="Órdenes de enlace de Mayer.")
    lv.add_argument("--json", default=None)

    bd = structure_command("bands", "Bandas a lo largo de un camino.", cmd_bands)
    bd.add_argument("--path", default=None, help="Puntos especiales (p. ej. GKMG).")
    bd.add_argument("--npoints", type=int, default=200)
    bd.add_argument("-o", "--out", default=None)

    ds = structure_command("dos", "Densidad de estados (y PDOS).", cmd_dos)
    ds.add_argument("--sigma", type=float, default=0.05)
    ds.add_argument("--points", type=int, default=3000)
    ds.add_argument("--pdos", choices=("element", "atom", "orbital"), default=None)
    ds.add_argument("-o", "--out", default=None)

    hb = structure_command("hubbard", "Hubbard de campo medio: momentos y magnetización.",
                           cmd_hubbard)
    hb.add_argument("--U", type=float, default=None, help="eV (por defecto, el del modelo).")
    hb.add_argument("--kT", type=float, default=0.005)
    hb.add_argument("--guess", choices=("antiferro", "ferro", "random", "paramagnetic"),
                    default="antiferro")
    hb.add_argument("--sigma", type=float, default=0.05)
    hb.add_argument("--m-energy", default=None, help="CSV con m(E) y dm/dE.")
    hb.add_argument("--field", type=float, nargs=3, metavar=("H0", "H1", "N"), default=None,
                    help="Barrido de energía Zeeman h = μB·B (eV).")
    hb.add_argument("--doping", type=float, nargs=3, metavar=("Q0", "Q1", "N"), default=None,
                    help="Barrido de carga añadida (e).")
    hb.add_argument("-o", "--out", default=None)

    ob = structure_command("orbital", "Un orbital molecular en un archivo cube.", cmd_orbital)
    ob.add_argument("--band", default="homo", help="homo, lumo o un índice.")
    ob.add_argument("--spacing", type=float, default=0.2)
    ob.add_argument("-o", "--out", default="orbital.cube")

    gp = sub.add_parser("gpaw-levels", help="Niveles de un gpaw.txt (para ajustar).")
    gp.add_argument("path")
    gp.set_defaults(func=cmd_gpaw_levels)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
