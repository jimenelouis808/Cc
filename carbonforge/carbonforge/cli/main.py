"""``carbonforge`` command line entry point.

Sub-commands:

* ``cnt``        — build a single CNT and export it.
* ``graphene``   — build a graphene supercell.
* ``ribbon``     — build a graphene nanoribbon.
* ``foam``       — build a 3D carbon foam.
* ``validate``   — run validation on an existing structure file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ase import io as ase_io

from ..builders import (
    build_cnt,
    build_graphene_supercell,
    build_nanoribbon,
    build_nanocoil,
    build_carbon_foam,
)
from ..calculations.spectroscopy import SpectroscopySpec
from ..calculations.spinorbit import SpinOrbitSpec
from ..dopants import dope_random
from ..defects import introduce_vacancies
from ..exports.qe import (
    QESettings,
    write_qe_bands,
    write_qe_input,
    write_qe_spectroscopy,
)
from ..exports.lammps import write_lammps
from ..exports.siesta import SiestaSettings, write_siesta
from ..validation.calculations import check_full_setup
from ..validation.checks import run_basic_checks

#: CLI task name → (QE calculation, spectroscopy mode or None)
_TASKS: dict[str, tuple[str, str | None]] = {
    "scf": ("scf", None),
    "relax": ("relax", None),
    "vc-relax": ("vc-relax", None),
    "bands": ("bands", None),
    "phonon": ("scf", "phonon"),
    "ir": ("scf", "ir"),
    "raman": ("scf", "ir+raman"),
}


def _apply_post(atoms, args):
    if getattr(args, "dopant", None):
        atoms = dope_random(atoms, args.dopant, args.dopant_conc, seed=args.seed)
    if getattr(args, "vacancies", 0):
        atoms = introduce_vacancies(atoms, n_defects=args.vacancies, seed=args.seed)
    return atoms


def _export(atoms, outdir: Path, args) -> int:
    """Export the structure in the requested formats for the requested task.

    Runs the calculation-level physics checks first and prints them. Errors
    abort unless ``--force`` was given, because writing an input that cannot
    possibly run is worse than refusing.
    """
    task = getattr(args, "task", "scf")
    calculation, spectro_mode = _TASKS[task]
    spectroscopy = (
        SpectroscopySpec(mode=spectro_mode) if spectro_mode else None
    )
    spinorbit = SpinOrbitSpec() if getattr(args, "spinorbit", False) else None
    settings = QESettings(
        calculation=calculation,
        spinorbit=spinorbit,
        cell_dofree=getattr(args, "cell_dofree", None),
    )

    report = check_full_setup(
        atoms,
        calculation=calculation,
        cell_dofree=getattr(args, "cell_dofree", None),
        spectroscopy=spectroscopy,
        spinorbit=spinorbit,
        pseudopotentials=None,
    )
    if report.warnings or report.errors:
        print(report.summary())
    if not report.ok and not args.force:
        print(
            "\nAbortado. Corrige lo anterior, o repite con --force si sabes "
            "lo que haces.",
            file=sys.stderr,
        )
        return 1

    fmt = args.format
    want_qe = fmt in ("qe", "both", "all")
    want_lammps = fmt in ("lammps", "both", "all")
    want_siesta = fmt in ("siesta", "all")

    if want_qe:
        qe_dir = outdir / "qe"
        if task == "bands":
            write_qe_bands(atoms, qe_dir, settings=settings, force=args.force)
        elif spectroscopy is not None:
            write_qe_spectroscopy(
                atoms, qe_dir, spectroscopy, settings=settings, force=args.force
            )
        else:
            write_qe_input(atoms, qe_dir, settings=settings, force=args.force)
    if want_siesta:
        run_type = "phonon" if spectroscopy is not None else calculation
        write_siesta(
            atoms, outdir / "siesta",
            settings=SiestaSettings(run_type=run_type, spinorbit=spinorbit),
            spectroscopy=spectroscopy,
            force=args.force,
        )
    if want_lammps:
        write_lammps(atoms, outdir / "lammps", force=args.force)
    return 0


def _cmd_cnt(args):
    atoms = build_cnt(n=args.n, m=args.m, length=args.length,
                      bond=args.bond, vacuum=args.vacuum)
    atoms = _apply_post(atoms, args)
    return _export(atoms, Path(args.out), args)


def _cmd_graphene(args):
    atoms = build_graphene_supercell(nx=args.nx, ny=args.ny,
                                     bond=args.bond, vacuum=args.vacuum)
    atoms = _apply_post(atoms, args)
    return _export(atoms, Path(args.out), args)


def _cmd_ribbon(args):
    atoms = build_nanoribbon(width=args.width, length=args.length,
                             edge=args.edge, bond=args.bond,
                             vacuum=args.vacuum, passivate=args.passivate)
    atoms = _apply_post(atoms, args)
    return _export(atoms, Path(args.out), args)


def _cmd_nanocoil(args):
    atoms = build_nanocoil(
        n=args.n, m=args.m,
        coil_radius=args.coil_radius, pitch=args.pitch,
        n_turns=args.turns, bond=args.bond, vacuum=args.vacuum,
        stone_wales_density=args.sw_density, seed=args.seed,
    )
    atoms = _apply_post(atoms, args)
    return _export(atoms, Path(args.out), args)


def _cmd_foam(args):
    atoms = build_carbon_foam(box_size=args.box, n_flakes=args.flakes,
                              flake_radius=args.radius, seed=args.seed)
    return _export(atoms, Path(args.out), args)


def _cmd_validate(args):
    atoms = ase_io.read(args.path)
    report = run_basic_checks(atoms)
    print(report.summary())
    return 0 if report.ok else 1


def _add_common(p):
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--format",
                   choices=["qe", "siesta", "lammps", "both", "all"],
                   default="qe",
                   help="'both' = qe+lammps; 'all' añade SIESTA.")
    p.add_argument("--task", default="scf", choices=list(_TASKS),
                   help="Qué calcular: scf, relax, vc-relax, bands, "
                        "phonon, ir, raman.")
    p.add_argument("--spinorbit", action="store_true",
                   help="Activa acoplamiento espín-órbita (necesita pseudos rel-).")
    p.add_argument("--cell-dofree", default=None,
                   help="Restringe la celda en vc-relax, p.ej. '2Dxy' o 'z'.")
    p.add_argument("--bond", type=float, default=1.42, help="C-C bond length (Å).")
    p.add_argument("--vacuum", type=float, default=15.0, help="Vacuum padding (Å).")
    p.add_argument("--dopant", choices=["N", "B", "S", "P"], default=None)
    p.add_argument("--dopant-conc", type=float, default=0.0)
    p.add_argument("--vacancies", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="Export even if validation reports errors.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="carbonforge",
                                     description="Generate and export nanocarbon structures.")
    sub = parser.add_subparsers(dest="command", required=True)

    cnt = sub.add_parser("cnt", help="Build a single-wall CNT.")
    cnt.add_argument("--n", type=int, required=True)
    cnt.add_argument("--m", type=int, required=True)
    cnt.add_argument("--length", type=float, default=10.0, help="Target length (Å).")
    _add_common(cnt)
    cnt.set_defaults(func=_cmd_cnt)

    gr = sub.add_parser("graphene", help="Build a graphene supercell.")
    gr.add_argument("--nx", type=int, default=2)
    gr.add_argument("--ny", type=int, default=2)
    _add_common(gr)
    gr.set_defaults(func=_cmd_graphene)

    rb = sub.add_parser("ribbon", help="Build a graphene nanoribbon.")
    rb.add_argument("--width", type=int, required=True)
    rb.add_argument("--length", type=int, required=True)
    rb.add_argument("--edge", choices=["armchair", "zigzag"], default="zigzag")
    rb.add_argument("--passivate", action="store_true")
    _add_common(rb)
    rb.set_defaults(func=_cmd_ribbon)

    nc = sub.add_parser("nanocoil", help="Build a helical carbon nanocoil.")
    nc.add_argument("--n", type=int, default=6)
    nc.add_argument("--m", type=int, default=6)
    nc.add_argument("--coil-radius", type=float, default=25.0,
                    help="Helix radius in Å (default 25).")
    nc.add_argument("--pitch", type=float, default=12.0,
                    help="Vertical advance per turn in Å (default 12).")
    nc.add_argument("--turns", type=float, default=1.0,
                    help="Number of helical turns (default 1).")
    nc.add_argument("--sw-density", type=float, default=0.0,
                    help="Stone-Wales defect density on outer wall (0-0.02).")
    _add_common(nc)
    nc.set_defaults(func=_cmd_nanocoil)

    fm = sub.add_parser("foam", help="Build a 3D carbon foam.")
    fm.add_argument("--box", type=float, default=30.0)
    fm.add_argument("--flakes", type=int, default=20)
    fm.add_argument("--radius", type=float, default=4.0)
    _add_common(fm)
    # Foams come out of a stochastic packing, so LAMMPS relaxation is the
    # realistic first step rather than a DFT run.
    fm.set_defaults(func=_cmd_foam, format="lammps")

    vl = sub.add_parser("validate",
                        help="Validate an existing structure file (CIF, XYZ, POSCAR…).")
    vl.add_argument("path", help="Path to a structure file readable by ASE.")
    vl.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
