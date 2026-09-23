"""``carbonforge vibspec`` sub-commands.

* ``presets`` -- list the functionalisation presets.
* ``build``   -- build a finite ribbon, apply a preset, check it and write it.
* ``check``   -- run the physical checks on an existing structure file.
* ``prepare`` -- validate a structure and settings, write a calculation directory.
* ``run``     -- relax and compute the IR spectrum of a prepared directory (GPAW).
* ``show``    -- report on one calculation, or every one under a directory.
* ``index``   -- put every calculation under a directory into an ASE database.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase import io as ase_io

from ..builders.nanoribbon import DEFAULT_VACUUM_PER_SIDE, build_finite_nanoribbon
from ..validation.checks import run_basic_checks
from .core import (
    CalcRecord,
    CalcSpec,
    VibspecError,
    apply_preset,
    check_structure,
    collect,
    describe_presets,
    find_records,
    index_records,
    prepare,
    run,
    suggest_spin,
)


def _position(text: str | None):
    if text is None:
        return None
    return int(text) if text.lstrip("-").isdigit() else text


def _report(atoms, charge: int) -> int:
    report = check_structure(atoms, charge=charge)
    print(report.summary())
    advice = suggest_spin(atoms, charge=charge)
    if advice.spinpol:
        moments = {i: round(float(m), 3) for i, m in enumerate(advice.magmoms) if m}
        print(f"\nMomentos magnéticos iniciales sugeridos (átomo: μB): {moments}")
    return 0 if report.ok else 1


def _cmd_presets(args) -> int:
    print(describe_presets())
    return 0


def _cmd_build(args) -> int:
    atoms = build_finite_nanoribbon(
        args.width, args.length, edge=args.edge, vacuum_per_side=args.vacuum_per_side,
    )
    atoms = apply_preset(atoms, args.preset, position=_position(args.site), edge=args.site_edge)
    basic = run_basic_checks(atoms)
    if not basic.ok and not args.force:
        print(basic.summary())
        print("\nLa estructura no pasa la validación básica; no se escribe (usa --force).")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ase_io.write(out, atoms)
    print(f"{atoms.get_chemical_formula()} ({len(atoms)} átomos) → {out}\n")
    return _report(atoms, args.charge)


def _cmd_check(args) -> int:
    return _report(ase_io.read(args.path), args.charge)


def _cmd_prepare(args) -> int:
    spinpol = {"auto": None, "on": True, "off": False}[args.spinpol]
    spec = CalcSpec(
        xc=args.xc, mode=args.mode, basis=args.basis, h=args.h, ecut=args.ecut, spinpol=spinpol,
        charge=args.charge, fmax=args.fmax, delta=args.delta, nfree=args.nfree,
        scale_factor=args.scale_factor,
    )
    try:
        record = prepare(ase_io.read(args.structure), spec, Path(args.directory),
                         force=args.force, overwrite=args.overwrite)
    except (VibspecError, FileExistsError) as exc:
        print(exc)
        return 1
    print(record.summary())
    warnings = record.checks["prepare"]["warnings"]
    for warning in warnings:
        print(f"  ⚠️  {warning}")
    print(f"\nPara correrlo donde esté GPAW:\n  cd {args.directory} && "
          "mpiexec -n 4 gpaw python run.py")
    return 0


def _cmd_run(args) -> int:
    try:
        record = run(Path(args.directory))
    except (VibspecError, ImportError) as exc:
        print(exc)
        return 1
    print(record.summary())
    return 0


def _cmd_show(args) -> int:
    path = Path(args.path)
    directories = find_records(path)
    if not directories:
        print(f"No hay cálculos en {path}.")
        return 1
    for directory in directories:
        record = CalcRecord.load(directory)
        print(record.summary())
        if record.status == "done" and len(directories) == 1:
            print()
            print(collect(directory).summary())
    return 0


def _cmd_index(args) -> int:
    count = index_records(Path(args.root), Path(args.db))
    print(f"{count} cálculo(s) indexado(s) en {args.db}.")
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``vibspec`` and its sub-commands on the main parser."""
    vs = sub.add_parser("vibspec", help="IR models of functionalised nanoribbons.")
    vsub = vs.add_subparsers(dest="vibspec_command", required=True)

    pr = vsub.add_parser("presets", help="List the functionalisation presets.")
    pr.set_defaults(func=_cmd_presets)

    bd = vsub.add_parser("build", help="Build a finite ribbon with one preset.")
    bd.add_argument("--edge", choices=("armchair", "zigzag"), default="armchair")
    bd.add_argument("--width", type=int, default=5)
    bd.add_argument("--length", type=int, default=3)
    bd.add_argument("--vacuum-per-side", type=float, default=DEFAULT_VACUUM_PER_SIDE)
    bd.add_argument("--preset", default="pristine")
    bd.add_argument(
        "--site", default=None,
        help="middle | center | near_edge | índice de átomo (por defecto, el del preset).",
    )
    bd.add_argument("--site-edge", choices=("armchair", "zigzag"), default=None)
    bd.add_argument("--charge", type=int, default=0)
    bd.add_argument("-o", "--out", default="gnr.xyz")
    bd.add_argument("--force", action="store_true",
                    help="Escribir aunque falle la validación básica.")
    bd.set_defaults(func=_cmd_build)

    ck = vsub.add_parser("check", help="Physical checks on an existing structure.")
    ck.add_argument("path")
    ck.add_argument("--charge", type=int, default=0)
    ck.set_defaults(func=_cmd_check)

    defaults = CalcSpec()
    pp = vsub.add_parser("prepare", help="Validate and write a calculation directory.")
    pp.add_argument("structure", help="Estructura finita (p. ej. de `vibspec build`).")
    pp.add_argument("-d", "--directory", required=True)
    pp.add_argument("--xc", default=defaults.xc)
    pp.add_argument("--mode", choices=("lcao", "fd", "pw"), default=defaults.mode)
    pp.add_argument("--basis", default=defaults.basis)
    pp.add_argument("--h", type=float, default=defaults.h)
    pp.add_argument("--ecut", type=float, default=defaults.ecut, help="eV, solo en modo pw.")
    pp.add_argument("--spinpol", choices=("auto", "on", "off"), default="auto")
    pp.add_argument("--charge", type=int, default=defaults.charge)
    pp.add_argument("--fmax", type=float, default=defaults.fmax)
    pp.add_argument("--delta", type=float, default=defaults.delta)
    pp.add_argument("--nfree", type=int, default=defaults.nfree)
    pp.add_argument("--scale-factor", type=float, default=defaults.scale_factor)
    pp.add_argument("--force", action="store_true",
                    help="Escribir aunque la validación falle (run lo seguirá rechazando).")
    pp.add_argument("--overwrite", action="store_true")
    pp.set_defaults(func=_cmd_prepare)

    rn = vsub.add_parser("run", help="Relax and compute the IR spectrum (needs GPAW).")
    rn.add_argument("directory")
    rn.set_defaults(func=_cmd_run)

    sh = vsub.add_parser("show", help="Report on a calculation or a tree of them.")
    sh.add_argument("path")
    sh.set_defaults(func=_cmd_show)

    ix = vsub.add_parser("index", help="Index every calculation into an ASE database.")
    ix.add_argument("root")
    ix.add_argument("--db", default="vibspec.db")
    ix.set_defaults(func=_cmd_index)
