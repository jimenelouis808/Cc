"""``carbonforge vibspec`` sub-commands.

* ``presets`` -- list the functionalisation presets.
* ``build``   -- build a finite ribbon, apply a preset, check it and write it.
* ``check``   -- run the physical checks on an existing structure file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase import io as ase_io

from ..builders.nanoribbon import DEFAULT_VACUUM_PER_SIDE, build_finite_nanoribbon
from ..validation.checks import run_basic_checks
from .core import apply_preset, check_structure, describe_presets, suggest_spin


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
