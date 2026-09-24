"""``carbonforge vibspec`` sub-commands.

* ``presets`` -- list the functionalisation presets.
* ``build``   -- build a finite ribbon, apply a preset, check it and write it.
* ``check``   -- run the physical checks on an existing structure file.
* ``import``  -- load your own geometry (any format), box it, optionally add a
  preset, check it and write it.
* ``prepare`` -- validate a structure and settings, write a calculation directory.
* ``run``     -- relax and compute the IR spectrum of a prepared directory (GPAW).
* ``show``    -- report on one calculation, or every one under a directory.
* ``index``   -- put every calculation under a directory into an ASE database.
* ``plot``    -- the computed IR spectrum, optionally against an FTIR, with a
  band-matching table and CSV export for ramancarbon.
* ``gui``     -- open the window (model, calculation queue, results).
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
    ImportRefused,
    VibspecError,
    apply_preset,
    check_structure,
    collect,
    describe_presets,
    export_csv,
    find_bands,
    find_records,
    match_bands,
    match_table,
    plot_ir_comparison,
    prepare_experiment,
    read_ftir,
    search_scale_factor,
    index_records,
    load_structure,
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


def _cmd_import(args) -> int:
    try:
        atoms, report = load_structure(args.path, vacuum_per_side=args.vacuum_per_side)
    except (ImportRefused, ValueError) as exc:
        print(exc)
        return 1
    print(report)
    if args.preset != "pristine" or args.site is not None:
        try:
            atoms = apply_preset(atoms, args.preset, position=_position(args.site),
                                 edge=args.site_edge)
        except ValueError as exc:
            print(f"\nNo se pudo aplicar '{args.preset}': {exc}")
            return 1
        print(f"\nPreset '{args.preset}' aplicado: {atoms.get_chemical_formula()}.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ase_io.write(out, atoms)
    print(f"\n→ {out}\n")
    return _report(atoms, args.charge)


def _cmd_prepare(args) -> int:
    spinpol = {"auto": None, "on": True, "off": False}[args.spinpol]
    spec = CalcSpec(
        xc=args.xc, mode=args.mode, basis=args.basis, h=args.h, ecut=args.ecut, spinpol=spinpol,
        charge=args.charge, fmax=args.fmax, delta=args.delta, nfree=args.nfree,
        scale_factor=args.scale_factor,
    )
    try:
        # Through load_structure, so an XYZ without a cell (Avogadro, GaussView)
        # is boxed instead of failing the vacuum check.
        atoms, _ = load_structure(args.structure)
        record = prepare(atoms, spec, Path(args.directory),
                         force=args.force, overwrite=args.overwrite)
    except (VibspecError, FileExistsError, ValueError) as exc:
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


def _cmd_plot(args) -> int:
    from .core.checks import SCALE_FACTOR_RANGE

    directory = Path(args.directory)
    try:
        spectrum = collect(directory)
    except VibspecError as exc:
        print(exc)
        return 1
    record = CalcRecord.load(directory)
    scale = args.scale if args.scale is not None else float(record.spec.get("scale_factor", 1.0))
    window = (args.xmin, args.xmax)

    experiment = None
    matches = None
    if args.ftir:
        try:
            measured = read_ftir(args.ftir, quantity=args.quantity)
        except (OSError, ValueError) as exc:
            print(f"No se pudo leer el FTIR: {exc}")
            return 1
        if measured.quantity_source == "values":
            print(f"⚠️  {measured.name}: sin cabecera que lo diga, se interpretó como "
                  f"{measured.quantity}. Si no es así, usa --quantity.")
        experiment = prepare_experiment(measured, baseline=not args.no_baseline, window=window)
        bands = find_bands(*experiment, prominence=args.prominence)
        matches = match_bands(spectrum, bands, scale, tolerance_cm1=args.tolerance,
                              min_relative_intensity=args.min_intensity)
        if args.fit_scale:
            try:
                fitted, fitted_matches = search_scale_factor(
                    spectrum, bands, tolerance_cm1=args.tolerance,
                    min_relative_intensity=args.min_intensity, bounds=SCALE_FACTOR_RANGE,
                )
            except ValueError as exc:
                print(f"⚠️  {exc} Se mantiene el factor {scale:.4f}.")
            else:
                n = sum(m.experimental_cm1 is not None for m in fitted_matches)
                print(f"Factor de escala ajustado: {fitted:.4f} con {n} bandas (antes "
                      f"{scale:.4f}). Guárdalo solo si las parejas de la tabla tienen sentido.")
                scale, matches = fitted, fitted_matches
        print(match_table(matches))

    figure, drawn = plot_ir_comparison(
        spectrum, title=record.name, experiment=experiment, fwhm_cm1=args.fwhm,
        profile=args.profile, scale_factor=scale, window=window, offset=args.offset,
        matches=matches, experiment_label=Path(args.ftir).stem if args.ftir else "FTIR",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    print(f"\nFigura → {out}")
    if args.csv:
        for path in export_csv(args.csv, drawn["grid"], drawn["computed"],
                               drawn["stick_positions"], drawn["stick_heights"], experiment):
            print(f"CSV → {path}")
    return 0


def _cmd_gui(args) -> int:
    from .gui import main

    return main(args.workdir)


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

    im = vsub.add_parser("import", help="Load your own geometry as a vibspec model.")
    im.add_argument("path", help="Cualquier formato que lea ASE (xyz, cif, pdb, mol, vasp...).")
    im.add_argument("-o", "--out", default="modelo.xyz")
    im.add_argument("--vacuum-per-side", type=float, default=None,
                    help="Vacío por lado, Å (por defecto 7, o el guardado en el archivo).")
    im.add_argument("--preset", default="pristine",
                    help="Funcionalización a añadir encima (por defecto, ninguna).")
    im.add_argument("--site", default=None)
    im.add_argument("--site-edge", choices=("armchair", "zigzag"), default=None)
    im.add_argument("--charge", type=int, default=0)
    im.set_defaults(func=_cmd_import)

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

    pl = vsub.add_parser("plot", help="Computed IR spectrum, optionally against an FTIR.")
    pl.add_argument("directory", help="Cálculo terminado (estado 'done').")
    pl.add_argument("--ftir", help="Espectro experimental, CSV/TXT de dos columnas.")
    pl.add_argument("--quantity", choices=("absorbance", "transmittance"), default=None,
                    help="Qué contiene el FTIR (por defecto, se deduce y se avisa).")
    pl.add_argument("--no-baseline", action="store_true",
                    help="No restar la línea base (envolvente convexa inferior).")
    pl.add_argument("--fwhm", type=float, default=10.0, help="Anchura a media altura, cm⁻¹.")
    pl.add_argument("--profile", choices=("lorentzian", "gaussian"), default="lorentzian")
    pl.add_argument("--scale", type=float, default=None,
                    help="Factor de escala (por defecto, el guardado con el cálculo).")
    pl.add_argument("--fit-scale", action="store_true",
                    help="Ajustar el factor de escala a las bandas emparejadas.")
    pl.add_argument("--tolerance", type=float, default=30.0,
                    help="Distancia máxima para emparejar una banda, cm⁻¹.")
    pl.add_argument("--min-intensity", type=float, default=0.05,
                    help="Intensidad relativa mínima de un modo calculado para emparejarlo.")
    pl.add_argument("--prominence", type=float, default=0.05,
                    help="Prominencia mínima de una banda experimental (fracción del máximo).")
    pl.add_argument("--xmin", type=float, default=400.0)
    pl.add_argument("--xmax", type=float, default=4000.0)
    pl.add_argument("--offset", type=float, default=0.0,
                    help="Desplazar el calculado hacia arriba (≈1.1 para apilar).")
    pl.add_argument("-o", "--out", default="ir.png")
    pl.add_argument("--csv", default=None,
                    help="Prefijo para exportar las curvas en CSV (para ramancarbon).")
    pl.set_defaults(func=_cmd_plot)

    gu = vsub.add_parser("gui", help="Open the vibspec window.")
    gu.add_argument("--workdir", default=None,
                    help="Directorio de cálculos (por defecto ./calculos).")
    gu.set_defaults(func=_cmd_gui)
