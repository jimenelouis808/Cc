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
from ..functionalization import (
    describe_groups,
    functionalize_bridges,
    functionalize_random,
    make_graphitic_n,
    make_pyridinic_n,
    make_pyridinic_n_oxide,
    make_pyrrolic_like,
    nitrogen_report,
    passivate_edges,
)
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


_NITROGEN_BUILDERS = {
    "graphitic": make_graphitic_n,
    "pyridinic": make_pyridinic_n,
    "pyrrolic": make_pyrrolic_like,
    "n-oxide": make_pyridinic_n_oxide,
}


def _apply_post(atoms, args):
    """Apply doping, nitrogen configurations, defects and functional groups.

    Order matters: lattice modifications (doping, N configurations, vacancies)
    come first, then groups are attached to whatever edges result. Doing it
    the other way round would attach groups to carbons that are later
    removed.
    """
    if getattr(args, "dopant", None):
        atoms = dope_random(atoms, args.dopant, args.dopant_conc, seed=args.seed)

    nitrogen = getattr(args, "nitrogen", None)
    if nitrogen:
        builder = _NITROGEN_BUILDERS[nitrogen]
        count = getattr(args, "nitrogen_count", 1)
        if nitrogen == "graphitic":
            atoms = builder(atoms, n_sites=count, seed=args.seed)
        elif nitrogen == "n-oxide":
            atoms = builder(atoms, n_defects=count, seed=args.seed)
        else:
            atoms = builder(atoms, n_defects=count, seed=args.seed)

    if getattr(args, "vacancies", 0):
        atoms = introduce_vacancies(atoms, n_defects=args.vacancies, seed=args.seed)

    if getattr(args, "passivate_edges", False):
        atoms = passivate_edges(atoms)

    group = getattr(args, "group", None)
    if group:
        if group == "epoxy":
            atoms = functionalize_bridges(
                atoms, n_groups=args.group_count, seed=args.seed
            )
        else:
            atoms = functionalize_random(
                atoms, group, n_groups=args.group_count,
                site_kind=args.group_site, seed=args.seed,
            )
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


def _cmd_groups(args):
    """List the functional groups available."""
    print(describe_groups())
    print(
        "\nOjo con la distinción: un grupo nitrogenado (-NH2, -NO2) se ANCLA "
        "al carbono,\nmientras que --nitrogen crea configuraciones DENTRO de "
        "la red (grafítico,\npiridínico, pirrólico). No son intercambiables: "
        "el XPS del N 1s las separa."
    )
    return 0


def _cmd_nitrogen_report(args):
    """Report the nitrogen content of an existing structure."""
    atoms = ase_io.read(args.path)
    print(nitrogen_report(atoms))
    return 0


def _cmd_pseudos(args):
    """Report which pseudopotentials a calculation needs, and check for them."""
    from ..exports.pseudos import check_directory, describe, requirements_for

    atoms = ase_io.read(args.structure)
    requirements = requirements_for(
        atoms, needs_raman=args.raman, needs_soc=args.spinorbit
    )
    print(describe(requirements))

    if args.dir:
        print()
        check = check_directory(args.dir, requirements)
        print(check.summary())
        return 0 if check.ok else 1
    return 0


def _cmd_bands(args):
    """Plot a finished band-structure calculation."""
    from ..results.bands import (
        attach_path_labels,
        plot_bands,
        read_qe_bands,
        read_qe_bands_gnu,
        read_siesta_bands,
    )

    path = Path(args.path)
    try:
        if path.suffix == ".bands":
            bands = read_siesta_bands(path)
        elif path.name.endswith(".gnu"):
            bands = read_qe_bands_gnu(path)
        else:
            bands = read_qe_bands(path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.labels:
        attach_path_labels(bands, args.labels.split(","))

    print(f"{bands.n_kpoints} puntos k x {bands.n_bands} bandas")
    reference = args.fermi if args.fermi is not None else bands.fermi_energy
    if reference is not None:
        print(f"Referencia de energía: {reference:.4f} eV")
        gap = bands.band_gap(fermi=reference)
        if gap is None:
            print("Gap: ninguno — las bandas cruzan la referencia (metálico).")
        else:
            print(f"Gap muestreado: {gap:.4f} eV")
            print("  (solo ve los puntos k del camino; un extremo fuera de él "
                  "no aparece)")
    else:
        print("Sin nivel de Fermi en el archivo; pásalo con --fermi para el gap.")

    window = tuple(args.window) if args.window else None
    figure = plot_bands(bands, reference=reference, energy_window=window)
    figure.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Figura guardada en {args.out}")
    return 0


def _cmd_spectrum(args):
    """Plot a finished phonon / IR / Raman calculation."""
    from ..results.spectra import plot_spectrum, read_dynmat

    try:
        spectrum = read_dynmat(args.path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(spectrum.summary())

    if args.kind == "raman" and not spectrum.has_raman:
        print("\nEste cálculo no trae actividades Raman.", file=sys.stderr)
        return 1
    if args.kind == "ir" and not spectrum.has_ir:
        print("\nEste cálculo no trae actividades IR.", file=sys.stderr)
        return 1

    figure = plot_spectrum(
        spectrum,
        kind=args.kind,
        width_cm1=args.width,
        laser_wavelength_nm=args.laser,
        temperature_k=args.temperature,
    )
    figure.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nFigura guardada en {args.out}")
    return 0


def _cmd_converge(args):
    """Generate a convergence sweep."""
    from ..workflows.convergence import cutoff_sweep, kpoint_sweep

    atoms = ase_io.read(args.structure)
    outdir = Path(args.out)
    if args.parameter == "cutoff":
        values = args.values or [40, 50, 60, 70, 80, 90, 100]
        written = cutoff_sweep(atoms, outdir, cutoffs=values, force=args.force)
    else:
        values = args.values or [0.40, 0.30, 0.25, 0.20, 0.15, 0.10]
        written = kpoint_sweep(atoms, outdir, densities=values, force=args.force)

    print(f"{len(written) - 1} entradas escritas en {outdir}")
    print(f"Ejecuta:  cd {outdir} && ./run_sweep.sh")
    print(f"Y luego:  carbonforge converge-report {outdir}")
    return 0


def _cmd_converge_report(args):
    """Analyse a finished convergence sweep."""
    from ..workflows.convergence import (
        convergence_table,
        plot_convergence,
        read_total_energies,
    )

    points = read_total_energies(args.directory)
    name = "ecutwfc (Ry)" if args.parameter == "cutoff" else "densidad k (1/Å)"
    print(convergence_table(points, tolerance_mev_per_atom=args.tolerance,
                            parameter_name=name))

    if args.out and len(points) >= 2:
        figure = plot_convergence(points, parameter_name=name,
                                  tolerance_mev_per_atom=args.tolerance)
        figure.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"\nFigura guardada en {args.out}")
    return 0


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
    p.add_argument("--nitrogen", default=None,
                   choices=["graphitic", "pyridinic", "pyrrolic", "n-oxide"],
                   help="Configuración de nitrógeno en la red (no es lo mismo "
                        "que un grupo funcional nitrogenado).")
    p.add_argument("--nitrogen-count", type=int, default=1,
                   help="Cuántos sitios de nitrógeno crear.")
    p.add_argument("--group", default=None,
                   help="Grupo funcional a anclar. Lista: carbonforge groups")
    p.add_argument("--group-count", type=int, default=1)
    p.add_argument("--group-site", default="edge", choices=["edge", "basal"],
                   help="Anclar en borde (habitual) o en plano basal (sp3).")
    p.add_argument("--passivate-edges", action="store_true",
                   help="Saturar los bordes con H antes de funcionalizar.")
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

    # --- analysis of finished calculations ------------------------------
    bd = sub.add_parser("plot-bands",
                        help="Plot a finished band structure (QE or SIESTA).")
    bd.add_argument("path", help="bands.dat, bands.dat.gnu or SystemLabel.bands")
    bd.add_argument("--out", default="bands.png")
    bd.add_argument("--fermi", type=float, default=None,
                    help="Nivel de Fermi en eV, si el archivo no lo trae.")
    bd.add_argument("--labels", default=None,
                    help="Etiquetas separadas por comas, p.ej. 'G,M,K,G'.")
    bd.add_argument("--window", type=float, nargs=2, default=None,
                    metavar=("BAJO", "ALTO"),
                    help="Rango de energía en eV respecto a la referencia.")
    bd.add_argument("--dpi", type=int, default=150)
    bd.set_defaults(func=_cmd_bands)

    sp = sub.add_parser("plot-spectrum",
                        help="Plot an IR or Raman spectrum from dynmat.x output.")
    sp.add_argument("path", help="Salida de dynmat.x (dynmat.out).")
    sp.add_argument("--kind", choices=["raman", "ir"], default="raman")
    sp.add_argument("--out", default="spectrum.png")
    sp.add_argument("--width", type=float, default=8.0,
                    help="Anchura lorentziana a media altura, en cm-1.")
    sp.add_argument("--laser", type=float, default=None,
                    help="Longitud de onda del láser en nm; aplica el factor "
                         "(v_laser - v)^4.")
    sp.add_argument("--temperature", type=float, default=None,
                    help="Temperatura en K; aplica el factor de Bose.")
    sp.add_argument("--dpi", type=int, default=150)
    sp.set_defaults(func=_cmd_spectrum)

    cv = sub.add_parser("converge",
                        help="Generate a cutoff or k-point convergence sweep.")
    cv.add_argument("structure", help="Estructura de entrada legible por ASE.")
    cv.add_argument("--parameter", choices=["cutoff", "kpoints"],
                    default="cutoff")
    cv.add_argument("--values", type=float, nargs="+", default=None,
                    help="Valores a barrer. Por defecto, una serie razonable.")
    cv.add_argument("--out", required=True)
    cv.add_argument("--force", action="store_true")
    cv.set_defaults(func=_cmd_converge)

    cr = sub.add_parser("converge-report",
                        help="Analyse a finished convergence sweep.")
    cr.add_argument("directory", help="Carpeta con las salidas de pw.x.")
    cr.add_argument("--parameter", choices=["cutoff", "kpoints"],
                    default="cutoff")
    cr.add_argument("--tolerance", type=float, default=1.0,
                    help="Tolerancia en meV/átomo (por defecto 1).")
    cr.add_argument("--out", default=None, help="Guardar figura en este archivo.")
    cr.add_argument("--dpi", type=int, default=150)
    cr.set_defaults(func=_cmd_converge_report)

    ps = sub.add_parser(
        "pseudos",
        help="Say which pseudopotentials are needed, and check a directory.",
    )
    ps.add_argument("structure", help="Estructura legible por ASE.")
    ps.add_argument("--raman", action="store_true",
                    help="El cálculo incluye Raman: fuerza norm-conserving.")
    ps.add_argument("--spinorbit", action="store_true",
                    help="El cálculo incluye SOC: fuerza pseudos relativistas.")
    ps.add_argument("--dir", default=None,
                    help="Carpeta a comprobar (tu pseudo_dir).")
    ps.set_defaults(func=_cmd_pseudos)

    gr_list = sub.add_parser(
        "groups", help="List the available functional groups."
    )
    gr_list.set_defaults(func=_cmd_groups)

    nr = sub.add_parser(
        "nitrogen-report",
        help="Report nitrogen content and configurations of a structure.",
    )
    nr.add_argument("path", help="Estructura legible por ASE.")
    nr.set_defaults(func=_cmd_nitrogen_report)

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
