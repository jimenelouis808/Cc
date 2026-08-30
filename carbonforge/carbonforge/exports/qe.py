"""Quantum ESPRESSO (pw.x) input writer.

Given an :class:`ase.Atoms`, we write a complete ``pw.x`` input containing
``&CONTROL``, ``&SYSTEM``, ``&ELECTRONS`` (+ ``&IONS`` / ``&CELL`` when the
calculation is a relaxation) and the ``ATOMIC_SPECIES``, ``ATOMIC_POSITIONS``,
``CELL_PARAMETERS`` and ``K_POINTS`` cards.

Dimensionality, vacuum and k-point mesh are **auto-detected** from the
``pbc`` flags and the cell:

* 0 periodic axes → 1 × 1 × 1 k-mesh, ``assume_isolated='mp'`` (makov-payne).
* 1 periodic axis → nk × 1 × 1 along that axis.
* 2 periodic axes → nk × nk × 1, ``assume_isolated='2D'``.
* 3 periodic axes → full Monkhorst-Pack mesh.

The k-point density (points per Å⁻¹ along a reciprocal axis) is configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from ase import Atoms

from ..calculations.dos import (
    DOSSpec,
    format_dos_input,
    format_dos_runner,
    format_projwfc_input,
)
from ..calculations.kpaths import BandPathSpec, format_qe_kpath, suggest_band_path
from ..calculations.spectroscopy import (
    SpectroscopySpec,
    format_dynmat_input,
    format_ph_input,
    format_runner_script,
)
from ..calculations.spinorbit import (
    SpinOrbitSpec,
    qe_system_fields,
    relativistic_pseudo_name,
)
from ..validation.checks import run_basic_checks


Calculation = Literal["scf", "relax", "vc-relax", "nscf", "bands"]


@dataclass
class QESettings:
    """Parameters inferred (or user-provided) for a QE pw.x run."""

    calculation: Calculation = "scf"
    prefix: str = "pwscf"
    pseudo_dir: str = "./pseudo"
    outdir: str = "./out"
    ecutwfc: float = 60.0
    ecutrho: float = 480.0
    kpoint_density: float = 0.20  # 1/Å, ~ 2π/kpoint_density Å-1 spacing
    occupations: str = "smearing"
    smearing: str = "mv"
    degauss: float = 0.01
    conv_thr: float = 1.0e-8
    mixing_beta: float = 0.4
    pseudopotentials: dict[str, str] | None = None
    assume_isolated: Optional[str] = None
    dimensionality: Optional[int] = None

    # --- extended capabilities ----------------------------------------
    spinorbit: Optional[SpinOrbitSpec] = None
    band_path: Optional[BandPathSpec] = None
    nbnd: Optional[int] = None
    #: Constrains which cell degrees of freedom ``vc-relax`` may change.
    #: Essential for slabs and wires, whose vacuum would otherwise collapse.
    cell_dofree: Optional[str] = None


_DEFAULT_PSEUDOS: dict[str, str] = {
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
    "B": "B.pbe-n-kjpaw_psl.1.0.0.UPF",
    "S": "S.pbe-n-kjpaw_psl.1.0.0.UPF",
    "P": "P.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}


_ATOMIC_MASSES: dict[str, float] = {
    "C": 12.011,
    "N": 14.007,
    "B": 10.811,
    "S": 32.065,
    "P": 30.974,
    "H": 1.008,
    "O": 15.999,
}


def _kpoint_mesh(atoms: Atoms, density: float) -> tuple[int, int, int]:
    """Monkhorst-Pack mesh proportional to reciprocal-lattice length."""
    pbc = atoms.get_pbc()
    cell = np.array(atoms.cell)
    recip = 2.0 * np.pi * np.linalg.inv(cell).T  # columns are b_i
    mesh = [1, 1, 1]
    for ax in range(3):
        if not pbc[ax]:
            continue
        b_len = np.linalg.norm(recip[ax])
        mesh[ax] = max(1, int(np.ceil(b_len / density)))
    return tuple(mesh)  # type: ignore[return-value]


def infer_qe_settings(atoms: Atoms, base: Optional[QESettings] = None) -> QESettings:
    """Build a :class:`QESettings` by introspecting the structure.

    Parameters
    ----------
    atoms
        Structure to export.
    base
        Optional user-provided baseline; fields that are already set on it
        are preserved.

    Returns
    -------
    QESettings
        Ready-to-write settings. ``dimensionality`` and ``assume_isolated``
        are filled in based on ``atoms.pbc``.
    """
    s = QESettings() if base is None else base
    dim = int(sum(atoms.get_pbc()))
    s.dimensionality = dim
    if s.assume_isolated is None:
        if dim == 2:
            s.assume_isolated = "2D"
        elif dim == 0:
            s.assume_isolated = "mp"
    if s.pseudopotentials is None:
        s.pseudopotentials = {
            sym: _DEFAULT_PSEUDOS.get(sym, f"{sym}.UPF")
            for sym in set(atoms.get_chemical_symbols())
        }
        # Spin-orbit needs fully-relativistic pseudopotentials; rewrite the
        # default names to their rel- counterparts so the input is at least
        # self-consistent. The files still have to be downloaded, and
        # validation.calculations warns if a scalar set slips through.
        if s.spinorbit is not None and s.spinorbit.enabled:
            s.pseudopotentials = {
                sym: relativistic_pseudo_name(sym, name)
                for sym, name in s.pseudopotentials.items()
            }
    return s


def _fmt_namelist(name: str, fields: dict[str, object]) -> str:
    lines = [f"&{name}"]
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, bool):
            val = ".true." if v else ".false."
        elif isinstance(v, str):
            val = f"'{v}'"
        elif isinstance(v, float):
            val = f"{v:.8g}"
        else:
            val = str(v)
        lines.append(f"    {k} = {val}")
    lines.append("/")
    return "\n".join(lines)


def write_qe_input(
    atoms: Atoms,
    outdir: str | Path,
    settings: Optional[QESettings] = None,
    filename: str = "pw.in",
    force: bool = False,
) -> Path:
    """Write a complete pw.x input file.

    Parameters
    ----------
    atoms
        Structure to export.
    outdir
        Directory where the file is written. Created if missing.
    settings
        Optional user-supplied :class:`QESettings`. Missing fields are
        filled in by :func:`infer_qe_settings`.
    filename
        Output file name (default ``pw.in``).
    force
        If ``False`` (default) the export is aborted when validation reports
        any error. Pass ``True`` to bypass (e.g. for batch pre-relaxation).

    Returns
    -------
    pathlib.Path
        Path to the written file.

    Raises
    ------
    ValueError
        If validation fails and ``force`` is ``False``.
    """
    report = run_basic_checks(atoms)
    if not report.ok and not force:
        raise ValueError(
            "Structure failed validation, refusing to export:\n" + report.summary()
        )

    s = infer_qe_settings(atoms, base=settings)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    symbols = atoms.get_chemical_symbols()
    unique = sorted(set(symbols))
    mesh = _kpoint_mesh(atoms, s.kpoint_density)

    control = {
        "calculation": s.calculation,
        "prefix": s.prefix,
        "pseudo_dir": s.pseudo_dir,
        "outdir": s.outdir,
        "tprnfor": True,
        "tstress": True,
        "verbosity": "high",
    }
    system = {
        "ibrav": 0,
        "nat": len(atoms),
        "ntyp": len(unique),
        "ecutwfc": s.ecutwfc,
        "ecutrho": s.ecutrho,
        "occupations": s.occupations,
        "smearing": s.smearing,
        "degauss": s.degauss,
        "assume_isolated": s.assume_isolated,
    }
    if s.nbnd is not None:
        system["nbnd"] = s.nbnd
    # Spin-orbit / non-collinear switches, when requested.
    if s.spinorbit is not None:
        system.update(qe_system_fields(s.spinorbit))
        for element, magnetisation in s.spinorbit.starting_magnetization.items():
            if element in unique:
                system[f"starting_magnetization({unique.index(element) + 1})"] = (
                    magnetisation
                )

    electrons = {
        "conv_thr": s.conv_thr,
        "mixing_beta": s.mixing_beta,
    }

    parts = [
        _fmt_namelist("CONTROL", control),
        _fmt_namelist("SYSTEM", system),
        _fmt_namelist("ELECTRONS", electrons),
    ]
    if s.calculation in ("relax", "vc-relax"):
        parts.append(_fmt_namelist("IONS", {"ion_dynamics": "bfgs"}))
    if s.calculation == "vc-relax":
        cell_fields: dict[str, object] = {"cell_dynamics": "bfgs"}
        if s.cell_dofree:
            cell_fields["cell_dofree"] = s.cell_dofree
        parts.append(_fmt_namelist("CELL", cell_fields))

    # Cards ---------------------------------------------------------------
    species_lines = ["ATOMIC_SPECIES"]
    pseudos = s.pseudopotentials or {}
    for sym in unique:
        mass = _ATOMIC_MASSES.get(sym, 1.0)
        pp = pseudos.get(sym, f"{sym}.UPF")
        species_lines.append(f"  {sym}  {mass:.4f}  {pp}")
    parts.append("\n".join(species_lines))

    cell_lines = ["CELL_PARAMETERS angstrom"]
    for row in atoms.cell:
        cell_lines.append(f"  {row[0]:20.12f} {row[1]:20.12f} {row[2]:20.12f}")
    parts.append("\n".join(cell_lines))

    pos_lines = ["ATOMIC_POSITIONS angstrom"]
    for sym, p in zip(symbols, atoms.get_positions()):
        pos_lines.append(f"  {sym}  {p[0]:20.12f} {p[1]:20.12f} {p[2]:20.12f}")
    parts.append("\n".join(pos_lines))

    # A 'bands' run walks a high-symmetry path; everything else samples a
    # uniform Monkhorst-Pack mesh.
    if s.calculation == "bands":
        path = s.band_path or suggest_band_path(atoms)
        parts.append(format_qe_kpath(path))
    else:
        parts.append(
            f"K_POINTS automatic\n  {mesh[0]} {mesh[1]} {mesh[2]} 0 0 0"
        )

    path = outdir / filename
    path.write_text("\n\n".join(parts) + "\n")
    return path


def write_qe_bands(
    atoms: Atoms,
    outdir: str | Path,
    settings: Optional[QESettings] = None,
    band_path: Optional[BandPathSpec] = None,
    npoints_per_segment: int = 30,
    nbnd: Optional[int] = None,
    force: bool = False,
) -> dict[str, Path]:
    """Write a complete band-structure workflow: scf → bands → ``bands.x``.

    A band structure needs three runs. The scf step converges the charge
    density on a uniform mesh; the bands step then reads that fixed density
    and diagonalises along the high-symmetry path (which is why it is
    non-self-consistent); ``bands.x`` finally collects the eigenvalues into a
    plottable file.

    Parameters
    ----------
    atoms
        Structure to compute.
    outdir
        Destination directory.
    settings
        Base settings; ``calculation`` is overridden per step.
    band_path
        Explicit path. Defaults to :func:`suggest_band_path`.
    npoints_per_segment
        Sampling density, used only when ``band_path`` is not given.
    nbnd
        Number of bands. Defaults to enough empty states to show the
        conduction bands: 1.5x the occupied count, estimated from the
        valence electrons of the species present.
    force
        Bypass structural validation.

    Returns
    -------
    dict
        Maps ``"scf"``, ``"bands"``, ``"bandsx"`` and ``"script"`` to the
        files written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = settings or QESettings()
    path = band_path or suggest_band_path(
        atoms, npoints_per_segment=npoints_per_segment
    )
    if nbnd is None:
        nbnd = _suggest_nbnd(atoms, base)

    written: dict[str, Path] = {}

    scf = replace(base, calculation="scf", band_path=None, nbnd=None)
    written["scf"] = write_qe_input(
        atoms, outdir, settings=scf, filename="pw.scf.in", force=force
    )

    bands = replace(base, calculation="bands", band_path=path, nbnd=nbnd)
    written["bands"] = write_qe_input(
        atoms, outdir, settings=bands, filename="pw.bands.in", force=True
    )

    bandsx = outdir / "bands.in"
    bandsx.write_text(
        "&BANDS\n"
        f"    prefix = '{base.prefix}'\n"
        f"    outdir = '{base.outdir}'\n"
        "    filband = 'bands.dat'\n"
        "    lsym = .true.\n"
        "/\n"
    )
    written["bandsx"] = bandsx

    script = outdir / "run_bands.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "# Generated by carbonforge.\n"
        "set -euo pipefail\n"
        'NCORES="${NCORES:-4}"\n'
        'MPI="${MPI:-mpirun -np $NCORES}"\n\n'
        f"# Band path: {path.path_string}  ({path.source})\n"
        "echo '[1/3] SCF'\n"
        "$MPI pw.x -in pw.scf.in > pw.scf.out\n\n"
        "echo '[2/3] Bands (non-self-consistent)'\n"
        "$MPI pw.x -in pw.bands.in > pw.bands.out\n\n"
        "echo '[3/3] Collecting eigenvalues'\n"
        "bands.x -in bands.in > bands.out\n\n"
        "echo 'Done. Plot bands.dat.gnu'\n"
    )
    script.chmod(0o755)
    written["script"] = script
    return written


def write_qe_spectroscopy(
    atoms: Atoms,
    outdir: str | Path,
    spectroscopy: SpectroscopySpec,
    settings: Optional[QESettings] = None,
    force: bool = False,
) -> dict[str, Path]:
    """Write a phonon / IR / Raman workflow: scf → ``ph.x`` → ``dynmat.x``.

    Parameters
    ----------
    atoms
        Structure to compute.
    outdir
        Destination directory.
    spectroscopy
        What to compute. See
        :class:`~carbonforge.calculations.spectroscopy.SpectroscopySpec`.
    settings
        Base pw.x settings.
    force
        Bypass structural validation.

    Returns
    -------
    dict
        Maps ``"scf"``, ``"ph"``, ``"dynmat"`` and ``"script"`` to the files
        written.

    Notes
    -----
    This writes the inputs; it does not check that they are physically
    runnable. Call
    :func:`carbonforge.validation.calculations.check_spectroscopy` for that —
    it catches the metal / pseudopotential prerequisites that make ``ph.x``
    stop.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = settings or QESettings()

    written: dict[str, Path] = {}
    scf = replace(base, calculation="scf")
    written["scf"] = write_qe_input(
        atoms, outdir, settings=scf, filename="pw.in", force=force
    )

    ph = outdir / "ph.in"
    ph.write_text(
        format_ph_input(spectroscopy, prefix=base.prefix, outdir=base.outdir)
    )
    written["ph"] = ph

    dynmat = outdir / "dynmat.in"
    dynmat.write_text(format_dynmat_input(spectroscopy))
    written["dynmat"] = dynmat

    script = outdir / "run_spectroscopy.sh"
    script.write_text(format_runner_script(spectroscopy))
    script.chmod(0o755)
    written["script"] = script
    return written


# Valence electron counts for the pseudopotentials this package defaults to.
# Used only to size nbnd; a wrong guess costs a few empty bands, nothing more.
_VALENCE_ELECTRONS: dict[str, int] = {
    "H": 1, "B": 3, "C": 4, "N": 5, "O": 6, "P": 5, "S": 6,
}


def _suggest_nbnd(atoms: Atoms, settings: QESettings) -> int:
    """Estimate a band count that includes a useful number of empty states."""
    electrons = sum(
        _VALENCE_ELECTRONS.get(symbol, 4) for symbol in atoms.get_chemical_symbols()
    )
    occupied = electrons / 2.0
    # Non-collinear runs use spinor states, doubling the band count.
    if settings.spinorbit is not None and settings.spinorbit.noncolin:
        occupied = float(electrons)
    return max(4, int(occupied * 1.5) + 4)


def write_qe_dos(
    atoms: Atoms,
    outdir: str | Path,
    spec: Optional[DOSSpec] = None,
    settings: Optional[QESettings] = None,
    force: bool = False,
) -> dict[str, Path]:
    """Write a density-of-states workflow: scf → nscf → dos.x → projwfc.x.

    The nscf step re-uses the converged charge density but samples a denser
    k-mesh, because a mesh that converges the density is not fine enough to
    resolve a DOS curve.

    Parameters
    ----------
    atoms
        Structure to compute.
    outdir
        Destination directory.
    spec
        DOS settings; defaults to :class:`~carbonforge.calculations.dos.DOSSpec`.
    settings
        Base pw.x settings.
    force
        Bypass structural validation.

    Returns
    -------
    dict
        Maps ``"scf"``, ``"nscf"``, ``"dos"``, optionally ``"projwfc"``, and
        ``"script"`` to the files written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dos_spec = spec or DOSSpec()
    base = settings or QESettings()

    written: dict[str, Path] = {}

    scf = replace(base, calculation="scf")
    written["scf"] = write_qe_input(
        atoms, outdir, settings=scf, filename="pw.scf.in", force=force
    )

    # A denser mesh means a smaller density parameter, since the mesh count
    # scales as 1/kpoint_density.
    nscf = replace(
        base,
        calculation="nscf",
        kpoint_density=base.kpoint_density / dos_spec.kmesh_factor,
        nbnd=base.nbnd or _suggest_nbnd(atoms, base),
    )
    written["nscf"] = write_qe_input(
        atoms, outdir, settings=nscf, filename="pw.nscf.in", force=True
    )

    dos_file = outdir / "dos.in"
    dos_file.write_text(
        format_dos_input(dos_spec, prefix=base.prefix, outdir=base.outdir)
    )
    written["dos"] = dos_file

    if dos_spec.projected:
        projwfc = outdir / "projwfc.in"
        projwfc.write_text(
            format_projwfc_input(dos_spec, prefix=base.prefix, outdir=base.outdir)
        )
        written["projwfc"] = projwfc

    script = outdir / "run_dos.sh"
    script.write_text(format_dos_runner(dos_spec))
    script.chmod(0o755)
    written["script"] = script
    return written
