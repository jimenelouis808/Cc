"""SIESTA (.fdf) input writer.

SIESTA reads a single Flexible Data Format file. Compared with Quantum
ESPRESSO the main conceptual differences that affect what we write:

* **Localised orbitals instead of plane waves.** There is no ``ecutwfc``;
  basis quality is set by ``PAO.BasisSize`` (SZ / DZ / DZP / TZP) and
  ``PAO.EnergyShift``. ``MeshCutoff`` controls only the real-space grid used
  for the density and potential, so it is *not* the plane-wave cutoff and
  the two are not interchangeable.
* **Different pseudopotentials.** SIESTA wants ``.psf`` or ``.psml`` files,
  not QE's ``.UPF``. They are not interchangeable either.
* **Always 3D-periodic.** Reduced dimensionality is expressed purely through
  vacuum plus a k-grid of 1 along the non-periodic axes, which is what we do.

On vibrational spectra, be aware of the asymmetry with QE: SIESTA has no
DFPT. Frequencies come from frozen phonons (``MD.TypeOfRun FC``, then the
``vibra`` post-processing utility). Born charges are reachable through the
Berry-phase polarisation machinery, but there is **no built-in Raman**. So
:func:`write_siesta` will set up force constants when asked for a phonon
run, and says plainly that Raman intensities need Quantum ESPRESSO instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from ase import Atoms

from ..calculations.kpaths import BandPathSpec, format_siesta_bandlines, suggest_band_path
from ..calculations.spectroscopy import SpectroscopySpec
from ..calculations.spinorbit import SpinOrbitSpec
from ..validation.checks import run_basic_checks

SiestaRunType = Literal["scf", "relax", "vc-relax", "bands", "phonon"]

_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16,
}


@dataclass
class SiestaSettings:
    """Parameters for a SIESTA run.

    Attributes
    ----------
    run_type
        ``"scf"``, ``"relax"`` (ions only), ``"vc-relax"`` (ions + cell),
        ``"bands"`` or ``"phonon"`` (force constants).
    system_name, system_label
        Free-text name and the prefix SIESTA gives its output files.
    basis_size
        ``"SZ"``, ``"DZ"``, ``"DZP"`` or ``"TZP"``. DZP is the usual
        production compromise; SZ is for quick tests only.
    energy_shift_ry
        Orbital confinement energy: sets the cutoff radius of the basis
        functions. Smaller means longer-ranged orbitals and higher cost.
    mesh_cutoff_ry
        Real-space grid cutoff for the density. **Not** a plane-wave cutoff.
    xc_functional, xc_authors
        Exchange-correlation choice, e.g. ``("GGA", "PBE")``.
    kpoint_density
        Same convention as the QE writer: points per Å⁻¹ of reciprocal
        lattice, applied only along periodic axes.
    electronic_temperature_k
        Fermi smearing width, in kelvin.
    max_scf_iterations, dm_tolerance, dm_mixing_weight
        SCF convergence controls.
    force_tolerance_ev_ang
        Relaxation convergence threshold.
    spinorbit
        Optional spin-orbit setup.
    band_path
        Optional explicit band path (defaults to the automatic suggestion).
    """

    run_type: SiestaRunType = "scf"
    system_name: str = "carbonforge structure"
    system_label: str = "carbon"
    basis_size: str = "DZP"
    energy_shift_ry: float = 0.02
    mesh_cutoff_ry: float = 300.0
    xc_functional: str = "GGA"
    xc_authors: str = "PBE"
    kpoint_density: float = 0.20
    electronic_temperature_k: float = 300.0
    max_scf_iterations: int = 200
    dm_tolerance: float = 1.0e-4
    dm_mixing_weight: float = 0.1
    force_tolerance_ev_ang: float = 0.02
    spinorbit: Optional[SpinOrbitSpec] = None
    band_path: Optional[BandPathSpec] = None
    write_forces: bool = True


def _kgrid(atoms: Atoms, density: float) -> tuple[int, int, int]:
    """Monkhorst-Pack subdivisions, 1 along non-periodic axes."""
    pbc = atoms.get_pbc()
    cell = np.array(atoms.cell)
    recip = 2.0 * np.pi * np.linalg.inv(cell).T
    mesh = [1, 1, 1]
    for axis in range(3):
        if not pbc[axis]:
            continue
        mesh[axis] = max(1, int(np.ceil(np.linalg.norm(recip[axis]) / density)))
    return tuple(mesh)  # type: ignore[return-value]


def write_siesta(
    atoms: Atoms,
    outdir: str | Path,
    settings: Optional[SiestaSettings] = None,
    spectroscopy: Optional[SpectroscopySpec] = None,
    filename: str = "input.fdf",
    force: bool = False,
) -> Path:
    """Write a complete SIESTA ``.fdf`` input.

    Parameters
    ----------
    atoms
        Structure to export.
    outdir
        Destination directory, created if missing.
    settings
        Optional :class:`SiestaSettings`; defaults are used otherwise.
    spectroscopy
        When ``run_type="phonon"``, controls the force-constant setup. Raman
        settings are accepted but produce a written warning, since SIESTA
        cannot compute Raman intensities.
    filename
        Output file name.
    force
        Bypass structural validation errors.

    Returns
    -------
    pathlib.Path
        Path to the written ``.fdf``.

    Raises
    ------
    ValueError
        If structural validation fails and ``force`` is False.
    """
    report = run_basic_checks(atoms)
    if not report.ok and not force:
        raise ValueError(
            "Structure failed validation, refusing to export:\n" + report.summary()
        )

    s = settings or SiestaSettings()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    symbols = atoms.get_chemical_symbols()
    unique = sorted(set(symbols))
    species_index = {symbol: i + 1 for i, symbol in enumerate(unique)}
    mesh = _kgrid(atoms, s.kpoint_density)

    lines: list[str] = [
        "# SIESTA input generated by carbonforge",
        f"SystemName      {s.system_name}",
        f"SystemLabel     {s.system_label}",
        "",
        f"NumberOfAtoms   {len(atoms)}",
        f"NumberOfSpecies {len(unique)}",
        "",
        "%block ChemicalSpeciesLabel",
    ]
    for symbol in unique:
        z = _ATOMIC_NUMBERS.get(symbol, 0)
        lines.append(f"  {species_index[symbol]:3d}  {z:3d}  {symbol}")
    lines += ["%endblock ChemicalSpeciesLabel", ""]

    # --- geometry ------------------------------------------------------
    lines += [
        "LatticeConstant 1.0 Ang",
        "%block LatticeVectors",
    ]
    for row in np.array(atoms.cell):
        lines.append(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}")
    lines += ["%endblock LatticeVectors", ""]

    lines += [
        "AtomicCoordinatesFormat Ang",
        "%block AtomicCoordinatesAndAtomicSpecies",
    ]
    for symbol, position in zip(symbols, atoms.get_positions()):
        lines.append(
            f"  {position[0]:16.10f} {position[1]:16.10f} {position[2]:16.10f}"
            f"  {species_index[symbol]:3d}"
        )
    lines += ["%endblock AtomicCoordinatesAndAtomicSpecies", ""]

    # --- basis and functional -------------------------------------------
    lines += [
        "# Basis set. DZP is the usual production choice; SZ is test-only.",
        f"PAO.BasisSize      {s.basis_size}",
        f"PAO.EnergyShift    {s.energy_shift_ry} Ry",
        "",
        f"XC.functional      {s.xc_functional}",
        f"XC.authors         {s.xc_authors}",
        "",
        "# Real-space grid for the density. NOT a plane-wave cutoff:",
        "# do not copy a QE ecutwfc value here.",
        f"MeshCutoff         {s.mesh_cutoff_ry} Ry",
        "",
        "SolutionMethod     diagon",
        f"ElectronicTemperature {s.electronic_temperature_k} K",
        f"MaxSCFIterations   {s.max_scf_iterations}",
        f"DM.Tolerance       {s.dm_tolerance:.1e}",
        f"DM.MixingWeight    {s.dm_mixing_weight}",
        "",
    ]

    # --- k-grid ---------------------------------------------------------
    dim = int(sum(atoms.get_pbc()))
    lines += [
        f"# {dim}D system: subdivisions are 1 along the non-periodic axes.",
        "%block kgrid_Monkhorst_Pack",
        f"  {mesh[0]:3d}   0   0   0.0",
        f"    0 {mesh[1]:3d}   0   0.0",
        f"    0   0 {mesh[2]:3d}   0.0",
        "%endblock kgrid_Monkhorst_Pack",
        "",
    ]

    # --- spin-orbit ------------------------------------------------------
    if s.spinorbit is not None and s.spinorbit.enabled:
        lines += [
            "# Spin-orbit coupling. Requires fully-relativistic",
            "# pseudopotentials (.psml); scalar ones give zero splitting.",
            "Spin               SO",
            "",
        ]
    elif s.spinorbit is not None and s.spinorbit.noncolin:
        lines += ["Spin               non-collinear", ""]

    # --- run type --------------------------------------------------------
    if s.run_type in ("relax", "vc-relax"):
        lines += [
            "MD.TypeOfRun       CG",
            "MD.NumCGsteps      200",
            f"MD.MaxForceTol     {s.force_tolerance_ev_ang} eV/Ang",
        ]
        if s.run_type == "vc-relax":
            lines += [
                "MD.VariableCell    .true.",
                "# NOTE: on a slab or wire this will also compress the vacuum.",
                "# Constrain it with MD.RemoveIntramolecularPressure or by",
                "# relaxing only the periodic directions.",
            ]
        lines.append("")
    elif s.run_type == "phonon":
        lines += [
            "# Frozen-phonon force constants. Post-process with the 'vibra'",
            "# utility shipped with SIESTA to get frequencies and modes.",
            "MD.TypeOfRun       FC",
            "MD.FCfirst         1",
            f"MD.FClast          {len(atoms)}",
            "MD.FCDispl         0.04 Bohr",
            "",
        ]
        if spectroscopy is not None and spectroscopy.needs_raman:
            lines += [
                "# WARNING: SIESTA has no Raman implementation. This run gives",
                "# frequencies only. For Raman intensities use Quantum ESPRESSO",
                "# (ph.x with lraman) via carbonforge.exports.qe.",
                "",
            ]

    # --- bands -----------------------------------------------------------
    if s.run_type == "bands":
        path = s.band_path or suggest_band_path(atoms)
        lines += [
            "BandLinesScale  ReciprocalLatticeVectors",
            format_siesta_bandlines(path),
            "",
        ]

    if s.write_forces:
        lines += ["WriteForces        .true.", "WriteCoorStep      .true.", ""]

    output = outdir / filename
    output.write_text("\n".join(lines) + "\n")
    return output
