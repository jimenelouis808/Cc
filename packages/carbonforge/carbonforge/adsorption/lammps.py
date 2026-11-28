"""The same grand-canonical run, done by LAMMPS (``fix gcmc``).

:mod:`carbonforge.adsorption.gcmc` is this package's own Monte Carlo,
checked against Levesque's tables. This module writes the **same model**
as a LAMMPS input so that an independent, widely used engine can check
it, and then run the structures nobody has published numbers for:

* the pair potentials are not re-typed. ``pair_style table`` reads them
  from :func:`~carbonforge.adsorption.potentials.feynman_hibbs`, so the
  Feynman-Hibbs correction -- 20 % of the loading at 77 K -- is the one
  the in-house run uses, at the run's own temperature;
* the reservoir enters as LAMMPS' ``pressure`` plus a
  ``fugacity_coeff`` from the same :class:`~.bulk.VirialGas`, so both
  engines see one activity ``beta f``;
* both interactions are **truncated** at :data:`~.gcmc.CUTOFF`, not
  shifted and without a tail correction, as the in-house run does;
* the framework is rigid: no integrator is defined, so the only moves
  are ``fix gcmc``'s insertions, deletions and displacements;
* ``overlap_cutoff`` rejects a trial that lands within
  :data:`OVERLAP_CUTOFF` of an atom before its energy is computed. At
  that distance the H2-C energy is ~10^6 K, so nothing that could be
  accepted is lost, and a table has an inner limit a random insertion
  would otherwise cross.

The run is a FOLDER (``carbonforge-gcmc-lammps/1``): a data file, the
table, the input and a ``request.json``. :func:`read_lammps_gcmc` turns
the count trace it leaves into a :class:`~.gcmc.GCMCResult`, block
error and drift warning included, so the two engines' results compare
field by field.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms

from .bulk import VirialGas, fit_virial
from .gcmc import AVOGADRO, CUTOFF, GCMCResult
from .potentials import H2_C, H2_H2, MASS_C, MASS_H2, feynman_hibbs

FORMAT = "carbonforge-gcmc-lammps/1"

#: k_B in kcal/mol/K: the potentials are in kelvin, LAMMPS ``real``
#: units in kcal/mol.
KCAL_PER_KELVIN = 0.0019872043

#: 1 MPa in atmospheres, LAMMPS ``real`` pressure units.
ATM_PER_MPA = 9.869232667

#: Trials closer than this to any atom are rejected outright (Å).
OVERLAP_CUTOFF = 1.5

#: The table's inner limit and its number of points. Linear
#: interpolation at 1.1e-4 Å spacing: the steepest part of the wall a
#: trial can reach (1.5 Å) moves by 0.3 % per step.
TABLE_INNER = 1.0
TABLE_POINTS = 100_000

#: Exchange (insert or delete) and displacement attempts per step. The
#: in-house run picks insert, delete and displace with a third each, so
#: two exchanges and one displacement per step is the same mix.
EXCHANGES_PER_STEP = 2
DISPLACEMENTS_PER_STEP = 1


def _table_section(name: str, pair, temperature: float, cutoff: float,
                   points: int = TABLE_POINTS) -> str:
    """One ``pair_style table`` section: index, r, energy, force."""
    r = np.linspace(TABLE_INNER, cutoff, points)
    if pair is None:
        energy = np.zeros_like(r)
    else:
        energy = feynman_hibbs(r, pair, temperature) * KCAL_PER_KELVIN
    force = -np.gradient(energy, r)
    lines = [name, f"N {points} R {TABLE_INNER} {cutoff}", ""]
    lines += [f"{i + 1} {x:.12f} {e:.12e} {f:.12e}"
              for i, (x, e, f) in enumerate(zip(r, energy, force))]
    return "\n".join(lines) + "\n\n"


def write_lammps_gcmc(
    framework: Atoms,
    folder: Path,
    temperature: float,
    pressure_mpa: float,
    equilibration: int = 200_000,
    production: int = 400_000,
    max_displacement: float = 1.0,
    seed: int = 1,
    gas: VirialGas | None = None,
    cutoff: float = CUTOFF,
    void_fraction: float | None = None,
    samples: int = 200,
) -> Path:
    """Write one state point as a LAMMPS ``fix gcmc`` folder.

    `equilibration` and `production` count **trial moves**, as
    :func:`~.gcmc.run_gcmc` does, so the same numbers give the same
    sampling; they are turned into steps of :data:`EXCHANGES_PER_STEP` +
    :data:`DISPLACEMENTS_PER_STEP` trials. The production count trace is
    sampled `samples` times.

    Raises
    ------
    ValueError
        For a cell the in-house run would refuse (not orthorhombic, or an
        edge under twice the cutoff), so both engines run the same set of
        structures, or a seed LAMMPS cannot take (it must be positive).
    """
    cell = np.asarray(framework.cell, dtype=float)
    edges = np.diag(cell)
    if not np.allclose(cell - np.diag(edges), 0.0):
        raise ValueError("the framework cell must be orthorhombic.")
    if edges.min() < 2.0 * cutoff:
        raise ValueError(
            f"the shortest cell edge is {edges.min():.2f} Å, under twice the "
            f"{cutoff:.1f} Å cutoff; build a supercell.")
    if seed < 1:
        raise ValueError("LAMMPS needs a positive random seed.")
    gas = gas or fit_virial(temperature)
    phi = gas.fugacity(pressure_mpa) / (pressure_mpa * 1e6)
    per_step = EXCHANGES_PER_STEP + DISPLACEMENTS_PER_STEP
    equilibration_steps = max(1, int(round(equilibration / per_step)))
    production_steps = max(1, int(round(production / per_step)))
    every = max(1, production_steps // samples)

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    positions = np.mod(framework.get_positions(), edges)
    data = [
        "carbonforge rigid framework for fix gcmc", "",
        f"{len(positions)} atoms", "2 atom types", "",
        f"0 {edges[0]:.8f} xlo xhi", f"0 {edges[1]:.8f} ylo yhi",
        f"0 {edges[2]:.8f} zlo zhi", "",
        "Masses", "", f"1 {MASS_C}", f"2 {MASS_H2}", "",
        "Atoms # atomic", "",
    ]
    data += [f"{i + 1} 1 {x:.8f} {y:.8f} {z:.8f}"
             for i, (x, y, z) in enumerate(positions)]
    (folder / "framework.data").write_text("\n".join(data) + "\n")
    (folder / "pair.table").write_text(
        _table_section("CC", None, temperature, cutoff)
        + _table_section("HC", H2_C, temperature, cutoff)
        + _table_section("HH", H2_H2, temperature, cutoff))
    script = f"""# {FORMAT}: H2 in a rigid framework, {temperature} K, {pressure_mpa} MPa
units real
atom_style atomic
boundary p p p
read_data framework.data
group frame type 1
group gas type 2
pair_style table linear {TABLE_POINTS}
pair_coeff 1 1 pair.table CC {cutoff}
pair_coeff 1 2 pair.table HC {cutoff}
pair_coeff 2 2 pair.table HH {cutoff}
neighbor 2.0 bin
neigh_modify exclude type 1 1 every 1 delay 0 check yes
variable n equal count(gas)
fix exchange gas gcmc 1 {EXCHANGES_PER_STEP} {DISPLACEMENTS_PER_STEP} 2 {seed} {temperature} 0.0 {max_displacement} pressure {pressure_mpa * ATM_PER_MPA:.8f} fugacity_coeff {phi:.8f} overlap_cutoff {OVERLAP_CUTOFF}
thermo {max(1, equilibration_steps // 20)}
thermo_style custom step v_n
run {equilibration_steps}
fix trace all print {every} "${{n}}" file counts.dat screen no
run {production_steps}
"""
    (folder / "in.gcmc").write_text(script)
    request = {
        "format": FORMAT, "temperature": temperature,
        "pressure_mpa": pressure_mpa, "equilibration": equilibration,
        "production": production, "seed": seed, "cutoff": cutoff,
        "fugacity_coefficient": phi, "edges": edges.tolist(),
        "carbon_atoms": len(positions),
        "bulk_density_g_cm3": gas.mass_density(pressure_mpa),
        "void_fraction": void_fraction,
    }
    (folder / "request.json").write_text(json.dumps(request, indent=1))
    return folder


def read_lammps_gcmc(folder: Path, blocks: int = 10) -> GCMCResult:
    """The finished folder's count trace as a :class:`~.gcmc.GCMCResult`.

    The same quantities and the same diagnostics as the in-house run:
    block-mean standard error, a warning when it is over 5 % of the
    mean, and one when the loading drifts between the first block and
    the last.
    """
    folder = Path(folder)
    request = json.loads((folder / "request.json").read_text())
    counts = np.loadtxt(folder / "counts.dat", comments="#", ndmin=1)
    usable = len(counts) - len(counts) % blocks
    if usable < blocks:
        raise ValueError(f"{folder} holds {len(counts)} samples, fewer than "
                         f"the {blocks} blocks the error needs.")
    block_means = counts[:usable].reshape(blocks, -1).mean(axis=1)
    mean_count = float(counts.mean())
    error = float(block_means.std(ddof=1) / np.sqrt(blocks))
    volume_cm3 = float(np.prod(request["edges"])) * 1e-24
    hydrogen_g = mean_count * MASS_H2 / AVOGADRO
    carbon_g = request["carbon_atoms"] * MASS_C / AVOGADRO
    absolute = hydrogen_g / volume_cm3
    bulk = float(request["bulk_density_g_cm3"])
    accessible = request.get("void_fraction")
    warnings = []
    if accessible is None:
        accessible = 1.0
        warnings.append("no void fraction was given, so the excess uses the "
                        "WHOLE cell volume and overstates the correction.")
    if error > 0.05 * max(mean_count, 1.0):
        warnings.append(f"the block standard error is "
                        f"{100 * error / max(mean_count, 1):.0f} % of the "
                        "mean: this run is too short to quote.")
    drift = block_means[-1] - block_means[0]
    if abs(drift) > 3.0 * error and abs(drift) > 0.02 * max(mean_count, 1.0):
        warnings.append(f"the loading drifted by {drift:+.1f} molecules from "
                        "the first production block to the last.")
    return GCMCResult(
        temperature=request["temperature"],
        pressure_mpa=request["pressure_mpa"],
        mean_count=mean_count, count_error=error,
        absolute_density_g_cm3=absolute,
        excess_density_g_cm3=absolute - bulk * accessible,
        weight_percent=100.0 * hydrogen_g / (hydrogen_g + carbon_g),
        bulk_density_g_cm3=bulk, carbon_mass_g=carbon_g,
        void_fraction=accessible,
        equilibration_steps=request["equilibration"],
        production_steps=request["production"],
        warnings=warnings,
    )


def lammps_executable() -> str:
    """The LAMMPS binary: ``$LAMMPS_EXECUTABLE``, else ``lmp`` on PATH.

    Raises
    ------
    FileNotFoundError
        If neither is there.
    """
    found = os.environ.get("LAMMPS_EXECUTABLE") or shutil.which("lmp")
    if not found:
        raise FileNotFoundError(
            "no LAMMPS executable: set LAMMPS_EXECUTABLE or put `lmp` on "
            "PATH. The PyPI wheel (`pip install lammps mpich`) includes "
            "fix gcmc.")
    return found


def run_lammps_gcmc(framework: Atoms, folder: Path, temperature: float,
                    pressure_mpa: float, **options) -> GCMCResult:
    """Write the folder, run LAMMPS in it, read the result."""
    folder = write_lammps_gcmc(framework, folder, temperature, pressure_mpa,
                               **options)
    with open(folder / "lammps.out", "w") as log:
        subprocess.run([lammps_executable(), "-in", "in.gcmc", "-log",
                        "log.lammps"], cwd=folder, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return read_lammps_gcmc(folder)
