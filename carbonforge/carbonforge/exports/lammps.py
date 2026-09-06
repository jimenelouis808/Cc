"""LAMMPS data + input script writer.

Writes an atom-style ``atomic`` data file (masses, types, box, coords) and a
companion input script with a basic minimisation → NVT → (optional) NPT
workflow using AIREBO for pure-C systems or Tersoff/LCBOP as fallback hints.

The focus is on correctness of the data file (box, tilt factors for
non-orthogonal cells, atom types, masses). Users are expected to customise
the input script for their specific potential / ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

from ..validation.checks import run_basic_checks


_ATOMIC_MASSES: dict[str, float] = {
    "C": 12.011,
    "N": 14.007,
    "B": 10.811,
    "S": 32.065,
    "P": 30.974,
    "H": 1.008,
    "O": 15.999,
}


@dataclass
class LAMMPSSettings:
    """User-facing knobs for the generated LAMMPS input script.

    The script separates **equilibration** from **production**, which is not
    cosmetic: averaging a thermodynamic quantity over the equilibration
    transient biases it, sometimes badly. Only the production stage writes a
    trajectory and accumulates averages.

    Attributes
    ----------
    mode
        ``"equilibrate"`` runs minimise → NVT equilibration → NVT production.
        ``"anneal"`` inserts a heat-and-quench cycle, which is how a
        realistic amorphous or foam-like carbon is actually produced: a
        stochastic packing is not a physical structure until it has been
        melted and cooled.
    equilibration_steps, production_steps
        Steps in each stage. Equilibration output is discarded.
    anneal_temperature_k, anneal_steps, quench_steps
        The annealing cycle: hold at high temperature, then cool to the
        target. Slower quenches give more ordered structures.
    dump_every
        Trajectory frame interval during production. 0 disables the dump.
    """

    timestep_fs: float = 0.5
    minimize_etol: float = 1.0e-6
    minimize_ftol: float = 1.0e-6
    nvt_temperature_k: float = 300.0
    mode: str = "equilibrate"
    equilibration_steps: int = 20000
    production_steps: int = 50000
    anneal_temperature_k: float = 3000.0
    anneal_steps: int = 20000
    quench_steps: int = 50000
    dump_every: int = 1000
    run_npt: bool = False
    npt_pressure_bar: float = 1.0
    npt_steps: int = 10000
    pair_style: str = "airebo 3.0 1 1"
    pair_coeff: Optional[str] = None  # if None, auto-generated for C-only

    # Kept for backwards compatibility with earlier scripts.
    nvt_steps: int = 10000

    def __post_init__(self) -> None:
        if self.mode not in ("equilibrate", "anneal", "minimize"):
            raise ValueError(
                f"mode desconocido: '{self.mode}'. "
                "Opciones: minimize, equilibrate, anneal."
            )
        if self.timestep_fs <= 0:
            raise ValueError("timestep_fs debe ser positivo.")
        if self.timestep_fs > 2.0:
            raise ValueError(
                f"timestep_fs={self.timestep_fs} fs es demasiado grande para "
                "carbono: las vibraciones C-C están cerca de 1600 cm-1 "
                "(~21 fs de periodo) y hacen falta al menos ~20 pasos por "
                "periodo. Usa 0.5-1.0 fs."
            )


def _cell_to_lammps_box(cell: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Convert an ASE (3,3) cell to LAMMPS box parameters.

    LAMMPS requires a lower-triangular cell (a along x, b in xy plane, c
    general). We assume the caller has already provided that convention
    (our builders always do). Otherwise we rotate via QR decomposition.
    """
    a, b, c = cell[0], cell[1], cell[2]
    # Rotate so a is along x, b in xy plane.
    xhi = float(np.linalg.norm(a))
    x_hat = a / xhi
    xy = float(np.dot(b, x_hat))
    yhi = float(np.linalg.norm(b - xy * x_hat))
    y_hat = (b - xy * x_hat) / (yhi if yhi > 1e-12 else 1.0)
    xz = float(np.dot(c, x_hat))
    yz = float(np.dot(c, y_hat))
    z_hat = np.cross(x_hat, y_hat)
    zhi = float(np.dot(c, z_hat))
    return (xhi, yhi, zhi), (xy, xz, yz), (x_hat, y_hat, z_hat)  # type: ignore[return-value]


def _rotate_positions(positions: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Rotate positions into LAMMPS basis."""
    R = np.vstack(basis)  # rows: x_hat, y_hat, z_hat
    return positions @ R.T


def write_lammps(
    atoms: Atoms,
    outdir: str | Path,
    settings: Optional[LAMMPSSettings] = None,
    data_filename: str = "data.lammps",
    input_filename: str = "in.lammps",
    force: bool = False,
) -> tuple[Path, Path]:
    """Write a LAMMPS data file and an input script for the structure.

    Parameters
    ----------
    atoms
        Structure to export.
    outdir
        Destination directory (created if missing).
    settings
        Optional :class:`LAMMPSSettings`.
    data_filename, input_filename
        Output file names.
    force
        If ``False``, abort when validation fails.

    Returns
    -------
    (Path, Path)
        Paths to the written data and input files.
    """
    report = run_basic_checks(atoms)
    if not report.ok and not force:
        raise ValueError(
            "Structure failed validation, refusing to export:\n" + report.summary()
        )

    s = settings or LAMMPSSettings()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    symbols = atoms.get_chemical_symbols()
    unique = sorted(set(symbols))
    type_of = {sym: i + 1 for i, sym in enumerate(unique)}

    (xhi, yhi, zhi), (xy, xz, yz), basis = _cell_to_lammps_box(np.array(atoms.cell))
    positions = _rotate_positions(atoms.get_positions(), basis)

    # ---- Data file -------------------------------------------------------
    data_lines = [
        "LAMMPS data file generated by carbonforge",
        "",
        f"{len(atoms)} atoms",
        f"{len(unique)} atom types",
        "",
        f"0.0 {xhi:.8f} xlo xhi",
        f"0.0 {yhi:.8f} ylo yhi",
        f"0.0 {zhi:.8f} zlo zhi",
    ]
    if abs(xy) > 1e-8 or abs(xz) > 1e-8 or abs(yz) > 1e-8:
        data_lines.append(f"{xy:.8f} {xz:.8f} {yz:.8f} xy xz yz")
    data_lines.extend(["", "Masses", ""])
    for sym in unique:
        data_lines.append(f"{type_of[sym]} {_ATOMIC_MASSES.get(sym, 1.0):.4f}  # {sym}")
    data_lines.extend(["", "Atoms  # atomic", ""])
    for idx, (sym, p) in enumerate(zip(symbols, positions), start=1):
        data_lines.append(
            f"{idx} {type_of[sym]} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}"
        )

    data_path = outdir / data_filename
    data_path.write_text("\n".join(data_lines) + "\n")

    # ---- Input script ----------------------------------------------------
    pair_coeff = s.pair_coeff
    if pair_coeff is None:
        if unique == ["C"]:
            # AIREBO potential file is called CH.airebo in the LAMMPS distro.
            pair_coeff = "* * CH.airebo C"
        else:
            pair_coeff = "* * REPLACE_ME " + " ".join(unique)

    boundaries = " ".join("p" if pb else "f" for pb in atoms.get_pbc())

    timestep_ps = s.timestep_fs / 1000.0
    input_lines = [
        "# LAMMPS input script generated by carbonforge",
        "units           metal",
        "atom_style      atomic",
        f"boundary        {boundaries}",
        "",
        f"read_data       {data_filename}",
        "",
        f"pair_style      {s.pair_style}",
        f"pair_coeff      {pair_coeff}",
        "",
        "neighbor        2.0 bin",
        "neigh_modify    every 1 delay 0 check yes",
        "",
        "thermo          100",
        "thermo_style    custom step temp pe ke etotal press vol",
        "",
        "# --- 1. Minimisation ---------------------------------------------",
        f"minimize        {s.minimize_etol} {s.minimize_ftol} 10000 100000",
        "write_data      minimised.data",
        "",
        f"timestep        {timestep_ps}",
        f"# {s.timestep_fs} fs: the C-C stretch near 1600 cm-1 has a ~21 fs",
        "# period, so this keeps ~40 steps per period.",
        "",
    ]

    if s.mode == "minimize":
        input_lines += ["write_data      final.data"]
        input_path = outdir / input_filename
        input_path.write_text("\n".join(input_lines) + "\n")
        return data_path, input_path

    input_lines += [
        f"velocity        all create {s.nvt_temperature_k} 12345 "
        "mom yes rot yes dist gaussian",
        "",
    ]

    if s.mode == "anneal":
        input_lines += [
            "# --- 2. Anneal -----------------------------------------------",
            "# A randomly packed structure is not physical until it has been",
            "# melted and cooled: this is what turns it into a real network.",
            f"fix             heat all nvt temp {s.nvt_temperature_k} "
            f"{s.anneal_temperature_k} $(100.0*dt)",
            f"run             {s.anneal_steps}",
            "unfix           heat",
            "",
            "# --- 3. Quench ------------------------------------------------",
            "# A slower quench gives a more ordered network. If the result",
            "# looks too amorphous, lengthen this stage rather than the heat.",
            f"fix             quench all nvt temp {s.anneal_temperature_k} "
            f"{s.nvt_temperature_k} $(100.0*dt)",
            f"run             {s.quench_steps}",
            "unfix           quench",
            "write_data      annealed.data",
            "",
        ]

    input_lines += [
        "# --- Equilibration (discarded) ------------------------------------",
        "# Nothing here is measured: averaging over the transient would bias",
        "# every quantity you take from the run.",
        f"fix             equil all nvt temp {s.nvt_temperature_k} "
        f"{s.nvt_temperature_k} $(100.0*dt)",
        f"run             {s.equilibration_steps}",
        "unfix           equil",
        "",
        "# --- Production (measured) ----------------------------------------",
        "reset_timestep  0",
    ]

    if s.dump_every > 0:
        input_lines += [
            f"dump            traj all custom {s.dump_every} "
            "traj.lammpstrj id type x y z",
            "dump_modify     traj sort id",
        ]

    ensemble = "npt" if s.run_npt else "nvt"
    if s.run_npt:
        input_lines.append(
            f"fix             prod all npt temp {s.nvt_temperature_k} "
            f"{s.nvt_temperature_k} $(100.0*dt) "
            f"iso {s.npt_pressure_bar} {s.npt_pressure_bar} $(1000.0*dt)"
        )
    else:
        input_lines.append(
            f"fix             prod all nvt temp {s.nvt_temperature_k} "
            f"{s.nvt_temperature_k} $(100.0*dt)"
        )

    input_lines += [
        "fix             averages all ave/time 10 100 1000 "
        "c_thermo_temp c_thermo_pe c_thermo_press file averages.dat",
        f"run             {s.production_steps}",
        "unfix           prod",
        "unfix           averages",
        "",
        f"# Ensemble: {ensemble.upper()}. Averages in averages.dat, "
        "trajectory in traj.lammpstrj.",
        "write_data      final.data",
    ]

    input_path = outdir / input_filename
    input_path.write_text("\n".join(input_lines) + "\n")

    return data_path, input_path
