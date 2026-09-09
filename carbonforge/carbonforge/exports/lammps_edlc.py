"""Writing a constant-potential EDLC simulation for LAMMPS.

This is a different kind of output from :mod:`carbonforge.exports.lammps`,
which writes a neutral reactive-potential run. An electric double layer is
electrostatics, so everything changes:

===================  ==========================  ============================
                     Neutral carbon (AIREBO)     EDLC (this module)
===================  ==========================  ============================
``atom_style``       ``atomic``                  ``full`` (carries charge)
Potential            AIREBO / Tersoff            ``lj/cut/coul/long``
Long range           none                        PPPM + slab correction
Electrodes           ordinary atoms              constant potential (ELECTRODE)
Rigid molecules      none                        SHAKE on water
===================  ==========================  ============================

A reactive potential cannot be used here at all: AIREBO and Tersoff carry no
charges, and without charges there is no double layer.

The generated script requires the **ELECTRODE package**, which is not built
into LAMMPS by default. The header says so, since discovering it from
``ERROR: Unrecognized fix style`` is a poor use of an afternoon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..forcefields.edlc import EDLCCell, check_edlc_setup
from ..forcefields.parameters import (
    describe_provenance,
    get_params,
    mixing_note,
)

_ATOMIC_MASSES: dict[str, float] = {
    "C": 12.011, "N": 14.007, "O": 15.999, "H": 1.008,
    "S": 32.065, "B": 10.811, "P": 30.974,
    "Na": 22.990, "Cl": 35.453, "K": 39.098, "Li": 6.941,
}

#: Coarse-grained ion masses, which are whole molecules rather than elements.
_CG_MASSES: dict[str, float] = {"BMIM": 139.22, "PF6": 144.96}


@dataclass
class EDLCSettings:
    """Run parameters for a constant-potential EDLC simulation.

    Attributes
    ----------
    temperature_k
        Thermostat temperature.
    timestep_fs
        1.0 fs is safe with rigid water; without SHAKE the O-H stretch needs
        0.5 fs or less.
    equilibration_steps, production_steps
        Only production is measured. An electrode double layer needs a long
        equilibration — the ions have to migrate across the cell — so the
        default is deliberately generous.
    coulomb_cutoff, lj_cutoff
        Real-space cutoffs in Å.
    pppm_accuracy
        Relative force accuracy for the k-space solver.
    electrode_eta
        Gaussian width of the electrode charge distribution, in 1/Å. 1.979
        is the value used in most constant-potential work and the ELECTRODE
        package default.
    slab_factor
        Vacuum multiplier for ``kspace_modify slab``. 3.0 is standard.
    rigid_water
        Constrain water with SHAKE, which is what allows a 1 fs timestep.
    dump_every
        Trajectory interval during production.
    """

    temperature_k: float = 298.0
    timestep_fs: float = 1.0
    equilibration_steps: int = 500000
    production_steps: int = 2000000
    coulomb_cutoff: float = 12.0
    lj_cutoff: float = 12.0
    pppm_accuracy: float = 1.0e-5
    electrode_eta: float = 1.979
    slab_factor: float = 3.0
    rigid_water: bool = True
    dump_every: int = 5000

    def __post_init__(self) -> None:
        if self.timestep_fs <= 0:
            raise ValueError("timestep_fs debe ser positivo.")
        if self.rigid_water and self.timestep_fs > 2.0:
            raise ValueError(
                f"timestep_fs={self.timestep_fs} fs es demasiado incluso con "
                "agua rígida. Con SHAKE lo habitual es 1-2 fs."
            )
        if not self.rigid_water and self.timestep_fs > 0.5:
            raise ValueError(
                f"Sin SHAKE, el estiramiento O-H (~3400 cm-1, periodo ~10 fs) "
                f"necesita un paso <=0.5 fs, no {self.timestep_fs} fs. "
                "Activa rigid_water o baja el paso."
            )
        if self.electrode_eta <= 0:
            raise ValueError("electrode_eta debe ser positivo.")


def _mass_for(type_name: str) -> float:
    if type_name in _CG_MASSES:
        return _CG_MASSES[type_name]
    element = get_params(type_name)
    # Types are named <Element>_<environment>, except the electrolyte ones.
    for symbol in ("Na", "Cl", "Li", "K"):
        if type_name == symbol:
            return _ATOMIC_MASSES[symbol]
    if type_name in ("OW",):
        return _ATOMIC_MASSES["O"]
    if type_name in ("HW",):
        return _ATOMIC_MASSES["H"]
    prefix = type_name.split("_")[0]
    return _ATOMIC_MASSES.get(prefix, 12.011)


def write_edlc_data(
    cell: EDLCCell,
    path: str | Path,
) -> Path:
    """Write the ``atom_style full`` data file.

    Unlike the neutral writer, every atom carries a charge and a molecule id,
    because SHAKE and the rigid-body machinery need to know which atoms
    belong to the same molecule.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    types = cell.types
    unique = cell.unique_types
    type_index = {name: i + 1 for i, name in enumerate(unique)}

    box = np.diag(np.array(cell.atoms.cell))
    positions = cell.atoms.get_positions()

    lines = [
        "LAMMPS data file for a constant-potential EDLC cell "
        "(carbonforge)",
        "",
        f"{len(cell.atoms)} atoms",
        f"{len(unique)} atom types",
        "",
        f"0.0 {box[0]:.8f} xlo xhi",
        f"0.0 {box[1]:.8f} ylo yhi",
        f"0.0 {box[2]:.8f} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for name in unique:
        lines.append(f"{type_index[name]} {_mass_for(name):.4f}  # {name}")

    lines += ["", "Atoms  # full", ""]
    for index, (name, position) in enumerate(zip(types, positions), start=1):
        params = get_params(name)
        molecule = cell.molecule_ids[index - 1] if cell.molecule_ids else 0
        lines.append(
            f"{index} {molecule} {type_index[name]} {params.charge:.6f} "
            f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}"
        )

    path.write_text("\n".join(lines) + "\n")
    return path


def write_edlc_input(
    cell: EDLCCell,
    path: str | Path,
    settings: Optional[EDLCSettings] = None,
    data_filename: str = "data.edlc",
) -> Path:
    """Write the LAMMPS input script for a constant-potential run."""
    s = settings or EDLCSettings()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    unique = cell.unique_types
    type_index = {name: i + 1 for i, name in enumerate(unique)}

    bottom_types = sorted({cell.types[i] for i in cell.groups.get("bottom", [])})
    water_types = [n for n in ("OW", "HW") if n in type_index]
    has_water = len(water_types) == 2

    lines = [
        "# Constant-potential EDLC simulation, generated by carbonforge.",
        "#",
        "# REQUIRES the ELECTRODE package:",
        "#   cmake -D PKG_ELECTRODE=yes ...   (or 'make yes-electrode')",
        "# Without it, 'fix electrode/conp' is an unrecognised fix style.",
        "#",
        "# A reactive potential (AIREBO, Tersoff) CANNOT be used here: those",
        "# carry no charges, and without charges there is no double layer.",
        "",
        "units           real",
        "atom_style      full",
        "# Periodic in the plane, finite along z: the slab geometry the",
        "# k-space correction below assumes.",
        "boundary        p p f",
        "",
        f"read_data       {data_filename}",
        "",
        "# --- interactions -------------------------------------------------",
        f"pair_style      lj/cut/coul/long {s.lj_cutoff} {s.coulomb_cutoff}",
        "pair_modify     mix geometric tail no",
    ]

    for name in unique:
        params = get_params(name)
        lines.append(
            f"pair_coeff      {type_index[name]} {type_index[name]} "
            f"{params.epsilon:.5f} {params.sigma:.4f}   # {name}"
        )

    lines += [
        "",
        f"kspace_style    pppm {s.pppm_accuracy:g}",
        "# The cell is periodic in x and y only, and develops a net dipole",
        "# once the double layer forms. Without this correction the Ewald sum",
        "# would include spurious interactions with slab images along z.",
        f"kspace_modify   slab {s.slab_factor}",
        "",
        "# --- groups ---------------------------------------------------------",
    ]

    bottom = cell.groups.get("bottom", [])
    top = cell.groups.get("top", [])
    electrolyte = cell.groups.get("electrolyte", [])
    if bottom:
        lines.append(f"group           bottom id 1:{len(bottom)}")
    if electrolyte:
        lines.append(
            f"group           electrolyte id {electrolyte[0] + 1}:"
            f"{electrolyte[-1] + 1}"
        )
    if top:
        lines.append(f"group           top id {top[0] + 1}:{top[-1] + 1}")
    lines.append("group           electrodes union bottom top")

    if has_water:
        lines += [
            "",
            f"group           water type {type_index['OW']} {type_index['HW']}",
        ]

    lines += [
        "",
        "neighbor        2.0 bin",
        "neigh_modify    every 1 delay 0 check yes",
        "",
        "thermo          1000",
        "thermo_style    custom step temp pe ke etotal press",
        "",
        "# --- constraints ------------------------------------------------",
        "# The electrodes are held fixed: their atoms provide the surface and",
        "# their CHARGES are what respond, not their positions.",
        "fix             freeze electrodes setforce 0.0 0.0 0.0",
    ]

    if has_water and s.rigid_water:
        lines += [
            "# SPC/E is a rigid model by construction; SHAKE enforces that and",
            "# is what makes a 1 fs timestep safe.",
            f"fix             rigid_water water shake 1.0e-4 20 0 "
            f"b 1 a 1",
            "# NOTE: adjust the 'b'/'a' indices to your bond and angle types.",
            "#       An alternative that needs no bond types is:",
            "#         fix rigid_water water rigid/small molecule",
        ]

    lines += [
        "",
        "# --- relax the starting configuration -----------------------------",
        "# The electrolyte was built by placing molecules on a jittered",
        "# lattice, which is close but not relaxed. Minimising first avoids a",
        "# first dynamics step with large forces. The electrodes are frozen,",
        "# so only the liquid moves.",
        "minimize        1.0e-4 1.0e-6 5000 50000",
        "",
        "# --- constant potential -------------------------------------------",
        "# This is the methodological heart of an EDLC simulation. In a real",
        "# capacitor the electrodes sit at fixed POTENTIAL and their charge",
        "# fluctuates in response to the electrolyte. Fixing the charge",
        "# instead samples a different ensemble and gives a capacitance that",
        "# can be wrong by tens of percent, worst at high potential.",
        "#",
        f"# The two electrodes are held at ±{cell.potential_v / 2:.3f} V so the",
        "# cell stays neutral overall.",
        f"fix             cpm bottom electrode/conp "
        f"{-cell.potential_v / 2:.4f} {s.electrode_eta} "
        f"couple top {cell.potential_v / 2:.4f} symm on",
        "",
        f"velocity        electrolyte create {s.temperature_k} 4928459 "
        "mom yes rot yes dist gaussian",
        f"timestep        {s.timestep_fs / 1000.0}",
        "",
        "# --- equilibration (discarded) -------------------------------------",
        "# The ions have to migrate across the cell to form the double layer,",
        "# which is slow. Measuring before that has settled gives a",
        "# capacitance for a half-formed interface.",
        f"fix             equil electrolyte nvt temp {s.temperature_k} "
        f"{s.temperature_k} $(100.0*dt)",
        f"run             {s.equilibration_steps}",
        "unfix           equil",
        "",
        "# --- production (measured) ------------------------------------------",
        "reset_timestep  0",
        f"fix             prod electrolyte nvt temp {s.temperature_k} "
        f"{s.temperature_k} $(100.0*dt)",
        "",
        "# The electrode charge is the observable: its mean gives the stored",
        "# charge, and C = Q / V follows.",
        "variable        qbot equal f_cpm[1]",
        "variable        qtop equal f_cpm[2]",
        "fix             qlog all ave/time 10 100 1000 v_qbot v_qtop "
        "file electrode_charge.dat",
        "",
        "# Density profiles along z show the double-layer structure: where the",
        "# counter-ions pile up and how the solvent orders against the wall.",
        "compute         chunkz all chunk/atom bin/1d z lower 0.5 units box",
        "fix             density all ave/chunk 100 100 10000 chunkz density/mass "
        "file density_profile.dat",
    ]

    if s.dump_every > 0:
        lines += [
            f"dump            traj all custom {s.dump_every} traj.lammpstrj "
            "id mol type q x y z",
            "dump_modify     traj sort id",
        ]

    lines += [
        f"run             {s.production_steps}",
        "",
        "write_data      final.data",
        "",
        "# Analysis:",
        "#   electrode_charge.dat -> mean charge Q; capacitance C = Q / V",
        f"#   with V = {cell.potential_v:.3f} V across the cell",
        "#   density_profile.dat  -> ion and solvent distribution vs z",
    ]

    path.write_text("\n".join(lines) + "\n")
    return path


def write_edlc(
    cell: EDLCCell,
    outdir: str | Path,
    settings: Optional[EDLCSettings] = None,
) -> dict[str, Path]:
    """Write a complete constant-potential EDLC project.

    Parameters
    ----------
    cell
        The assembled cell from
        :func:`~carbonforge.forcefields.edlc.build_edlc_cell`.
    outdir
        Destination directory.
    settings
        Run parameters.

    Returns
    -------
    dict
        Maps ``"data"``, ``"input"`` and ``"notes"`` to the files written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    s = settings or EDLCSettings()

    written = {
        "data": write_edlc_data(cell, outdir / "data.edlc"),
        "input": write_edlc_input(cell, outdir / "in.edlc", settings=s),
    }

    # A record of where the parameters came from and what to watch out for,
    # so the choices survive past this session.
    notes = [
        cell.summary(),
        "",
        "=" * 70,
        describe_provenance(cell.unique_types),
        "",
        "=" * 70,
        mixing_note(),
    ]
    warnings = check_edlc_setup(cell)
    if warnings:
        notes += ["", "=" * 70, "AVISOS SOBRE ESTE MONTAJE:", ""]
        notes += [f"  ⚠️  {w}\n" for w in warnings]

    notes += [
        "",
        "=" * 70,
        "REQUISITOS PARA EJECUTARLO",
        "",
        "  1. LAMMPS compilado con el paquete ELECTRODE.",
        "  2. Los índices de tipo de enlace y ángulo en la línea de SHAKE",
        "     dependen de cómo definas el agua: ajústalos, o usa",
        "     'fix rigid/small molecule', que no los necesita.",
        "  3. La capacitancia sale de electrode_charge.dat: C = <Q> / V.",
        "     Usa solo la etapa de producción, nunca la de equilibración.",
    ]

    notes_path = outdir / "NOTAS_EDLC.txt"
    notes_path.write_text("\n".join(notes) + "\n")
    written["notes"] = notes_path
    return written
