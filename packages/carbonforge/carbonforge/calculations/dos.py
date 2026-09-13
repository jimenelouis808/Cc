"""Density of states, total and projected.

For a doped or functionalised carbon the band structure answers "is there a
gap?", but the question that usually matters is "**which atoms** put states
near the Fermi level?". That is the projected density of states, and for
nitrogen in graphene it is what distinguishes the configurations: graphitic N
donates into the π* conduction states, while pyridinic N leaves a localised
level associated with its lone pair.

The Quantum ESPRESSO workflow is::

    pw.x (scf)  →  pw.x (nscf, denser mesh)  →  dos.x  →  projwfc.x

Two details bite people:

* **The nscf step needs a denser k-mesh than the scf.** A mesh good enough to
  converge the charge density is not good enough to resolve a DOS; too coarse
  and the curve is a row of smearing bumps rather than a density. The factor
  defaults to 2 here.
* **Summed PDOS does not exactly equal the total DOS.** The projection is
  onto atomic orbitals, which do not span the full plane-wave basis, so a few
  percent goes missing. Reporting "N contributes 12 % of the states at E_F"
  is fine; claiming the projections sum to unity is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: QE smearing types for dos.x / projwfc.x, by ``ngauss`` code.
SMEARING_TYPES: dict[int, str] = {
    0: "Simple Gaussian",
    1: "Methfessel-Paxton de primer orden",
    -1: "Marzari-Vanderbilt (cold smearing)",
    -99: "Fermi-Dirac",
}


@dataclass
class DOSSpec:
    """Settings for a density-of-states calculation.

    Attributes
    ----------
    energy_min, energy_max
        Energy window in eV, **absolute** (not relative to E_F), because that
        is what ``dos.x`` expects. ``None`` lets QE choose from the computed
        eigenvalue range.
    delta_e
        Energy step of the output grid, in eV. 0.01-0.02 eV is usual; finer
        than the smearing width buys nothing.
    degauss
        Broadening in Ry. 0.01 Ry ≈ 0.14 eV. Too large smooths away the
        features you are looking for; too small turns the finite k-mesh into
        visible spikes.
    ngauss
        Smearing type; see :data:`SMEARING_TYPES`.
    kmesh_factor
        How much denser the nscf mesh is than the scf one, per periodic axis.
    projected
        Also run ``projwfc.x`` for the per-atom, per-orbital projection.
    """

    energy_min: Optional[float] = None
    energy_max: Optional[float] = None
    delta_e: float = 0.02
    degauss: float = 0.01
    ngauss: int = 0
    kmesh_factor: int = 2
    projected: bool = True

    def __post_init__(self) -> None:
        if self.delta_e <= 0:
            raise ValueError("delta_e debe ser positivo.")
        if self.degauss <= 0:
            raise ValueError("degauss debe ser positivo.")
        if self.kmesh_factor < 1:
            raise ValueError("kmesh_factor debe ser >= 1.")
        if self.ngauss not in SMEARING_TYPES:
            raise ValueError(
                f"ngauss={self.ngauss} no es válido. "
                f"Opciones: {sorted(SMEARING_TYPES)}."
            )
        if (
            self.energy_min is not None
            and self.energy_max is not None
            and self.energy_min >= self.energy_max
        ):
            raise ValueError("energy_min debe ser menor que energy_max.")

    @property
    def degauss_ev(self) -> float:
        """The broadening expressed in eV, which is how people think of it."""
        return self.degauss * 13.605693122994

    def describe(self) -> str:
        """Human-readable summary of what this will compute."""
        window = (
            f"{self.energy_min} a {self.energy_max} eV"
            if self.energy_min is not None and self.energy_max is not None
            else "rango automático"
        )
        lines = [
            f"Ventana de energía: {window}, paso {self.delta_e} eV",
            f"Ensanchamiento: {self.degauss} Ry ({self.degauss_ev:.3f} eV), "
            f"{SMEARING_TYPES[self.ngauss]}",
            f"Malla k del nscf: {self.kmesh_factor}x la del scf",
            f"Proyectada por átomo y orbital: {'sí' if self.projected else 'no'}",
        ]
        if self.delta_e > self.degauss_ev:
            lines.append(
                "⚠️  El paso de energía es mayor que el ensanchamiento: "
                "estarás submuestreando la curva."
            )
        return "\n".join(lines)


def format_dos_input(
    spec: DOSSpec,
    prefix: str = "pwscf",
    outdir: str = "./out",
    fildos: str = "dos.dat",
) -> str:
    """Render a ``dos.x`` input for the total density of states."""
    lines = ["&DOS"]
    lines.append(f"    prefix = '{prefix}'")
    lines.append(f"    outdir = '{outdir}'")
    lines.append(f"    fildos = '{fildos}'")
    lines.append(f"    DeltaE = {spec.delta_e}")
    lines.append(f"    degauss = {spec.degauss}")
    lines.append(f"    ngauss = {spec.ngauss}")
    if spec.energy_min is not None:
        lines.append(f"    Emin = {spec.energy_min}")
    if spec.energy_max is not None:
        lines.append(f"    Emax = {spec.energy_max}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def format_projwfc_input(
    spec: DOSSpec,
    prefix: str = "pwscf",
    outdir: str = "./out",
    filpdos: str = "pdos",
) -> str:
    """Render a ``projwfc.x`` input for the projected density of states.

    ``projwfc.x`` writes one file per atom and orbital
    (``pdos.pdos_atm#1(C)_wfc#1(s)`` and so on) plus a summed
    ``pdos.pdos_tot``. :mod:`carbonforge.results.dos` reads the lot and
    groups it by element.
    """
    lines = ["&PROJWFC"]
    lines.append(f"    prefix = '{prefix}'")
    lines.append(f"    outdir = '{outdir}'")
    lines.append(f"    filpdos = '{filpdos}'")
    lines.append(f"    DeltaE = {spec.delta_e}")
    lines.append(f"    degauss = {spec.degauss}")
    lines.append(f"    ngauss = {spec.ngauss}")
    if spec.energy_min is not None:
        lines.append(f"    Emin = {spec.energy_min}")
    if spec.energy_max is not None:
        lines.append(f"    Emax = {spec.energy_max}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def format_dos_runner(spec: DOSSpec, ncores: int = 4) -> str:
    """Render a shell script chaining the DOS workflow."""
    steps = [
        "#!/usr/bin/env bash",
        "# Generated by carbonforge. Run from this directory.",
        "set -euo pipefail",
        "",
        f'NCORES="${{NCORES:-{ncores}}}"',
        'MPI="${MPI:-mpirun -np $NCORES}"',
        "",
        "echo '[1/3] SCF'",
        "$MPI pw.x -in pw.scf.in > pw.scf.out",
        "",
        f"echo '[2/3] NSCF (malla {spec.kmesh_factor}x más densa)'",
        "$MPI pw.x -in pw.nscf.in > pw.nscf.out",
        "",
        "echo '[3/3] Densidad de estados'",
        "dos.x -in dos.in > dos.out",
    ]
    if spec.projected:
        steps += [
            "projwfc.x -in projwfc.in > projwfc.out",
            "",
            "echo 'Listo. Analiza con:'",
            "echo '  carbonforge plot-dos . --fermi <E_F de pw.scf.out>'",
        ]
    else:
        steps += [
            "",
            "echo 'Listo. Analiza con:'",
            "echo '  carbonforge plot-dos dos.dat --fermi <E_F de pw.scf.out>'",
        ]
    return "\n".join(steps) + "\n"
