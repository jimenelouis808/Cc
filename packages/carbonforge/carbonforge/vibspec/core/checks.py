"""Physical sanity checks for a vibrational calculation on a finite flake.

These encode, as errors and warnings rather than as advice in a README, the
mistakes that produce a wrong IR spectrum while every step appears to run
fine:

* **Periodic model.** ``ase.vibrations.Infrared`` differentiates the dipole
  moment, which is not defined along a periodic axis. Only finite models.
* **Too little vacuum.** Under :data:`MIN_VACUUM_PER_SIDE` per side the
  molecule interacts with its periodic images through the cell.
* **Spin.** An odd electron count cannot be closed-shell, and zigzag edges
  of four or more sites carry a spin-polarised edge state. A spin-paired
  calculation of either gives the wrong potential energy surface, so the
  wrong frequencies, with no error. :func:`suggest_spin` says when, and
  gives initial magnetic moments (antiferromagnetic across the two
  sublattices for the edge state, which is where it converges). As for the
  periodic zigzag ribbons elsewhere in carbonforge, running such a system
  without spin is refused, not merely discouraged: if the moments vanish
  in the spin-polarised run, the system was closed-shell and nothing was
  lost.
* **Unrelaxed geometry.** Harmonic frequencies are only meaningful at a
  minimum. Residual forces show up as spurious low or imaginary modes and
  shift the rest; 0.05 eV/Å, a common relaxation default, is too loose.
* **Unformed pyrrolic ring.** The pyrrolic preset is a precursor; its N-H
  must sit in a pentagon after relaxation, or the calculation is of a
  different defect.
* **Numerical settings.** Displacement too small (noise) or too large
  (anharmonic), an LCAO grid coarse enough for the egg-box effect to break
  translational symmetry, a frequency scale factor outside any published
  range.

Every function returns a :class:`~carbonforge.validation.checks.ValidationReport`
or plain data; nothing here needs a DFT code installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from ase import Atoms

from ...builders.nanoribbon import MIN_VACUUM_PER_SIDE
from ...topology.graph import build_bond_graph
from ...validation.checks import ValidationReport, check_minimum_distances
from .sites import zigzag_runs

#: Recommended and largest acceptable residual force before vibrations, eV/Å.
FMAX_RECOMMENDED: float = 0.01
FMAX_LIMIT: float = 0.05

#: Finite-difference displacement window, Å. ASE's default is 0.01.
DELTA_RANGE: tuple[float, float] = (0.005, 0.03)

#: Largest real-space grid spacing, Å, before the LCAO egg-box effect
#: noticeably breaks the translational invariance the acoustic modes need.
LCAO_MAX_H: float = 0.18

#: Frequency scale factors outside this range do not correspond to any
#: functional-basis combination in the usual tabulations (CCCBDB, Merrick
#: et al. 2007); a value outside it is almost certainly a typo or the inverse.
SCALE_FACTOR_RANGE: tuple[float, float] = (0.90, 1.05)

#: Relaxation moves the outermost atoms, so the vacuum measured after it may
#: be a little under what was prepared. Before relaxing, a margin this large
#: above the minimum is asked for (as a warning); after relaxing, the
#: minimum is enforced this much more leniently. Re-boxing after the
#: relaxation is not an option: a new cell means a new grid, and the relaxed
#: forces would no longer be zero on it.
RELAX_VACUUM_SLACK: float = 0.5

#: Zigzag edges at least this many sites long get a spin-polarised check.
ZIGZAG_MAGNETIC_RUN: int = 4

_FRAMEWORK = ("C", "N", "B")


def electron_count(atoms: Atoms, charge: int = 0) -> int:
    """Total number of electrons (all-electron count; parity equals the valence one)."""
    return int(atoms.get_atomic_numbers().sum()) - int(charge)


def vacuum_per_side(atoms: Atoms) -> dict[int, float]:
    """Smallest gap between the atoms and the cell boundary, per non-periodic axis."""
    positions = atoms.get_positions()
    cell = np.array(atoms.cell)
    gaps = {}
    for axis, periodic in enumerate(atoms.get_pbc()):
        if periodic:
            continue
        low = float(positions[:, axis].min())
        high = float(cell[axis, axis] - positions[:, axis].max())
        gaps[axis] = min(low, high)
    return gaps


def undercoordinated_atoms(atoms: Atoms) -> list[int]:
    """Carbons with fewer than three bonds: dangling bonds, i.e. radical sites.

    The sp carbon of a nitrile has two neighbours and a full valence (its
    triple bond to a terminal N), so it is not counted.
    """
    graph = build_bond_graph(atoms)
    symbols = atoms.get_chemical_symbols()

    def is_nitrile_carbon(i: int) -> bool:
        return any(symbols[n] == "N" and graph.degree[n] == 1 for n in graph.neighbors(i))

    return [
        i for i, s in enumerate(symbols)
        if s == "C" and graph.degree[i] < 3 and not is_nitrile_carbon(i)
    ]


def unformed_pyrrolic_nitrogens(atoms: Atoms) -> list[int]:
    """N-H nitrogens of a pyrrolic precursor that are not yet in a five-membered ring.

    Empty when the structure has no pyrrolic precursor, or when every one of
    them has closed its pentagon.
    """
    wanted = {
        int(i)
        for entry in atoms.info.get("nitrogen_configurations", [])
        if entry.get("type") == "pyrrolic_precursor"
        for i in entry.get("indices", [])
    }
    if not wanted:
        return []
    graph = build_bond_graph(atoms)
    symbols = atoms.get_chemical_symbols()
    framework = graph.subgraph(i for i, s in enumerate(symbols) if s in _FRAMEWORK)
    in_pentagon = {i for cycle in nx.minimum_cycle_basis(framework) if len(cycle) == 5
                   for i in cycle}
    return sorted(i for i in wanted if i not in in_pentagon)


@dataclass
class SpinAdvice:
    """Whether to run spin-polarised, and where to start the moments.

    ``magmoms`` is per atom, ready for ``atoms.set_initial_magnetic_moments``.
    """

    spinpol: bool
    magmoms: np.ndarray
    reasons: list[str] = field(default_factory=list)


def suggest_spin(atoms: Atoms, charge: int = 0) -> SpinAdvice:
    """Decide whether the system needs a spin-polarised calculation.

    Three independent reasons, all reported:

    * an **odd electron count** -- a doublet at least, no choice about it;
    * **dangling bonds** (two-coordinated carbons) -- radical centres;
    * **zigzag edges** of :data:`ZIGZAG_MAGNETIC_RUN` sites or more -- the
      edge state is spin-polarised, antiferromagnetically between opposite
      edges. Whether a given flake keeps the polarisation depends on its
      size; only a spin-polarised run can tell, and a spin-paired one would
      never find out.

    Initial moments: ±0.5 on zigzag-edge carbons with the sign of their
    sublattice (so opposite edges start antiparallel), and one extra
    electron's worth spread over the nitrogens (or the dangling carbons)
    when the count is odd.
    """
    symbols = atoms.get_chemical_symbols()
    magmoms = np.zeros(len(atoms))
    reasons: list[str] = []

    runs = [run for run in zigzag_runs(atoms) if len(run) >= ZIGZAG_MAGNETIC_RUN]
    if runs:
        graph = build_bond_graph(atoms)
        framework = graph.subgraph(i for i, s in enumerate(symbols) if s in _FRAMEWORK)
        try:
            colour = nx.bipartite.color(framework)
        except nx.NetworkXError:
            colour = {}
        for run in runs:
            for carbon in run:
                magmoms[carbon] = 0.5 if colour.get(carbon, 0) == 0 else -0.5
        lengths = ", ".join(str(len(r)) for r in runs)
        reasons.append(
            f"Bordes zigzag de {lengths} sitios: pueden tener un estado de borde "
            "polarizado (antiferromagnético entre bordes). Calcula con espín, con "
            "estos momentos iniciales; si convergen a cero, el sistema es de capa "
            "cerrada."
        )
        if not colour:
            reasons.append(
                "La red no es bipartita (hay anillos impares): los momentos "
                "iniciales se pusieron todos paralelos."
            )

    dangling = undercoordinated_atoms(atoms)
    if dangling:
        magmoms[dangling] += 1.0 / len(dangling)
        reasons.append(
            f"{len(dangling)} carbono(s) con enlaces colgantes {dangling}: son "
            "centros radicales."
        )

    n_electrons = electron_count(atoms, charge)
    if n_electrons % 2:
        hosts = [i for i, s in enumerate(symbols) if s == "N"] or dangling
        hosts = hosts or [i for i, s in enumerate(symbols) if s == "C"]
        magmoms[hosts] += 1.0 / len(hosts)
        reasons.append(
            f"Número impar de electrones ({n_electrons}): el sistema es de capa "
            "abierta y el cálculo sin espín no es válido."
        )

    return SpinAdvice(spinpol=bool(reasons), magmoms=magmoms, reasons=reasons)


def check_structure(
    atoms: Atoms,
    charge: int = 0,
    min_vacuum_per_side: float = MIN_VACUUM_PER_SIDE,
    comfortable_vacuum: Optional[float] = None,
) -> ValidationReport:
    """Everything about the *structure* that must hold before an IR calculation.

    Vacuum under ``min_vacuum_per_side`` is an error; under
    ``comfortable_vacuum`` (default: the minimum plus
    :data:`RELAX_VACUUM_SLACK`) a warning, since relaxation may still move
    the outermost atoms outwards.
    """
    if comfortable_vacuum is None:
        comfortable_vacuum = min_vacuum_per_side + RELAX_VACUUM_SLACK
    report = ValidationReport()

    if any(atoms.get_pbc()):
        report.errors.append(
            "La estructura es periódica: el IR por diferencias finitas del dipolo "
            "(ase.vibrations.Infrared) necesita un sistema finito. Usa "
            "build_finite_nanoribbon()."
        )

    for axis, gap in vacuum_per_side(atoms).items():
        report.info[f"vacuum_per_side_axis_{axis}"] = round(gap, 2)
        if gap < min_vacuum_per_side - 1e-6:
            report.errors.append(
                f"Eje {axis}: solo {gap:.2f} Å de vacío por lado (mínimo "
                f"{min_vacuum_per_side} Å). La molécula interactúa con sus imágenes."
            )
        elif gap < comfortable_vacuum - 1e-6:
            report.warnings.append(
                f"Eje {axis}: {gap:.2f} Å de vacío por lado, justo por encima del "
                f"mínimo; deja al menos {comfortable_vacuum} Å para que la relajación "
                "no se lo coma."
            )

    # Only its errors: its warnings are tuned for C-C and fire on every C-H.
    report.errors.extend(check_minimum_distances(atoms).errors)

    unformed = unformed_pyrrolic_nitrogens(atoms)
    if unformed:
        report.warnings.append(
            f"N pirrólico {unformed} todavía sin pentágono: es el precursor. "
            "Relájalo y vuelve a comprobarlo antes de calcular vibraciones."
        )

    advice = suggest_spin(atoms, charge)
    report.info["n_electrons"] = electron_count(atoms, charge)
    report.info["spin_polarised_recommended"] = str(advice.spinpol)
    report.warnings.extend(advice.reasons)
    return report


def check_vibration_settings(
    fmax: float,
    delta: float = 0.01,
    nfree: int = 2,
    mode: Optional[str] = None,
    h: Optional[float] = None,
    scale_factor: Optional[float] = None,
) -> ValidationReport:
    """Numerical settings of the relaxation and the finite differences.

    Parameters
    ----------
    fmax
        Force criterion of the relaxation that precedes the vibrations, eV/Å.
    delta
        Finite-difference displacement, Å.
    nfree
        Displacements per coordinate: 2 (central differences) or 4.
    mode
        GPAW mode, ``"lcao"``, ``"pw"`` or ``"fd"``, when known.
    h
        Real-space grid spacing, Å, when known.
    scale_factor
        Frequency scale factor that will be applied to the result.
    """
    report = ValidationReport()

    if fmax > FMAX_LIMIT:
        report.errors.append(
            f"fmax={fmax} eV/Å es demasiado alto para vibraciones: las fuerzas "
            "residuales crean modos espurios. Usa como máximo "
            f"{FMAX_LIMIT} y preferiblemente {FMAX_RECOMMENDED}."
        )
    elif fmax > FMAX_RECOMMENDED:
        report.warnings.append(
            f"fmax={fmax} eV/Å: aceptable para un barrido, pero para asignar "
            f"bandas usa {FMAX_RECOMMENDED} eV/Å o menos."
        )

    low, high = DELTA_RANGE
    if delta < low:
        report.warnings.append(
            f"delta={delta} Å es muy pequeño: la diferencia de fuerzas queda "
            "dominada por el ruido numérico de la SCF."
        )
    elif delta > high:
        report.warnings.append(
            f"delta={delta} Å es grande: entra anarmonicidad en la derivada."
        )

    if nfree not in (2, 4):
        report.errors.append(f"nfree={nfree}: ASE solo admite 2 o 4.")

    if mode is not None and mode.lower() == "lcao" and h is not None and h > LCAO_MAX_H:
        report.warnings.append(
            f"LCAO con h={h} Å: el efecto huevera rompe la invariancia "
            f"traslacional y ensucia los modos bajos. Usa h <= {LCAO_MAX_H} Å."
        )

    if scale_factor is not None:
        low, high = SCALE_FACTOR_RANGE
        if not low <= scale_factor <= high:
            report.warnings.append(
                f"Factor de escala {scale_factor} fuera de [{low}, {high}]: "
                "revisa que no sea el inverso o un error de tecleo."
            )
    return report


def check_ready_for_vibrations(
    atoms: Atoms,
    relaxed_fmax: Optional[float],
    spinpol: Optional[bool] = None,
    charge: int = 0,
) -> ValidationReport:
    """The gate in front of a vibrational calculation.

    Stricter than :func:`check_structure`: here an unrelaxed structure, a
    pyrrolic ring that did not close and a spin-paired run of a system that
    needs spin (odd electrons, dangling bonds, long zigzag edges) are errors,
    not warnings.

    Parameters
    ----------
    atoms
        The structure the vibrations will be computed on.
    relaxed_fmax
        Largest residual force after its relaxation, eV/Å, or ``None`` if
        it was not relaxed.
    spinpol
        Whether the calculation is spin-polarised, when known.
    """
    # After relaxing, the vacuum minimum is enforced with some slack (see
    # RELAX_VACUUM_SLACK) and "comfortable" means the minimum itself.
    report = check_structure(
        atoms, charge=charge,
        min_vacuum_per_side=MIN_VACUUM_PER_SIDE - RELAX_VACUUM_SLACK,
        comfortable_vacuum=MIN_VACUUM_PER_SIDE,
    )

    if relaxed_fmax is None:
        report.errors.append(
            "La estructura no está relajada: las frecuencias armónicas solo "
            "tienen sentido en un mínimo. Relaja antes de las vibraciones."
        )
    elif relaxed_fmax > FMAX_LIMIT:
        report.errors.append(
            f"Fuerza residual {relaxed_fmax:.3f} eV/Å > {FMAX_LIMIT}: la "
            "relajación no terminó."
        )
    elif relaxed_fmax > FMAX_RECOMMENDED:
        report.warnings.append(
            f"Fuerza residual {relaxed_fmax:.3f} eV/Å: para asignar bandas conviene "
            f"<= {FMAX_RECOMMENDED}."
        )

    unformed = unformed_pyrrolic_nitrogens(atoms)
    if unformed and relaxed_fmax is not None:
        report.errors.append(
            f"Tras relajar, el N pirrólico {unformed} no está en un pentágono: la "
            "estructura no es pirrólica. No calcules su espectro como si lo fuera."
        )

    advice = suggest_spin(atoms, charge)
    if spinpol is False and advice.spinpol:
        report.errors.append(
            "El sistema necesita un cálculo con espín y se configuró sin espín: "
            + " ".join(advice.reasons)
        )
    return report
