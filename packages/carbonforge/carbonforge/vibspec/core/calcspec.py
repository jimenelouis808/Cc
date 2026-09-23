"""Settings of one relax-then-IR calculation, and their physical validation.

One :class:`CalcSpec` describes everything that decides the numbers: the
engine, the functional, the basis or grid, spin, the relaxation criterion and
the finite-difference settings. It is saved with every calculation (see
:mod:`carbonforge.vibspec.core.record`), so a spectrum can always be traced
back to exactly what produced it.

Engines and modes
-----------------
Only GPAW runs the IR workflow here, in three modes:

``lcao``
    Localised basis (``dzp`` by default). Fast; the mode for screening many
    functionalisations. Its real-space grid must be fine enough
    (``h <= 0.18`` Å) or the egg-box effect breaks translational symmetry and
    contaminates the low-frequency modes.
``fd``
    Real-space finite differences. Systematically improvable with ``h``;
    the mode for confirming the bands you assign.
``pw``
    Plane waves (``ecut``). Converges cleanly with one number, but the
    Hartree potential is solved with periodic boundary conditions even when
    ``pbc`` is False: the molecule's periodic images interact through their
    dipoles, which shifts both forces and dipole derivatives of a polar
    group. Usable with more vacuum (:data:`PW_MIN_VACUUM_PER_SIDE`); LCAO and
    FD use zero boundary conditions and do not have the problem.

Quantum ESPRESSO reaches IR and Raman through DFPT (``ph.x``), which
carbonforge already writes as input files; it is not driven from here.

Finite systems are sampled at Γ only; there is no k-point setting to get
wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Optional

from ase import Atoms

from ...validation.checks import ValidationReport
from .checks import check_structure, check_vibration_settings, suggest_spin, vacuum_per_side

Mode = Literal["lcao", "fd", "pw"]

#: Vacuum per side asked for in PW mode, where periodic images interact.
PW_MIN_VACUUM_PER_SIDE: float = 8.0

#: Plane-wave cutoff below which forces on first-row atoms are not converged, eV.
PW_MIN_ECUT: float = 400.0

#: Default SCF convergence for vibrations. Finite differences of forces over
#: a 0.01 Å displacement need forces far better converged than GPAW's
#: defaults, which are tuned for total energies.
DEFAULT_CONVERGENCE: dict[str, float] = {"energy": 1e-6, "density": 1e-6, "forces": 1e-4}

#: Loosest density criterion that still gives clean finite differences.
LOOSEST_DENSITY: float = 1e-5


@dataclass
class CalcSpec:
    """Settings of a relaxation followed by an IR calculation.

    Attributes
    ----------
    engine
        Only ``"gpaw"`` for the IR workflow.
    xc
        Exchange-correlation functional, as GPAW names it (``"PBE"``,
        ``"BLYP"``, ...). Hybrids are not available in LCAO/FD forces.
    mode
        ``"lcao"``, ``"fd"`` or ``"pw"``; see the module docstring.
    basis
        LCAO basis set. Ignored in ``fd``.
    h
        Real-space grid spacing, Å (LCAO and FD).
    ecut
        Plane-wave cutoff, eV (PW only).
    spinpol
        ``None`` lets :func:`~carbonforge.vibspec.core.checks.suggest_spin`
        decide, which is the safe default. ``False`` on a system that needs
        spin is refused before anything runs.
    charge
        Net charge of the system, in units of e.
    fmax
        Relaxation criterion, eV/Å.
    max_steps
        Relaxation step limit.
    delta
        Finite-difference displacement, Å.
    nfree
        Displacements per coordinate: 2 or 4.
    ir_method
        How ASE builds the force constants from the displacements.
        ``"frederiksen"`` (default) imposes that the forces of each displaced
        structure sum to zero, i.e. exact translational invariance, which a
        real-space grid breaks (the egg-box effect). On N2 in GPAW LCAO it
        brings the translational modes from 150-270 cm^-1 to zero and moves
        the stretch by under 1 %. ``"standard"`` uses the raw forces. Both
        read the same displacement cache, so switching costs nothing.
    convergence
        GPAW SCF criteria. Defaults to :data:`DEFAULT_CONVERGENCE`.
    scale_factor
        Frequency scale factor to apply when comparing with experiment.
        Stored with the calculation; the raw frequencies are never scaled in
        place.
    """

    engine: Literal["gpaw"] = "gpaw"
    xc: str = "PBE"
    mode: Mode = "lcao"
    basis: str = "dzp"
    h: float = 0.18
    ecut: float = 600.0
    spinpol: Optional[bool] = None
    charge: int = 0
    fmax: float = 0.01
    max_steps: int = 500
    delta: float = 0.01
    nfree: int = 2
    ir_method: Literal["frederiksen", "standard"] = "frederiksen"
    convergence: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CONVERGENCE))
    scale_factor: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalcSpec":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Parámetros desconocidos en la especificación: {sorted(unknown)}.")
        return cls(**data)

    def resolved_spinpol(self, atoms: Atoms) -> bool:
        """Spin polarisation actually used: the explicit choice, or the advice."""
        if self.spinpol is not None:
            return self.spinpol
        return suggest_spin(atoms, charge=self.charge).spinpol

    def validate(self, atoms: Optional[Atoms] = None) -> ValidationReport:
        """Check these settings, and the structure they will be used on.

        Errors here mean the calculation must not run. With ``atoms`` the
        structure checks and the spin decision are included.
        """
        report = check_vibration_settings(
            fmax=self.fmax, delta=self.delta, nfree=self.nfree,
            mode=self.mode, h=self.h, scale_factor=self.scale_factor,
        )

        if self.engine != "gpaw":
            report.errors.append(
                f"Motor '{self.engine}': el flujo IR solo corre con GPAW. Para QE usa "
                "la exportación DFPT (ph.x) de carbonforge."
            )
        if self.mode not in ("lcao", "fd", "pw"):
            report.errors.append(f"Modo '{self.mode}': solo 'lcao', 'fd' o 'pw'.")
        if self.mode == "pw":
            report.warnings.append(
                "Modo PW: el potencial de Hartree es periódico aunque pbc=False, así que "
                "la molécula interactúa con sus imágenes a través del dipolo. Usa al "
                f"menos {PW_MIN_VACUUM_PER_SIDE} Å de vacío por lado, o LCAO/FD."
            )
            if self.ecut < PW_MIN_ECUT:
                report.warnings.append(
                    f"ecut={self.ecut} eV es bajo para fuerzas con C, N y O: usa >= "
                    f"{PW_MIN_ECUT} eV."
                )
        if self.mode == "fd" and self.h > 0.20:
            report.warnings.append(
                f"FD con h={self.h} Å: para frecuencias usa h <= 0.20 Å."
            )
        if self.ir_method not in ("frederiksen", "standard"):
            report.errors.append(
                f"ir_method='{self.ir_method}': solo 'frederiksen' o 'standard'."
            )
        if self.max_steps < 1:
            report.errors.append("max_steps debe ser >= 1.")

        density = self.convergence.get("density")
        if density is None or density > LOOSEST_DENSITY:
            report.warnings.append(
                f"Convergencia de densidad {density}: para diferencias finitas de "
                f"fuerzas usa <= {LOOSEST_DENSITY}."
            )

        if atoms is not None:
            report.merge(check_structure(atoms, charge=self.charge))
            if self.mode == "pw":
                thin = {axis: gap for axis, gap in vacuum_per_side(atoms).items()
                        if gap < PW_MIN_VACUUM_PER_SIDE - 1e-6}
                if thin:
                    report.warnings.append(
                        f"Modo PW con {min(thin.values()):.1f} Å de vacío por lado: las "
                        "imágenes periódicas alteran el dipolo. Reconstruye con "
                        f"vacuum_per_side >= {PW_MIN_VACUUM_PER_SIDE}."
                    )
            advice = suggest_spin(atoms, charge=self.charge)
            if self.spinpol is False and advice.spinpol:
                report.errors.append(
                    "spinpol=False en un sistema que necesita espín: "
                    + " ".join(advice.reasons)
                )
            report.info["spinpol"] = str(self.resolved_spinpol(atoms))
        return report
