"""First-order, non-resonant Raman from the tight-binding model alone.

Pipeline (all from one :class:`~tbkit.params.TBModel` with a repulsive term):

1. Γ phonons: force constants by finite displacements (``ase.vibrations``),
   modes as Cartesian displacement per unit normal coordinate (e/√m).
2. Polarizability of every displaced geometry (±δ along x, y, z of each
   atom, central differences): for a finite system with self-consistent
   charges (:func:`tbkit.optics.polarizability_linear_response`, screening
   included) when the model has U for all its elements, otherwise and for
   crystals by sum over states (:mod:`tbkit.optics`, independent particles).
3. Raman tensor of each mode ``dα/dQ_k = Σ_ai (∂α/∂x_ai) L_k[a,i]``, its
   invariants (mean a', anisotropy γ'²), the activity
   ``S = 45 a'² + 7 γ'²`` (Å⁴/amu) and the depolarization ratio
   ``ρ = 3γ'² / (45a'² + 4γ'²)``: 0 for a totally symmetric mode, 0.75 for
   a non-totally-symmetric one.

A measured intensity also carries (ν_L - ν)⁴ and the Bose factor; they are
applied by :func:`spectrum`, never stored.

Scope: the non-resonant limit (laser well below the gap). Metals and
graphene are refused: their Raman is resonant. Second-order bands (2D) need
the double-resonance theory of the next step, not this.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from ase import Atoms

from .hamiltonian import System
from .params import TBModel

#: hc/k_B in cm·K, for the Bose factor.
_HC_OVER_KB = 1.438777


@dataclass
class RamanResult:
    """Raman-active modes of a structure."""

    frequencies: np.ndarray            # cm⁻¹, internal/optical modes only
    activities: np.ndarray             # Å⁴/amu
    depolarization: np.ndarray
    tensors: np.ndarray                # (n, 3, 3) dα/dQ, Å²/amu^½
    alpha: np.ndarray                  # polarizability of the structure, Å³
    method: str
    all_frequencies: np.ndarray        # every Γ mode, cm⁻¹ (negative = imaginary)
    warnings: list[str] = field(default_factory=list)

    def groups(self, tolerance: float = 1.0, threshold: float = 1e-3) -> list[dict]:
        """Degenerate sets (within ``tolerance`` cm⁻¹) with non-negligible activity.

        ``threshold`` is relative to the strongest set. Returns frequency,
        degeneracy, summed activity and depolarization of each active set.
        """
        order = np.argsort(self.frequencies)
        sets: list[list[int]] = []
        for index in order:
            if sets and abs(self.frequencies[index] - self.frequencies[sets[-1][-1]]) < tolerance:
                sets[-1].append(index)
            else:
                sets.append([index])
        rows = []
        for members in sets:
            activity = float(self.activities[members].sum())
            rows.append({"frequency_cm1": float(np.mean(self.frequencies[members])),
                         "degeneracy": len(members), "activity": activity,
                         "depolarization": _group_depolarization(self.tensors[members])})
        top = max((r["activity"] for r in rows), default=0.0)
        return [r for r in rows if top > 0 and r["activity"] >= threshold * top]

    def summary(self) -> str:
        lines = [f"Raman no resonante ({self.method}); α = "
                 f"{np.trace(self.alpha) / 3:.2f} Å³ (media)",
                 f"{'cm⁻¹':>9} {'deg':>4} {'actividad':>11} {'ρ':>6}"]
        # Down to 1e-5 of the strongest: weak but real lines (C60's Hg(3) is
        # ~5e-5); finite-difference noise of inactive modes stays below ~1e-6.
        for row in self.groups(threshold=1e-5):
            lines.append(f"{row['frequency_cm1']:9.1f} {row['degeneracy']:4d} "
                         f"{row['activity']:11.4g} {row['depolarization']:6.3f}")
        lines += [f"AVISO: {w}" for w in self.warnings]
        return "\n".join(lines)


def invariants(tensor: np.ndarray) -> tuple[float, float]:
    """Mean ``a'`` and anisotropy ``γ'²`` of a (symmetrised) Raman tensor."""
    t = 0.5 * (tensor + tensor.T)
    mean = float(np.trace(t) / 3.0)
    gamma2 = 0.5 * ((t[0, 0] - t[1, 1]) ** 2 + (t[1, 1] - t[2, 2]) ** 2
                    + (t[2, 2] - t[0, 0]) ** 2
                    + 6.0 * (t[0, 1] ** 2 + t[1, 2] ** 2 + t[2, 0] ** 2))
    return mean, float(gamma2)


def _depolarization(mean: float, gamma2: float) -> float:
    denominator = 45 * mean ** 2 + 4 * gamma2
    return 3 * gamma2 / denominator if denominator > 1e-20 else 0.0


def _group_depolarization(tensors: np.ndarray) -> float:
    """ρ of a degenerate set: invariants summed over its members (basis-independent)."""
    means, gammas = zip(*(invariants(t) for t in tensors), strict=True)
    mean2 = float(np.sum(np.square(means)))
    gamma2 = float(np.sum(gammas))
    denominator = 45 * mean2 + 4 * gamma2
    return 3 * gamma2 / denominator if denominator > 1e-20 else 0.0


def _alpha_function(system_of: Callable[[Atoms], System], atoms: Atoms, model: TBModel,
                    kmesh: int, kT: float, omega: float, screening: str):
    from .optics import polarizability, polarizability_linear_response

    finite = not atoms.get_pbc().any()
    has_u = all(model.hubbard_u.get(el, 0) > 0 for el in model.orbitals)
    use_scc = finite and (screening == "scc" or (screening == "auto" and has_u))
    if use_scc and omega:
        raise ValueError("El α apantallado (SCC) es estático: usa omega = 0 o "
                         "screening='none'.")
    if use_scc:
        return (lambda a: polarizability_linear_response(system_of(a), kT=kT),
                "α apantallado (respuesta lineal SCC, finito)")
    if not finite and screening == "scc":
        # Crystals keep the independent-particle α under "auto" (their validated
        # results); the local-field α is asked for by name.
        from .optics import polarizability_periodic_screened

        if omega:
            raise ValueError("El α apantallado (SCC) es estático: usa omega = 0.")
        return (lambda a: polarizability_periodic_screened(system_of(a), kmesh=kmesh, kT=kT),
                "α apantallado (respuesta lineal SCC con campos locales, cristal)")
    kind = "suma sobre estados, partículas independientes"
    kind += "" if finite else ", cristal"
    return (lambda a: polarizability(system_of(a), kmesh=kmesh, kT=kT, omega=omega),
            f"α por {kind}")


def _model_phonons(atoms: Atoms, model: TBModel, kmesh: int, kT: float, phonon_delta: float,
                   warnings: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Γ frequencies (cm⁻¹, imaginary as negative) and e/√m modes from the model."""
    from ase.units import invcm
    from ase.vibrations import Vibrations

    from .calculator import TBCalculator

    atoms.calc = TBCalculator(model, kpts=kmesh, kT=kT)
    residual = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    if residual > 0.05:
        warnings.append(f"Fuerza residual {residual:.3f} eV/Å: la geometría no está relajada; "
                        "frecuencias y tensores no son los del mínimo.")
    with tempfile.TemporaryDirectory() as directory:
        vibrations = Vibrations(atoms, name=os.path.join(directory, "vib"), delta=phonon_delta)
        vibrations.run()
        energies, modes = vibrations.get_vibrations().get_energies_and_modes(all_atoms=True)
    frequencies = np.where(np.abs(energies.imag) > np.abs(energies.real),
                           -np.abs(energies.imag), np.abs(energies.real)) / invcm
    return frequencies, modes


def phonons_for(atoms: Atoms, model: TBModel, kmesh: int, kT: float, phonon_delta: float,
                phonons: Optional[tuple], warnings: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The model's Γ phonons, or the given ``(frequencies, L)`` after checking them."""
    if phonons is None:
        return _model_phonons(atoms, model, kmesh, kT, phonon_delta, warnings)
    frequencies = np.asarray(phonons[0], dtype=float)
    modes = np.asarray(phonons[1], dtype=float)
    if modes.shape != (len(frequencies), len(atoms), 3):
        raise ValueError(f"Fonones externos con forma {modes.shape}; se esperaba "
                         f"({len(frequencies)}, {len(atoms)}, 3).")
    warnings.append("Frecuencias y modos importados: el modelo TB solo da la "
                    "polarizabilidad, a la geometría dada.")
    return frequencies, modes


def internal_modes(atoms: Atoms, frequencies: np.ndarray, warnings: list[str]) -> np.ndarray:
    """Indices of the vibrational (non rigid-body) modes; warns about imaginary ones."""
    finite = not atoms.get_pbc().any()
    n_rigid = 6 if finite else 3
    if finite and len(atoms) > 2:
        inertia = np.sort(atoms.get_moments_of_inertia())
        n_rigid = 5 if inertia[0] < 1e-3 * inertia[-1] else 6
    elif finite:
        n_rigid = 5
    order = np.argsort(np.abs(frequencies))
    internal = np.sort(order[n_rigid:])
    imaginary = frequencies[internal][frequencies[internal] < -20]
    if len(imaginary):
        warnings.append(f"{len(imaginary)} modo(s) imaginario(s) (hasta {imaginary.min():.0f} "
                        "cm⁻¹): la estructura no está en un mínimo.")
    return internal


def raman(atoms: Atoms, model: TBModel, kmesh: int = 12, kT: float = 0.01,
          delta: float = 0.01, omega: float = 0.0, screening: str = "auto",
          phonon_delta: float = 0.005,
          phonons: Optional[tuple[np.ndarray, np.ndarray]] = None) -> RamanResult:
    """Non-resonant Raman activities of ``atoms`` (relaxed) under ``model``.

    Parameters
    ----------
    kmesh, kT
        k points per periodic axis and Fermi-Dirac width (eV).
    delta
        Displacement for ∂α/∂x, Å.
    omega
        Laser photon energy for a pre-resonant α (eV); 0 = static. Must stay
        well below the gap.
    screening
        ``"auto"`` (SCC screening for finite systems when the model has U),
        ``"scc"`` or ``"none"``.
    phonon_delta
        Displacement for the force constants, Å.
    phonons
        ``(frequencies_cm1, L)`` from elsewhere (QE: :func:`raman_from_qe`),
        ``L`` the displacement per unit normal coordinate, ``(n, N, 3)``. The
        model then supplies only the polarizability and needs no repulsive
        term; ``atoms`` must be the geometry the phonons were computed at.
    """
    warnings: list[str] = []
    atoms = atoms.copy()
    frequencies, modes = phonons_for(atoms, model, kmesh, kT, phonon_delta, phonons, warnings)
    internal = internal_modes(atoms, frequencies, warnings)

    def system_of(a: Atoms) -> System:
        return System.build(a, model)

    alpha_of, method = _alpha_function(system_of, atoms, model, kmesh, kT, omega, screening)
    alpha0 = alpha_of(atoms)
    base = atoms.get_positions()
    derivative = np.zeros((len(atoms), 3, 3, 3))
    probe = atoms.copy()
    probe.calc = None
    for a in range(len(atoms)):
        for i in range(3):
            pair = []
            for step in (delta, -delta):
                positions = base.copy()
                positions[a, i] += step
                probe.set_positions(positions)
                pair.append(alpha_of(probe))
            derivative[a, i] = (pair[0] - pair[1]) / (2 * delta)

    tensors = np.einsum("aijk,mai->mjk", derivative, modes[internal])
    activities, ratios = [], []
    for tensor in tensors:
        mean, gamma2 = invariants(tensor)
        activities.append(45 * mean ** 2 + 7 * gamma2)
        ratios.append(_depolarization(mean, gamma2))
    return RamanResult(frequencies[internal], np.array(activities), np.array(ratios), tensors,
                       alpha0, method, frequencies, warnings)


def raman_from_qe(atoms: Atoms, model: TBModel, modes_file, kind: str = "auto",
                  **kwargs) -> RamanResult:
    """Raman with Quantum ESPRESSO's Γ phonons and the TB polarizability.

    ``modes_file`` is a ``dynmat.x``/``matdyn.x`` mode file (:mod:`tbkit.qe`);
    ``atoms`` the structure of that phonon run, in the same atom order. Masses
    are ``atoms``' (ASE's standard ones unless set: match them to the run's
    if it used isotopes).
    """
    from .qe import modes_for_raman, read_qe_modes

    modes = read_qe_modes(modes_file)
    vectors = modes_for_raman(modes, atoms.get_masses(), kind)
    return raman(atoms, model, phonons=(modes.frequencies, vectors), **kwargs)


def spectrum(result: RamanResult, grid: Optional[np.ndarray] = None, fwhm: float = 8.0,
             laser_nm: Optional[float] = 532.0, temperature_k: Optional[float] = 300.0
             ) -> tuple[np.ndarray, np.ndarray]:
    """Lorentzian-broadened Raman spectrum (Stokes).

    With ``laser_nm`` and ``temperature_k`` each line is weighted by
    ``(ν_L - ν)⁴ / ν · [1 + n_B(ν)]``, the factors that turn an activity into
    a cross-section; pass ``None`` for bare activities.
    """
    if grid is None:
        grid = np.linspace(0.0, max(1800.0, result.frequencies.max() + 200), 4000)
    frequencies, activities = result.frequencies, result.activities.copy()
    keep = frequencies > 1.0
    frequencies, activities = frequencies[keep], activities[keep]
    if laser_nm is not None:
        nu_laser = 1e7 / laser_nm
        activities = activities * (nu_laser - frequencies) ** 4 / frequencies
    if temperature_k is not None:
        activities = activities / (1 - np.exp(-_HC_OVER_KB * frequencies / temperature_k))
    half = fwhm / 2
    lines = (half / np.pi) / ((grid[:, None] - frequencies[None, :]) ** 2 + half ** 2)
    return grid, lines @ activities
