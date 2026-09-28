"""Resonant first-order Raman: Raman tensors from the complex α(ω_L + iη).

When the laser approaches an electronic transition, the static
polarizability derivative (:mod:`tbkit.raman`) no longer describes the
intensity. The frozen-phonon generalisation used for resonant Raman in DFT
(e.g. Gillet, Giantomassi and Gonze, Phys. Rev. B 88, 094305 (2013)) takes
the derivative of the frequency-dependent, complex polarizability at the
laser energy:

    R_k(ω_L) = Σ_ai (∂α(ω_L + iη)/∂x_ai) L_k[a, i]

with η the lifetime broadening of the excited states. The activity uses the
moduli of the complex invariants, ``S = 45|a'|² + 7γ'²``, and the excitation
profile ``S_k(ω_L)`` shows which electronic transition enhances which mode.
With ω_L far below the gap it reduces to the non-resonant result.

What this is not: the vibronic (Franck-Condon, Albrecht A-term) structure of
resonance Raman in molecules, where the excited-state potential surface
shapes the intensities and overtones appear. Here the excited states enter
only through their energies and transition densities at the ground-state
geometry and its small displacements. Graphene and other semimetals are
allowed (their Raman is always resonant); only interband transitions count.

Polarizabilities: finite systems use the screened response of
:func:`tbkit.optics.dynamic_polarizability_finite` (charges, intra-atomic
and extra dipoles); crystals the independent-particle interband sum of
:func:`tbkit.optics.dynamic_polarizability_periodic`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ase import Atoms

from .hamiltonian import System
from .params import TBModel
from .raman import RamanResult, internal_modes, phonons_for


def complex_invariants(tensor: np.ndarray) -> tuple[float, float]:
    """``|a'|²`` and ``γ'²`` of a complex (symmetric) Raman tensor."""
    t = 0.5 * (np.asarray(tensor) + np.asarray(tensor).T)
    mean = np.trace(t) / 3.0
    gamma2 = 0.5 * (abs(t[0, 0] - t[1, 1]) ** 2 + abs(t[1, 1] - t[2, 2]) ** 2
                    + abs(t[2, 2] - t[0, 0]) ** 2) \
        + 3.0 * (abs(t[0, 1]) ** 2 + abs(t[1, 2]) ** 2 + abs(t[2, 0]) ** 2)
    return float(abs(mean) ** 2), float(gamma2)


def _activity(tensor) -> tuple[float, float]:
    mean2, gamma2 = complex_invariants(tensor)
    denominator = 45 * mean2 + 4 * gamma2
    return 45 * mean2 + 7 * gamma2, (3 * gamma2 / denominator if denominator > 1e-30 else 0.0)


@dataclass
class ResonantRamanResult:
    """Raman activities of every mode at every laser energy."""

    frequencies: np.ndarray            # (n_modes,) cm⁻¹
    lasers: np.ndarray                 # (n_lasers,) eV
    eta: float                         # eV
    tensors: np.ndarray                # (n_lasers, n_modes, 3, 3) complex, Å²/amu^½
    activities: np.ndarray             # (n_lasers, n_modes) Å⁴/amu
    depolarization: np.ndarray         # (n_lasers, n_modes)
    alpha: np.ndarray                  # (n_lasers, 3, 3) complex α at the structure
    method: str
    warnings: list[str] = field(default_factory=list)

    def at(self, laser_ev: float) -> RamanResult:
        """The result at the laser energy closest to ``laser_ev``, as a RamanResult
        (for :func:`tbkit.raman.spectrum` and :meth:`RamanResult.groups`)."""
        i = int(np.argmin(np.abs(self.lasers - laser_ev)))
        return RamanResult(self.frequencies, self.activities[i], self.depolarization[i],
                           np.abs(self.tensors[i]), np.abs(self.alpha[i]),
                           f"{self.method}; ħω_L = {self.lasers[i]:.3f} eV, η = {self.eta} eV",
                           self.frequencies, list(self.warnings))

    def profile(self, frequency_cm1: float, tolerance: float = 1.0) -> np.ndarray:
        """Excitation profile: activity summed over the modes within ``tolerance``
        cm⁻¹ of ``frequency_cm1`` (a degenerate set), for every laser energy."""
        members = np.abs(self.frequencies - frequency_cm1) <= tolerance
        if not members.any():
            raise ValueError(f"Ningún modo a {frequency_cm1} ± {tolerance} cm⁻¹.")
        return self.activities[:, members].sum(axis=1)

    def summary(self, threshold: float = 1e-3) -> str:
        lines = [f"Raman resonante ({self.method}); η = {self.eta} eV"]
        header = "     cm⁻¹ " + "".join(f"{e:>10.2f}" for e in self.lasers) + "   (eV)"
        lines.append(header)
        strongest = self.activities.max()
        for k, freq in enumerate(self.frequencies):
            if self.activities[:, k].max() < threshold * strongest:
                continue
            lines.append(f"{freq:9.1f} " + "".join(f"{a:10.3g}" for a in self.activities[:, k]))
        lines += [f"AVISO: {w}" for w in self.warnings]
        return "\n".join(lines)


def resonant_raman(atoms: Atoms, model: TBModel, lasers_ev, eta: float = 0.1,
                   kmesh: int = 24, kT: float = 0.01, delta: float = 0.01,
                   phonon_delta: float = 0.005, phonons: Optional[tuple] = None,
                   screened: bool = True) -> ResonantRamanResult:
    """Resonant Raman activities for each laser energy in ``lasers_ev`` (eV).

    Parameters as in :func:`tbkit.raman.raman`; ``eta`` is the broadening of
    the electronic excitations (eV, > 0). ``phonons=(frequencies, L)`` takes
    phonons from elsewhere (QE, or another model: graphene's G mode from Xu
    with the α of the π model, for instance).
    """
    if eta <= 0:
        raise ValueError("η debe ser > 0 en resonancia (si no, α diverge en cada transición).")
    from .optics import dynamic_polarizability_finite, dynamic_polarizability_periodic

    lasers = np.atleast_1d(np.asarray(lasers_ev, dtype=float))
    warnings: list[str] = []
    atoms = atoms.copy()
    frequencies, modes = phonons_for(atoms, model, kmesh, kT, phonon_delta, phonons, warnings)
    internal = internal_modes(atoms, frequencies, warnings)
    finite = not atoms.get_pbc().any()
    if not finite:
        from .kpoints import mesh
        from .solver import solve

        k, w = mesh(atoms, min(kmesh, 12))
        if solve(System.build(atoms, model), k, w, kT=kT).gap() < 0.05:
            warnings.append("Sin gap (semimetal): cerca de los puntos de Dirac la suma en k "
                            "converge despacio y las diferencias finitas mezclan simetrías. "
                            "Para el grafeno usa tbkit.graphene (perturbativo, analítico).")

    def alpha_of(a: Atoms) -> np.ndarray:
        system = System.build(a, model)
        if finite:
            return dynamic_polarizability_finite(system, lasers, eta, kT=kT, screened=screened)
        return dynamic_polarizability_periodic(system, lasers, eta, kmesh=kmesh, kT=kT)

    method = ("α(ω + iη) apantallado (cargas y dipolos), finito" if finite and screened
              else "α(ω + iη) de partículas independientes" + ("" if finite else ", cristal"))
    alpha0 = alpha_of(atoms)
    base = atoms.get_positions()
    derivative = np.zeros((len(lasers), len(atoms), 3, 3, 3), dtype=complex)
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
            derivative[:, a, i] = (pair[0] - pair[1]) / (2 * delta)
    tensors = np.einsum("laijk,mai->lmjk", derivative, modes[internal])
    activities = np.zeros(tensors.shape[:2])
    ratios = np.zeros(tensors.shape[:2])
    for li in range(len(lasers)):
        for k in range(tensors.shape[1]):
            activities[li, k], ratios[li, k] = _activity(tensors[li, k])
    return ResonantRamanResult(frequencies[internal], lasers, float(eta), tensors, activities,
                               ratios, alpha0, method, warnings)
