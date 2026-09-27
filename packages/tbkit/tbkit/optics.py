"""Electronic polarizability from the tight-binding states (sum over states).

For independent electrons in linear response,

    α_ab(ω) = e² Σ_k w_k Σ_{n≠m} (f_n - f_m) Re[r^a_nm r^b_mn] E_mn / (E_mn² - (ħω)²)

with E_mn = E_m - E_n and occupations f summed over spin. ω = 0 gives the
static polarizability; a laser energy ħω below the gap gives the
pre-resonant one used for non-resonant Raman.

Position matrix elements (point-dipole approximation: orbitals are centred
on their atoms and intra-atomic dipoles s-p are neglected):

* finite systems: ``r = C† R C`` with ``R_μν = ½ (R_μ + R_ν) S_μν`` (``S = 1``
  for an orthogonal model). This is exactly the linear response of the model
  to a uniform field added on the diagonal, which the tests check.
* periodic systems: the interband ``r_nm = i ⟨n|∂_k H|m⟩ / (E_m - E_n)``
  (``∂_k H - E ∂_k S`` with overlap, an approximation), with H(k) in the
  convention where Bloch phases include the positions inside the cell, so
  that ∂_k H is the velocity operator. α is then per unit cell.

Units: e²Å²/eV × 14.3996 = Å³ (Gaussian polarizability volume). For a
crystal, ``ε∞ = 1 + 4π α / V_cell``.

What it cannot do: metals and semimetals have no static polarizability
(the intraband response diverges; graphene's interband one diverges as
1/ω), so they are refused. Local-field and excitonic effects are absent
(independent particles): TB polarizabilities of extended systems are
indicative. With ħω close to the gap the expression diverges; that is the
resonant regime, not described here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import eigh

from .hamiltonian import System
from .kpoints import gamma, mesh
from .solver import Solution, _fermi_dirac, fermi_level, solve

#: e²/(4 π ε0), eV Å.
COULOMB = 14.399645


def _check_gapped(solution: Solution, omega: float) -> float:
    gap = solution.gap()
    if gap < max(0.05, 20 * solution.kT):
        raise ValueError(
            f"Gap de {gap:.3f} eV: un metal o semimetal no tiene polarizabilidad estática "
            "(la respuesta intrabanda diverge). Su Raman es resonante, no el de este modelo.")
    if omega >= 0.8 * gap:
        raise ValueError(f"ħω = {omega} eV está a menos del 20 % del gap ({gap:.3f} eV): "
                         "régimen resonante, fuera del alcance de la aproximación no resonante.")
    return gap


def _response(energies, occupations, r_matrices, omega: float) -> np.ndarray:
    """Σ_{n≠m} (f_n - f_m) Re[r^a_nm r^b_mn] E_mn/(E_mn² - ω²) for one k (e² omitted)."""
    diff_f = occupations[:, None] - occupations[None, :]
    e_mn = energies[None, :] - energies[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(np.abs(e_mn) > 1e-9, diff_f * e_mn / (e_mn ** 2 - omega ** 2), 0.0)
    out = np.zeros((3, 3))
    for a in range(3):
        for b in range(a, 3):
            value = float(np.sum(factor * np.real(r_matrices[a] * r_matrices[b].T)))
            out[a, b] = out[b, a] = value
    return out


def polarizability_finite(solution: Solution, omega: float = 0.0) -> np.ndarray:
    """α (Å³) of a finite system from its solution."""
    system = solution.system
    if system.periodic:
        raise ValueError("Sistema periódico: usa polarizability().")
    _check_gapped(solution, omega)
    positions = system.atoms.get_positions()
    centres = np.array([positions[o.atom] for o in system.basis.orbitals])
    centres = centres - centres.mean(axis=0)
    overlap = solution.overlaps[0] if solution.overlaps is not None else None
    alpha = np.zeros((3, 3))
    for s in range(solution.nspin):
        c = solution.vectors[s, 0]
        occupations = solution.occupations[s, 0]
        r = []
        for a in range(3):
            if overlap is None:
                operator = np.diag(centres[:, a])
            else:
                operator = 0.5 * (centres[:, a][:, None] + centres[:, a][None, :]) * overlap
            r.append(c.conj().T @ operator @ c)
        alpha += _response(solution.energies[s, 0], occupations, r, omega)
    return alpha * COULOMB


def _bloch_ii(system: System, k_fractional: np.ndarray):
    """H(k), S(k) and ∂H/∂k, ∂S/∂k (Cartesian) in the phase convention with positions."""
    n = system.basis.size
    k_cart = system.kpoint_cartesian(k_fractional)
    h = np.zeros((n, n), dtype=complex)
    s = None if system.model.orthogonal else np.eye(n, dtype=complex)
    dh = np.zeros((3, n, n), dtype=complex)
    ds = None if s is None else np.zeros((3, n, n), dtype=complex)
    for bond, hb, sb in zip(system.bonds, system.h_blocks, system.s_blocks, strict=True):
        phase = np.exp(1j * float(k_cart @ bond.vector))
        ri, rj = system.basis.of_atom(bond.i), system.basis.of_atom(bond.j)
        h[ri.start:ri.stop, rj.start:rj.stop] += hb * phase
        for a in range(3):
            dh[a, ri.start:ri.stop, rj.start:rj.stop] += 1j * bond.vector[a] * hb * phase
        if s is not None and sb is not None:
            s[ri.start:ri.stop, rj.start:rj.stop] += sb * phase
            for a in range(3):
                ds[a, ri.start:ri.stop, rj.start:rj.stop] += 1j * bond.vector[a] * sb * phase
    h[np.diag_indices(n)] += system.onsite
    return h, s, dh, ds


def polarizability_periodic(system: System, kpts: np.ndarray, weights: np.ndarray,
                            kT: float = 0.01, omega: float = 0.0,
                            charge: float = 0.0) -> tuple[np.ndarray, float]:
    """α per unit cell (Å³) of a periodic, gapped system; also returns the gap (eV)."""
    energies, blocks = [], []
    for k in kpts:
        h, s, dh, ds = _bloch_ii(system, k)
        e, c = eigh(h, s) if s is not None else eigh(h)
        energies.append(e)
        blocks.append((c, dh, ds, e))
    energies = np.array(energies)[None]
    mu = fermi_level(energies, weights, system.electrons - charge, kT, 2.0)
    occupations = 2.0 * _fermi_dirac(energies, mu, kT)
    full = energies[0][occupations[0] > 1.0]
    empty = energies[0][occupations[0] <= 1.0]
    fractional = (occupations > 0.04) & (occupations < 1.96)
    gap = 0.0 if fractional.any() else float(empty.min() - full.max())
    if gap < max(0.05, 20 * kT):
        raise ValueError(
            f"Gap de {gap:.3f} eV: un metal o semimetal no tiene polarizabilidad estática. "
            "Su Raman es resonante, no el de este modelo.")
    if omega >= 0.8 * gap:
        raise ValueError(f"ħω = {omega} eV demasiado cerca del gap ({gap:.3f} eV): resonante.")
    alpha = np.zeros((3, 3))
    for index, (c, dh, ds, e) in enumerate(blocks):
        e_mn = e[None, :] - e[:, None]
        r = []
        for a in range(3):
            velocity = c.conj().T @ dh[a] @ c
            if ds is not None:
                velocity = velocity - (c.conj().T @ ds[a] @ c) * e[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                r.append(np.where(np.abs(e_mn) > 1e-9, 1j * velocity / e_mn, 0.0))
        alpha += weights[index] * _response(e, occupations[0, index], r, omega)
    return alpha * COULOMB, gap


def polarizability(system: System, kmesh: int = 12, kT: float = 0.01, omega: float = 0.0,
                   charge: float = 0.0, kpts: Optional[np.ndarray] = None,
                   weights: Optional[np.ndarray] = None) -> np.ndarray:
    """α (Å³): per molecule for a finite system, per unit cell for a periodic one."""
    if system.periodic:
        if kpts is None:
            kpts, weights = mesh(system.atoms, kmesh)
        return polarizability_periodic(system, kpts, weights, kT, omega, charge)[0]
    solution = solve(system, *gamma(), charge=charge, kT=kT)
    return polarizability_finite(solution, omega)


def charge_response(solution: Solution) -> np.ndarray:
    """Independent-particle charge response χ_AB = ∂q_A/∂V_B (e²/eV, finite systems).

    ``χ_AB = Σ_{n≠m} (f_n - f_m) q_A^{nm} q_B^{nm} / (E_n - E_m)`` with the
    Mulliken transition charges ``q_A^{nm} = ½ Σ_{μ∈A} [c_μn (Sc)_μm + (Sc)_μn c_μm]``
    and V_B a potential energy added to the orbitals of atom B. Negative
    semidefinite: electrons leave where the potential is raised.
    """
    system = solution.system
    atoms = system.basis.atoms
    owner = np.array([atoms.index(o.atom) for o in system.basis.orbitals])
    projector = np.zeros((len(atoms), system.basis.size))
    projector[owner, np.arange(system.basis.size)] = 1.0
    chi = np.zeros((len(atoms), len(atoms)))
    for s in range(solution.nspin):
        c = np.real(solution.vectors[s, 0])
        sc = c if solution.overlaps is None else np.real(solution.overlaps[0]) @ c
        f = solution.occupations[s, 0]
        e = solution.energies[s, 0]
        active = np.flatnonzero(f > 1e-12)
        passive = np.flatnonzero(f < (2.0 if solution.nspin == 1 else 1.0) - 1e-12)
        # q_A^{nm} for n active, m passive: (A, n, m)
        q = 0.5 * np.einsum("am,mn,mk->ank", projector, c[:, active], sc[:, passive]) \
            + 0.5 * np.einsum("am,mn,mk->ank", projector, sc[:, active], c[:, passive])
        de = e[active][:, None] - e[passive][None, :]
        df = f[active][:, None] - f[passive][None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(np.abs(de) > 1e-9, df / de, 0.0)
        # The ordered pair (m, n) gives the same term as (n, m). It is already
        # in the sum when n is also "passive" and m also "active" (both
        # partially occupied, e.g. a smeared HOMO-LUMO pair); otherwise count
        # (n, m) twice.
        in_passive = np.isin(active, passive)
        in_active = np.isin(passive, active)
        multiplicity = np.where(in_passive[:, None] & in_active[None, :], 1.0, 2.0)
        chi += np.einsum("ank,nk,bnk->ab", q, weight * multiplicity, q)
    return chi


def polarizability_linear_response(system: System, kT: float = 0.01, U=None,
                                   screened: bool = True) -> np.ndarray:
    """α (Å³) of a finite system by linear response of the SCC model.

    ``δq = (1 - χ γ)⁻¹ χ V_ext`` with ``V_ext,A = e E·R_A`` and ``μ = -Σ δq_A R_A``:
    exactly the derivative of the self-consistent dipole with respect to the
    field, from one diagonalisation of the ground state. ``screened=False``
    drops γ (independent particles).
    """
    from .scc import gamma_matrix, self_consistent

    if system.periodic:
        raise ValueError("Respuesta lineal SCC: solo sistemas finitos (sin Ewald).")
    reference = self_consistent(system, U=U, kT=kT, tol=1e-10) if screened else None
    solution = reference.solution if screened else solve(system, *gamma(), kT=kT)
    _check_gapped(solution, 0.0)
    chi = charge_response(solution)
    positions = system.atoms.get_positions()[system.basis.atoms]
    centred = positions - positions.mean(axis=0)
    if screened:
        gamma_ = gamma_matrix(system, U)
        response = np.linalg.solve(np.eye(len(chi)) - chi @ gamma_, chi)
    else:
        response = chi
    alpha = -centred.T @ response @ centred
    return 0.5 * (alpha + alpha.T) * COULOMB


def polarizability_screened(system: System, field: float = 0.01, kT: float = 0.01,
                            U=None) -> np.ndarray:
    """α (Å³) of a finite system with the self-consistent charge response,
    by finite field (the check of :func:`polarizability_linear_response`).

    ``α_ab = ∂μ_a/∂E_b`` by central differences of the SCC dipole under ±E
    (V/Å). The induced charges screen the applied field through γ (the
    monopole part of local-field effects), which the independent-particle
    :func:`polarizability_finite` lacks and overestimates by a large factor
    in molecules (C60: about 3x). Needs U for every element
    (:mod:`tbkit.scc`).
    """
    from .scc import self_consistent

    if system.periodic:
        raise ValueError("Polarizabilidad apantallada: solo sistemas finitos (sin Ewald).")
    reference = self_consistent(system, U=U, kT=kT, tol=1e-10)
    _check_gapped(reference.solution, 0.0)
    positions = system.atoms.get_positions()[system.basis.atoms]
    centred = positions - positions.mean(axis=0)
    alpha = np.zeros((3, 3))
    for b in range(3):
        dipoles = []
        for sign in (1.0, -1.0):
            vector = np.zeros(3)
            vector[b] = sign * field
            result = self_consistent(system, U=U, kT=kT, tol=1e-10, field=vector,
                                     initial_dq=reference.dq)
            if not result.converged:
                raise RuntimeError("SCC en campo sin converger: " + result.summary())
            # Δq counts electrons (charge -e): μ = -Σ Δq_A R_A, e·Å.
            dipoles.append(-(result.dq @ centred))
        alpha[:, b] = (dipoles[0] - dipoles[1]) / (2 * field)
    return 0.5 * (alpha + alpha.T) * COULOMB


def dielectric_constant(system: System, **kwargs) -> np.ndarray:
    """ε∞ tensor of a 3D crystal, ``1 + 4π α / V`` (independent particles)."""
    if not system.atoms.get_pbc().all():
        raise ValueError("ε∞ solo para cristales 3D (pbc en los tres ejes).")
    alpha = polarizability(system, **kwargs)
    return np.eye(3) + 4 * np.pi * alpha / system.atoms.get_volume()
