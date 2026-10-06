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
1/ω), so they are refused. Excitonic effects are absent. Local fields: the
SCC linear response screens finite systems and, through Ewald,
crystals (:func:`polarizability_periodic_screened`); the sum over states
here is independent particles. With ħω close to the gap the expression diverges; that is the
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


def polarizability_finite(solution: Solution, omega: float = 0.0,
                          onsite_dipoles: bool = True,
                          extra_polarizability: bool = True) -> np.ndarray:
    """α (Å³) of a finite system from its solution.

    With ``onsite_dipoles`` (and ``model.onsite_dipole`` set) the position
    operator includes the intra-atomic s-p dipoles (:mod:`tbkit.dipoles`);
    ``extra_polarizability`` adds the model's atomic polarizabilities, here
    without interaction (independent particles).
    """
    from .dipoles import dipole_matrices, has_dipoles

    system = solution.system
    if system.periodic:
        raise ValueError("Sistema periódico: usa polarizability().")
    _check_gapped(solution, omega)
    positions = system.atoms.get_positions()
    centres = np.array([positions[o.atom] for o in system.basis.orbitals])
    centres = centres - centres.mean(axis=0)
    overlap = solution.overlaps[0] if solution.overlaps is not None else None
    onsite = dipole_matrices(system) if onsite_dipoles and has_dipoles(system.model) else None
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
            if onsite is not None:
                operator = operator + onsite[a]
            r.append(c.conj().T @ operator @ c)
        alpha += _response(solution.energies[s, 0], occupations, r, omega)
    return alpha * COULOMB + _extra_sum(system, extra_polarizability)


def _extra_sum(system: System, enabled: bool) -> np.ndarray:
    """Σ_A α_A^extra · 1 (Å³): the unscreened extra polarizability."""
    from .dipoles import extra_alphas, has_extra

    if not (enabled and has_extra(system.model)):
        return np.zeros((3, 3))
    return np.eye(3) * float(extra_alphas(system).sum())


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
                            charge: float = 0.0,
                            onsite_dipoles: bool = True,
                            extra_polarizability: bool = True) -> tuple[np.ndarray, float]:
    """α per unit cell (Å³) of a periodic, gapped system; also returns the gap (eV).

    The intra-atomic dipoles, when the model has them, add ``⟨n|D|m⟩`` to the
    interband position elements (the velocity ``∂_k H + i[H, D]`` gives
    exactly that). The extra atomic polarizabilities are added per cell,
    without local-field effects (no screening in crystals yet).
    """
    from .dipoles import dipole_matrices, has_dipoles

    onsite = dipole_matrices(system) if onsite_dipoles and has_dipoles(system.model) else None
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
                element = np.where(np.abs(e_mn) > 1e-9, 1j * velocity / e_mn, 0.0)
            if onsite is not None:
                element = element + np.where(np.abs(e_mn) > 1e-9,
                                             c.conj().T @ onsite[a] @ c, 0.0)
            r.append(element)
        alpha += weights[index] * _response(e, occupations[0, index], r, omega)
    return alpha * COULOMB + _extra_sum(system, extra_polarizability), gap


def polarizability(system: System, kmesh: int = 12, kT: float = 0.01, omega: float = 0.0,
                   charge: float = 0.0, kpts: Optional[np.ndarray] = None,
                   weights: Optional[np.ndarray] = None,
                   onsite_dipoles: bool = True,
                   extra_polarizability: bool = True) -> np.ndarray:
    """α (Å³): per molecule for a finite system, per unit cell for a periodic one."""
    if system.periodic:
        if kpts is None:
            kpts, weights = mesh(system.atoms, kmesh)
        return polarizability_periodic(system, kpts, weights, kT, omega, charge,
                                       onsite_dipoles, extra_polarizability)[0]
    solution = solve(system, *gamma(), charge=charge, kT=kT)
    return polarizability_finite(solution, omega, onsite_dipoles, extra_polarizability)


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


def multipole_response(solution: Solution) -> np.ndarray:
    """χ between atomic charges and intra-atomic dipoles, (4N, 4N) (finite systems).

    As :func:`charge_response`, with transition multipoles (Mulliken charge,
    then the three components of ``Σ_{μν∈A} c_μn D_μν c_νm``) in place of
    transition charges.
    """
    from .dipoles import dipole_matrices, transition_multipoles

    system = solution.system
    d_matrices = dipole_matrices(system)
    size = 4 * len(system.basis.atoms)
    chi = np.zeros((size, size))
    for s in range(solution.nspin):
        c = np.real(solution.vectors[s, 0])
        sc = c if solution.overlaps is None else np.real(solution.overlaps[0]) @ c
        f = solution.occupations[s, 0]
        e = solution.energies[s, 0]
        active = np.flatnonzero(f > 1e-12)
        passive = np.flatnonzero(f < (2.0 if solution.nspin == 1 else 1.0) - 1e-12)
        m = transition_multipoles(system, c, sc, active, passive, d_matrices)
        de = e[active][:, None] - e[passive][None, :]
        df = f[active][:, None] - f[passive][None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(np.abs(de) > 1e-9, df / de, 0.0)
        in_passive = np.isin(active, passive)
        in_active = np.isin(passive, active)
        multiplicity = np.where(in_passive[:, None] & in_active[None, :], 1.0, 2.0)
        chi += np.einsum("ink,nk,jnk->ij", m, weight * multiplicity, m)
    return chi


def _field_coupling(system: System, dipoles: bool, extra: bool) -> np.ndarray:
    """X (channels, 3): a uniform field couples to charges (E·R_A) and dipoles (E)."""
    atoms = system.basis.atoms
    positions = system.atoms.get_positions()[atoms]
    blocks = [positions - positions.mean(axis=0)]
    identity = np.tile(np.eye(3), (len(atoms), 1))
    blocks += [identity] * (int(dipoles) + int(extra))
    return np.vstack(blocks)


def polarizability_linear_response(system: System, kT: float = 0.01, U=None,
                                   screened: bool = True,
                                   ground_scc: Optional[bool] = None,
                                   onsite_dipoles: bool = True,
                                   extra_polarizability: bool = True) -> np.ndarray:
    """α (Å³) of a finite system by linear response of the SCC model.

    ``δq = (1 - χ γ)⁻¹ χ V_ext`` with ``V_ext,A = e E·R_A`` and ``μ = -Σ δq_A R_A``:
    exactly the derivative of the self-consistent dipole with respect to the
    field, from one diagonalisation of the ground state. ``screened=False``
    drops γ (independent particles).

    The ground state is self-consistent when ``ground_scc`` is True (default:
    the model's ``scc`` flag). A model fitted without SCC (Xu, π) keeps its
    own ground state and only the response is screened; with a symmetric
    pure-carbon molecule (C60) the two coincide.

    With ``onsite_dipoles`` (and ``model.onsite_dipole`` set) the induced
    density carries atomic dipoles as well as charges, screened by the
    multipole kernel of :mod:`tbkit.dipoles`: α⊥ of chains and out-of-plane
    α of planar molecules are then no longer zero. With
    ``extra_polarizability`` (and ``model.extra_polarizability`` set) each
    atom also carries the polarizable dipole of what the minimal basis
    misses, screened by the same kernel.
    """
    from .scc import self_consistent

    if system.periodic:
        return polarizability_periodic_screened(system, kT=kT, U=U, screened=screened,
                                                ground_scc=ground_scc,
                                                onsite_dipoles=onsite_dipoles,
                                                extra_polarizability=extra_polarizability)
    if ground_scc is None:
        ground_scc = system.model.scc
    reference = self_consistent(system, U=U, kT=kT, tol=1e-10) if ground_scc else None
    solution = reference.solution if ground_scc else solve(system, *gamma(), kT=kT)
    _check_gapped(solution, 0.0)
    from .dipoles import extra_alphas, has_dipoles, has_extra, response_kernel

    dipoles = onsite_dipoles and has_dipoles(system.model)
    extra = extra_polarizability and has_extra(system.model)
    chi = multipole_response(solution) if dipoles else charge_response(solution)
    if extra:
        alphas = np.repeat(extra_alphas(system), 3)
        size = len(chi)
        chi = np.block([[chi, np.zeros((size, len(alphas)))],
                        [np.zeros((len(alphas), size)), np.diag(-alphas / COULOMB)]])
    coupling = _field_coupling(system, dipoles, extra)
    kernel = response_kernel(system, U, dipoles, extra) if screened else None
    response = chi if kernel is None else np.linalg.solve(np.eye(len(chi)) - chi @ kernel, chi)
    alpha = -coupling.T @ response @ coupling
    return 0.5 * (alpha + alpha.T) * COULOMB


def polarizability_periodic_screened(system: System, kmesh: int | tuple = 12,
                                     kT: float = 0.01, U=None, screened: bool = True,
                                     ground_scc: Optional[bool] = None,
                                     onsite_dipoles: bool = True,
                                     extra_polarizability: bool = True,
                                     kpts: Optional[np.ndarray] = None,
                                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    """α per unit cell (Å³) of a gapped crystal with the SCC local fields (q → 0).

    A uniform field is not ``E·R_A`` in a crystal, so the field enters through
    the interband position elements ``r_nm`` (as in :func:`polarizability_periodic`)
    and the charges it moves are screened by the periodic γ (Ewald, its G = 0
    term absent: the applied field is the macroscopic one). With the
    independent-particle responses ``χ_XY = Σ_k w Σ_{n≠m} (f_n - f_m)/(E_n - E_m)
    X_nm Y_mn`` between Mulliken transition charges ``q`` and ``r``,

        α = -[χ_rr + χ_rq γ (1 - χ_qq γ)⁻¹ χ_qr]

    which for a molecule in a large box is the finite
    :func:`polarizability_linear_response` (tested). Intra-atomic dipoles
    enter ``r`` but are not themselves screened in crystals (the dipole kernel
    has no Ewald sum yet), and the extra atomic polarizabilities are added per
    cell unscreened.

    What that leaves: on an atom whose site has a C3 (or higher) axis a uniform
    field induces no net charge, so in pristine h-BN or diamond the charge local
    fields are exactly zero (tested) and this equals the independent-particle α;
    in those crystals the local fields are dipolar and not included. They
    appear where the symmetry is broken: dopants, functional groups, strain.
    """
    from .dipoles import dipole_matrices, has_dipoles
    from .scc import gamma_matrix, self_consistent

    if kpts is None:
        kpts, weights = mesh(system.atoms, kmesh)
    if ground_scc is None:
        ground_scc = system.model.scc
    atoms = system.basis.atoms
    owner = np.array([atoms.index(o.atom) for o in system.basis.orbitals])
    shift = np.zeros(system.basis.size)
    if ground_scc:
        result = self_consistent(system, U=U, kT=kT, tol=1e-10, kpts=kpts, weights=weights)
        if not result.converged:
            raise RuntimeError("SCC sin converger: " + result.summary())
        shift = result.shift
    onsite = dipole_matrices(system) if onsite_dipoles and has_dipoles(system.model) else None
    states = []
    for k in kpts:
        h, s, dh, ds = _bloch_ii(system, k)
        if s is None:
            h = h + np.diag(shift)
            e, c = eigh(h)
        else:
            h = h + 0.5 * s * (shift[:, None] + shift[None, :])
            dh = [d + 0.5 * dsa * (shift[:, None] + shift[None, :]) for d, dsa in zip(dh, ds)]
            e, c = eigh(h, s)
        states.append((e, c, s, dh, ds))
    energies = np.array([st[0] for st in states])[None]
    mu = fermi_level(energies, weights, system.electrons, kT, 2.0)
    occupations = 2.0 * _fermi_dirac(energies, mu, kT)[0]
    fractional = (occupations > 0.04) & (occupations < 1.96)
    if fractional.any():
        raise ValueError("Sistema sin gap: un metal no tiene polarizabilidad estática.")
    n_atoms = len(atoms)
    chi_qq = np.zeros((n_atoms, n_atoms))
    chi_qr = np.zeros((n_atoms, 3))
    chi_rr = np.zeros((3, 3))
    for (e, c, s, dh, ds), f, w in zip(states, occupations, weights, strict=True):
        de = e[:, None] - e[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(np.abs(de) > 1e-9, (f[:, None] - f[None, :]) / de, 0.0)
        sc = c if s is None else s @ c
        # q[A, n, m] = ½ Σ_{μ∈A} [c*_μn (Sc)_μm + (Sc)*_μn c_μm]
        q = np.zeros((n_atoms, len(e), len(e)), dtype=complex)
        pair = 0.5 * (c.conj()[:, :, None] * sc[:, None, :] + sc.conj()[:, :, None] * c[:, None, :])
        np.add.at(q, owner, pair)
        r = []
        for a in range(3):
            velocity = c.conj().T @ dh[a] @ c
            if ds is not None:
                velocity = velocity - (c.conj().T @ ds[a] @ c) * e[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                element = np.where(np.abs(de.T) > 1e-9, 1j * velocity / (e[None, :] - e[:, None]),
                                   0.0)
            if onsite is not None:
                element = element + np.where(np.abs(de) > 1e-9, c.conj().T @ onsite[a] @ c, 0.0)
            r.append(element)
        r = np.array(r)
        chi_qq += w * np.real(np.einsum("anm,nm,bnm->ab", q, weight, q.conj()))
        chi_qr += w * np.real(np.einsum("anm,nm,bnm->ab", q, weight, r.conj()))
        chi_rr += w * np.real(np.einsum("anm,nm,bnm->ab", r, weight, r.conj()))
    alpha = -chi_rr
    if screened:
        gamma = gamma_matrix(system, U)
        response = np.linalg.solve(np.eye(n_atoms) - chi_qq @ gamma, chi_qr)
        alpha = alpha - chi_qr.T @ gamma @ response
    alpha = 0.5 * (alpha + alpha.T) * COULOMB
    return alpha + _extra_sum(system, extra_polarizability)


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
        raise ValueError("Polarizabilidad apantallada: solo sistemas finitos (no implementada con Ewald).")
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


def dielectric_constant(system: System, screened: bool = False, **kwargs) -> np.ndarray:
    """ε∞ tensor of a 3D crystal, ``1 + 4π α / V``: independent particles, or with
    the SCC local fields (``screened=True``, :func:`polarizability_periodic_screened`)."""
    if not system.atoms.get_pbc().all():
        raise ValueError("ε∞ solo para cristales 3D (pbc en los tres ejes).")
    alpha = polarizability_periodic_screened(system, **kwargs) if screened \
        else polarizability(system, **kwargs)
    return np.eye(3) + 4 * np.pi * alpha / system.atoms.get_volume()


# --------------------------------------------------------------------------
# Frequency-dependent (complex) polarizability: resonant Raman
# --------------------------------------------------------------------------

def _dynamic_weights(de: np.ndarray, df: np.ndarray, multiplicity: np.ndarray,
                     z: complex) -> np.ndarray:
    """Both time orderings of each (active, passive) pair at complex frequency z.

    Static limit: 2 df/de (with multiplicity 2), as in :func:`charge_response`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        w = df * (1.0 / (de + z) + 1.0 / (de - z)) * (multiplicity / 2.0)
    return np.where(np.abs(de) > 1e-9, w, 0.0)


def _transition_data(solution: Solution, dipoles: bool):
    """Transition multipoles (charges[, dipoles]) and energy/occupation differences."""
    from .dipoles import atom_projector, dipole_matrices, transition_multipoles

    system = solution.system
    d_matrices = dipole_matrices(system) if dipoles else None
    projector = atom_projector(system)
    out = []
    for s in range(solution.nspin):
        c = np.real(solution.vectors[s, 0])
        sc = c if solution.overlaps is None else np.real(solution.overlaps[0]) @ c
        f = solution.occupations[s, 0]
        e = solution.energies[s, 0]
        active = np.flatnonzero(f > 1e-12)
        passive = np.flatnonzero(f < (2.0 if solution.nspin == 1 else 1.0) - 1e-12)
        if dipoles:
            m = transition_multipoles(system, c, sc, active, passive, d_matrices)
        else:
            m = 0.5 * np.einsum("am,mn,mk->ank", projector, c[:, active], sc[:, passive]) \
                + 0.5 * np.einsum("am,mn,mk->ank", projector, sc[:, active], c[:, passive])
        de = e[active][:, None] - e[passive][None, :]
        df = f[active][:, None] - f[passive][None, :]
        in_passive = np.isin(active, passive)
        in_active = np.isin(passive, active)
        multiplicity = np.where(in_passive[:, None] & in_active[None, :], 1.0, 2.0)
        out.append((m, de, df, multiplicity))
    return out


def dynamic_polarizability_finite(system: System, energies, eta: float = 0.1,
                                  kT: float = 0.01, U=None, screened: bool = True,
                                  onsite_dipoles: bool = True,
                                  extra_polarizability: bool = True) -> np.ndarray:
    """Complex α(ω + iη) (Å³) of a finite system, one 3x3 per photon energy (eV).

    The same response as :func:`polarizability_linear_response` (charges,
    intra-atomic and extra dipoles, screened by the multipole kernel), at
    complex frequency: ``χ(z) = Σ (f_n - f_m) M^{nm} M^{mn} / (E_n - E_m + z)``
    over both time orderings, ``α = -Xᵀ (1 - χK)⁻¹ χ X``. η is the lifetime
    broadening of the excited states (eV): it keeps α finite at resonance.
    ``energies = [0]`` with η = 0 is the static α. The extra atomic
    polarizability is taken as frequency independent (its own resonances
    lie far above the valence ones).
    """
    from .dipoles import extra_alphas, has_dipoles, has_extra, response_kernel
    from .scc import self_consistent

    if system.periodic:
        raise ValueError("Sistema periódico: usa dynamic_polarizability_periodic().")
    if system.model.scc:
        solution = self_consistent(system, U=U, kT=kT, tol=1e-10).solution
    else:
        solution = solve(system, *gamma(), kT=kT)
    if solution.gap() < max(0.05, 20 * kT):
        raise ValueError("Molécula sin gap (capa abierta o metálica): fuera del alcance.")
    dipoles = onsite_dipoles and has_dipoles(system.model)
    extra = extra_polarizability and has_extra(system.model)
    data = _transition_data(solution, dipoles)
    coupling = _field_coupling(system, dipoles, extra)
    kernel = response_kernel(system, U, dipoles, extra) if screened else None
    alphas = np.repeat(extra_alphas(system), 3) if extra else None
    out = []
    for energy in np.atleast_1d(np.asarray(energies, dtype=float)):
        z = complex(energy, eta)
        chi = sum(np.einsum("ink,nk,jnk->ij", m, _dynamic_weights(de, df, mult, z), m)
                  for m, de, df, mult in data)
        if extra:
            size = len(chi)
            chi = np.block([[chi, np.zeros((size, len(alphas)))],
                            [np.zeros((len(alphas), size)), np.diag(-alphas / COULOMB)]])
        if kernel is not None:
            chi = np.linalg.solve(np.eye(len(chi)) - chi @ kernel, chi)
        alpha = -coupling.T @ chi @ coupling
        out.append(0.5 * (alpha + alpha.T) * COULOMB)
    return np.array(out)


def dynamic_polarizability_periodic(system: System, energies, eta: float = 0.1,
                                    kpts: Optional[np.ndarray] = None,
                                    weights: Optional[np.ndarray] = None, kmesh: int = 24,
                                    kT: float = 0.01, onsite_dipoles: bool = True,
                                    extra_polarizability: bool = True,
                                    ground_scc: Optional[bool] = None,
                                    scc_state: Optional[dict] = None) -> np.ndarray:
    """Complex interband α(ω + iη) per unit cell (Å³), independent particles.

    The states are those of the model's ground state: self-consistent charges when
    ``ground_scc`` (default: the model's ``scc``), so a set fitted with SCC gets its
    levels with the charge shifts it was fitted with (an N dopant's donor level,
    the pentagon/heptagon charge transfer of a 5-7 net). The response itself has no
    local fields (independent particles); that is what the SCC shift does not change.
    ``scc_state`` (a dict the caller keeps) starts the SCC from the charges of the
    previous call and stores the new ones: across the small displacements of a
    Raman derivative that saves most of the iterations, not the result.

    ``α_ab(z) = Σ_k w_k Σ_{nm} (f_n - f_m) r^a_nm r^b_mn / (E_m - E_n - z)``
    with the interband position elements of :func:`polarizability_periodic`.
    Metals and semimetals are allowed here (graphene's Raman is resonant at
    any laser), but only interband transitions are counted: the intraband
    (Drude) term is left out, which is what the Raman tensor of a Γ phonon
    needs away from ω = 0.
    """
    from .dipoles import dipole_matrices, has_dipoles

    if kpts is None:
        kpts, weights = mesh(system.atoms, kmesh)
    if ground_scc is None:
        ground_scc = system.model.scc
    shift = None
    if ground_scc:
        from .scc import self_consistent

        guess = None if scc_state is None else scc_state.get("dq")
        result = self_consistent(system, kT=kT, tol=1e-10, kpts=kpts, weights=weights,
                                 initial_dq=guess)
        if not result.converged:
            raise RuntimeError("SCC sin converger: " + result.summary())
        shift = result.shift
        if scc_state is not None:
            scc_state["dq"] = result.dq
    onsite = dipole_matrices(system) if onsite_dipoles and has_dipoles(system.model) else None
    zs = np.array([complex(e, eta) for e in np.atleast_1d(np.asarray(energies, float))])
    spectra, blocks = [], []
    for k in kpts:
        h, s, dh, ds = _bloch_ii(system, k)
        if shift is not None and s is None:
            h = h + np.diag(shift)
        elif shift is not None:
            h = h + 0.5 * s * (shift[:, None] + shift[None, :])
            dh = np.array([d + 0.5 * dsa * (shift[:, None] + shift[None, :])
                           for d, dsa in zip(dh, ds, strict=True)])
        e, c = eigh(h, s) if s is not None else eigh(h)
        spectra.append(e)
        blocks.append((c, dh, ds, e))
    energies_k = np.array(spectra)[None]
    mu = fermi_level(energies_k, weights, system.electrons, kT, 2.0)
    occupations = 2.0 * _fermi_dirac(energies_k, mu, kT)
    alpha = np.zeros((len(zs), 3, 3), dtype=complex)
    for index, (c, dh, ds, e) in enumerate(blocks):
        e_mn = e[None, :] - e[:, None]
        r = []
        for a in range(3):
            velocity = c.conj().T @ dh[a] @ c
            if ds is not None:
                velocity = velocity - (c.conj().T @ ds[a] @ c) * e[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                element = np.where(np.abs(e_mn) > 1e-9, 1j * velocity / e_mn, 0.0)
            if onsite is not None:
                element = element + np.where(np.abs(e_mn) > 1e-9, c.conj().T @ onsite[a] @ c, 0.0)
            r.append(element)
        f = occupations[0, index]
        diff_f = f[:, None] - f[None, :]
        for iz, z in enumerate(zs):
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(np.abs(e_mn) > 1e-9, diff_f / (e_mn - z), 0.0)
            for a in range(3):
                for b in range(3):
                    alpha[iz, a, b] += weights[index] * np.sum(factor * r[a] * r[b].T)
    alpha = 0.5 * (alpha + alpha.transpose(0, 2, 1)) * COULOMB
    return alpha + _extra_sum(system, extra_polarizability)[None]
