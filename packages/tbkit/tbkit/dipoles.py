"""Intra-atomic dipoles: the part of the optical response point charges miss.

With orbitals as points, the position operator is ``R_μν = ½(R_μ + R_ν) S_μν``
and the screened response moves only atomic charges. An atom then cannot
polarise perpendicular to a chain or out of the plane of a flat molecule:
α⊥ is exactly zero in HCN, α_zz in benzene. What is missing is the dipole
between the s and p orbitals of the same atom,

    ⟨s|r_α|p_β⟩ = d δ_αβ,    d = (1/√3) ∫ R_s(r) R_p(r) r³ dr,

one number per element, computed from the free atom's valence radial
functions (GPAW's all-electron atom, PBE; :func:`tbkit.references.gpaw_onsite_dipole`)
and stored as ``onsite_dipole`` in the parameter file, as a magnitude.

**Its sign** depends on the phase convention of the model's p orbitals
relative to its s orbitals, which a parameter set fixes only implicitly
(through the signs of ssσ and spσ). It is therefore taken from the model
itself: in the lowest σ orbital of a two-atom probe, the electrons of each
atom shift towards the bond; ``sign(d) = sign(c_s c_pz)`` on the atom whose
partner lies along +z. The same model with its p orbitals flipped (every
spσ negated) gives the same polarizability.

**Screening.** The induced density is represented by atomic charges δq_A and
atomic dipoles δp_A (electrons; δp_A = Σ_{μν∈A} δρ_μν D_μν). Their
interaction extends the Klopman-Ohno γ of :mod:`tbkit.scc`: with
``v(R) = C/√(|R|² + a²)`` between the smeared distributions of A and B,

* charge-charge ``v``, charge(A)-dipole(B) ``-∇v(R_A - R_B)``,
* dipole(A)-charge(B) ``+∇v``, dipole-dipole ``-∇∇v``,
* on the same atom, dipole-dipole ``C/a³`` (the self-energy of a dipole) and
  no charge-dipole term.

No parameter is added: ``a`` is the one γ already uses (from U). The ground
state is left as the model was fitted (monopole SCC); the dipole terms act
on the changes δM the field induces. :func:`tbkit.scc.self_consistent` with
``dipoles=True`` solves the same functional by finite field, the check of
the linear response.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms

from .hamiltonian import System
from .params import TBModel

COULOMB = 14.399645
_AXIS = {"px": 0, "py": 1, "pz": 2}


def _probe_sign(model: TBModel, element: str) -> float:
    """sign(c_s c_pz) of ``element`` in the lowest σ orbital of a two-atom probe."""
    from .solver import solve

    partner = "H" if "H" in model.orbitals else element
    distance = 1.1 if partner == "H" else 1.4
    probe = Atoms([element, partner], positions=[[0, 0, 0], [0, 0, distance]])
    solution = solve(System.build(probe, model))
    basis = System.build(probe, model).basis
    first = basis.of_atom(0)
    names = model.orbitals[element]
    s_index = first.start + names.index("s")
    z_index = first.start + names.index("pz")
    c = np.real(solution.vectors[0, 0])
    for n in range(c.shape[1]):             # lowest orbital with s and pz on the atom
        if abs(c[s_index, n]) > 1e-3 and abs(c[z_index, n]) > 1e-3:
            return float(np.sign(c[s_index, n] * c[z_index, n]))
    raise ValueError(f"{element}: no hay orbital σ con s y p_z para fijar el signo de d.")


_SIGNED: dict[int, tuple[TBModel, dict[str, float]]] = {}


def signed_dipoles(model: TBModel) -> dict[str, float]:
    """``onsite_dipole`` with the sign the model's orbital convention requires (Å).

    Cached per model object (the probe costs two small diagonalisations).
    """
    cached = _SIGNED.get(id(model))
    if cached is not None and cached[0] is model:
        return cached[1]
    out = {}
    for element, magnitude in model.onsite_dipole.items():
        orbitals = model.orbitals.get(element, ())
        if magnitude and "s" in orbitals and "pz" in orbitals:
            out[element] = abs(float(magnitude)) * _probe_sign(model, element)
    if len(_SIGNED) > 64:
        _SIGNED.clear()
    _SIGNED[id(model)] = (model, out)
    return out


def has_dipoles(model: TBModel) -> bool:
    return any(model.onsite_dipole.get(el) for el in model.orbitals
               if {"s", "pz"} <= set(model.orbitals[el]))


def dipole_matrices(system: System) -> np.ndarray:
    """``D[α]_μν = ⟨μ|r_α - R_A|ν⟩`` for μ, ν on the same atom A: shape (3, n, n), Å."""
    n = system.basis.size
    out = np.zeros((3, n, n))
    signed = signed_dipoles(system.model)
    symbols = system.atoms.get_chemical_symbols()
    for atom in system.basis.atoms:
        d = signed.get(symbols[atom])
        if not d:
            continue
        block = system.basis.of_atom(atom)
        names = system.model.orbitals[symbols[atom]]
        s = block.start + names.index("s")
        for name, axis in _AXIS.items():
            if name in names:
                p = block.start + names.index(name)
                out[axis, s, p] = out[axis, p, s] = d
    return out


def atom_projector(system: System) -> np.ndarray:
    """(n_atoms, n_orbitals): 1 where the orbital belongs to the atom."""
    atoms = system.basis.atoms
    owner = np.array([atoms.index(o.atom) for o in system.basis.orbitals])
    projector = np.zeros((len(atoms), system.basis.size))
    projector[owner, np.arange(system.basis.size)] = 1.0
    return projector


def atomic_dipoles(system: System, density: np.ndarray, d_matrices: Optional[np.ndarray] = None
                   ) -> np.ndarray:
    """``p_A = Σ_{μν∈A} ρ_μν D_μν`` (electrons·Å), shape (n_atoms, 3)."""
    d_matrices = dipole_matrices(system) if d_matrices is None else d_matrices
    projector = atom_projector(system)
    return np.einsum("am,xmn,mn->ax", projector, d_matrices, np.real(density))


def multipole_kernel(system: System, U=None) -> np.ndarray:
    """Interaction between atomic charges and dipoles, (4N, 4N), eV per e² (Å)."""
    from .scc import gamma_matrix

    gamma = gamma_matrix(system, U)
    atoms = system.basis.atoms
    n = len(atoms)
    positions = system.atoms.get_positions()[atoms]
    diagonal = np.diag(gamma)                              # γ_AA = U_A = C/a_A
    ubar = 0.5 * (diagonal[:, None] + diagonal[None, :])
    a2 = (COULOMB / ubar) ** 2
    vectors = positions[:, None, :] - positions[None, :, :]          # R_A - R_B
    r2 = np.sum(vectors ** 2, axis=-1)
    base = r2 + a2
    grad = -COULOMB * vectors * base[..., None] ** -1.5               # ∇v(R_A - R_B)
    hessian = COULOMB * (3 * vectors[..., :, None] * vectors[..., None, :]
                         * base[..., None, None] ** -2.5
                         - np.eye(3) * base[..., None, None] ** -1.5)
    kernel = np.zeros((4 * n, 4 * n))
    kernel[:n, :n] = gamma
    for x in range(3):
        kernel[:n, n + x::3][:, :n] = -grad[..., x]                   # q_A with p_B
        kernel[n + x::3, :n][:n, :] = grad[..., x]                    # p_A with q_B
        for y in range(3):
            kernel[n + x::3, n + y::3][:n, :n] = -hessian[..., x, y]
    return kernel


def transition_multipoles(system: System, c: np.ndarray, sc: np.ndarray, active: np.ndarray,
                          passive: np.ndarray, d_matrices: np.ndarray) -> np.ndarray:
    """(4N, n_active, n_passive): Mulliken charges then dipoles of each transition n→m."""
    projector = atom_projector(system)
    q = 0.5 * np.einsum("am,mn,mk->ank", projector, c[:, active], sc[:, passive]) \
        + 0.5 * np.einsum("am,mn,mk->ank", projector, sc[:, active], c[:, passive])
    # p_A^{nm} = Σ_{μν∈A} c_μn D_μν c_νm (D is block-diagonal by atom)
    dc = np.einsum("xmv,vk->xmk", d_matrices, c[:, passive])
    p = np.einsum("am,mn,xmk->axnk", projector, c[:, active], dc)
    n_atoms = projector.shape[0]
    return np.concatenate([q, p.reshape(n_atoms * 3, len(active), len(passive))], axis=0)


# --------------------------------------------------------------------------
# Finite field: the check of the linear response
# --------------------------------------------------------------------------

def _solve(system: System, shift: np.ndarray, extra: np.ndarray, charge: float, kT: float):
    """Γ solution with orbital shifts (Mulliken form) and an extra on-site matrix."""
    from scipy.linalg import eigh

    from .scc import _package

    h, s = system.hamiltonian((0, 0, 0))
    if s is None:
        h = h + np.diag(shift) + extra
        e, c = eigh(h)
    else:
        h = h + 0.5 * s * (shift[:, None] + shift[None, :]) + extra
        e, c = eigh(h, s)
    return _package(system, e, c, s, charge, kT)


def _multipoles(solution, system: System, neutral_atom: np.ndarray, d_matrices: np.ndarray):
    from .analysis import populations

    pops = populations(solution).sum(axis=0)
    projector = atom_projector(system)
    dq = projector @ pops - neutral_atom
    c = np.real(solution.vectors[0, 0])
    density = (c * solution.occupations[0, 0]) @ c.T
    return dq, atomic_dipoles(system, density, d_matrices)


def polarizability_finite_field(system: System, field: float = 0.002, kT: float = 0.01,
                                U=None, ground_scc: Optional[bool] = None, tol: float = 1e-10,
                                mixing: float = 0.3, max_iter: int = 500) -> np.ndarray:
    """α (Å³) by ±E with self-consistent charges AND intra-atomic dipoles.

    The functional is the ground state's (monopole SCC when ``ground_scc``,
    the model's ``scc`` flag by default; bare otherwise) plus the multipole
    interaction of the changes δM = M - M_ground, charge-charge excluded
    when the ground state already counts it. Its second derivative is what
    :func:`tbkit.optics.polarizability_linear_response` computes in one step.
    """
    from .hubbard import _neutral
    from .scc import anderson, gamma_matrix, self_consistent
    from .solver import solve

    if system.periodic:
        raise ValueError("Campo finito con dipolos: solo sistemas finitos.")
    ground_scc = system.model.scc if ground_scc is None else ground_scc
    d_matrices = dipole_matrices(system)
    kernel = multipole_kernel(system, U)
    atoms = system.basis.atoms
    n = len(atoms)
    gamma = gamma_matrix(system, U)
    projector = atom_projector(system)
    owner = np.argmax(projector, axis=0)
    n0 = _neutral(system)
    neutral_atom = projector @ n0
    if ground_scc:
        ground = self_consistent(system, U=U, kT=kT, tol=tol).solution
        kernel = kernel.copy()
        kernel[:n, :n] = 0.0                     # counted by the ground functional
    else:
        ground = solve(system, kT=kT)
    dq0, p0 = _multipoles(ground, system, neutral_atom, d_matrices)
    m0 = np.concatenate([dq0, p0.ravel()])
    positions = system.atoms.get_positions()[atoms]
    centred = positions - positions.mean(axis=0)
    alpha = np.zeros((3, 3))
    for axis in range(3):
        dipoles = []
        for sign in (1.0, -1.0):
            e_field = np.zeros(3)
            e_field[axis] = sign * field
            m_in = m0.copy()
            inputs, residuals = [], []
            for _ in range(max_iter):
                delta = m_in - m0
                potential = kernel @ delta
                shift_atom = potential[:n] + centred @ e_field
                if ground_scc:
                    shift_atom = shift_atom + gamma @ m_in[:n]
                w = potential[n:].reshape(n, 3) + e_field
                extra = np.einsum("am,ax,xmn->mn", projector, w, d_matrices)
                extra = 0.5 * (extra + extra.T)
                solution = _solve(system, shift_atom[owner], extra, 0.0, kT)
                dq, p = _multipoles(solution, system, neutral_atom, d_matrices)
                m_out = np.concatenate([dq, p.ravel()])
                residual = m_out - m_in
                if np.abs(residual).max() < tol:
                    break
                inputs = (inputs + [m_in])[-6:]
                residuals = (residuals + [residual])[-6:]
                m_in = anderson(inputs, residuals, mixing)
            else:
                raise RuntimeError("Campo finito con dipolos: sin converger.")
            dipoles.append(-(dq @ centred + p.sum(axis=0)))
        alpha[:, axis] = (dipoles[0] - dipoles[1]) / (2 * field)
    return 0.5 * (alpha + alpha.T) * COULOMB
