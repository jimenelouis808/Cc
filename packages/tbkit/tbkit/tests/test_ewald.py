"""Periodic SCC: Ewald γ against closed forms, the finite limit and finite differences."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from tbkit.ewald import ewald_coulomb, periodic_gamma
from tbkit.hamiltonian import System
from tbkit.params import load_parameters
from tbkit.scc import energy_and_forces, self_consistent

MADELUNG_NACL = 1.747564594633


def _rock_salt(a=2.0):
    fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    return np.vstack([fcc, fcc + [.5, 0, 0]]) * a, np.array([1] * 4 + [-1] * 4)


def test_madelung_constant_of_rock_salt_in_two_cells():
    a = 2.0
    positions, q = _rock_salt(a)
    phi, _ = ewald_coulomb(positions, np.eye(3) * a)
    assert -0.5 * q @ phi @ q / 4 * (a / 2) == pytest.approx(MADELUNG_NACL, abs=1e-9)
    primitive = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * a
    phi, _ = ewald_coulomb(np.array([[0, 0, 0], [a / 2, 0, 0]]), primitive)
    q = np.array([1, -1])
    assert -0.5 * q @ phi @ q * (a / 2) == pytest.approx(MADELUNG_NACL, abs=1e-9)


def test_ewald_and_periodic_gamma_gradients_are_derivatives():
    rng = np.random.default_rng(0)
    positions, q = _rock_salt(3.0)
    positions = positions + rng.normal(scale=0.05, size=positions.shape)
    cell = np.eye(3) * 3.0
    u = np.where(q > 0, 8.0, 11.0)
    for function in (lambda p: ewald_coulomb(p, cell), lambda p: periodic_gamma(p, cell, u)):
        _, grad = function(positions)
        analytic = np.einsum("a,b,abx->ax", q, q, grad)
        numeric = np.zeros_like(positions)
        for i in range(len(positions)):
            for k in range(3):
                e = []
                for s in (1, -1):
                    moved = positions.copy()
                    moved[i, k] += s * 1e-5
                    e.append(0.5 * q @ function(moved)[0] @ q)
                numeric[i, k] = (e[0] - e[1]) / 2e-5
        assert np.allclose(analytic, numeric, atol=1e-6)


def test_periodic_gamma_is_ko_on_site_and_converged_in_the_short_range_cutoff():
    positions, _ = _rock_salt(3.0)
    u = np.full(len(positions), 9.0)
    g30, _ = periodic_gamma(positions, np.eye(3) * 3.0, u)
    g60, _ = periodic_gamma(positions, np.eye(3) * 3.0, u, r_short=60.0)
    q = np.array([1] * 4 + [-1] * 4)
    # ±1 charges 1.5 Å apart, the hardest case: the r⁻⁵ rest is converged
    assert abs(q @ (g30 - g60) @ q) < 1e-4
    assert np.allclose(g30, g30.T)


def test_molecule_in_a_large_box_gives_the_finite_result():
    model = load_parameters("xu_chno")
    mol = molecule("CH3OH")
    mol.center(vacuum=12.0)
    finite = self_consistent(System.build(mol, model), tol=1e-10)
    box = mol.copy()
    box.pbc = True
    periodic = self_consistent(System.build(box, model), kpts=np.zeros((1, 3)),
                               weights=np.ones(1), tol=1e-10)
    assert energy_and_forces(periodic, False)[0] == pytest.approx(
        energy_and_forces(finite, False)[0], abs=1e-3)
    assert max(abs(finite.charges[i] - periodic.charges[i]) for i in finite.charges) < 1e-3


def test_periodic_scc_forces_are_energy_derivatives_in_hexagonal_bn():
    from tbkit.kpoints import mesh

    model = load_parameters("xu_chnob")
    a = 2.504
    bn = Atoms("BN", scaled_positions=[[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]],
               cell=[[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 15]],
               pbc=(True, True, False))
    bn.positions[0] += [0.03, -0.02, 0.04]
    k, w = mesh(bn, 4)
    result = self_consistent(System.build(bn, model), kpts=k, weights=w, tol=1e-11)
    assert result.converged and result.charges[0] > 0.3          # B gives charge to N
    assert sum(result.charges.values()) == pytest.approx(0, abs=1e-8)
    _, forces, _ = energy_and_forces(result)
    numeric = np.zeros_like(forces)
    for i in range(2):
        for c in range(3):
            e = []
            for s in (1, -1):
                moved = bn.copy()
                moved.positions[i, c] += s * 1e-4
                r = self_consistent(System.build(moved, model), kpts=k, weights=w, tol=1e-11)
                e.append(energy_and_forces(r, need_forces=False)[0])
            numeric[i, c] = -(e[0] - e[1]) / 2e-4
    assert np.allclose(forces, numeric, atol=1e-5)
