"""tbkit.double_resonance against the analytic graphene module.

The supercell route (frozen phonons, all bands, any tbkit model) must give exactly
the amplitudes of :mod:`tbkit.graphene` for graphene's π model: the G band at first
order on the primitive cell, and the two-phonon (overtone) amplitude at q = K in a
3×3 supercell, where K folds to Γ. Meshes are shifted off the Dirac point, where the
two degenerate states have no preferred basis in either code.
"""

from __future__ import annotations

import numpy as np
import pytest

from tbkit import double_resonance as dr
from tbkit import graphene as gr
from tbkit.params import pi_model

A_CC = 1.42


@pytest.fixture(scope="module")
def phonons():
    return gr.load_phonons("gpaw")


@pytest.fixture(scope="module")
def electrons():
    return gr.PiElectrons.from_parameters(a_cc=A_CC)


def _mesh(n: int) -> np.ndarray:
    grid = (np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1)
            .reshape(-1, 2) + 0.5) / n
    return np.c_[grid, np.zeros(len(grid))]


def test_first_order_is_the_g_band(phonons, electrons):
    laser, nk = 2.0, 31
    f, v = phonons.modes(np.zeros(3))
    optical = [i for i in gr._in_plane(v) if f[i] > 500]
    modes = dr.real_modes(f[optical], v[optical])
    atoms = gr.graphene_cell(A_CC)
    k = _mesh(nk)
    data = dr.electron_phonon(atoms, pi_model(strain_beta=3.37), modes, k,
                              np.full(len(k), 1 / len(k)), window_ev=10.0, step=0.0005)
    ours = np.sum(np.abs(dr.first_order(data, modes, laser, 0.1)) ** 2)
    kc = k @ atoms.cell.reciprocal() * 2 * np.pi
    e, _ = electrons.states(kc)
    optics = gr._optical(electrons, kc, gr._POLARIZATIONS)
    reference = 0.0
    for nu in optical:
        u = v[nu] / np.sqrt(atoms.get_masses())[:, None] * gr.zero_point(f[nu])
        g = electrons.coupling(kc, np.zeros(3), u)
        diff = gr._band_elements(electrons, kc, kc, g, (1, 1)) - \
            gr._band_elements(electrons, kc, kc, g, (0, 0))
        den = (laser - (e[:, 1] - e[:, 0]) + 0.1j) * \
              (laser - f[nu] * gr.CM1_TO_EV - (e[:, 1] - e[:, 0]) + 0.1j)
        for s in optics:
            for i in optics:
                reference += abs(np.mean(np.conj(s) * diff * i / den)) ** 2
    assert ours == pytest.approx(reference, rel=1e-4)


def test_overtone_at_k_matches_the_four_processes(phonons, electrons):
    laser, gamma, n_super, ns = 2.0, 0.1, 3, 6
    primitive = gr.graphene_cell(A_CC)
    supercell = primitive.repeat((n_super, n_super, 1))
    K = gr._dirac_points(phonons)[0]
    f, v = phonons.modes(K)
    k_prim = _mesh(n_super * ns) @ primitive.cell.reciprocal() * 2 * np.pi
    e_k, _ = electrons.states(k_prim)
    optics = gr._optical(electrons, k_prim, gr._POLARIZATIONS)
    k_super = _mesh(ns)
    positions = supercell.positions
    checked = 0
    for nu in gr._in_plane(v):
        if f[nu] < 1000:
            continue
        omega = f[nu] * gr.CM1_TO_EV
        u1 = v[nu] / np.sqrt(primitive.get_masses())[:, None] * gr.zero_point(f[nu])
        amplitude = np.zeros((2, 2), complex)
        for sign in (1.0, -1.0):
            first, second = (u1, np.conj(u1)) if sign > 0 else (np.conj(u1), u1)
            amplitude += gr._four_processes(electrons, k_prim, e_k, optics, sign * K, first,
                                            second, omega, omega, laser, gamma) / len(k_prim)
        reference = np.sum(np.abs(amplitude) ** 2)
        z = np.array([v[nu][a % 2] * np.exp(1j * K @ positions[a])
                      for a in range(len(supercell))]) / n_super
        modes = dr.real_modes(np.array([f[nu], f[nu]]), np.array([z, np.conj(z)]))
        data = dr.electron_phonon(supercell, pi_model(strain_beta=3.37), modes, k_super,
                                  np.full(len(k_super), 1 / len(k_super)), window_ev=10.0,
                                  step=0.0005)
        pairs = [(0, 0), (1, 1), (0, 1)]
        ours = dr.two_phonon_intensity(dr.second_order(data, modes, pairs, laser, gamma),
                                       pairs).sum()
        assert ours == pytest.approx(reference, rel=1e-4)
        checked += 1
    assert checked >= 2


def test_real_modes_span_the_degenerate_set():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    z /= np.linalg.norm(z)
    modes = dr.real_modes(np.array([1300.0, 1300.0]), np.array([z, np.conj(z)]))
    basis = np.array([m.vector.ravel() for m in modes])
    assert np.allclose(basis @ basis.T, np.eye(2))
    assert np.allclose(basis.T @ (basis @ z.real.ravel()), z.real.ravel())
