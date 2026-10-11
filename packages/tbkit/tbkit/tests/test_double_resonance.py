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


def test_overtone_at_any_q_matches_the_four_processes(phonons, electrons):
    """General q (no supercell): q one mesh step off K, on a 13×13 mesh that avoids K."""
    from tbkit.hamiltonian import System

    laser, gamma, n = 2.0, 0.1, 13
    atoms = gr.graphene_cell(A_CC)
    system = System.build(atoms, pi_model(strain_beta=3.37))
    bands, _ = dr.mesh_bands(system, (n, n, 1), 10.0, gr._POLARIZATIONS)
    coupling = dr.Coupling(system)
    qi = (9, 4, 0)                                    # (2/3, 1/3) ≈ (8.67, 4.33)/13
    qc = np.array(qi, float) / n @ atoms.cell.reciprocal() * 2 * np.pi
    k_frac = np.array([[i / n, j / n, 0] for i in range(n) for j in range(n)])
    kc = k_frac @ atoms.cell.reciprocal() * 2 * np.pi
    e_k, _ = electrons.states(kc)
    optics = gr._optical(electrons, kc, gr._POLARIZATIONS)
    f, v = phonons.modes(qc)
    masses = atoms.get_masses()
    nu = max(gr._in_plane(v), key=lambda i: f[i])             # the TO branch
    u1 = v[nu] / np.sqrt(masses)[:, None] * gr.zero_point(f[nu])
    reference = np.zeros((2, 2), complex)
    for sign in (1.0, -1.0):
        first, second = (u1, np.conj(u1)) if sign > 0 else (np.conj(u1), u1)
        w = f[nu] * gr.CM1_TO_EV
        reference += gr._four_processes(electrons, kc, e_k, optics, sign * qc, first, second,
                                        w, w, laser, gamma) / len(kc)
    e_lattice = v[nu] * np.exp(1j * (atoms.positions @ qc))[:, None]
    ours = dr.overtone_q(bands, (n, n, 1), coupling, qi, f[nu], e_lattice, masses, laser, gamma)
    assert np.allclose(ours, reference, rtol=1e-6, atol=1e-12 * np.abs(reference).max())


def test_defect_vertex_is_the_local_potential_between_bloch_states():
    from tbkit.hamiltonian import System

    atoms = gr.graphene_cell(A_CC)
    system = System.build(atoms, pi_model(strain_beta=3.37))
    bands, _ = dr.mesh_bands(system, (5, 5, 1), 10.0, gr._POLARIZATIONS)
    v0 = 0.7
    vertex = dr.DefectVertex(system, {(0, 0, (0, 0, 0)): (np.array([[v0]]), None)})
    to, frm = bands[7], bands[3]
    kt, kf = (system.kpoint_cartesian(b.k) for b in (to, frm))
    expected = v0 * np.exp(-1j * (kt - kf) @ atoms.positions[0]) * \
        np.conj(to.cc[0])[:, None] * frm.cc[0][None, :]
    assert np.allclose(vertex(to, frm, None, "cc"), expected)
    # a defect-activated amplitude is linear in the defect potential
    coupling = dr.Coupling(system)
    phonons = gr.load_phonons("gpaw")
    q = (2, 1, 0)
    qc = np.array(q, float) / 5 @ atoms.cell.reciprocal() * 2 * np.pi
    f, v = phonons.modes(qc)
    nu = int(np.argmax(f))
    e_lattice = v[nu] * np.exp(1j * (atoms.positions @ qc))[:, None]
    phonon = dr.PhononVertex(coupling, e_lattice, atoms.get_masses(), f[nu])
    one = dr.two_vertices_q(bands, (5, 5, 1), q, phonon, vertex, 2.0)
    double = dr.DefectVertex(system, {(0, 0, (0, 0, 0)): (np.array([[2 * v0]]), None)})
    assert np.allclose(dr.two_vertices_q(bands, (5, 5, 1), q, phonon, double, 2.0), 2 * one)


def test_one_vertex_equals_first_order(phonons):
    from tbkit.hamiltonian import System

    atoms = gr.graphene_cell(A_CC)
    model = pi_model(strain_beta=3.37)
    f, v = phonons.modes(np.zeros(3))
    optical = [i for i in gr._in_plane(v) if f[i] > 500]
    modes = dr.real_modes(f[optical], v[optical])
    n = 7
    k = np.array([[i / n, j / n, 0.0] for i in range(n) for j in range(n)])
    data = dr.electron_phonon(atoms, model, modes, k, np.full(len(k), 1 / len(k)),
                              window_ev=10.0, step=0.0005)
    reference = dr.first_order(data, modes, 2.0, 0.1)
    system = System.build(atoms, model)
    bands, _ = dr.mesh_bands(system, (n, n, 1), 10.0, gr._POLARIZATIONS)
    coupling = dr.Coupling(system)
    for mu, mode in enumerate(modes):
        ours = dr.one_vertex(bands, dr.PhononVertex(coupling, mode.vector, atoms.get_masses(),
                                                    mode.frequency), 2.0, 0.1)
        assert np.allclose(np.abs(ours), np.abs(reference[mu]), rtol=1e-4, atol=1e-9)


def test_phonon_pairs_diagonal_is_the_overtone(phonons):
    from tbkit.hamiltonian import System

    atoms = gr.graphene_cell(A_CC)
    system = System.build(atoms, pi_model(strain_beta=3.37))
    n = 7
    bands, _ = dr.mesh_bands(system, (n, n, 1), 10.0, gr._POLARIZATIONS)
    coupling = dr.Coupling(system)
    qi = (5, 2, 0)
    qc = np.array(qi, float) / n @ atoms.cell.reciprocal() * 2 * np.pi
    f, v = phonons.modes(qc)
    keep = [i for i in gr._in_plane(v) if f[i] > 1000]
    vectors = np.array([v[i] * np.exp(1j * (atoms.positions @ qc))[:, None] for i in keep])
    masses = atoms.get_masses()
    w = dr.phonon_pairs_q(bands, (n, n, 1), coupling, qi, f[keep], vectors, masses, 2.0)
    for a, nu in enumerate(keep):
        single = dr.overtone_q(bands, (n, n, 1), coupling, qi, f[nu], vectors[a], masses, 2.0)
        assert w[a, a] == pytest.approx(np.sum(np.abs(single) ** 2), rel=1e-8)
    assert w.shape == (len(keep), len(keep)) and np.all(w >= 0)


def test_fast_pairs_match_the_exact_ones(phonons):
    """Many bands (a 2×1 supercell: 4 π bands) so that every index of the chain is
    exercised with unequal sizes, and branches spread over ~200 cm⁻¹."""
    from tbkit.hamiltonian import System

    primitive = gr.graphene_cell(A_CC)
    atoms = primitive.repeat((2, 1, 1))
    system = System.build(atoms, pi_model(strain_beta=3.37))
    n = 5
    bands, _ = dr.mesh_bands(system, (n, n, 1), 3.0, gr._POLARIZATIONS, fermi=0.8)
    assert any(len(b.ec) != len(b.ev) for b in bands)
    coupling = dr.Coupling(system)
    qi = (2, 1, 0)
    qc = np.array(qi, float) / n @ atoms.cell.reciprocal() * 2 * np.pi
    f, v = phonons.modes(qc)
    keep = [i for i in gr._in_plane(v) if f[i] > 500]
    # primitive modes placed on both cells of the supercell (a valid q of the supercell)
    vectors = np.array([np.concatenate([v[i], v[i] * np.exp(1j * qc @ primitive.cell[0])])
                        * np.exp(1j * (atoms.positions @ qc))[:, None] / np.sqrt(2)
                        for i in keep])
    args = (bands, (n, n, 1), coupling, qi, f[keep], vectors, atoms.get_masses(), 2.0)
    assert np.allclose(dr.phonon_pairs_q_fast(*args), dr.phonon_pairs_q(*args), rtol=1e-3)


def test_fast_chain_algebra_with_random_states():
    """The factorised chain equals the direct one for random states and couplings with
    different numbers of bands at k, k+q and k−q (catches any transposed index)."""
    rng = np.random.default_rng(5)

    def bands(nv, nc):
        return dr.Bands(np.zeros(3), np.sort(rng.uniform(-3, -0.1, nv)),
                        np.sort(rng.uniform(0.1, 3, nc)), None, None,
                        rng.normal(size=(2, nc, nv)) + 1j * rng.normal(size=(2, nc, nv)))

    k, a, b = bands(3, 4), bands(5, 2), bands(2, 6)
    sizes = {id(k): (len(k.ec), len(k.ev)), id(a): (len(a.ec), len(a.ev)),
             id(b): (len(b.ec), len(b.ev))}

    class Random:
        energy = 0.0

        def __init__(self, seed):
            self.cache, self.seed = {}, seed

        def __call__(self, to, frm, q, block):
            key = (id(to), id(frm), block)
            if key not in self.cache:
                r = np.random.default_rng(hash((self.seed,) + key) % 2 ** 32)
                i = 0 if block == "cc" else 1
                shape = (sizes[id(to)][i], sizes[id(frm)][i])
                self.cache[key] = r.normal(size=shape) + 1j * r.normal(size=shape)
            return self.cache[key]

    first = [Random(s) for s in range(3)]
    second = [Random(10 + s) for s in range(3)]
    w = np.array([0.155, 0.16, 0.168])
    direct = dr._pairs_ordered(k, a, b, np.zeros(3), first, second, w, 2.0, 0.1)
    fast = dr._pairs_ordered_fast(k, a, b, np.zeros(3), first, second, w, 2.0, 0.1, order=9)
    assert np.allclose(fast, direct, rtol=1e-6, atol=1e-9 * np.abs(direct).max())
