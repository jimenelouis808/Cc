"""Graphene Raman by perturbation theory (tbkit.graphene).

Closed-form and symmetry checks: the electron-phonon matrix element equals an
explicit lattice sum of ΔH; the Dirac point has f(K) = 0; phonons obey the
acoustic sum rule, D(-q) = D(q)* and the LA/LO degeneracy at K; the G band
is E2g (parallel = cross polarization); and, in the slow test, the 2D band
comes from the TO branch at |q - K| ≈ E_L/(ħ v_F) and shifts up with the
laser (double resonance).
"""

from __future__ import annotations

import numpy as np
import pytest

from tbkit.graphene import GraphenePhonons, PiElectrons, g_band, graphene_cell
from tbkit.params import xu_carbon


@pytest.fixture(scope="module")
def phonons():
    return GraphenePhonons.from_model(xu_carbon(), n=4, kmesh=4, kT=0.05, a_cc=1.4176)


@pytest.fixture(scope="module")
def electrons(phonons):
    a = float(np.linalg.norm(phonons.atoms.positions[1] - phonons.atoms.positions[0]))
    return PiElectrons.from_parameters(a_cc=a)


def test_coupling_equals_explicit_lattice_sum():
    el = PiElectrons.from_parameters()
    cell = graphene_cell(el.a_cc)
    lattice, tau = cell.cell.array[:2], cell.get_positions()
    b = 2 * np.pi * np.linalg.inv(cell.cell.array).T[:2]
    n = 6
    rng = np.random.default_rng(1)
    k = (rng.integers(0, n) * b[0] + rng.integers(0, n) * b[1]) / n
    q = (rng.integers(0, n) * b[0] + rng.integers(0, n) * b[1]) / n
    eps = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    g_ab = g_ba = 0.0
    for i1 in range(n):
        for i2 in range(n):
            r_a = i1 * lattice[0] + i2 * lattice[1] + tau[0]
            for d in el.deltas:
                r_b = r_a + d
                change = el.dt * (d / el.a_cc) @ (eps[1] * np.exp(1j * q @ r_b)
                                                  - eps[0] * np.exp(1j * q @ r_a))
                g_ab += np.exp(-1j * (k + q) @ r_a) * change * np.exp(1j * k @ r_b) / n ** 2
                g_ba += np.exp(-1j * (k + q) @ r_b) * change * np.exp(1j * k @ r_a) / n ** 2
    g = el.coupling(k, q, eps)
    assert g[0, 1] == pytest.approx(g_ab, abs=1e-10)
    assert g[1, 0] == pytest.approx(g_ba, abs=1e-10)


def test_dirac_point(electrons, phonons):
    assert abs(electrons.f(phonons.special_points()["K"])) < 1e-10
    assert electrons.t == pytest.approx(-2.7 * np.exp(-3.37 * (1.4176 / 1.42 - 1)), rel=1e-9)


class TestPhonons:
    def test_acoustic_sum_rule(self, phonons):
        frequencies, _ = phonons.modes(np.zeros(3))
        assert np.abs(frequencies[:3]).max() < 5.0

    def test_time_reversal(self, phonons):
        q = 0.3 * phonons.reciprocal()[0] + 0.1 * phonons.reciprocal()[1]
        assert phonons.dynamical_matrix(-q) == pytest.approx(
            phonons.dynamical_matrix(q).conj(), abs=1e-8)

    def test_degeneracies_at_k(self, phonons):
        frequencies, _ = phonons.modes(phonons.special_points()["K"])
        pairs = np.diff(np.sort(frequencies))
        assert (pairs < 1.0).sum() >= 2          # ZA/ZO and LA/LO meet at K

    def test_round_trip(self, phonons):
        again = GraphenePhonons.from_dict(phonons.to_dict())
        q = phonons.special_points()["M"]
        assert again.modes(q)[0] == pytest.approx(phonons.modes(q)[0], abs=1e-4)


def test_g_band_is_e2g(electrons, phonons):
    result = g_band(electrons, phonons, 2.41, nk=240)
    assert result["intensity"] > 0
    assert result["parallel"] == pytest.approx(result["cross"], rel=0.02)


@pytest.mark.slow
def test_2d_band_double_resonance(electrons, phonons):
    from tbkit.graphene import double_resonance, second_order_spectrum

    peaks = []
    for laser in (1.96, 2.54):
        result = double_resonance(electrons, phonons, laser, dk=0.015, dq=0.05, workers=2)
        k = phonons.special_points()["K"]
        kp = (phonons.reciprocal()[0] + 2 * phonons.reciprocal()[1]) / 3
        distance = np.minimum(np.linalg.norm(result["q"] - k, axis=1),
                              np.linalg.norm(result["q"] - kp, axis=1))
        near_k = distance < 0.7
        mean_q = np.average(distance[near_k], weights=result["weights"][near_k])
        hbar_vf = 1.5 * abs(electrons.t) * electrons.a_cc
        assert mean_q == pytest.approx(laser / hbar_vf, rel=0.25)
        grid, intensity = second_order_spectrum(result, np.arange(2400, 3200, 1.0))
        peaks.append(grid[np.argmax(intensity)])
    assert peaks[1] > peaks[0]                   # the 2D band disperses upwards
