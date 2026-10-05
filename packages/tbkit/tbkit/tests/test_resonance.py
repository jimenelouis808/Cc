"""Resonant first-order Raman (tbkit.resonance) and the complex α(ω + iη).

Checks with known answers: the static limit is the existing linear response;
far below the gap the resonant activities are the non-resonant ones; Im α is
an absorption (positive); symmetry-forbidden modes stay forbidden at any
laser; and a polyene's C=C stretch is enhanced by orders of magnitude when
the laser meets its π→π* transition.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk, molecule
from ase.optimize import BFGS

from tbkit.calculator import TBCalculator
from tbkit.hamiltonian import System
from tbkit.kpoints import mesh
from tbkit.optics import (
    dynamic_polarizability_finite,
    dynamic_polarizability_periodic,
    polarizability_linear_response,
    polarizability_periodic,
)
from tbkit.params import load_parameters, xu_carbon
from tbkit.raman import raman
from tbkit.resonance import complex_invariants, resonant_raman


@pytest.fixture(scope="module")
def chn():
    return load_parameters("xu_chn")


@pytest.fixture(scope="module")
def butadiene(chn):
    atoms = molecule("butadiene")
    atoms.calc = TBCalculator(chn)
    BFGS(atoms, logfile=None).run(fmax=0.002, steps=400)
    return atoms.copy()


class TestComplexPolarizability:
    def test_static_limit_finite(self, chn):
        system = System.build(molecule("C5H5N"), chn)
        static = dynamic_polarizability_finite(system, [0.0], eta=0.0)[0]
        assert static.real == pytest.approx(polarizability_linear_response(system), abs=1e-10)
        assert np.abs(static.imag).max() < 1e-12

    def test_static_limit_crystal(self):
        system = System.build(bulk("C", "diamond", a=3.567), xu_carbon())
        k, w = mesh(system.atoms, 4)
        static = dynamic_polarizability_periodic(system, [0.0], eta=0.0, kpts=k, weights=w)[0]
        assert static.real == pytest.approx(polarizability_periodic(system, k, w)[0],
                                            abs=1e-10)

    def test_absorption_is_positive(self, chn):
        system = System.build(molecule("C6H6"), chn)
        alpha = dynamic_polarizability_finite(system, np.linspace(0.5, 8, 16), eta=0.1)
        assert np.all(np.trace(alpha.imag, axis1=1, axis2=2) > -1e-9)
        assert np.trace(alpha.imag, axis1=1, axis2=2).max() > 1.0

    def test_invariants_of_a_real_tensor(self):
        from tbkit.raman import invariants

        tensor = np.array([[1.0, 0.2, 0.0], [0.2, -0.5, 0.1], [0.0, 0.1, 0.3]])
        mean, gamma2 = invariants(tensor)
        assert complex_invariants(tensor) == pytest.approx((mean ** 2, gamma2))
        assert complex_invariants(1j * tensor) == pytest.approx((mean ** 2, gamma2))


class TestResonantRaman:
    @pytest.fixture(scope="class")
    @staticmethod
    def result(chn, butadiene):
        return resonant_raman(butadiene, chn, [0.2, 2.0, 4.0], eta=0.1)

    def test_far_below_the_gap_is_the_non_resonant_result(self, chn, butadiene, result):
        static = raman(butadiene, chn)
        assert result.frequencies == pytest.approx(static.frequencies)
        strong = static.activities > 1e-3 * static.activities.max()
        assert result.activities[0][strong] == pytest.approx(static.activities[strong],
                                                             rel=0.02)

    def test_polyene_cc_stretch_is_resonantly_enhanced(self, result):
        cc = int(np.argmin(np.abs(result.frequencies - 1690)))
        profile = result.profile(result.frequencies[cc])
        assert profile[2] > 100 * profile[0]                   # at the π→π* transition
        assert np.argmax(result.activities[2]) == cc           # and it dominates

    def test_forbidden_modes_stay_forbidden(self, result):
        """Butadiene is C2h: its Au and Bu modes are Raman-inactive at any laser."""
        strongest = result.activities.max(axis=1, keepdims=True)
        inactive = result.activities < 1e-6 * strongest
        assert inactive.sum(axis=1).min() >= 8                 # 9 Au + Bu modes of 24

    def test_at_gives_a_raman_result(self, result):
        from tbkit.raman import spectrum

        at = result.at(4.0)
        grid, intensity = spectrum(at, laser_nm=None, temperature_k=None)
        assert intensity.max() > 0 and "η" in at.method

    def test_needs_broadening(self, chn, butadiene):
        with pytest.raises(ValueError, match="η"):
            resonant_raman(butadiene, chn, [3.0], eta=0.0)


@pytest.mark.parametrize("case", ["benzene", "diamond"])
def test_derivative_along_modes_equals_the_full_raman_tensor(case, tmp_path):
    """Σ ∂α/∂x·L_k (6N α) and (α(x + hL_k) - α(x - hL_k))/2h (2 α per mode) are the
    same derivative: finite (screened) and periodic α, with the per-mode cache. Small
    steps: the two differ at O(δ²), and moving every atom of a mode at once makes
    that error larger (4 % in diamond at δ = 0.01 Å, near resonance)."""
    from ase.build import bulk, molecule

    from tbkit.modes import vibrations
    from tbkit.params import load_parameters, xu_carbon
    from tbkit.resonance import resonant_raman

    if case == "benzene":
        atoms, model, kmesh = molecule("C6H6"), load_parameters("xu_chn"), 1
    else:
        atoms, model, kmesh = bulk("C", "diamond", a=3.56), xu_carbon(), 4
    vib = vibrations(atoms, model, kmesh=kmesh, kT=0.05)
    phonons = (vib.frequencies, vib.modes)
    lasers = [2.0, 2.5]
    common = {"eta": 0.2, "kmesh": kmesh, "kT": 0.05, "phonons": phonons, "delta": 0.002}
    full = resonant_raman(atoms, model, lasers, **common)
    from tbkit.raman import internal_modes

    order = list(internal_modes(atoms, vib.frequencies, []))        # full's mode order
    chosen = [int(order[k]) for k in np.argsort(full.activities[1])[-3:]]
    part = resonant_raman(atoms, model, lasers, select=chosen, cache_dir=tmp_path, **common)
    for j, k in enumerate(part.mode_indices):
        i = order.index(k)
        assert np.allclose(part.tensors[:, j], full.tensors[:, i],
                           atol=2e-3 * np.abs(full.tensors).max())
    again = resonant_raman(atoms, model, lasers, select=chosen, cache_dir=tmp_path, **common)
    assert np.array_equal(again.tensors, part.tensors)
    with pytest.raises(ValueError):
        resonant_raman(atoms, model, [2.0], select=chosen, cache_dir=tmp_path, **common)
