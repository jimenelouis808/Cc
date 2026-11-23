"""H2/D2 quantum sieving through the Feynman-Hibbs term."""

import numpy as np
import pytest

from carbonforge.adsorption.henry import henry_constant
from carbonforge.adsorption.potentials import (D2_SELF, H2_SELF, MASS_D2, MASS_H2,
                                               feynman_hibbs, with_carbon)
from carbonforge.builders import build_graphene_supercell


def _slit(h):
    sheet = build_graphene_supercell(nx=10, ny=6, vacuum=10.0)
    sheet.set_cell(np.diag([sheet.cell[0, 0], sheet.cell[1, 1], h]))
    sheet.center(axis=2)
    stack = sheet * (1, 1, int(np.ceil(24.0 / h)))
    stack.set_pbc(True)
    stack.wrap()
    return stack


def test_d2_is_h2_with_twice_the_mass():
    assert D2_SELF.epsilon == H2_SELF.epsilon
    assert D2_SELF.sigma == H2_SELF.sigma
    assert D2_SELF.reduced_mass / H2_SELF.reduced_mass == pytest.approx(
        MASS_D2 / MASS_H2)


def test_the_quantum_correction_halves_with_the_mass():
    r = np.linspace(3.0, 6.0, 20)
    h, d = with_carbon(H2_SELF), with_carbon(D2_SELF)
    lj = feynman_hibbs(r, h, 1e9)          # classical limit
    dh = feynman_hibbs(r, h, 77.0) - lj
    dd = feynman_hibbs(r, d, 77.0) - lj
    assert np.allclose(dd / dh, MASS_H2 / MASS_D2)


def test_d2_is_preferred_and_the_preference_fades_when_hot():
    atoms = _slit(6.0)
    def ratio(t):
        h = henry_constant(atoms, t, samples=40_000, seed=1)
        d = henry_constant(atoms, t, samples=40_000, seed=1, adsorbate=D2_SELF)
        return d.henry_coefficient / h.henry_coefficient
    cold, hot = ratio(77.0), ratio(1000.0)
    assert cold > 1.2
    assert cold > hot > 1.0
    assert hot < 1.05
