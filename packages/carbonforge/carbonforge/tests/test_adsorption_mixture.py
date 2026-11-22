"""CO2/N2 mixture GCMC: the reservoir, the moves, the bookkeeping."""

import numpy as np
import pytest

from carbonforge.adsorption import co2
from carbonforge.adsorption.mixture import (peng_robinson_mixture,
                                            run_mixture_gcmc)


class TestTheReservoir:
    def test_a_pure_component_is_the_pure_equation_of_state(self):
        z, phi = peng_robinson_mixture(300.0, 2.0, (co2.CO2, co2.N2),
                                       (1.0, 0.0))
        z_pure, phi_pure = co2.peng_robinson(300.0, 2.0)
        assert z == pytest.approx(z_pure, rel=1e-6)
        assert phi[0] == pytest.approx(phi_pure, rel=1e-6)

    def test_flue_gas_at_one_bar_is_nearly_ideal(self):
        z, phi = peng_robinson_mixture(298.0, 0.1, (co2.CO2, co2.N2),
                                       (0.15, 0.85))
        assert abs(z - 1) < 2e-3
        assert np.all(np.abs(phi - 1) < 5e-3)

    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError):
            peng_robinson_mixture(298.0, 0.1, (co2.CO2, co2.N2), (0.2, 0.7))


class TestTheSimulation:
    def test_the_dilute_empty_box_is_the_ideal_gas_mixture(self):
        """No wall, low pressure: each species at its partial density,
        so the selectivity is 1 and the composition is the gas's."""
        box = np.array([30.0, 30.0, 30.0])
        r = run_mixture_gcmc(None, 300.0, 0.5, (0.3, 0.7), edges=box,
                             equilibration=4000, production=40000, seed=1)
        ideal = 0.5e6 * np.prod(box) * 1e-30 / (co2.BOLTZMANN * 300.0)
        assert sum(r.mean_counts) == pytest.approx(ideal, rel=0.12)
        assert r.adsorbed_fractions[0] == pytest.approx(0.3, abs=0.06)
        assert r.energy_drift_k < 1e-6

    def test_swaps_keep_the_energy_bookkeeping(self):
        box = np.array([25.0, 25.0, 25.0])
        r = run_mixture_gcmc(None, 250.0, 3.0, (0.5, 0.5), edges=box,
                             equilibration=1000, production=3000, seed=2)
        assert r.acceptance["swap"] > 0.0
        assert r.energy_drift_k < 1e-6

    def test_n2_is_a_neutral_quadrupole(self):
        assert sum(co2.N2.charge) == pytest.approx(0.0)
        # Its quadrupole is about a third of CO2's (same sign).
        def theta(sp):
            return 2 * sp.charge[0] * sp.half ** 2
        assert 0.25 < theta(co2.N2) / theta(co2.CO2) < 0.4


class TestRigidMolecules:
    def test_a_linear_frame_is_the_old_molecule(self):
        rot = co2.random_rotation(np.random.default_rng(0))
        centre = np.array([3.0, 4.0, 5.0])
        sites = co2.place(co2.CO2, centre, rot)
        assert np.allclose(sites, co2.molecule(centre, sites[2] - sites[1]))

    def test_water_is_spce(self):
        f = co2.WATER.frame()
        oh = np.linalg.norm(f[[0, 2]] - f[1], axis=1)
        assert np.allclose(oh, 1.0)
        cos = f[0] @ f[2] / (oh[0] * oh[1])
        assert np.degrees(np.arccos(cos)) == pytest.approx(109.47, abs=1e-6)
        assert sum(co2.WATER.charge) == pytest.approx(0.0)

    def test_rotations_are_proper(self):
        rng = np.random.default_rng(1)
        for _ in range(5):
            r = co2.random_rotation(rng)
            assert np.allclose(r @ r.T, np.eye(3))
            assert np.linalg.det(r) == pytest.approx(1.0)

    def test_a_ternary_with_water_keeps_its_bookkeeping(self):
        box = np.array([25.0, 25.0, 25.0])
        r = run_mixture_gcmc(None, 298.0, 0.1, (0.15, 0.82, 0.03),
                             species=(co2.CO2, co2.N2, co2.WATER), edges=box,
                             equilibration=500, production=2000, seed=4)
        assert r.energy_drift_k < 1e-6
        assert len(r.mean_counts) == 3
