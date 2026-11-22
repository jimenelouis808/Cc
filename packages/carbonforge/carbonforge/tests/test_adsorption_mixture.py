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
