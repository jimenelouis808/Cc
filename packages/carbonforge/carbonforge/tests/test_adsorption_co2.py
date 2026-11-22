"""Tests for the three-site CO2 GCMC (adsorption/co2.py).

Checks the parts a plausible-looking isotherm can hide: the Ewald sum
converged in its own parameter, the incremental energy against a full
recomputation, the reservoir's ideal-gas limit, and the empty box
returning the reservoir's own density.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from carbonforge.adsorption import co2


def _random_molecules(count: int, box: np.ndarray, seed: int = 1):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        sites = co2.molecule(rng.random(3) * box, co2.random_axis(rng))
        if all(np.min(np.linalg.norm(co2._wrap(m[:, None] - sites[None], box),
                                     axis=-1)) > 3.0 for m in out):
            out.append(sites)
    return out


class TestEwald:
    def test_the_energy_does_not_depend_on_the_split(self):
        """alpha moves energy between real and reciprocal space only."""
        box = np.array([30.0, 31.0, 32.0])
        molecules = _random_molecules(20, box)
        totals = []
        for precision in (1e-5, 1e-7):
            system = co2.CO2System(None, box)
            system.ewald = co2.Ewald(box, co2.CUTOFF, precision)
            system.constants = [system.ewald.molecule_constant()]
            system.molecules = list(molecules)
            totals.append(system.total_energy())
        assert abs(totals[0] - totals[1]) < 1e-3 * abs(totals[1])

    def test_the_quadrupole_prefers_the_t_shape(self):
        """Electrostatics alone: T-shaped attracts, side by side repels.

        LJ switched off so only the charges speak. CO2's quadrupole is
        why its dimer is T-shaped (or slipped) and not parallel; a sign
        error in the Ewald terms would swap these two.
        """
        box = np.array([30.0, 30.0, 30.0])
        centre = np.array([10.0, 10.0, 10.0])

        def coulomb(other_centre, other_axis):
            system = co2.CO2System(None, box)
            system.pairs = {k: (sig, eps * 0.0, qq)
                            for k, (sig, eps, qq) in system.pairs.items()}
            system.molecules = [co2.molecule(centre, np.array([1.0, 0, 0]))]
            system.ewald.structure = system.ewald.terms(
                system.molecules[0], co2.SITE_CHARGE)
            energy, _ = system.insertion(co2.molecule(other_centre,
                                                      other_axis))
            return energy - system.constant

        t_shape = coulomb(centre + [0.0, 4.2, 0.0], np.array([0.0, 1, 0]))
        parallel = coulomb(centre + [0.0, 4.2, 0.0], np.array([1.0, 0, 0]))
        assert t_shape < 0.0 < parallel


class TestTheSimulation:
    def test_incremental_energy_matches_a_recomputation(self):
        result = co2.run_co2_gcmc(None, 300.0, 2.0, equilibration=1500,
                                  production=1500, seed=3,
                                  edges=np.array([30.0, 30.0, 30.0]))
        assert result.energy_drift_k < 1e-6

    def test_the_dilute_empty_box_is_the_ideal_gas(self):
        """At 0.1 MPa CO2 is 0.5 % off ideal: N must be zeta V."""
        box = np.array([30.0, 30.0, 30.0])
        result = co2.run_co2_gcmc(None, 300.0, 0.1, equilibration=3000,
                                  production=20000, seed=4, edges=box)
        expected = co2.gas_density(300.0, 0.1) * np.prod(box)
        assert abs(result.mean_count - expected) < 0.35 * expected

    def test_a_wall_adsorbs(self):
        """Between two graphene-like planes, more than the bulk gas."""
        a = 2.46
        nx, ny = 13, 13
        base = np.array([[0, 0], [a / 2, a * np.sqrt(3) / 6]])
        cells = [[i * a + j * a / 2, j * a * np.sqrt(3) / 2]
                 for i in range(nx) for j in range(ny)]
        layer = np.array([c + b for c in cells for b in base])
        width = 7.0
        planes = np.vstack([np.column_stack((layer, np.zeros(len(layer)))),
                            np.column_stack((layer, np.full(len(layer),
                                                            width)))])
        cell = [nx * a, ny * a * np.sqrt(3) / 2 * 1.0, 25.0]
        frame = Atoms(f"C{len(planes)}", positions=planes,
                      cell=[cell[0], cell[1], 25.0], pbc=True)
        # The cell above is not exactly periodic in y for a skewed grid;
        # for this test only the attraction matters, not the lattice.
        k_h, q_st = co2.henry_co2(frame, 298.0, samples=4000, seed=0)
        assert k_h > 0.0 and q_st > 10.0


class TestTheReservoir:
    def test_the_dilute_limit_is_ideal(self):
        z, phi = co2.peng_robinson(300.0, 1e-4)
        assert abs(z - 1) < 1e-3 and abs(phi - 1) < 1e-3

    def test_the_liquid_is_refused(self):
        with pytest.raises(ValueError, match="saturation"):
            co2.peng_robinson(280.0, 5.0)

    def test_saturation_is_near_the_real_one(self):
        """PR's vapour pressure at 280 K; the real one is 4.16 MPa."""
        assert 3.9 < co2._pr_saturation_pressure(280.0) < 4.4


def _hydroxylated_slit():
    from carbonforge.builders import build_graphene_supercell
    from carbonforge.functionalization.attach import functionalize_random

    sheet = functionalize_random(
        build_graphene_supercell(nx=10, ny=6, vacuum=10.0), "OH",
        n_groups=3, site_kind="basal", seed=0)
    sheet.set_cell(np.diag([sheet.cell[0, 0], sheet.cell[1, 1], 6.6]))
    sheet.center(axis=2)
    stack = sheet * (1, 1, 4)
    stack.set_pbc(True)
    stack.wrap()
    return stack


class TestTheChargedWall:
    def test_it_is_neutralised_over_the_sheet(self):
        # The electrode table's hydroxyl is not neutral (-0.03 e), which
        # is what exercises the neutralisation.
        wall = co2.wall_parameters(_hydroxylated_slit(),
                                   forcefield="electrode")
        assert abs(wall["net_charge"]) > 0.01
        assert abs(float(np.sum(wall["charge"]))) < 1e-9
        assert wall["shift"] != 0.0

    def test_incremental_energy_holds_with_wall_charges(self):
        atoms = _hydroxylated_slit()
        result = co2.run_co2_gcmc(atoms, 298.0, 0.1, equilibration=600,
                                  production=600, seed=1,
                                  wall=co2.wall_parameters(atoms))
        assert result.energy_drift_k < 1e-6

    def test_a_neutral_typed_wall_is_the_plain_wall(self):
        """Typed pristine graphene is the default wall, to its rounding.

        The parameter table stores Steele's epsilon as 0.0559 kcal/mol,
        28.13 K against the default's 28.0, and K_H is exponential in it:
        2.5 % apart, measured. Equal to 1e-6 would mean the typed path
        was silently not used; 5 % apart would mean it was used wrongly.
        """
        from carbonforge.builders import build_graphene_supercell

        sheet = build_graphene_supercell(nx=10, ny=6, vacuum=10.0)
        sheet.set_cell(np.diag([sheet.cell[0, 0], sheet.cell[1, 1], 6.6]))
        sheet.center(axis=2)
        stack = sheet * (1, 1, 4)
        stack.set_pbc(True)
        typed = co2.henry_co2(stack, 298.0, samples=3000, seed=0,
                              wall=co2.wall_parameters(stack))
        plain = co2.henry_co2(stack, 298.0, samples=3000, seed=0)
        assert typed[0] == pytest.approx(plain[0], rel=0.05)
        assert typed[0] != pytest.approx(plain[0], rel=1e-6)
