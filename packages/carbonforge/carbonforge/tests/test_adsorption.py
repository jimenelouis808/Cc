"""Adsorption: the geometry, the reservoir, the potentials, the sampling.

Most of these pin a published number rather than a self-consistent one.
That is deliberate: the whole module exists to be checked against
Levesque et al., J. Phys.: Condens. Matter 14 (2002) 9285 before it is
pointed at a structure nobody has an answer for.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from scipy.spatial import cKDTree

from carbonforge.adsorption import (LEVESQUE_BULK, analyse_porosity,
                                    feynman_hibbs, fit_virial,
                                    levesque_bundle, quantum_fraction,
                                    run_gcmc, zigzag_radius)
from carbonforge.adsorption.bundle import (AVOGADRO,
                                           LEVESQUE_BUNDLE_ISOTHERMS,
                                           REFERENCE_DIAMETER)
from carbonforge.adsorption.potentials import (H2_C, H2_H2, MASS_C,
                                               lennard_jones)


class TestBundleReproducesThePaper:
    """Four numbers the paper prints, and all four have to come out."""

    def test_the_diameter_comes_from_the_papers_own_formula(self):
        assert zigzag_radius(17) * 2 == pytest.approx(13.385, abs=0.002)

    @pytest.mark.parametrize(
        "gap, cell, density",
        [(3.4, (57.9, 34.08, 66.8), 1.31), (6.0, (66.8, 34.08, 77.2), 0.98)],
    )
    def test_the_cell_and_the_adsorbent_density(self, gap, cell, density):
        """8704 carbons, the cell edges, and 1.31 / 0.98 g/cm3.

        The cell needs the paper's ROUNDED diameter: its own formula
        gives 13.385 Å and it sized the lattice from "13.3". That 0.6 %
        is the paper's, not this builder's, and
        :data:`REFERENCE_DIAMETER` is how it is reproduced.
        """
        atoms = levesque_bundle(gap=gap, diameter=REFERENCE_DIAMETER)
        assert len(atoms) == 8704
        assert np.allclose(atoms.info["cell_edges"], cell, atol=0.1)
        assert atoms.info["adsorbent_density_g_cm3"] == pytest.approx(
            density, abs=0.01)

    def test_every_carbon_has_three_neighbours_at_the_bond_length(self):
        """A tube rolled wrong still has the right atom count."""
        atoms = levesque_bundle(gap=6.0)
        edges = np.diag(np.asarray(atoms.cell))
        positions = atoms.get_positions()
        tree = cKDTree(positions, boxsize=edges)
        pairs = tree.query_pairs(1.7, output_type="ndarray")
        assert len(pairs) == 3 * len(atoms) // 2
        delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
        delta -= edges * np.round(delta / edges)
        lengths = np.linalg.norm(delta, axis=1)
        assert lengths.min() > 1.41
        assert lengths.max() < 1.43

    def test_the_positions_are_inside_the_cell(self):
        """Every neighbour search downstream is a tree with a boxsize,
        and one negative coordinate makes all of them raise."""
        atoms = levesque_bundle(gap=6.0)
        assert (atoms.get_positions() >= 0).all()
        assert (atoms.get_positions() < np.diag(np.asarray(atoms.cell))).all()


class TestTheReferenceIsothermTable:
    """The published tables, and the convention their wt% column uses.

    These tests do not run a simulation. They check that the numbers
    the prototype is validated against are transcribed right and that
    the quantity being compared is the same quantity — which is where
    this kind of comparison usually goes wrong, not in the sampling.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def bundle():
        return levesque_bundle(gap=6.0, diameter=REFERENCE_DIAMETER)

    @pytest.fixture(scope="class")
    @staticmethod
    def cell_volume_cm3(bundle):
        return float(np.prod(np.diag(np.asarray(bundle.cell, float)))) * 1e-24

    @pytest.fixture(scope="class")
    @staticmethod
    def carbon_mass_g(bundle):
        return len(bundle) * MASS_C / AVOGADRO

    def test_every_table_has_the_eight_published_pressures(self):
        expected = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0]
        assert sorted(LEVESQUE_BUNDLE_ISOTHERMS) == [77.0, 150.0, 293.0]
        for rows in LEVESQUE_BUNDLE_ISOTHERMS.values():
            assert [row[0] for row in rows] == expected

    def test_the_loading_rises_with_pressure_at_every_temperature(self):
        """Monotonic in both columns. A transcription slip in a
        five-decimal number is otherwise invisible."""
        for rows in LEVESQUE_BUNDLE_ISOTHERMS.values():
            densities = [row[1] for row in rows]
            assert densities == sorted(densities)
            percentages = [row[2] for row in rows]
            assert percentages == sorted(percentages)

    def test_the_bulk_table_and_the_isotherm_table_agree_on_where_they_cross(
            self):
        """At 77 K the bulk overtakes the adsorbed phase by 20 MPa, and
        the paper says so in words: adsorption becomes disadvantageous
        above 15 MPa. Two independently transcribed tables have to
        reproduce that crossing or one of them is wrong."""
        bulk = dict(LEVESQUE_BULK[77.0])
        adsorbed = {p: rho for p, rho, _ in LEVESQUE_BUNDLE_ISOTHERMS[77.0]}
        assert bulk[15.0] < adsorbed[15.0]
        assert bulk[20.0] > adsorbed[20.0]

    def test_the_wt_column_is_hydrogen_per_carbon_not_per_total_mass(
            self, cell_volume_cm3, carbon_mass_g):
        """The finding this table carries a paragraph about.

        Converting the published ``rho_o`` at 77 K with the reproduced
        cell reaches the published ``wt%`` on the per-carbon basis and
        misses it on the total-mass basis, consistently and in one
        direction. 1.3 % against 3.2 % mean error over eight points.
        """
        per_carbon = total = 0.0
        rows = LEVESQUE_BUNDLE_ISOTHERMS[77.0]
        for _, rho, published in rows:
            hydrogen = rho * cell_volume_cm3
            per_carbon += abs(
                100 * hydrogen / carbon_mass_g - published) / published
            total += abs(
                100 * hydrogen / (hydrogen + carbon_mass_g) - published
            ) / published
        per_carbon /= len(rows)
        total /= len(rows)
        assert per_carbon < 0.02
        assert total > 0.025
        assert per_carbon < total / 2.0

    @pytest.mark.parametrize("pressure, published", [(0.1, 3.2), (20.0, 5.2)])
    def test_the_two_ends_of_the_77_K_isotherm_land_on_the_printed_digits(
            self, pressure, published, cell_volume_cm3, carbon_mass_g):
        """Not a tolerance — the per-carbon conversion of ``rho_o``
        rounds to the printed value at both ends."""
        rho = {p: r for p, r, _ in LEVESQUE_BUNDLE_ISOTHERMS[77.0]}[pressure]
        hydrogen = rho * cell_volume_cm3
        assert round(100 * hydrogen / carbon_mass_g, 1) == published


class TestPotentials:
    def test_the_correction_is_repulsive_where_the_core_is(self):
        """Short range only, and that is the physics, not a bug.

        Feynman-Hibbs smears the potential over the molecule's thermal
        spread, so it RAISES the energy where the potential is convex
        — the repulsive core, which is what keeps molecules apart —
        and lowers it slightly past the inflection point. An earlier
        version of this test asserted repulsion everywhere and was
        simply wrong about the shape.
        """
        close = np.array([2.6, 2.8, 3.0])
        far = np.array([4.5, 5.0])
        assert (feynman_hibbs(close, H2_H2, 77.0)
                > lennard_jones(close, H2_H2)).all()
        assert (feynman_hibbs(far, H2_H2, 77.0)
                < lennard_jones(far, H2_H2)).all()

    def test_the_correction_fades_as_the_temperature_rises(self):
        """It scales as 1/T, which is why 293 K barely notices it and
        77 K changes by a fifth."""
        r = np.array([2.8, 3.0, 3.2])
        classical = lennard_jones(r, H2_H2)
        cold = np.abs(feynman_hibbs(r, H2_H2, 77.0) - classical)
        warm = np.abs(feynman_hibbs(r, H2_H2, 293.0) - classical)
        assert (cold > warm).all()
        assert np.allclose(cold / warm, 293.0 / 77.0, rtol=1e-9)

    def test_it_becomes_classical_in_the_hot_limit(self):
        r = np.array([3.5])
        assert feynman_hibbs(r, H2_H2, 1e9)[0] == pytest.approx(
            lennard_jones(r, H2_H2)[0], rel=1e-8)

    def test_the_framework_pair_does_not_let_the_wall_recoil(self):
        """A fixed carbon has infinite mass, so the reduced mass of the
        H2-C pair is the molecule's own, not the two-body value."""
        assert H2_C.reduced_mass == pytest.approx(2.01588)
        assert H2_H2.reduced_mass == pytest.approx(2.01588 / 2)

    def test_the_quantum_parameter_matches_what_the_paper_quotes(self):
        """Levesque: about 0.07 at 70 K for a pair of H2."""
        assert quantum_fraction(70.0) == pytest.approx(0.07, abs=0.02)

    def test_a_negative_temperature_is_refused(self):
        with pytest.raises(ValueError):
            feynman_hibbs(np.array([3.0]), H2_H2, -1.0)


class TestBulk:
    @pytest.mark.parametrize("temperature", [77.0, 150.0, 293.0])
    def test_it_reproduces_the_densities_it_was_fitted_to(self, temperature):
        """Within the rounding of the published table, which is the
        honest accuracy of everything downstream of it."""
        eos = fit_virial(temperature)
        assert eos.residual_pct < 2.0
        for pressure, density in LEVESQUE_BULK[temperature]:
            assert eos.mass_density(pressure) == pytest.approx(
                density, rel=0.05)

    def test_the_fugacity_leaves_the_ideal_gas_in_the_right_direction(self):
        """Above the Boyle temperature f > P, below it f < P.

        Getting this backwards would shift every isotherm, and nothing
        downstream would look wrong.
        """
        hot = fit_virial(293.0)
        cold = fit_virial(77.0)
        assert hot.fugacity(20.0) > 20.0e6
        assert cold.fugacity(20.0) < 20.0e6

    def test_fitting_two_coefficients_to_two_points_is_refused(self):
        with pytest.raises(ValueError, match="at least three"):
            fit_virial(293.0, reference=[(1.0, 0.0008), (2.0, 0.0016)])


class TestPorosity:
    def test_a_solid_block_has_no_void(self):
        """The null case, and the one that catches a sign error."""
        rng = np.random.default_rng(0)
        positions = rng.random((400, 3)) * 8.0
        atoms = Atoms("C400", positions=positions, cell=[8.0] * 3, pbc=True)
        report = analyse_porosity(atoms, samples=4000, area_points=40, seed=0)
        assert report.void_fraction < 0.05
        assert any("closed to it" in w or "cannot cross" in w
                   for w in report.warnings) or report.void_fraction == 0.0

    def test_an_empty_cell_is_all_void(self):
        atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)],
                      cell=[40.0] * 3, pbc=True)
        report = analyse_porosity(atoms, samples=4000, area_points=40, seed=0)
        assert report.void_fraction > 0.98

    def test_closing_the_gap_removes_the_outer_surface(self):
        """The bundle's own check, and a real result.

        At d = 6 Å both faces of every tube are reachable; at 3.4 Å the
        tubes touch and the outside is gone. The area has to fall by
        much more than the void fraction does, because the volume lost
        is a thin shell and the surface lost is a whole side.
        """
        wide = analyse_porosity(
            levesque_bundle(gap=6.0, diameter=REFERENCE_DIAMETER),
            samples=20_000, area_points=80, seed=1)
        tight = analyse_porosity(
            levesque_bundle(gap=3.4, diameter=REFERENCE_DIAMETER),
            samples=20_000, area_points=80, seed=1)
        assert wide.accessible_area_m2_g > 2.0 * tight.accessible_area_m2_g
        assert wide.void_fraction < 2.0 * tight.void_fraction

    def test_the_area_is_in_square_metres_per_gram(self):
        """A nanotube bundle is thousands, not 1e23.

        Dropping the Å² to m² conversion gives 1e23 m²/g, which is the
        kind of wrong that is obvious once and invisible in a table.
        """
        report = analyse_porosity(
            levesque_bundle(gap=6.0, diameter=REFERENCE_DIAMETER),
            samples=8000, area_points=60, seed=1)
        assert 500.0 < report.accessible_area_m2_g < 4000.0

    def test_the_largest_cavity_is_the_tube_interior(self):
        """13.3 Å of tube minus a carbon radius each side."""
        report = analyse_porosity(
            levesque_bundle(gap=6.0, diameter=REFERENCE_DIAMETER),
            samples=20_000, area_points=40, seed=1)
        assert report.largest_included_sphere == pytest.approx(10.1, abs=0.6)

    def test_the_helium_void_exceeds_the_hard_sphere_one(self):
        """Because a helium atom is attracted as well as excluded, and
        it is the helium one an experiment's excess is defined with."""
        report = analyse_porosity(
            levesque_bundle(gap=6.0, diameter=REFERENCE_DIAMETER),
            samples=8000, area_points=40, seed=1)
        assert report.helium_void_fraction > report.void_fraction

    def test_a_triclinic_cell_is_refused(self):
        atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)], pbc=True,
                      cell=[[10, 0, 0], [0, 10, 0], [2, 0, 10]])
        with pytest.raises(ValueError, match="triclinic"):
            analyse_porosity(atoms, samples=100, area_points=10, seed=0)

    def test_a_sealed_cage_holds_a_probe_it_cannot_let_in(self):
        """Cavity large, channel zero: the distinction the report makes.

        A shell of atoms around an empty middle has somewhere for a
        probe to sit and no way for one to arrive. Reporting only the
        cavity size would call this porous.
        """
        rng = np.random.default_rng(1)
        directions = rng.normal(size=(600, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        shell = 20.0 + 6.0 * directions
        atoms = Atoms(f"C{len(shell)}", positions=shell,
                      cell=[40.0] * 3, pbc=True)
        report = analyse_porosity(atoms, samples=20_000, area_points=40,
                                  seed=0)
        assert report.largest_included_sphere > 4.0
        assert report.largest_free_sphere < report.largest_included_sphere


class TestGCMC:
    def test_an_empty_cell_fills_to_about_the_bulk_density(self):
        """With no framework there is nothing to adsorb on, so the cell
        must come to the reservoir's own density. It is the one case
        with an answer that needs no simulation, which makes it the
        test that catches a wrong activity or a wrong volume."""
        atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)],
                      cell=[30.0] * 3, pbc=True)
        gas = fit_virial(293.0)
        result = run_gcmc(atoms, temperature=293.0, pressure_mpa=10.0,
                          equilibration=20_000, production=40_000, seed=2,
                          void_fraction=1.0)
        assert result.absolute_density_g_cm3 == pytest.approx(
            gas.mass_density(10.0), rel=0.25)

    def test_it_says_so_when_the_run_was_too_short(self):
        atoms = levesque_bundle(gap=6.0, diameter=REFERENCE_DIAMETER)
        result = run_gcmc(atoms, temperature=293.0, pressure_mpa=10.0,
                          equilibration=200, production=400, seed=1,
                          blocks=5)
        assert result.warnings, "una tirada de 600 movimientos no converge"

    def test_the_excess_uses_the_void_fraction_it_was_given(self):
        atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)],
                      cell=[25.0] * 3, pbc=True)
        shared = dict(temperature=293.0, pressure_mpa=10.0,
                      equilibration=2000, production=2000, seed=3, blocks=4)
        whole = run_gcmc(atoms, void_fraction=1.0, **shared)
        half = run_gcmc(atoms, void_fraction=0.5, **shared)
        assert half.excess_density_g_cm3 > whole.excess_density_g_cm3
        assert whole.void_fraction == 1.0

    def test_not_passing_a_void_fraction_is_said_out_loud(self):
        atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)],
                      cell=[25.0] * 3, pbc=True)
        result = run_gcmc(atoms, temperature=293.0, pressure_mpa=1.0,
                          equilibration=500, production=500, seed=4, blocks=4)
        assert any("void fraction" in w for w in result.warnings)

    def test_a_triclinic_framework_is_refused(self):
        atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)], pbc=True,
                      cell=[[10, 0, 0], [0, 10, 0], [2, 0, 10]])
        with pytest.raises(ValueError, match="orthorhombic"):
            run_gcmc(atoms, temperature=293.0, pressure_mpa=1.0,
                     equilibration=10, production=10, seed=0, blocks=2)

    def test_the_same_seed_gives_the_same_answer(self):
        atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)],
                      cell=[20.0] * 3, pbc=True)
        shared = dict(temperature=293.0, pressure_mpa=5.0,
                      equilibration=500, production=500, blocks=4)
        first = run_gcmc(atoms, seed=7, **shared)
        again = run_gcmc(atoms, seed=7, **shared)
        assert first.mean_count == again.mean_count
