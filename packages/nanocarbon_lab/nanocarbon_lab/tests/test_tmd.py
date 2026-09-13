"""Tests for transition-metal dichalcogenide structures.

These have a luxury the carbon builders do not: the answers are in the
crystallography literature. MoS2's Mo-S bond is 2.41 Å, its bulk c axis
is 12.29 Å, its metal is six-coordinate. So the tests check named numbers
rather than ranges, and the ones that check ranges say why.

The recurring theme is the thing that makes a TMD not a decorated
graphene: it is a **sandwich** of finite thickness, so rolling it strains
the outer plane, cutting it exposes two chemically different edges, and
the phase is a statement about where the third plane sits.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from ase.neighborlist import neighbor_list

from nanocarbon_lab.tmd import (
    MATERIALS,
    build_tmd_bulk,
    build_tmd_layers,
    build_tmd_monolayer,
    build_tmd_nanotube,
    build_tmd_ribbon,
    geometry_report,
    get_material,
    tmd_quality,
    tube_radius,
)


def metal_coordination(atoms) -> set[int]:
    """How many chalcogens each metal atom is bonded to."""
    report = geometry_report(atoms)
    return {report["metal_coordination_min"], report["metal_coordination_max"]}


class TestMaterials:
    def test_bond_length_follows_from_a_and_h(self):
        """Storing a, h and d separately invites them to disagree, so d is
        derived. It must land on the literature value."""
        assert get_material("MoS2").bond_length == pytest.approx(2.41, abs=0.01)
        assert get_material("WSe2").bond_length == pytest.approx(2.54, abs=0.02)
        assert get_material("TiS2").bond_length == pytest.approx(2.43, abs=0.03)

    def test_every_material_is_physically_sensible(self):
        """Bounds on the whole MX2 family, not on the rows we happen to have.

        The upper bound on `a` was 3.8 Å when the table held thirteen
        compounds, none heavier than ZrS2. That was the range of the data
        rather than of the chemistry: the heavy tellurides genuinely go
        past it -- ZrTe2 and HfTe2 sit at 3.95 Å and PtTe2 at 4.03 -- so
        the bound was describing an accident of the table's contents.
        4.2 Å is where a hexagonal MX2 lattice constant really stops.
        """
        for name, material in MATERIALS.items():
            assert 3.0 < material.a < 4.2, name
            assert 2.5 < material.h < 3.8, name
            # The van der Waals gap is what holds layers apart; graphite's
            # is 3.35 Å and TMDs are in the same family of contact. The
            # platinum dichalcogenides are the tight end at ~2.4 Å, which
            # is real and is why their gap is so layer-dependent.
            assert 2.0 < material.vdw_gap < 3.8, f"{name}: {material.vdw_gap}"

    def test_unknown_material_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="MoS2"):
            get_material("MoS3")


class TestMonolayer:
    def test_mos2_matches_the_literature_geometry(self):
        atoms = build_tmd_monolayer("MoS2")
        report = geometry_report(atoms)
        assert report["bond_min"] == pytest.approx(2.404, abs=0.005)
        assert report["bond_max"] == pytest.approx(2.404, abs=0.005)
        assert report["stoichiometry"] == pytest.approx(2.0)

    @pytest.mark.parametrize("phase", ["2H", "1T"])
    def test_the_metal_is_six_coordinate_in_both_phases(self, phase):
        """Trigonal prismatic and octahedral differ in the *arrangement*
        of the six chalcogens, not in how many there are."""
        assert metal_coordination(build_tmd_monolayer("MoS2", phase=phase)) == {6}

    def test_2h_eclipses_its_chalcogens_and_1t_staggers_them(self):
        """This is the entire difference between the phases: 2H puts both
        chalcogen planes over the same column, 1T over different ones."""
        def offset(phase):
            atoms = build_tmd_monolayer("MoS2", phase=phase)
            positions = atoms.get_positions()
            symbols = np.array(atoms.get_chemical_symbols())
            metal_z = positions[symbols == "Mo"][0][2]
            sulphur = positions[symbols == "S"]
            top = sulphur[sulphur[:, 2] > metal_z][0]
            bottom = sulphur[sulphur[:, 2] < metal_z][0]
            return float(np.linalg.norm((top - bottom)[:2]))

        assert offset("2H") == pytest.approx(0.0, abs=1e-6)
        assert offset("1T") == pytest.approx(get_material("MoS2").a / np.sqrt(3),
                                             abs=1e-3)

    def test_1t_prime_dimerises_the_metals(self):
        """1T' is 1T with the metals paired -- the distortion that opens
        the gap. Literature puts the MoS2 dimer near 2.8 Å against the
        undistorted 3.16."""
        atoms = build_tmd_monolayer("MoS2", phase="1T'", nx=3, ny=3)
        symbols = np.array(atoms.get_chemical_symbols())
        first, second, distance = neighbor_list("ijd", atoms, cutoff=4.2)
        metal_pairs = distance[(symbols[first] == "Mo") & (symbols[second] == "Mo")]
        assert metal_pairs.min() == pytest.approx(2.80, abs=0.05)
        # And it must still be labelled as the idealised form.
        assert "idealised" in atoms.info["phase_note"]

    def test_supercells_scale_the_atom_count(self):
        single = len(build_tmd_monolayer("MoS2"))
        assert len(build_tmd_monolayer("MoS2", nx=3, ny=2)) == single * 6


class TestStackingAndBulk:
    def test_bulk_2h_reproduces_the_measured_cell(self):
        """a = 3.16 Å and c = 12.29 Å for 2H-MoS2, two layers per period."""
        bulk = build_tmd_bulk("MoS2", stacking="2H")
        lengths = bulk.cell.lengths()
        assert lengths[0] == pytest.approx(3.160, abs=0.005)
        assert lengths[2] == pytest.approx(12.29, abs=0.02)
        assert bulk.cell.angles()[2] == pytest.approx(120.0, abs=0.1)
        assert all(bulk.get_pbc())

    def test_the_stackings_have_their_own_repeats(self):
        """2H repeats every two layers, 3R every three -- that is what
        makes them different crystals rather than different slabs."""
        assert len(build_tmd_bulk("MoS2", stacking="2H")) == 6
        assert len(build_tmd_bulk("MoS2", stacking="3R")) == 9
        assert len(build_tmd_bulk("MoS2", stacking="AA")) == 3

    @pytest.mark.parametrize(
        ("stacking", "over"), [("2H", "S"), ("3R", "S"), ("AA", "Mo")]
    )
    def test_what_each_stacking_puts_the_metal_above(self, stacking, over):
        """2H and 3R both put the metal of one layer over the chalcogen of
        the next; AA stacks metal on metal. Rotating about the metal site
        instead of applying the full 6_3 screw silently gives AA for all
        three, which is why this is checked rather than assumed."""
        bilayer = build_tmd_layers("MoS2", n_layers=2, stacking=stacking)
        positions = bilayer.get_positions()
        symbols = np.array(bilayer.get_chemical_symbols())
        metals = sorted(positions[symbols == "Mo"], key=lambda p: p[2])
        lower_metal, upper_metal = metals[0], metals[1]
        lower_chalcogen = positions[symbols == "S"]
        lower_chalcogen = lower_chalcogen[lower_chalcogen[:, 2] < lower_metal[2] + 2]

        to_metal = float(np.linalg.norm((upper_metal - lower_metal)[:2]))
        to_chalcogen = min(
            float(np.linalg.norm((upper_metal - x)[:2])) for x in lower_chalcogen
        )
        if over == "S":
            assert to_chalcogen < 0.01 and to_metal > 1.0
        else:
            assert to_metal < 0.01 and to_chalcogen > 1.0

    def test_2h_and_3r_diverge_at_the_third_layer(self):
        """They are identical as bilayers -- both metal-over-chalcogen --
        and differ only in the repeat: ABAB against ABCABC."""
        def layer3_offset(stacking):
            slab = build_tmd_layers("MoS2", n_layers=3, stacking=stacking)
            positions = slab.get_positions()
            symbols = np.array(slab.get_chemical_symbols())
            metals = sorted(positions[symbols == "Mo"], key=lambda p: p[2])
            return float(np.linalg.norm((metals[2] - metals[0])[:2]))

        assert layer3_offset("2H") == pytest.approx(0.0, abs=1e-6)
        assert layer3_offset("3R") > 1.0

    def test_a_slab_keeps_vacuum_and_a_bulk_does_not(self):
        slab = build_tmd_layers("MoS2", n_layers=2, vacuum=18.0)
        bulk = build_tmd_bulk("MoS2")
        assert not slab.get_pbc()[2]
        assert slab.cell.lengths()[2] > 18.0
        assert all(bulk.get_pbc())

    def test_layers_sit_at_the_measured_interlayer_distance(self):
        bilayer = build_tmd_layers("MoS2", n_layers=2)
        positions = bilayer.get_positions()
        symbols = np.array(bilayer.get_chemical_symbols())
        metal_z = np.unique(np.round(positions[symbols == "Mo"][:, 2], 3))
        assert float(np.diff(metal_z)[0]) == pytest.approx(6.147, abs=0.01)

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            build_tmd_layers("MoS2", n_layers=0)
        with pytest.raises(ValueError):
            build_tmd_layers("MoS2", nx=0)
        with pytest.raises(ValueError, match="stacking"):
            build_tmd_layers("MoS2", n_layers=2, stacking="ZZ")


class TestRibbon:
    def test_interior_metals_keep_full_coordination(self):
        """Only the edge should be under-coordinated; if the interior is
        too, the cut has gone through the middle of the sandwich."""
        ribbon = build_tmd_ribbon("MoS2", width=8, length=2)
        report = geometry_report(ribbon)
        assert report["metal_coordination_max"] == 6
        assert report["metal_coordination_min"] < 6  # the edge, as expected

    def test_a_plain_cut_leaves_one_edge_of_each_element(self):
        ribbon = build_tmd_ribbon("MoS2", width=6, termination="mixed")
        assert _edge_elements(ribbon) == ({"S"}, {"Mo"})

    @pytest.mark.parametrize(
        ("termination", "element"), [("metal", "Mo"), ("chalcogen", "S")]
    )
    def test_termination_makes_both_edges_alike(self, termination, element):
        """MoS2's two zigzag edges are chemically different -- one
        metallic and magnetic, one not -- so which one you get is a
        choice, not an accident of the cut."""
        ribbon = build_tmd_ribbon("MoS2", width=6, termination=termination)
        assert _edge_elements(ribbon) == ({element}, {element})

    def test_a_terminated_edge_is_deliberately_off_stoichiometry(self):
        """A metal-terminated ribbon really is chalcogen-poor. That is the
        point of asking for it, so the verdict must not call it broken."""
        ribbon = build_tmd_ribbon("MoS2", width=6, termination="metal")
        report = geometry_report(ribbon)
        assert report["stoichiometry"] < 2.0
        assert tmd_quality(report, expect_stoichiometric=False)[0] == "clean"
        assert tmd_quality(report, expect_stoichiometric=True)[0] == "broken"

    def test_the_two_edge_types_are_perpendicular_cuts(self):
        """A zigzag ribbon runs along a lattice vector and an armchair one
        across it, so they are periodic along different axes. Getting this
        wrong yields the other ribbon type in a different cell."""
        zigzag = build_tmd_ribbon("MoS2", width=6, edge="zigzag")
        armchair = build_tmd_ribbon("MoS2", width=6, edge="armchair")
        assert list(zigzag.get_pbc()) == [True, False, False]
        assert list(armchair.get_pbc()) == [False, True, False]

    def test_width_grows_with_the_row_count(self):
        widths = [build_tmd_ribbon("MoS2", width=w).info["width_angstrom"]
                  for w in (4, 6, 8)]
        assert widths[0] < widths[1] < widths[2]

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            build_tmd_ribbon("MoS2", width=1)
        with pytest.raises(ValueError, match="edge"):
            build_tmd_ribbon("MoS2", edge="diagonal")


def _edge_elements(ribbon) -> tuple[set[str], set[str]]:
    """Elements on each edge of a ribbon, found along the cut axis."""
    axis = 1 if ribbon.get_pbc()[0] else 0
    coords = ribbon.get_positions()[:, axis]
    symbols = np.array(ribbon.get_chemical_symbols())
    tol = 0.4
    return (
        set(symbols[np.abs(coords - coords.min()) < tol]),
        set(symbols[np.abs(coords - coords.max()) < tol]),
    )


class TestNanotube:
    def test_radius_follows_the_chiral_indices(self):
        material = get_material("MoS2")
        expected = material.a * np.sqrt(20**2) / (2 * np.pi)
        assert tube_radius(material, 20, 0) == pytest.approx(expected)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tube = build_tmd_nanotube("MoS2", n=60)
        assert tube.info["radius"] == pytest.approx(
            material.a * 60 / (2 * np.pi), rel=1e-6
        )

    def test_rolling_keeps_every_metal_six_coordinate(self):
        """A seam in the roll would show up here immediately."""
        tube = build_tmd_nanotube("MoS2", n=60)
        assert metal_coordination(tube) == {6}
        assert geometry_report(tube)["stoichiometry"] == pytest.approx(2.0)

    def test_the_three_planes_land_on_three_cylinders(self):
        """The whole reason a TMD tube is not a carbon tube: the sandwich
        has thickness, so the chalcogen planes roll onto different radii
        from the metal."""
        tube = build_tmd_nanotube("MoS2", n=60)
        positions = tube.get_positions()
        centre = positions[:, :2].mean(axis=0)
        radial = np.linalg.norm(positions[:, :2] - centre, axis=1)
        symbols = np.array(tube.get_chemical_symbols())
        metal_r = radial[symbols == "Mo"].mean()
        inner = radial[symbols == "S"][radial[symbols == "S"] < metal_r].mean()
        outer = radial[symbols == "S"][radial[symbols == "S"] > metal_r].mean()
        h = get_material("MoS2").h
        assert outer - inner == pytest.approx(h, abs=0.05)
        assert metal_r == pytest.approx(tube.info["radius"], abs=0.05)

    def test_strain_falls_as_the_radius_grows(self):
        """Strain goes as h/2R, which is why real MoS2 tubes are tens of
        nanometres across where carbon ones are one."""
        h = get_material("MoS2").h
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tight = build_tmd_nanotube("MoS2", n=20)
        wide = build_tmd_nanotube("MoS2", n=60)
        for tube in (tight, wide):
            assert tube.info["roll_strain"] == pytest.approx(
                h / (2 * tube.info["radius"]), rel=1e-9
            )
        assert tight.info["roll_strain"] > 2.5 * wide.info["roll_strain"]
        assert tmd_quality(geometry_report(tight))[0] == "strained"
        assert tmd_quality(geometry_report(wide))[0] == "clean"

    def test_a_tube_too_tight_to_be_physical_warns(self):
        with pytest.warns(UserWarning, match="outer"):
            build_tmd_nanotube("MoS2", n=12)

    @pytest.mark.parametrize(
        ("n", "m", "family"),
        [(60, 0, "zigzag"), (40, 40, "armchair"), (30, 15, "chiral")],
    )
    def test_chirality_families_and_periods(self, n, m, family):
        tube = build_tmd_nanotube("MoS2", n=n, m=m)
        assert tube.info["chirality"] == family
        a = get_material("MoS2").a
        if family == "zigzag":
            assert tube.info["period"] == pytest.approx(a * np.sqrt(3), abs=0.01)
        elif family == "armchair":
            assert tube.info["period"] == pytest.approx(a, abs=0.01)

    def test_invalid_indices_raise(self):
        with pytest.raises(ValueError):
            build_tmd_nanotube("MoS2", n=0)
        with pytest.raises(ValueError):
            build_tmd_nanotube("MoS2", n=10, m=20)  # m must not exceed n


class TestQualityVerdict:
    def test_a_perfect_monolayer_is_clean(self):
        verdict, _ = tmd_quality(geometry_report(build_tmd_monolayer("MoS2")))
        assert verdict == "clean"

    def test_overlapping_atoms_read_broken(self):
        atoms = build_tmd_monolayer("MoS2", nx=2, ny=2)
        positions = atoms.get_positions()
        positions[1] = positions[0] + np.array([0.4, 0.0, 0.0])
        atoms.set_positions(positions)
        verdict, why = tmd_quality(geometry_report(atoms))
        assert verdict == "broken"
        # Overlapping atoms must read as a fold, not merely as a short
        # bond, even when the overlapping pair happens to be M and X.
        assert "folded" in why, why

    @pytest.mark.parametrize("name", sorted(MATERIALS))
    def test_every_material_builds_a_clean_monolayer(self, name):
        material = get_material(name)
        atoms = build_tmd_monolayer(name, phase=material.natural_phase)
        verdict, why = tmd_quality(geometry_report(atoms))
        assert verdict == "clean", f"{name}: {why}"
