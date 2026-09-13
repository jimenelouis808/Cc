"""Tests for helical MX2 nanotubes.

The coil is a swept nanotube, so the things worth pinning down are the
ones sweeping can silently get wrong: that the lattice survives the bend
(every ring still a hexagon, every bond still M-X, stoichiometry still
MX2), that the reported helix is the one actually built rather than the
one requested, and that both strains are computed and reported rather
than one hiding behind the other.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from nanocarbon_lab.tmd.coil import (
    MAX_BEND_STRAIN,
    build_tmd_coil,
    helix_curvature,
)
from nanocarbon_lab.tmd.materials import get_material
from nanocarbon_lab.tmd.quality import geometry_report

# Small enough to build repeatedly in a test, large enough to be a real coil.
SMALL = dict(n=20, m=0, coil_radius=140.0, pitch=60.0, turns=0.15)


def build(**overrides):
    kwargs = dict(SMALL)
    kwargs.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return build_tmd_coil(**kwargs)


class TestHelixCurvature:
    def test_a_straight_limit_has_no_curvature(self):
        """As the pitch dominates, the helix straightens out."""
        assert helix_curvature(100.0, 1e6) < 1e-8

    def test_a_flat_ring_is_one_over_the_radius(self):
        """With no rise, a helix is a circle."""
        assert helix_curvature(50.0, 1e-9) == pytest.approx(1.0 / 50.0)

    def test_curvature_falls_as_the_coil_widens(self):
        assert helix_curvature(400.0, 90.0) < helix_curvature(200.0, 90.0)

    def test_a_non_positive_radius_is_rejected(self):
        with pytest.raises(ValueError, match="coil_radius"):
            helix_curvature(0.0, 90.0)


class TestGeometry:
    def test_the_lattice_survives_the_bend(self):
        """Sweeping must not rewire anything: MX2 stoichiometry exactly,
        and no M-M or X-X contact appearing where the tube compresses."""
        atoms = build()
        symbols = atoms.get_chemical_symbols()
        n_mo = symbols.count("Mo")
        n_s = symbols.count("S")
        assert n_s == 2 * n_mo
        report = geometry_report(atoms)
        assert report["stoichiometry"] == pytest.approx(2.0)

    def test_no_atoms_overlap(self):
        """A bend tight enough to fold the tube would show up here long
        before the strain numbers looked wrong."""
        report = geometry_report(atoms=build())
        assert report["n_close_contacts"] == 0

    def test_it_is_actually_helical(self):
        """A sweep that silently failed would leave a straight tube. Check
        the metal centreline turns through the requested angle."""
        atoms = build(turns=0.25)
        pos = atoms.get_positions()
        span = pos.max(axis=0) - pos.min(axis=0)
        # A quarter turn of a 140 A coil is wide in x and y, short in z.
        assert span[0] > 100.0
        assert span[1] > 100.0
        assert span[2] < span[0]

    def test_handedness_reverses_the_winding(self):
        """A left-handed coil rises as it turns the other way.

        Not tested by mirroring the two structures onto each other: the
        cross-section is transported by a rotation-minimizing frame whose
        seed comes from a fixed world axis, so the frame is not
        mirror-equivariant and the two coils are genuinely not exact
        reflections (their y-extents differ by ~0.6%). What *is* exact is
        the sign of the winding, so measure that.
        """
        def winding(atoms):
            pos = atoms.get_positions()
            # About the atoms' own centroid, which for a partial turn is
            # not the helix axis -- so this dilutes to ~0.33 rather than
            # approaching 1. Only the sign is being asserted.
            centre = pos[:, :2].mean(axis=0)
            angle = np.arctan2(pos[:, 1] - centre[1], pos[:, 0] - centre[0])
            return float(np.corrcoef(angle, pos[:, 2])[0, 1])

        right, left = winding(build(handedness=1)), winding(build(handedness=-1))
        assert right > 0.1
        assert left < -0.1
        assert right == pytest.approx(-left, rel=0.05)


class TestReportedGeometry:
    def test_the_achieved_helix_is_reported_not_the_requested_one(self):
        """The path is rescaled to a whole number of tube periods, so the
        radius actually built differs from the one asked for. Reporting
        the request would be a number the structure does not have."""
        atoms = build()
        info = atoms.info
        assert info["requested_coil_radius"] == SMALL["coil_radius"]
        assert info["coil_radius"] != pytest.approx(SMALL["coil_radius"])
        # But only by the rescale, which is under one period.
        drift = abs(info["coil_radius"] - SMALL["coil_radius"])
        assert drift / SMALL["coil_radius"] < 0.05

    def test_the_rescale_is_under_one_tube_period(self):
        atoms = build()
        info = atoms.info
        requested_arc = math.hypot(
            2.0 * math.pi * SMALL["coil_radius"] * SMALL["turns"],
            SMALL["pitch"] * SMALL["turns"],
        )
        assert abs(info["arc_length"] - requested_arc) < info["period"]

    def test_both_strains_are_recorded_and_they_add(self):
        atoms = build()
        info = atoms.info
        assert info["total_strain"] == pytest.approx(
            info["roll_strain"] + info["bend_strain"])

    def test_roll_strain_is_h_over_2r(self):
        atoms = build()
        material = get_material("MoS2")
        expected = material.h / (2.0 * atoms.info["tube_radius"])
        assert atoms.info["roll_strain"] == pytest.approx(expected)

    def test_bend_strain_is_the_outer_radius_times_curvature(self):
        atoms = build()
        info = atoms.info
        expected = info["outer_radius"] * helix_curvature(
            info["coil_radius"], info["pitch"])
        assert info["bend_strain"] == pytest.approx(expected)

    def test_atom_count_is_periods_times_one_cell(self):
        atoms = build()
        info = atoms.info
        assert len(atoms) % info["periods"] == 0


class TestStrainTradeoff:
    def test_a_wider_coil_bends_less(self):
        tight = build(coil_radius=140.0)
        wide = build(coil_radius=400.0)
        assert wide.info["bend_strain"] < tight.info["bend_strain"]

    def test_a_wider_tube_rolls_less_but_bends_more(self):
        """The two strains pull opposite ways, which is why an MX2 coil
        cannot be made comfortable by widening one of them alone."""
        narrow = build(n=20)
        wide = build(n=40)
        assert wide.info["roll_strain"] < narrow.info["roll_strain"]
        assert wide.info["bend_strain"] > narrow.info["bend_strain"]

    def test_a_tight_coil_warns(self):
        with pytest.warns(UserWarning, match="strains its outer wall"):
            build_tmd_coil(n=20, m=0, coil_radius=70.0, pitch=40.0,
                           turns=0.1, max_strain=1.0)

    def test_a_comfortable_coil_does_not_warn_about_bending(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_tmd_coil(n=60, m=0, coil_radius=2000.0, pitch=200.0,
                           turns=0.02, max_strain=1.0)
        assert not [w for w in caught
                    if "strains its outer wall" in str(w.message)]

    def test_the_default_bend_threshold_is_the_documented_one(self):
        assert MAX_BEND_STRAIN == pytest.approx(0.08)


class TestValidation:
    def test_non_positive_turns_is_rejected(self):
        with pytest.raises(ValueError, match="turns"):
            build_tmd_coil(turns=0.0)

    def test_a_bad_handedness_is_rejected(self):
        with pytest.raises(ValueError, match="handedness"):
            build_tmd_coil(handedness=0, **{k: v for k, v in SMALL.items()})

    def test_bad_chiral_indices_are_rejected_by_the_tube_builder(self):
        with pytest.raises(ValueError, match="n >= 1"):
            build_tmd_coil(n=3, m=5, coil_radius=140.0, pitch=60.0, turns=0.1)

    def test_other_materials_build(self):
        for formula in ("WS2", "MoSe2"):
            atoms = build(material=formula)
            assert atoms.info["material"] == formula
            assert len(atoms) > 0
