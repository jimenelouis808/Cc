"""Tests for defects and corrugation on the two lattice builders.

The open tube and the ribbon place their atoms directly on the graphene
net, so a defect on them is an edit to a finished structure rather than to
a mesh. That makes the failure mode specific and worth pinning: an edit
can leave the ring *topology* exactly right while the geometry clashes, or
leave the geometry spotless while the topology is not what was asked for.
Both are checked here, separately, on every case.

The ring census is checked against Euler rather than against a number
typed in by hand: on a tube periodic along its axis the surface is a
torus, so the faces must come to ``E - V`` exactly, and both edits must
leave the angular deficit at zero because both are Euler-neutral by
construction.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from nanocarbon_lab.builders import build_cnt, build_nanoribbon
from nanocarbon_lab.builders.fullerene_mesh import minimum_image
from nanocarbon_lab.builders.lattice_edits import (
    MIN_DEFECT_SEPARATION,
    apply_lattice_edits,
    periodic_box,
    ring_census,
)
from nanocarbon_lab.utils.geometry import guess_bonds
from nanocarbon_lab.validation.quality import sp2_quality


def _deficit(ring_counts: dict[int, int]) -> int:
    return sum((6 - size) * count for size, count in ring_counts.items())


def _census(atoms):
    bonds = [(i, j) for i, j, _ in guess_bonds(atoms)]
    return ring_census(atoms.get_positions(), bonds, periodic_box(atoms))


def _carbon_coordination(atoms) -> list[int]:
    symbols = atoms.get_chemical_symbols()
    counts = [0] * len(atoms)
    for i, j, _ in guess_bonds(atoms):
        if symbols[i] == "C" and symbols[j] == "C":
            counts[i] += 1
            counts[j] += 1
    return [c for c, s in zip(counts, symbols, strict=True) if s == "C"]


class TestRingCensus:
    def test_a_pristine_tube_is_all_hexagons(self):
        """Every atom of a closed hexagonal net belongs to three rings, so
        a tube of N atoms has exactly N/2 of them."""
        atoms = build_cnt(6, 6, length=20.0)
        assert _census(atoms) == {6: len(atoms) // 2}

    def test_a_pristine_ribbon_is_all_hexagons(self):
        """The boundary of an open sheet traces as one long cycle. It is
        not a ring and must not be counted as one."""
        atoms = build_nanoribbon(6, 8, edge="zigzag")
        assert set(_census(atoms)) == {6}

    def test_faces_obey_euler_on_the_torus(self):
        """A tube periodic along its axis is a torus: chi = 0, so the
        number of faces is exactly E - V. This is the check that catches a
        trace wandering out of one ring and into the next -- which is what
        a mis-signed surface normal made it do."""
        atoms = build_cnt(6, 6, length=20.0,
                          defects=[{"type": "divacancy", "count": 1}], seed=0)
        bonds = [tuple(pair) for pair in atoms.info["bonds"]]
        assert sum(atoms.info["ring_counts"].values()) == len(bonds) - len(atoms)


class TestStoneWales:
    def test_it_makes_two_pentagons_and_two_heptagons(self):
        atoms = build_cnt(6, 6, length=20.0,
                          defects=[{"type": "stone_wales", "count": 1}], seed=0)
        counts = atoms.info["ring_counts"]
        assert counts[5] == 2
        assert counts[7] == 2
        assert 8 not in counts

    def test_it_removes_no_atoms(self):
        pristine = build_cnt(6, 6, length=20.0)
        defective = build_cnt(6, 6, length=20.0,
                              defects=[{"type": "stone_wales", "count": 2}],
                              seed=1)
        assert len(defective) == len(pristine)

    def test_it_is_euler_neutral(self):
        """Two pentagons and two heptagons cancel: the panel says the
        defect changes ring types and not the pentagon budget, and this is
        that claim."""
        atoms = build_cnt(6, 6, length=20.0,
                          defects=[{"type": "stone_wales", "count": 3}], seed=2)
        assert _deficit(atoms.info["ring_counts"]) == 0

    def test_the_rotation_actually_moves_the_atoms(self):
        """The rotation axis is the local surface normal. Derived from one
        atom's own bond vectors it can come out antiparallel at the two
        ends of the bond, average to nothing, and normalise to a direction
        along the bond -- where a 90 degree rotation is the identity and
        the 'defect' is a pristine lattice."""
        atoms = build_cnt(6, 6, length=20.0,
                          defects=[{"type": "stone_wales", "count": 1}], seed=0)
        assert set(atoms.info["ring_counts"]) != {6}

    def test_the_wall_does_not_clash(self):
        """Topology right and geometry broken is the combination that
        matters here: rotating the bond without relaxing around it leaves
        four non-bonded pairs inside 2 Å."""
        atoms = build_cnt(6, 6, length=20.0,
                          defects=[{"type": "stone_wales", "count": 2}], seed=0)
        assert atoms.info["geometry"]["n_close_contacts"] == 0
        verdict, why = sp2_quality(atoms.info["geometry"],
                                   atoms.info["quality_family"])
        assert verdict != "broken", why


class TestDivacancy:
    def test_it_makes_two_pentagons_and_one_octagon(self):
        atoms = build_cnt(6, 6, length=20.0,
                          defects=[{"type": "divacancy", "count": 1}], seed=0)
        counts = atoms.info["ring_counts"]
        assert counts[5] == 2
        assert counts[8] == 1
        assert 7 not in counts

    def test_it_removes_exactly_two_atoms_each(self):
        pristine = len(build_cnt(6, 6, length=20.0))
        atoms = build_cnt(6, 6, length=20.0,
                          defects=[{"type": "divacancy", "count": 2}], seed=0)
        assert len(atoms) == pristine - 4

    def test_it_is_euler_neutral(self):
        atoms = build_cnt(6, 6, length=20.0,
                          defects=[{"type": "divacancy", "count": 2}], seed=0)
        assert _deficit(atoms.info["ring_counts"]) == 0

    def test_it_leaves_no_dangling_carbon(self):
        """The four atoms around the hole are paired and relaxed, so every
        carbon still has three carbon neighbours. An unreconstructed
        divacancy is a hole, not the 5-8-5 the control is labelled with."""
        atoms = build_cnt(6, 6, length=20.0,
                          defects=[{"type": "divacancy", "count": 2}], seed=3)
        assert min(_carbon_coordination(atoms)) == 3


class TestSeparation:
    def test_two_defects_are_kept_apart(self):
        atoms = build_cnt(10, 0, length=40.0,
                          defects=[{"type": "stone_wales", "count": 2}], seed=0)
        positions = atoms.get_positions()
        box = periodic_box(atoms)
        centres = [
            positions[spec["bond"][0]]
            + 0.5 * minimum_image(
                positions[spec["bond"][1]] - positions[spec["bond"][0]], box)
            for spec in atoms.info["defects"] if spec["type"] == "stone_wales"
        ]
        gap = np.linalg.norm(minimum_image(centres[1] - centres[0], box))
        assert gap >= MIN_DEFECT_SEPARATION

    def test_asking_for_more_than_fit_is_refused(self):
        with pytest.raises(ValueError, match="fit with"):
            build_cnt(6, 6, length=10.0,
                      defects=[{"type": "stone_wales", "count": 40}], seed=0)

    def test_an_unknown_defect_type_is_refused(self):
        with pytest.raises(ValueError, match="Unknown defect type"):
            build_cnt(6, 6, length=20.0, defects=[{"type": "adatom"}])


class TestRibbonEdges:
    @pytest.mark.parametrize("edge", ["zigzag", "armchair"])
    @pytest.mark.parametrize("kind", ["stone_wales", "divacancy"])
    def test_a_defect_never_strands_an_edge_atom(self, edge, kind):
        """A bond one row in from the edge passes a three-neighbour test on
        its own two atoms, and removing it leaves the edge carbon behind it
        with a single neighbour.

        Compared against the pristine ribbon rather than against 3: ASE's
        own ribbon already ends in a pair of one-coordinated corner atoms,
        so an absolute floor would be asserting something the builder never
        provided. What must not change is how many such atoms there are.
        """
        pristine = Counter(_carbon_coordination(
            build_nanoribbon(8, 10, edge=edge)))
        atoms = build_nanoribbon(8, 10, edge=edge,
                                 defects=[{"type": kind, "count": 1}], seed=0)
        under = Counter(_carbon_coordination(atoms))
        assert [under[n] for n in (1, 2)] == [pristine[n] for n in (1, 2)]

    def test_passivation_is_not_judged_as_an_sp2_bond(self):
        """C-H is 1.09 Å and correct; the sp2 window starts at 1.30 Å. The
        report describes the carbon skeleton, so a passivated ribbon is not
        called broken on the strength of its hydrogens."""
        atoms = build_nanoribbon(8, 10, edge="zigzag", passivate=True,
                                 defects=[{"type": "stone_wales", "count": 1}],
                                 seed=0)
        assert "H" in atoms.get_chemical_symbols()
        assert atoms.info["geometry"]["bond_min"] > 1.30


class TestRoughness:
    def test_it_changes_geometry_and_not_topology(self):
        pristine = build_cnt(6, 6, length=20.0)
        rough = build_cnt(6, 6, length=20.0, roughness=0.15, seed=0)
        assert rough.info["ring_counts"] == {6: len(pristine) // 2}
        displacement = np.linalg.norm(
            rough.get_positions() - pristine.get_positions(), axis=1)
        assert displacement.max() > 0.05

    def test_a_negative_amplitude_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            build_cnt(6, 6, length=20.0, roughness=-0.1)


class TestPristineIsUntouched:
    def test_no_edits_means_no_relaxation(self):
        """The default path must cost nothing: a pristine tube is returned
        as it was built, not relaxed 'just in case'."""
        atoms = build_cnt(6, 6, length=20.0)
        assert apply_lattice_edits(atoms) is atoms
        assert "geometry" not in atoms.info

    def test_the_quality_window_follows_the_rings(self):
        """A flat octagon's interior angle is 135 degrees exactly, which is
        the edge of the pristine sp2 window. A structure asked to carry
        one is judged against the window that admits it."""
        pristine = build_cnt(6, 6, length=20.0, roughness=0.1, seed=0)
        defective = build_cnt(6, 6, length=20.0,
                              defects=[{"type": "divacancy", "count": 1}],
                              seed=0)
        assert pristine.info["quality_family"] == "sp2"
        assert defective.info["quality_family"] == "haeckelite"
