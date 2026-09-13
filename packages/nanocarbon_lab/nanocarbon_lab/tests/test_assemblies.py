"""Tests for multi-wall tubes, bundles, surface finish and the sp2 verdict.

What distinguishes an assembly from a single tube is not its topology --
each shell is an ordinary capped tube -- but the **van der Waals spacing**
between walls, which the covalent relaxation knows nothing about. So that
is what these tests measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import build_bundle, build_capped_cnt, build_multiwall_cnt
from nanocarbon_lab.builders import fullerene_mesh as fm
from nanocarbon_lab.validation.quality import SP2_BOND_RANGE, sp2_quality


def _deficit(ring_counts: dict[int, int]) -> int:
    return sum((6 - size) * count for size, count in ring_counts.items())


class TestMultiWall:
    @pytest.fixture(scope="class")
    def mwcnt(self):
        return build_multiwall_cnt(n_shells=2, inner_freq=2, n_body_rings=6)

    def test_each_shell_pays_its_own_euler_budget(self, mwcnt):
        """Two closed shells owe 12 pentagons apiece, so 24 in total."""
        assert _deficit(mwcnt.info["ring_counts"]) == 24
        assert mwcnt.info["n_shells"] == 2

    def test_walls_sit_near_the_graphite_interlayer_distance(self, mwcnt):
        """The relaxation has no dispersion term, so this must be checked.

        The lattice quantises tube radius in ~1.96 Å steps, so the walls
        cannot land on graphite's 3.35 Å exactly; what matters is that they
        neither interpenetrate nor drift far past it.
        """
        gap = mwcnt.info["geometry"]["min_wall_separation"]
        assert 3.0 < gap < 4.5

    def test_a_lone_tube_reports_no_wall_separation(self):
        """One shell has no second wall, and must say so rather than
        reporting an intra-wall distance (a hexagon's 1-4 diagonal is
        2.84 Å, which would look like a catastrophic wall gap)."""
        single = build_multiwall_cnt(n_shells=1, inner_freq=2, n_body_rings=6)
        assert np.isnan(single.info["geometry"]["min_wall_separation"])

    def test_invalid_shell_counts_raise(self):
        with pytest.raises(ValueError):
            build_multiwall_cnt(n_shells=0)
        with pytest.raises(ValueError):
            build_multiwall_cnt(freq_step=0)


class TestBundle:
    @pytest.fixture(scope="class")
    def rope(self):
        return build_bundle(n_rings_across=1, freq=2, n_body_rings=5)

    def test_a_full_hexagonal_shell_holds_seven_tubes(self, rope):
        assert rope.info["n_tubes"] == 7
        assert _deficit(rope.info["ring_counts"]) == 7 * 12

    def test_tubes_do_not_interpenetrate(self, rope):
        gap = rope.info["geometry"]["min_wall_separation"]
        assert gap > 3.0
        assert rope.info["geometry"]["n_close_contacts"] == 0

    def test_lattice_constant_follows_the_requested_gap(self, rope):
        expected = 2.0 * rope.info["tube_radius"] + rope.info["gap"]
        assert rope.info["lattice_constant"] == pytest.approx(expected)

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            build_bundle(n_rings_across=-1)
        with pytest.raises(ValueError):
            build_bundle(gap=0.0)


class TestSurfaceRoughness:
    """Roughness is optional and must stay inside the sp2 range.

    Displacing atoms along the local surface normal is the soft direction;
    isotropic jitter would just strain bonds and be undone by the
    re-relaxation. So corrugation should grow with sigma while the bond
    statistics stay chemically valid.
    """

    def test_zero_roughness_leaves_an_ideally_smooth_wall(self):
        atoms = build_capped_cnt(n_body_rings=8, freq=3, roughness=0.0, seed=1)
        assert _radial_spread(atoms) < 0.01

    def test_corrugation_grows_with_sigma_and_bonds_survive(self):
        spreads = []
        for sigma in (0.1, 0.3):
            atoms = build_capped_cnt(n_body_rings=8, freq=3, roughness=sigma, seed=1)
            spreads.append(_radial_spread(atoms))
            geometry = atoms.info["geometry"]
            low, high = SP2_BOND_RANGE
            assert low <= geometry["bond_min"] and geometry["bond_max"] <= high
            assert geometry["n_close_contacts"] == 0
        assert spreads[1] > spreads[0] > 0.02

    def test_roughness_never_changes_the_topology(self):
        smooth = build_capped_cnt(n_body_rings=8, freq=3, roughness=0.0, seed=1)
        rough = build_capped_cnt(n_body_rings=8, freq=3, roughness=0.4, seed=1)
        assert smooth.info["ring_counts"] == rough.info["ring_counts"]
        assert smooth.info["bonds"] == rough.info["bonds"]


def _radial_spread(atoms) -> float:
    """RMS deviation of the body wall from a perfect cylinder, in Å.

    The window is the middle 15% of the length, matching what
    ``build_capped_cnt`` itself uses to measure tube radius. A wider one
    reaches into the hemispherical caps, where the radius tapers for
    entirely legitimate reasons, and reports that taper as corrugation: at
    30% of the length a perfectly smooth 5-ring tube measures 0.025 Å of
    "roughness" against 0.003 Å here.
    """
    positions = atoms.get_positions()
    centred = positions - positions.mean(axis=0)
    body = np.abs(centred[:, 2]) < 0.15 * (centred[:, 2].max() - centred[:, 2].min())
    return float(np.linalg.norm(centred[body][:, :2], axis=1).std())


class TestNeighbourListSkin:
    """The non-bonded list is frozen for a whole L-BFGS run.

    Without a skin, two atoms further apart than the cutoff when the list
    was built are invisible to each other for thousands of iterations and
    pass straight through. This reproduces that in miniature: two sheets
    5 Å apart -- outside the 2.2 Å cutoff, inside a 4 Å skin -- driven
    together by anchors, in a **single** cycle so the list is never
    rebuilt. That single frozen cycle is the real failure mode; with the
    default three cycles the rebuild between them papers over it, which
    is exactly why the bug survived so long in compact structures.
    """

    def test_skin_catches_walls_that_approach_mid_run(self):
        grid = np.stack(np.meshgrid(np.arange(4) * 1.42, np.arange(4) * 1.42,
                                    indexing="ij"), axis=-1).reshape(-1, 2)
        lower = np.column_stack([grid, np.zeros(len(grid))])
        upper = np.column_stack([grid, np.full(len(grid), 5.0)])
        positions = np.vstack([lower, upper])
        # Bond each sheet internally along one direction so the shells hold
        # together, and pull the sheets toward each other with anchors.
        bonds = set()
        for offset in (0, len(grid)):
            for i in range(len(grid)):
                for j in range(i + 1, len(grid)):
                    if np.linalg.norm(grid[i] - grid[j]) < 1.5:
                        bonds.add((i + offset, j + offset))
        anchors = np.arange(len(positions))
        targets = positions.copy()
        targets[len(grid):, 2] = 0.0   # ask the top sheet to sit on the bottom

        def closest_approach(result):
            from scipy.spatial import cKDTree

            tree = cKDTree(result[:len(grid)])
            distance, _ = tree.query(result[len(grid):], k=1)
            return float(distance.min())

        without = fm.relax_shell(positions, bonds, anchors=anchors,
                                 anchor_targets=targets, k_anchor=30.0,
                                 repel_skin=0.0, outer_cycles=1,
                                 max_iterations=4000)
        with_skin = fm.relax_shell(positions, bonds, anchors=anchors,
                                   anchor_targets=targets, k_anchor=30.0,
                                   repel_skin=4.0, outer_cycles=1,
                                   max_iterations=4000)
        # With no skin the sheets do not merely touch, they interpenetrate
        # completely: measured closest approach 0.000 Å.
        assert closest_approach(without) < 0.1
        assert closest_approach(with_skin) > 1.0


class TestSp2Verdict:
    def test_clean_geometry_reads_clean(self):
        verdict, _ = sp2_quality(
            {"bond_min": 1.41, "bond_max": 1.43, "angle_min": 108.0,
             "angle_max": 120.0, "n_close_contacts": 0}
        )
        assert verdict == "clean"

    def test_overlapping_atoms_read_broken_even_with_perfect_bonds(self):
        verdict, why = sp2_quality(
            {"bond_min": 1.42, "bond_max": 1.42, "angle_min": 119.0,
             "angle_max": 121.0, "n_close_contacts": 12}
        )
        assert verdict == "broken"
        assert "folded" in why

    def test_a_stretched_bond_reads_broken_even_with_no_contacts(self):
        """The case the verdict was added for: an over-tight coil keeps its
        atoms apart while stretching its bonds past any real C-C."""
        verdict, why = sp2_quality(
            {"bond_min": 1.23, "bond_max": 1.69, "angle_min": 104.0,
             "angle_max": 128.0, "n_close_contacts": 0}
        )
        assert verdict == "broken"
        assert "1.69" in why

    def test_the_edge_of_the_window_reads_strained(self):
        verdict, _ = sp2_quality(
            {"bond_min": 1.33, "bond_max": 1.54, "angle_min": 103.0,
             "angle_max": 129.0, "n_close_contacts": 0}
        )
        assert verdict == "strained"
