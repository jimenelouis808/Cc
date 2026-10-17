"""Nanoparticles cut out of an imported crystal, and coated with carbon.

The strongest tests here are the ones that check against something
outside this framework: ``ase.cluster`` builds fcc clusters by its own
route and its atom counts have to agree with ours, and an fcc cube's
count has to be the lattice's own ``4n^3``. Those two pin the whole chain
-- Miller family, half-space polyhedron, supercell, cut -- without this
package marking its own work.

The rest protect the four things that decided a design here, each of which
was measured rather than assumed. They are listed in the modules'
docstrings and each has a test below:

* a polyhedron's extent has to come from its vertices, not from sampled
  directions, or a sharp corner is clipped away;
* a facet that lands on a lattice plane makes the atom count
  tie-break-sensitive to the last digit of the size;
* a hollow particle is almost all surface **by design** and must not be
  judged by the solid thresholds;
* a coated shell's ring census has to satisfy Euler for the surface it
  was meshed on, whatever the shape.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from ase.build import bulk
from ase.cluster import Octahedron
from ase.io import write

from nanocarbon_lab.nanoparticle import shapes as sh
from nanocarbon_lab.nanoparticle.build import (
    build_nanoparticle,
    estimate_nanoparticle_atoms,
)
from nanocarbon_lab.nanoparticle.carve import (
    TIE_TOLERANCE,
    carve_nanoparticle,
    coordination,
)
from nanocarbon_lab.nanoparticle.coat import (
    CARBIDE_FORMERS,
    GRAPHITIC_GAP,
    encapsulate,
    graphitic_coat,
    vdw_envelope,
)
from nanocarbon_lab.nanoparticle.crystal import (
    interplanar_spacing,
    lattice_point_group,
    miller_family,
    miller_normal,
    read_crystal,
    repeats_for,
)
from nanocarbon_lab.nanoparticle.quality import particle_quality

GOLD = 4.078
MAGNESIA = 4.212


@pytest.fixture(scope="module")
def gold():
    atoms = bulk("Au", "fcc", a=GOLD, cubic=True)
    atoms.pbc = True
    return atoms


@pytest.fixture(scope="module")
def magnesia():
    atoms = bulk("MgO", "rocksalt", a=MAGNESIA, cubic=True)
    atoms.pbc = True
    return atoms


# ----------------------------------------------------------------------
# shapes
# ----------------------------------------------------------------------
class TestTheShapeCatalogue:
    """Every shape, against the geometry it claims to be."""

    def test_every_catalogue_shape_builds_and_encloses_a_volume(self):
        for name in sh.SHAPES:
            shape = sh.make_shape(name)
            assert shape.extent > 0, name
            assert math.isfinite(shape.extent), name
            assert shape.volume(48) > 0, name
            assert shape.info["catalogue_name"] == name

    def test_nothing_reaches_outside_its_declared_extent(self):
        """``extent`` sizes the carver's block and the coater's grid.

        Too large only wastes memory; too small clips the shape and
        nothing downstream can tell. So it is checked as a bound, on a
        grid deliberately wider than the bound itself.
        """
        for name in sh.SHAPES:
            shape = sh.make_shape(name)
            axis = np.linspace(-1.4 * shape.extent, 1.4 * shape.extent, 49)
            grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"),
                            axis=-1)
            inside = shape.field(grid) <= 0
            assert inside.any(), name
            reach = float(np.abs(grid[inside]).max())
            assert reach <= shape.extent + 1e-6, (name, reach, shape.extent)

    @pytest.mark.parametrize(("name", "maker", "exact"), [
        ("sphere", lambda: sh.ball(15.0), 4 / 3 * math.pi * 15.0 ** 3),
        ("cube", lambda: sh.cube(24.0), 24.0 ** 3),
        ("octahedron", lambda: sh.octahedron(14.0),
         4 / 3 * (14.0 * math.sqrt(3.0)) ** 3),
        ("torus", lambda: sh.torus(18.0, 6.0),
         2 * math.pi ** 2 * 18.0 * 6.0 ** 2),
        ("capsule", lambda: sh.capsule(10.0, 20.0),
         math.pi * 100.0 * 40.0 + 4 / 3 * math.pi * 1000.0),
        ("cone", lambda: sh.cone(14.0, 16.0),
         math.pi * 196.0 * 32.0 / 3.0),
        ("hex prism", lambda: sh.prism(14.0, 16.0, 6),
         2 * math.sqrt(3.0) * 196.0 * 32.0),
        ("hex bipyramid", lambda: sh.bipyramid(14.0, 20.0, 6),
         2 / 3 * (2 * math.sqrt(3.0) * 196.0) * 20.0),
    ])
    def test_the_measured_volume_is_the_analytic_one(self, name, maker, exact):
        """The field has to describe the solid its name claims.

        A sign error or a swapped parameter leaves a field that is still
        a perfectly good shape -- just not the one asked for -- and no
        other test here would notice.

        The tolerance is the grid's own, one voxel over the surface, and
        it is looser for the flat-faced shapes on purpose: a face lying
        parallel to the sampling planes is either in or out for a whole
        sheet of voxels at once, so its error does not average away the
        way a curved surface's does.
        """
        measured = maker().volume(170)
        assert measured == pytest.approx(exact, rel=0.02), (
            f"{name}: measured {measured:.0f} vs analytic {exact:.0f}")

    def test_a_polyhedron_extent_comes_from_its_vertices(self):
        """A regular tetrahedron's circumradius is 3x its inradius.

        Sampling 512 directions returned 44.4 Å for faces at 16 Å, where
        the real solid reaches 48: short of its own corners, which the
        sampling misses precisely because they are sharp. A facetted
        particle is nothing but sharp corners.
        """
        shape = sh.tetrahedron(16.0)
        assert shape.extent == pytest.approx(48.0, abs=1e-6)
        vertices = sh.polyhedron_vertices(
            np.asarray(shape.info.get("normals",
                                      [(1, 1, 1), (1, -1, -1), (-1, 1, -1),
                                       (-1, -1, 1)]), dtype=float)
            / math.sqrt(3.0),
            np.full(4, 16.0))
        assert len(vertices) == 4

    def test_the_octahedron_and_the_cube_are_ends_of_one_parameter(self):
        """The cuboctahedron's truncation runs between them.

        Not three constructions: one, which is also how a real particle
        moves between the habits as the growth conditions change.
        """
        octa = sh.octahedron(14.0).volume(140)
        cube_like = sh.cuboctahedron(14.0, truncation=1 / math.sqrt(3.0))
        high = sh.cuboctahedron(14.0, truncation=math.sqrt(3.0))
        assert high.volume(140) == pytest.approx(octa, rel=0.02)
        # At the other end the {100} faces bind and the solid is the cube
        # of half-width d/sqrt(3).
        assert cube_like.volume(140) == pytest.approx(
            (2 * 14.0 / math.sqrt(3.0)) ** 3, rel=0.02)

    def test_a_wulff_facet_with_too_high_an_energy_disappears(self):
        """That is the construction working, not a failure.

        With gamma(110) five times gamma(111) the {110} faces never touch
        the solid, and what is left is exactly the octahedron.
        """
        shape = sh.wulff({"111": 1.0, "110": 5.0}, scale=14.0)
        assert shape.info["inactive_faces"] == 12
        assert shape.volume(140) == pytest.approx(
            sh.octahedron(14.0).volume(140), rel=1e-9)

    def test_a_hollow_shape_knows_it_is_hollow(self):
        """Marked, not inferred.

        Guessing from ``operation == "difference"`` would also catch the
        concave cube, which is solid -- and the verdict needs to tell a
        cage from a particle cut too small.
        """
        cage = sh.hollowed("sphere", 6.0, radius=18.0)
        assert cage.info["hollow"] is True
        assert cage.info["wall_thickness"] == pytest.approx(6.0)
        assert not cage.contains(np.zeros((1, 3)))[0]
        exact = 4 / 3 * math.pi * (18.0 ** 3 - 12.0 ** 3)
        assert cage.volume(200) == pytest.approx(exact, rel=0.03)
        assert sh.concave_cube(26.0, 5.0).info.get("hollow") is None

    def test_a_half_space_is_unbounded_and_says_so(self):
        """A finite extent here would quietly clip whatever it met."""
        assert not math.isfinite(sh.half_space((1, 0, 0), 5.0).extent)
        # And it costs nothing once intersected with something closed.
        clipped = sh.intersection(sh.half_space((0, 0, 1), 5.0),
                                  sh.ball(10.0))
        assert clipped.extent == pytest.approx(10.0)

    def test_an_elongated_shape_is_aligned_not_rewritten(self):
        """Every elongated primitive is written along z and turned once."""
        along_x = sh.aligned_to(sh.capsule(4.0, 12.0), (1, 0, 0))
        assert along_x.contains(np.array([[15.0, 0.0, 0.0]]))[0]
        assert not along_x.contains(np.array([[0.0, 15.0, 0.0]]))[0]

    def test_a_rotation_has_to_be_rigid(self):
        """A general linear map rescales the distance units."""
        with pytest.raises(ValueError, match="orthogonal"):
            sh.rotated(sh.ball(5.0), np.diag([2.0, 1.0, 1.0]))

    def test_the_refusals_name_the_number(self):
        with pytest.raises(ValueError, match="positive"):
            sh.ball(-1.0)
        with pytest.raises(ValueError, match="Unknown shape"):
            sh.make_shape("banana")
        with pytest.raises(ValueError, match="two particles"):
            sh.dumbbell(10.0, 60.0)
        with pytest.raises(ValueError, match="hole closes"):
            sh.torus(6.0, 8.0)
        with pytest.raises(ValueError, match="takes"):
            sh.make_shape("sphere", edge=3.0)


# ----------------------------------------------------------------------
# the crystal
# ----------------------------------------------------------------------
class TestTheImportedCrystal:
    """Reading the file, and the lattice symmetry read off it."""

    @pytest.mark.parametrize(("cell", "order", "system"), [
        (np.eye(3) * 4.08, 48, "cubic"),
        # The fcc primitive cell looks rhombohedral and is cubic. Getting
        # this one right is the point of deriving the group from the
        # metric rather than from the cell's shape.
        (np.array([[0.0, 2.04, 2.04], [2.04, 0.0, 2.04],
                   [2.04, 2.04, 0.0]]), 48, "cubic"),
        (np.array([[2.46, 0.0, 0.0], [-1.23, 2.1304, 0.0],
                   [0.0, 0.0, 6.70]]), 24, "hexagonal"),
        (np.diag([3.0, 3.0, 5.0]), 16, "tetragonal"),
        (np.diag([3.0, 4.0, 5.0]), 8, "orthorhombic"),
        (np.array([[3.0, 0.0, 0.0], [0.4, 4.0, 0.0],
                   [0.3, 0.7, 5.0]]), 2, "triclinic"),
    ])
    def test_the_lattice_point_group_comes_out_right(self, cell, order,
                                                     system):
        """An integer matrix preserving the metric tensor, found by search.

        Exact statements both: an operation maps the lattice onto itself
        iff it is an integer matrix in the cell's basis, and it is a
        rotation or reflection iff it preserves the metric. No spglib and
        no tolerance on the symmetry itself.
        """
        group = lattice_point_group(cell)
        assert len(group) == order
        from nanocarbon_lab.nanoparticle.crystal import lattice_system

        assert lattice_system(len(group)) == system
        assert any(np.array_equal(item, np.eye(3, dtype=int))
                   for item in group)

    @pytest.mark.parametrize(("indices", "count"), [
        ((1, 0, 0), 6), ((1, 1, 0), 12), ((1, 1, 1), 8), ((2, 1, 1), 24),
    ])
    def test_the_cubic_families_have_their_textbook_sizes(self, indices,
                                                          count):
        assert len(miller_family(np.eye(3) * 4.08, indices)) == count

    def test_a_miller_normal_comes_from_the_reciprocal_lattice(self):
        """For anything but a cubic cell it is not parallel to (hkl).

        In a calcite-like rhombohedral cell the (104) cleavage normal --
        the plane every calcite rhomb is bounded by -- is 15.5 degrees
        from the direction its indices would point if read as Cartesian
        components. A 15-degree error in a facet normal is a different
        facet.
        """
        from ase.lattice import RHL

        cell = np.asarray(RHL(6.375, 46.1).tocell()[:])
        normal = miller_normal(cell, (1, 0, 4))
        naive = np.array([1.0, 0.0, 4.0])
        naive = naive / np.linalg.norm(naive)
        angle = math.degrees(math.acos(abs(float(normal @ naive))))
        assert 14.0 < angle < 17.0
        # The cubic case is the one where the two do agree.
        cubic = miller_normal(np.eye(3) * 4.0, (1, 0, 4))
        assert float(cubic @ naive) == pytest.approx(1.0, abs=1e-9)

    def test_graphite_s_basal_spacing_is_the_c_axis(self, tmp_path):
        cell = np.array([[2.46, 0.0, 0.0], [-1.23, 2.1304, 0.0],
                         [0.0, 0.0, 6.70]])
        assert interplanar_spacing(cell, (0, 0, 1)) == pytest.approx(6.70)
        assert len(miller_family(cell, (0, 0, 1))) == 2

    def test_a_file_with_no_cell_is_refused_by_name(self, tmp_path):
        """A molecule has no lattice to repeat."""
        from ase import Atoms

        molecule = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0],
                                           [-0.24, 0.93, 0]])
        path = tmp_path / "water.xyz"
        write(path, molecule)
        with pytest.raises(ValueError, match="no usable unit cell"):
            read_crystal(path)

    def test_the_repeat_count_follows_the_inverse_cell(self):
        """``extent / |a_i|`` is right for a cubic cell and short for an
        oblique one, which shows up as a particle with two flat sides."""
        oblique = np.array([[4.0, 0.0, 0.0], [2.0, 3.5, 0.0],
                            [1.0, 1.0, 4.5]])
        repeats = repeats_for(oblique, (12.0, 12.0, 12.0), margin=1.0)
        naive = [int(math.ceil(2 * 13.0 / np.linalg.norm(row)))
                 for row in oblique]
        assert all(a >= b for a, b in zip(repeats, naive, strict=True))
        # And the block really does cover the box it was asked for.
        corners = np.asarray([[sx * 13.0, sy * 13.0, sz * 13.0]
                              for sx in (1, -1) for sy in (1, -1)
                              for sz in (1, -1)])
        fractional = corners @ np.linalg.inv(oblique)
        span = fractional.max(axis=0) - fractional.min(axis=0)
        assert all(count >= s for count, s in zip(repeats, span, strict=True))

    def test_an_unbounded_shape_has_no_block(self):
        with pytest.raises(ValueError, match="no finite extent"):
            repeats_for(np.eye(3) * 4.0, (math.inf, 1.0, 1.0))


# ----------------------------------------------------------------------
# the cut
# ----------------------------------------------------------------------
class TestTheCut:
    """What comes out, checked against the lattice and against ASE."""

    @pytest.mark.parametrize("length", [4, 5, 6, 7, 9])
    def test_an_octahedron_matches_ase_cluster(self, gold, length):
        """The one check here that is not this package marking its own work.

        ``ase.cluster.Octahedron`` builds the same solid by a completely
        different route -- shells of lattice sites rather than a cut
        through a block -- so agreement to the atom pins the whole chain:
        Miller family, half-space polyhedron, supercell, cut.

        The face distance follows from the geometry: an octahedron with
        ``length`` atoms along an edge has its vertices at
        ``(length - 1) * a / 2``, and a regular octahedron's inradius is
        its circumradius over sqrt(3).
        """
        reference = Octahedron("Au", length, latticeconstant=GOLD)
        distance = (length - 1) * GOLD / (2.0 * math.sqrt(3.0))
        particle, report = carve_nanoparticle(
            gold, sh.octahedron(distance + 1e-4), centre="atom:Au",
            trim_loose=False)
        assert report.n_atoms == len(reference)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_a_cube_of_fcc_holds_exactly_four_n_cubed(self, gold, n):
        """With its faces BETWEEN the lattice planes there is no tie.

        Offset by a quarter of the lattice constant the cut is
        unambiguous, and then the count is the lattice's own.
        """
        edge = n * GOLD
        shape = sh.translated(sh.cube(edge), (edge / 2 + GOLD / 4,) * 3)
        _, report = carve_nanoparticle(gold, shape, centre="cell",
                                       trim_loose=False)
        assert report.n_atoms == 4 * n ** 3

    def test_a_facet_on_a_lattice_plane_is_reported_as_such(self, gold):
        """The trap, and the warning that exists for it.

        A 4a cube of fcc gold centred on a lattice point has its six
        faces exactly on planes of atoms, and a change of one part in a
        million in the requested edge moves the count between 171, 256
        and 365 -- a factor of 2.1. Nothing is wrong with any of the
        three; they are three different terminations. What would be
        wrong is not saying so.
        """
        edge = 4 * GOLD
        counts = []
        for epsilon in (-1e-6, 0.0, 1e-6):
            _, report = carve_nanoparticle(gold, sh.cube(edge + 2 * epsilon),
                                           centre="cell", trim_loose=False)
            counts.append(report.n_atoms)
        assert counts == [171, 256, 365]

        _, on_plane = carve_nanoparticle(gold, sh.cube(edge), centre="cell",
                                         trim_loose=False)
        assert on_plane.atoms_on_the_cut > 0
        assert any("commensurate" in text for text in on_plane.warnings)

        offset = sh.translated(sh.cube(edge), (edge / 2 + GOLD / 4,) * 3)
        _, between = carve_nanoparticle(gold, offset, centre="cell",
                                        trim_loose=False)
        assert between.atoms_on_the_cut == 0
        assert not any("commensurate" in text for text in between.warnings)
        assert TIE_TOLERANCE < 1e-2

    def test_the_centring_decides_a_compound_s_composition(self, magnesia):
        """And therefore its charge, which is not a detail.

        Centred on the cation the cube comes out cation-rich and centred
        on the anion, anion-rich. Two different structures, and the
        choice is which one was asked for.
        """
        _, on_mg = carve_nanoparticle(magnesia, sh.cube(16.0),
                                      centre="atom:Mg")
        _, on_o = carve_nanoparticle(magnesia, sh.cube(16.0),
                                     centre="atom:O")
        assert on_mg.composition != on_o.composition
        assert on_mg.net_excess["O"] == on_o.net_excess["Mg"]
        with pytest.raises(ValueError, match="has no Fe"):
            carve_nanoparticle(magnesia, sh.cube(16.0), centre="atom:Fe")
        with pytest.raises(ValueError, match="not one of"):
            carve_nanoparticle(magnesia, sh.cube(16.0), centre="middle")

    def test_trimming_to_stoichiometry_closes_the_gap(self, magnesia):
        _, before = carve_nanoparticle(magnesia, sh.cube(16.0),
                                       centre="atom:Mg")
        _, after = carve_nanoparticle(magnesia, sh.cube(16.0),
                                      centre="atom:Mg",
                                      stoichiometric=True)
        assert after.stoichiometry_deviation <= before.stoichiometry_deviation
        assert after.composition["Mg"] == after.composition["O"]
        assert after.trimmed_stoichiometry >= 0

    def test_loose_atoms_are_trimmed_iteratively(self, gold):
        """Removing one lowers its neighbour's count, so it cascades."""
        thin = sh.plate(16.0, 1.2)
        kept, report = carve_nanoparticle(gold, thin, centre="atom:Au",
                                          trim_loose=True)
        counts = coordination(kept)
        assert counts.min() > 1
        _, loose = carve_nanoparticle(gold, thin, centre="atom:Au",
                                      trim_loose=False)
        assert report.n_atoms <= loose.n_atoms

    def test_the_surface_is_measured_against_the_bulk_not_a_threshold(
            self, gold, magnesia):
        """Which is what makes the number mean the same thing for a
        metal and for an oxide without any per-material tuning."""
        _, au = carve_nanoparticle(gold, sh.ball(20.0), centre="atom:Au")
        assert au.bulk_coordination["Au"] == pytest.approx(12.0)
        _, mgo = carve_nanoparticle(magnesia, sh.ball(20.0),
                                    centre="atom:Mg")
        assert mgo.bulk_coordination["Mg"] == pytest.approx(6.0)
        assert mgo.bulk_coordination["O"] == pytest.approx(6.0)
        assert 0.0 < au.surface_fraction < 1.0

    def test_the_cutoff_reproduces_textbook_coordination(self):
        """The reason it is derived from the structure and not looked up.

        ASE's covalent radii put the Mg-Mg contact of rock salt (2.978 Å)
        inside any usable slack, so a radius-based cutoff reports
        magnesium as **18**-coordinate in MgO -- twelve magnesiums plus
        the six oxygens. An ionic crystal has no covalent radii to speak
        of and a metal has no bonds in that sense at all.
        """
        from ase.spacegroup import crystal as from_spacegroup

        from nanocarbon_lab.nanoparticle.carve import neighbour_cutoff

        rutile = from_spacegroup(
            ["Ti", "O"], basis=[(0, 0, 0), (0.3053, 0.3053, 0.0)],
            spacegroup=136, cellpar=[4.594, 4.594, 2.959, 90, 90, 90])
        cases = [
            (bulk("Au", "fcc", a=GOLD, cubic=True), {"Au": 12.0}),
            (bulk("MgO", "rocksalt", a=MAGNESIA, cubic=True),
             {"Mg": 6.0, "O": 6.0}),
            (rutile, {"Ti": 6.0, "O": 3.0}),
            (bulk("C", "diamond", a=3.567), {"C": 4.0}),
            # bcc's second shell is only 15 % beyond its first, so no
            # single slack separates them -- and 8+6 is how bcc is
            # described anyway.
            (bulk("Fe", "bcc", a=2.866, cubic=True), {"Fe": 14.0}),
        ]
        for crystal, expected in cases:
            crystal.pbc = True
            cutoff = neighbour_cutoff(crystal)
            _, report = carve_nanoparticle(crystal, sh.ball(12.0),
                                           centre="cell")
            assert cutoff > 0
            for element, value in expected.items():
                assert report.bulk_coordination[element] == pytest.approx(
                    value), (crystal.get_chemical_formula(), element)

    def test_a_cut_that_keeps_nothing_is_refused_with_the_numbers(self,
                                                                  gold):
        with pytest.raises(ValueError, match="nothing was kept"):
            carve_nanoparticle(gold, sh.ball(0.9), centre="void")

    def test_the_facet_census_is_empty_for_a_round_cut(self, gold):
        """A sphere has no facets, and a census that listed some would be
        describing the sampling rather than the particle."""
        _, sphere = carve_nanoparticle(gold, sh.ball(12.0), centre="atom:Au")
        assert sphere.facets == {}
        _, cube = carve_nanoparticle(gold, sh.octahedron(12.0),
                                     centre="atom:Au")
        assert len(cube.facets) == 8


# ----------------------------------------------------------------------
# the verdict
# ----------------------------------------------------------------------
class TestTheVerdict:
    def test_a_big_particle_is_clean_and_a_tiny_one_is_not(self, gold):
        _, big = carve_nanoparticle(gold, sh.ball(20.0), centre="atom:Au")
        assert particle_quality(big)[0] == "clean"
        _, small = carve_nanoparticle(gold, sh.ball(5.0), centre="atom:Au")
        verdict, why = particle_quality(small)
        assert verdict == "broken"
        assert "no interior" in why

    def test_a_cage_is_judged_as_a_cage(self, gold):
        """It is almost all surface because that is what a cage is.

        Judging one by the solid particle's thresholds is the mistake
        ``sp2_quality`` made when its hexagonal window called every sound
        haeckelite broken: the window was right and the family was wrong.
        """
        cage = sh.hollowed("sphere", 6.0, radius=16.0)
        _, report = carve_nanoparticle(gold, cage, centre="atom:Au")
        assert report.hollow is True
        verdict, why = particle_quality(report)
        assert verdict == "strained"
        assert "hollow by design" in why

    def test_an_off_stoichiometric_compound_is_called_out(self, magnesia):
        _, report = carve_nanoparticle(magnesia, sh.ball(9.0),
                                       centre="atom:Mg")
        if report.stoichiometry_deviation > 0.02:
            assert particle_quality(report)[0] in ("strained", "broken")


# ----------------------------------------------------------------------
# the coating
# ----------------------------------------------------------------------
class TestTheEnvelope:
    def test_the_envelope_sits_at_the_gap_from_the_nearest_atom(self, gold):
        core, _ = carve_nanoparticle(gold, sh.ball(8.0), centre="atom:Au")
        field, extent = vdw_envelope(core, 3.35)
        positions = core.get_positions()
        from scipy.spatial import cKDTree

        tree = cKDTree(positions)
        # On the level set, every point is exactly `gap` from the core.
        probe = positions[0] + np.array([[0.0, 0.0, 0.0]])
        assert float(field(probe)[0]) == pytest.approx(-3.35, abs=1e-9)
        assert extent > float(np.linalg.norm(positions, axis=1).max())
        far = positions[0] + np.array([[0.0, 0.0, 10.0]])
        distance, _ = tree.query(far)
        assert float(field(far)[0]) == pytest.approx(
            float(distance[0]) - 3.35, abs=1e-9)

    def test_per_element_gaps_take_the_binding_surface(self, magnesia):
        """A nearer small atom can be further from its own surface than a
        more distant large one, so one nearest-neighbour query will not
        do once the radii differ."""
        core, _ = carve_nanoparticle(magnesia, sh.ball(8.0),
                                     centre="atom:Mg")
        uniform, _ = vdw_envelope(core, 3.35)
        mixed, _ = vdw_envelope(core, 3.35, {"O": 4.5})
        probe = core.get_positions()[:40] * 1.6
        assert np.all(mixed(probe) <= uniform(probe) + 1e-9)
        assert float(np.min(mixed(probe) - uniform(probe))) < -0.1

    def test_a_gap_inside_the_core_is_refused(self, gold):
        core, _ = carve_nanoparticle(gold, sh.ball(8.0), centre="atom:Au")
        with pytest.raises(ValueError, match="shorter than"):
            graphitic_coat(core, gap=1.0)
        with pytest.raises(ValueError, match="merges consecutive"):
            graphitic_coat(core, shells=2, interlayer=1.0)
        with pytest.raises(ValueError, match="at least 1"):
            graphitic_coat(core, shells=0)

    def test_the_carbide_formers_get_a_warning_either_way(self):
        assert "Fe" in CARBIDE_FORMERS
        assert "Au" not in CARBIDE_FORMERS
        assert GRAPHITIC_GAP == pytest.approx(3.35)


@pytest.mark.slow
class TestTheCoating:
    """One build, shared: meshing and relaxing a shell takes seconds."""

    @pytest.fixture(scope="class")
    def coated(self):
        gold = bulk("Au", "fcc", a=GOLD, cubic=True)
        gold.pbc = True
        core, carve = carve_nanoparticle(gold, sh.ball(8.0),
                                         centre="atom:Au")
        shells, report = graphitic_coat(core, gap=3.35, shells=1)
        return core, shells, report

    def test_the_ring_census_satisfies_euler_for_the_surface_meshed(
            self, coated):
        """Nothing is placed: the field's topology fixes the census and
        the shared finisher verifies it against the mesh's own genus. An
        envelope of a solid is sphere-like, so the budget is +12."""
        _, shells, report = coated
        deficit, expected = report.ring_budget[0]
        assert deficit == expected == 12
        counts = report.ring_counts[0]
        assert counts.get(5, 0) > 0
        assert counts.get(6, 0) > counts.get(5, 0)
        assert shells[0].get_chemical_symbols() == ["C"] * len(shells[0])

    def test_the_achieved_gap_is_reported_in_three_numbers(self, coated):
        """The wall of a closed cage is corrugated, so one will not do:
        the mean says where the shell sits, the minimum is one atom of
        several hundred, and the fifth percentile is what compares to a
        van der Waals contact."""
        core, _, report = coated
        assert report.achieved_gap_min <= report.achieved_gap_p5
        assert report.achieved_gap_p5 <= report.achieved_gap_mean
        assert 2.5 < report.achieved_gap_mean < 3.6
        assert report.achieved_gap_min > 1.0

    def test_the_wall_is_carbon_of_the_right_density(self, coated):
        """A graphitic sheet is one atom thick, so the shell's atom count
        follows its AREA. Within a fifth of the ideal is all this claims:
        the envelope of a facetted core is not a sphere."""
        core, shells, _ = coated
        shell = shells[0]
        radius = float(np.linalg.norm(
            shell.get_positions() - shell.get_positions().mean(axis=0),
            axis=1).mean())
        area = 4.0 * math.pi * radius ** 2
        per_atom = area / len(shell)
        assert 2.1 < per_atom < 3.2, per_atom

    def test_curvature_placement_beats_annealing_on_a_convex_envelope(
            self, coated):
        """The measured claim the defaults rest on.

        An envelope is the offset of a solid, so it is convex everywhere:
        pentagons belong on it and heptagons are defects. Placement cuts
        the disclination count without touching the budget; annealing
        optimises the census and makes the wall worse. Here the weaker,
        cheaper assertion: placement is on, annealing is off, and the
        census has far fewer strays than the un-placed route's 39/25/1.
        """
        _, shells, report = coated
        assert shells[0].info["place_curvature"] is True
        assert shells[0].info["anneal_sweeps"] == 0
        counts = report.ring_counts[0]
        strays = sum(count for size, count in counts.items() if size != 6)
        assert strays < 40, counts


@pytest.mark.slow
class TestEncapsulation:
    """Core plus shell, with the chemistry on the shell only."""

    @pytest.fixture(scope="class")
    def encapsulated(self):
        gold = bulk("Au", "fcc", a=GOLD, cubic=True)
        gold.pbc = True
        core, _ = carve_nanoparticle(gold, sh.ball(8.0), centre="atom:Au")
        return core, *encapsulate(core, gap=3.35, shells=1, dopant="N",
                                  dopant_conc=0.04, dopant_site="pentagon",
                                  graft="hydroxyl", graft_coverage=0.05,
                                  seed=0)

    def test_the_core_is_never_substituted(self, encapsulated):
        """Putting a nitrogen where an iron was is not doping -- it is a
        different compound. The dopant chemistry here is about what an
        sp2 carbon lattice tolerates."""
        core, atoms, report = encapsulated
        symbols = atoms.get_chemical_symbols()
        start, stop = atoms.info["core_indices"]
        assert set(symbols[start:stop]) == {"Au"}
        assert symbols[start:stop].count("Au") == len(core)
        assert report.dopant == "N"
        assert "N" in symbols[stop:]

    def test_the_groups_go_on_the_outer_face(self, encapsulated):
        """A hydroxyl pointing into a gold particle is not a structure."""
        _, atoms, report = encapsulated
        assert report.graft == "hydroxyl"
        assert report.graft_count > 0
        assert "O" in atoms.get_chemical_symbols()

    def test_the_shell_can_be_co_doped_too(self):
        """The window offers the control, so it has to work.

        Both species against the same denominator, in one pass -- and
        refused together with `dopant`, because running one substitution
        after another makes the second fraction a fraction of what the
        first left.
        """
        gold = bulk("Au", "fcc", a=GOLD, cubic=True)
        gold.pbc = True
        core, _ = carve_nanoparticle(gold, sh.ball(7.0), centre="atom:Au")
        atoms, report = encapsulate(
            core, gap=3.35, shells=1,
            codope_spec=[("N", 0.04), ("B", 0.04)],
            codope_affinity="seek", seed=0)
        symbols = atoms.get_chemical_symbols()
        start, stop = atoms.info["core_indices"]
        assert set(symbols[start:stop]) == {"Au"}
        assert "N" in symbols[stop:] and "B" in symbols[stop:]
        assert report.dopant == "N+B"
        with pytest.raises(ValueError, match="not both"):
            encapsulate(core, shells=1, dopant="N", dopant_conc=0.05,
                        codope_spec=[("B", 0.05)])

    def test_the_split_between_core_and_shell_is_recorded(self,
                                                          encapsulated):
        """Almost every analysis in this framework is about carbon and
        would otherwise run over the metal too."""
        core, atoms, _ = encapsulated
        start, stop = atoms.info["core_indices"]
        shell_start, shell_stop = atoms.info["shell_indices"]
        assert (start, stop) == (0, len(core))
        assert shell_start == stop
        assert shell_stop == len(atoms)


# ----------------------------------------------------------------------
# the one-call build
# ----------------------------------------------------------------------
class TestTheSharedBuilder:
    def test_a_bare_core_carries_all_three_reports(self, gold, tmp_path):
        path = tmp_path / "au.cif"
        write(path, gold)
        atoms, crystal, carve, coat = build_nanoparticle(
            path, shape="cube", shape_params={"edge": 20.0},
            centre="atom:Au")
        assert coat is None
        assert crystal.formula == "Au"
        assert carve.n_atoms == len(atoms)
        assert atoms.info["crystal"]["lattice_system"] == "cubic"
        assert atoms.info["particle_verdict"] in ("clean", "strained",
                                                  "broken")
        assert atoms.info["shape"]["shape"] == "cube"

    def test_the_estimate_is_a_size_not_a_count(self, gold, tmp_path):
        """A volume argument cannot know that a facetted cut loses a
        whole plane of atoms per face, so the cube reads high."""
        path = tmp_path / "au.cif"
        write(path, gold)
        for shape, params, tolerance in [
            ("sphere", {"radius": 15.0}, 0.10),
            ("cube", {"edge": 24.0}, 0.25),
        ]:
            estimate = estimate_nanoparticle_atoms(path, shape, params)
            atoms, _, _, _ = build_nanoparticle(path, shape, params,
                                                centre="atom:Au")
            assert estimate == pytest.approx(len(atoms),
                                             rel=tolerance), shape

    def test_an_unreadable_file_estimates_zero_rather_than_raising(self,
                                                                   tmp_path):
        """An estimate is shown while the path is still being typed."""
        assert estimate_nanoparticle_atoms(tmp_path / "nothing.cif") == 0

    def test_a_hollow_request_reaches_the_shape(self, gold, tmp_path):
        path = tmp_path / "au.cif"
        write(path, gold)
        atoms, _, carve, _ = build_nanoparticle(
            path, shape="sphere", shape_params={"radius": 16.0},
            hollow=6.0, centre="atom:Au")
        assert carve.hollow is True
        assert atoms.info["shape"]["hollow"] is True


class TestTheJobIntegration:
    def test_the_mode_is_its_own_family_and_builds(self, gold, tmp_path):
        from nanocarbon_lab.jobs import (
            FAMILIES,
            NANOPARTICLE_MODES,
            Job,
            estimate_atoms,
            estimate_cost,
            family_of,
            to_cli,
        )

        path = tmp_path / "au.cif"
        write(path, gold)
        assert FAMILIES["nanoparticle"] == NANOPARTICLE_MODES
        job = Job(mode="nanoparticle",
                  params={"cif": str(path), "shape": "sphere",
                          "shape_params": {"radius": 10.0},
                          "centre": "atom:Au", "shells": 0})
        assert family_of("nanoparticle") == "nanoparticle"
        assert estimate_atoms(job) > 0
        severity, text = estimate_cost(job)
        assert severity == "fast"
        assert "no meshing" in text
        line = to_cli(job, out="out/np")
        assert "nanocarbon nanoparticle" in line
        assert "--set radius=10" in line
        assert "--centre atom:Au" in line

    def test_a_co_doped_job_emits_a_command_that_parses(self, gold,
                                                        tmp_path):
        """The equivalent command line has to be runnable, not just
        printable -- it is what reproduces a result found by dragging
        sliders, and a flag the parser rejects is worse than none."""
        from nanocarbon_lab.cli.main import build_parser
        from nanocarbon_lab.jobs import Job, to_cli

        path = tmp_path / "au.cif"
        write(path, gold)
        job = Job(mode="nanoparticle",
                  params={"cif": str(path), "shape": "sphere",
                          "shape_params": {"radius": 8.0},
                          "centre": "atom:Au", "shells": 1},
                  codope="N:0.04,B:0.04", codope_affinity="seek", seed=5)
        line = to_cli(job, out="out/x")
        assert "--codope N:0.04,B:0.04" in line
        args = build_parser().parse_args(line.split()[1:])
        assert args.codope == "N:0.04,B:0.04"
        assert args.codope_affinity == "seek"

    def test_a_coated_job_is_costed_as_meshed(self, gold, tmp_path):
        from nanocarbon_lab.jobs import Job, estimate_cost

        path = tmp_path / "au.cif"
        write(path, gold)
        job = Job(mode="nanoparticle",
                  params={"cif": str(path), "shape": "sphere",
                          "shape_params": {"radius": 10.0},
                          "centre": "atom:Au", "shells": 2},
                  dopant="N", dopant_conc=0.05)
        severity, text = estimate_cost(job)
        assert severity in ("slow", "very slow")
        assert "meshed and relaxed" in text
