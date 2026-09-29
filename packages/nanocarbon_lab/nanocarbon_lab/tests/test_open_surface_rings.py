"""A rim is what a boundary looks like, not evidence against a surface.

``is_surface_net`` gated face tracing on "at least 90% of bonded atoms
have degree 3", and that threshold was deciding real cases on its
margin: a planar X junction sat at 89.74% and fell back to shortest-path
rings, which silently lost all sixteen of its heptagons. A nanocone and
a nanoribbon are *defined* by having an edge and fell back too.

The replacement asks the question that actually distinguishes a surface:
no atom may have **more** than three neighbours, and at least one must
have three.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nanocarbon_lab.analyse.rings import (
    MAX_RING_SIZE,
    is_surface_net,
    ring_report,
    trace_faces,
)
from nanocarbon_lab.jobs import Job, build
from nanocarbon_lab.utils.geometry import guess_bonds


def _bonds(atoms) -> np.ndarray:
    recorded = atoms.info.get("bonds")
    if recorded is not None and len(recorded):
        return np.asarray(recorded, dtype=int)
    return np.asarray(guess_bonds(atoms), dtype=int)


def _built(mode: str, **params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build(Job(mode, dict(params)))


class TestAnOpenSurfaceIsStillASurface:
    """The three structures the threshold was excluding, and why."""

    CASES = [
        ("junction (knees)", {"kind": "x"}),
        ("nanocone", {"n_pentagons": 1, "radius": 22.0}),
        ("nanoribbon", {"width": 6, "length": 6}),
    ]

    @pytest.mark.parametrize("mode,params", CASES)
    def test_it_traces_faces_despite_the_rim(self, mode, params):
        atoms = _built(mode, **params)
        pairs = _bonds(atoms)
        assert is_surface_net(atoms, pairs)
        assert ring_report(atoms, pairs)["method"] == "faces"

    def test_the_x_junction_gets_its_heptagons_back(self):
        # The bug this exists for: shortest-path rings never emit a
        # heptagon every one of whose bonds also borders a hexagon, so
        # the census read {5: 4, 6: 328} and sum(6-n) = +4 where the
        # structure's own is {5: 4, 6: 328, 7: 16} and -12.
        atoms = _built("junction (knees)", kind="x")
        report = ring_report(atoms, _bonds(atoms))
        assert report["counts"] == atoms.info["ring_counts"]
        assert report["counts"].get(7) == 16
        assert report["euler_deficit"] == atoms.info["ring_deficit"] == -12

    @pytest.mark.parametrize("mode,params,expected", [
        ("junction (knees)", {"kind": "y"}, {6: 240, 7: 6}),
        ("junction (knees)", {"kind": "tetrahedral"}, {6: 328, 7: 12}),
        ("nanocone", {"n_pentagons": 1, "radius": 22.0}, {5: 1, 6: 210}),
    ])
    def test_the_perceived_census_is_the_builder_s_own(self, mode, params,
                                                       expected):
        atoms = _built(mode, **params)
        report = ring_report(atoms, _bonds(atoms))
        assert report["counts"] == expected == atoms.info["ring_counts"]


class TestWhatMustStayExcluded:
    """Widening the gate must not let in what faces mean nothing for."""

    def test_a_bulk_dichalcogenide_is_not_a_surface(self):
        # Its metal reaches degree 15 by the bond list, which is exactly
        # the thing "no more than three neighbours" rules out.
        atoms = _built("TMD bulk", material="MoS2")
        pairs = _bonds(atoms)
        degree = np.bincount(pairs.ravel(), minlength=len(atoms))
        assert degree.max() > 3
        assert not is_surface_net(atoms, pairs)

    def test_a_decorated_net_still_goes_through_its_backbone(self):
        # A carboxylated tube has degree-4 anchors, so it must not be
        # traced directly -- the wall is the thing with rings.
        atoms = _built("nanotube (open)", n=6, m=6, length=12.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grafted = build(Job("nanotube (open)",
                                {"n": 6, "m": 6, "length": 12.0},
                                graft="carboxyl", graft_coverage=0.15,
                                seed=1))
        pairs = _bonds(grafted)
        assert not is_surface_net(grafted, pairs)
        assert len(grafted) > len(atoms)

    def test_a_structure_with_no_trivalent_atom_is_not_a_surface(self):
        from ase import Atoms

        chain = Atoms("C3", positions=[[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0]])
        pairs = np.array([[0, 1], [1, 2]])
        assert not is_surface_net(chain, pairs)


class TestARimShortEnoughToPassAsARing:
    """`trace_faces` tells a face from the boundary by SIZE, which is
    exact only while the rim is longer than a ring can be."""

    def test_a_long_ribbon_lands_on_euler_exactly(self):
        # A ribbon is periodic along its length, so it is an annulus:
        # chi = 0 and therefore faces = E - V, with the two rims
        # discarded.
        for width, length in ((6, 6), (8, 8), (10, 10)):
            atoms = _built("nanoribbon", width=width, length=length)
            pairs = _bonds(atoms)
            edges = {(min(a, b), max(a, b)) for a, b in pairs.tolist()}
            faces, discarded = trace_faces(atoms, pairs)
            assert discarded == 2, (width, length)
            assert len(faces) == len(edges) - len(atoms), (width, length)

    def test_the_short_one_counts_its_rims_and_says_so(self):
        # 6 wide by 3 long: the rims come out SIX atoms long, a hexagon's
        # size, so both are counted as rings and the census is two heavy.
        # There is no way to tell them apart by size, so the report says
        # so rather than guessing which two to drop.
        atoms = _built("nanoribbon", width=6, length=3)
        pairs = _bonds(atoms)
        edges = {(min(a, b), max(a, b)) for a, b in pairs.tolist()}
        faces, discarded = trace_faces(atoms, pairs)
        assert discarded == 0
        assert len(faces) == len(edges) - len(atoms) + 2

        report = ring_report(atoms, pairs)
        assert not report["reliable"]
        assert "indistinguishable from a ring" in report["caveat"]

    def test_a_closed_structure_never_trips_the_caveat(self):
        # No rim, so nothing can be mistaken for one.
        atoms = _built("toroid (knees)")
        report = ring_report(atoms, _bonds(atoms))
        assert report["reliable"]
        assert report["n_boundary_walks"] == 0

    def test_an_open_structure_with_a_long_rim_is_reliable(self):
        atoms = _built("junction (knees)", kind="x")
        report = ring_report(atoms, _bonds(atoms))
        assert report["reliable"]
        # Four mouths, each far longer than a ring may be.
        assert report["n_boundary_walks"] == 4
        assert report["max_size_searched"] == MAX_RING_SIZE
