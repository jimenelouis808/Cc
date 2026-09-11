"""Tests for the helical nanocoil builder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from carbonforge.builders import build_nanocoil
from carbonforge.validation import run_basic_checks
from carbonforge.topology import coordination_numbers


class TestNanocoilGeometry:
    def test_arc_length_matches_formula(self):
        coil = build_nanocoil(n=5, m=5, coil_radius=30.0, pitch=15.0, n_turns=1.0)
        expected = math.sqrt((2 * math.pi * 30.0) ** 2 + 15.0 ** 2)
        assert coil.info["arc_length"] == pytest.approx(expected, rel=1e-10)

    def test_reasonable_bond_distortion(self):
        coil = build_nanocoil(n=5, m=5, coil_radius=30.0, pitch=12.0, n_turns=1.0)
        dmat = coil.get_all_distances(mic=False)
        np.fill_diagonal(dmat, np.inf)
        # Nearest-neighbour distance must stay in the physical [1.2, 1.8] Å window.
        assert 1.2 < dmat.min() < 1.8

    def test_coordination_is_mostly_three(self):
        coil = build_nanocoil(n=6, m=6, coil_radius=30.0, pitch=12.0, n_turns=1.0)
        coord = coordination_numbers(coil)
        # Most atoms sp2 (coordination 3); edges at the coil tips are coord 2.
        assert (coord == 3).mean() > 0.85

    def test_tight_coil_rejected(self):
        with pytest.raises(ValueError):
            # R = 5 Å is smaller than 2× a (6,6) CNT radius (~4 Å).
            build_nanocoil(n=6, m=6, coil_radius=5.0, pitch=12.0, n_turns=1.0)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            build_nanocoil(n=5, m=5, coil_radius=0.0, pitch=12.0, n_turns=1.0)
        with pytest.raises(ValueError):
            build_nanocoil(n=5, m=5, coil_radius=25.0, pitch=12.0, n_turns=0.0)

    def test_validation_passes(self):
        coil = build_nanocoil(n=5, m=5, coil_radius=25.0, pitch=12.0, n_turns=1.0)
        rep = run_basic_checks(coil)
        assert rep.ok, rep.summary()

    def test_metadata_records_params(self):
        coil = build_nanocoil(n=5, m=5, coil_radius=20.0, pitch=10.0, n_turns=1.5)
        info = coil.info
        assert info["structure_type"] == "nanocoil"
        assert info["coil_radius"] == 20.0
        assert info["pitch"] == 10.0
        assert info["n_turns"] == 1.5

    def test_stone_wales_density_bounds(self):
        with pytest.raises(ValueError):
            build_nanocoil(n=5, m=5, coil_radius=30.0, pitch=12.0,
                           n_turns=1.0, stone_wales_density=0.5)

    def test_atoms_span_full_height(self):
        """One turn at pitch P must span ~P along z."""
        pitch = 15.0
        coil = build_nanocoil(n=6, m=6, coil_radius=25.0, pitch=pitch, n_turns=1.0)
        z = coil.get_positions()[:, 2]
        span = float(z.max() - z.min())
        # Allow for tube diameter contribution on both ends.
        assert span >= pitch - 1.0
