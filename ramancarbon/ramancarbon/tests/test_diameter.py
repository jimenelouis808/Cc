"""RBM diameters, chirality, G-splitting and wall pairing."""

from __future__ import annotations

import pytest

from ramancarbon.analysis.diameter import (
    assign_chirality,
    chiral_angle,
    chiral_diameter,
    chirality_kind,
    compare_parameterisations,
    diameter_from_g_splitting,
    diameter_to_rbm,
    electronic_type,
    find_wall_pairs,
    rbm_diameter_with_spread,
    rbm_to_diameter,
)


def test_rbm_round_trip():
    for diameter in (0.8, 1.2, 1.8, 2.4):
        omega = diameter_to_rbm(diameter)
        assert rbm_to_diameter(omega).diameter_nm == pytest.approx(diameter, rel=1e-9)


@pytest.mark.parametrize(
    "n, m, diameter",
    [(10, 10, 1.356), (6, 5, 0.747), (9, 0, 0.705), (16, 0, 1.253), (7, 6, 0.882)],
)
def test_known_tube_diameters(n, m, diameter):
    assert chiral_diameter(n, m) == pytest.approx(diameter, abs=0.002)


@pytest.mark.parametrize(
    "n, m, kind, electronic",
    [
        (10, 10, "armchair", "metallic"),
        (9, 0, "zigzag", "quasi-metallic"),
        (16, 0, "zigzag", "semiconducting"),
        (6, 5, "chiral", "semiconducting"),
        (12, 6, "chiral", "quasi-metallic"),
    ],
)
def test_chirality_classification(n, m, kind, electronic):
    assert chirality_kind(n, m) == kind
    assert electronic_type(n, m) == electronic


def test_chiral_angles_span_zero_to_thirty():
    assert chiral_angle(10, 0) == pytest.approx(0.0)
    assert chiral_angle(10, 10) == pytest.approx(30.0)


def test_chirality_assignment_is_a_set_not_a_unique_answer():
    """Diameter alone cannot fix (n,m); the code must return candidates."""
    candidates = assign_chirality(1.24, tolerance_nm=0.02)
    assert len(candidates) > 1
    assert all(abs(c.diameter_nm - 1.24) <= 0.02 for c in candidates)
    assert candidates == sorted(candidates, key=lambda c: c.mismatch_nm)


def test_chirality_can_be_restricted_by_metallicity():
    only = assign_chirality(1.24, tolerance_nm=0.05, only="metallic")
    assert only and all(c.electronic != "semiconducting" for c in only)


def test_parameterisations_disagree_by_more_than_the_peak_precision():
    """The reason the spread is reported rather than one number."""
    values = [e.diameter_nm for e in compare_parameterisations(200.0).values()]
    assert max(values) - min(values) > 0.05
    assert rbm_diameter_with_spread(200.0).uncertainty_nm > 0.02


def test_rbm_below_the_offset_is_rejected():
    with pytest.raises(ValueError, match="not an RBM"):
        rbm_to_diameter(5.0)


def test_g_splitting_diameter():
    estimate = diameter_from_g_splitting(1591.0, 1570.0, metallic=False)
    assert estimate.diameter_nm == pytest.approx((47.7 / 21.0) ** 0.5, rel=1e-6)


def test_g_splitting_refuses_multiwalled_material():
    """There, the feature above G is D' — a defect band, not a split G."""
    with pytest.raises(ValueError, match="multi-walled"):
        diameter_from_g_splitting(1591.0, 1570.0, walls=5)


def test_g_splitting_reports_the_metallicity_alternative():
    estimate = diameter_from_g_splitting(1591.0, 1570.0, metallic=False)
    assert any("metálico" in w for w in estimate.warnings)


def test_wall_pairing_accepts_a_real_double_wall():
    pairs = [p for p in find_wall_pairs([158.0, 265.0]) if p.plausible]
    assert pairs
    assert 0.33 <= pairs[0].spacing_nm <= 0.37


def test_wall_pairing_rejects_two_similar_single_walls():
    """A mixture of two SWCNT diameters must not read as a DWCNT."""
    assert not [p for p in find_wall_pairs([158.0, 165.0]) if p.plausible]
