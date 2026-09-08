"""Diffraction: symmetry, structures, calculated patterns, phase ID."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ramancarbon.examples.demo_data import XRD_DEMOS, make_xrd_demo
from ramancarbon.xrd.cif import CIFError, parse_cif, read_cif, write_cif
from ramancarbon.xrd.io import PatternIOError, read_pattern, write_pattern
from ramancarbon.xrd.pattern import Pattern, PatternError
from ramancarbon.xrd.powder import (
    instrumental_correction,
    lorentz_polarisation,
    march_dollase,
    pseudo_voigt,
    reflections,
    scherrer,
    simulate,
)
from ramancarbon.xrd.preprocess import kalpha2_offset, strip_kalpha2
from ramancarbon.xrd.reference import find_phase, load_library
from ramancarbon.xrd.scattering import (
    ScatteringError,
    atomic_number,
    fluorescence_risk,
    form_factor,
    wavelength_for,
)
from ramancarbon.xrd.search import find_peaks, identify_phases
from ramancarbon.xrd.structure import Lattice, StructureError
from ramancarbon.xrd.symmetry import SymmetryError, close_group, orbit, parse_xyz

CU = wavelength_for("Cu", "ka1")


# -- symmetry ---------------------------------------------------------


@pytest.mark.parametrize(
    "generators, order",
    [
        (["-y+1/2,x,z", "-x,y+1/2,-z", "-x,-y,-z"], 16),            # P4/nmm
        (["x-y,x,z+1/2", "-x,-y,z+1/2", "y,x,-z", "-x,-y,-z"], 24),  # P6_3/mmc
        (["-x,-y,z", "-x+1/2,y+1/2,-z+1/2", "-x,-y,-z"], 8),         # Pnnm
        (["-y,x-y,z+1/3", "y,x,-z"], 6),                             # P3_121
        (["-x+1/2,-y,z+1/2", "-x,y+1/2,-z", "-x,-y,-z"], 8),         # Pnma
        (["-y,x-y,z", "y,x,-z+1/2", "-x,-y,-z", "x+2/3,y+1/3,z+1/3"], 36),  # R-3c
        (["-x,y+1/2,-z+1/2", "-x,-y,-z"], 4),                        # P2_1/c
    ],
)
def test_group_closure_reproduces_the_known_order(generators, order):
    """The order of the closed group is an exact, independent check on the
    generators: a wrong operation almost always changes it."""
    assert len(close_group(generators)) == order


def test_operation_strings_round_trip():
    for text in ("x,y,z", "-x+1/2, y, -z+1/2", "1/2-x, 1/2+y, z", "y,x,-z"):
        operation = parse_xyz(text)
        assert parse_xyz(operation.to_xyz()) == operation


def test_a_malformed_operation_is_refused_rather_than_closed_forever():
    with pytest.raises(SymmetryError):
        close_group(["x, y"])
    with pytest.raises(SymmetryError):
        close_group(["0.37*x, y, z"], max_order=50)


def test_special_positions_are_merged_not_multiplied():
    """An atom on a symmetry element maps onto itself; counting each image
    separately would multiply its scattering power by the site symmetry."""
    group = close_group(
        ["-x+3/4,-y+1/4,z+1/2", "-x+1/4,y+1/2,-z+3/4", "z,x,y",
         "y+3/4,x+1/4,-z+1/2", "-x,-y,-z", "x,y+1/2,z+1/2", "x+1/2,y,z+1/2"]
    )
    assert len(group) == 192
    assert len(orbit([0.125, 0.125, 0.125], group)) == 8    # 8a
    assert len(orbit([0.5, 0.5, 0.5], group)) == 16         # 16d
    assert len(orbit([0.2549, 0.2549, 0.2549], group)) == 32  # 32e


# -- lattice and structure --------------------------------------------


def test_impossible_cells_are_refused():
    with pytest.raises(StructureError):
        Lattice(0.0, 1.0, 1.0)
    with pytest.raises(StructureError):
        Lattice(1.0, 1.0, 1.0, alpha=200.0)
    with pytest.raises(StructureError):
        Lattice(1.0, 1.0, 1.0, alpha=20.0, beta=20.0, gamma=170.0)


def test_d_spacing_matches_the_closed_form_for_a_cubic_cell():
    lattice = Lattice(5.0, 5.0, 5.0)
    hkl = [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]]
    expected = [5.0 / math.sqrt(h * h + k * k + l * l) for h, k, l in hkl]
    assert np.allclose(lattice.d_spacing(hkl), expected)


def test_lattice_constraints_come_from_the_operations():
    assert find_phase("grafito_2H").lattice_constraint() == ("a", "a", "c")
    assert find_phase("Si").lattice_constraint() == ("a", "a", "a")
    assert find_phase("Fe3C_cementita").lattice_constraint() == ("a", "b", "c")


# -- the bundled library ----------------------------------------------

#: Literature densities in g/cm3. Density validates the cell and the cell
#: CONTENTS at once, which is why it is the check used on the whole
#: bundled library rather than a spot check on one structure.
DENSITIES = {
    "grafito_2H": 2.26, "FeSe_tetragonal": 5.70, "FeSe2_marcasita": 7.14,
    "Se_trigonal": 4.81, "Fe_alfa": 7.874, "Fe3C_cementita": 7.67,
    "Fe3O4_magnetita": 5.20, "Fe2O3_hematita": 5.26, "Si": 2.329,
    "MoS2_2H": 5.02, "MoSe2_2H": 6.90, "MoO2": 6.47,
}

#: Strongest reflection, degrees 2theta with Cu Ka1.
STRONGEST = {
    "grafito_2H": 26.5, "FeSe_tetragonal": 28.6, "FeSe_hexagonal": 32.2,
    "Se_trigonal": 29.7, "Fe_alfa": 44.7, "Fe3C_cementita": 45.0,
    "Fe3O4_magnetita": 35.5, "Fe2O3_hematita": 33.2, "Si": 28.44,
    "MoS2_2H": 14.4, "WS2_2H": 14.4, "MoO2": 26.0,
}


def test_the_library_loads():
    entries = load_library()
    assert len(entries) >= 12
    assert all(entry.crystal.sites for entry in entries)


@pytest.mark.parametrize("name, density", sorted(DENSITIES.items()))
def test_bundled_densities_match_the_literature(name, density):
    crystal = find_phase(name)
    assert crystal is not None
    assert crystal.density == pytest.approx(density, rel=0.04), crystal.describe()


@pytest.mark.parametrize("name, angle", sorted(STRONGEST.items()))
def test_bundled_strongest_reflections_match_their_powder_cards(name, angle):
    crystal = find_phase(name)
    lines = reflections(crystal, wavelength=CU, two_theta_range=(5.0, 90.0))
    strongest = max(lines, key=lambda r: r.intensity)
    assert strongest.two_theta == pytest.approx(angle, abs=0.6)


def test_reflection_indices_are_reported_conventionally():
    lines = reflections(find_phase("grafito_2H"), wavelength=CU,
                        two_theta_range=(10.0, 60.0))
    assert lines[0].hkl == (0, 0, 2)
    assert all(all(v >= 0 for v in r.hkl) or r.is_overlap for r in lines)


# -- scattering -------------------------------------------------------


def test_form_factor_is_exactly_z_at_zero_angle():
    for element in ("C", "O", "Fe", "Se", "Mo", "W"):
        assert form_factor(element, 0.0)[0] == pytest.approx(atomic_number(element))


def test_form_factor_decreases_monotonically_in_range():
    for element in ("C", "Fe", "Se", "Mo"):
        values = form_factor(element, np.array([0.0, 0.1, 0.25, 0.5, 0.649]))
        assert np.all(np.diff(values) < 0)


def test_oxidation_states_in_cif_symbols_are_tolerated():
    assert form_factor("Fe2+", 0.2) == pytest.approx(form_factor("Fe", 0.2))
    with pytest.raises(ScatteringError):
        form_factor("Xx", 0.1)


def test_iron_under_a_copper_tube_is_flagged():
    assert fluorescence_risk(["Fe", "Se"], "Cu")
    assert fluorescence_risk(["Fe", "Se"], "Co") is None
    assert fluorescence_risk(["C", "O"], "Cu") is None


# -- profile and corrections ------------------------------------------


def test_pseudo_voigt_is_area_normalised():
    x = np.linspace(20.0, 30.0, 20001)
    for eta in (0.0, 0.5, 1.0):
        area = np.trapezoid(pseudo_voigt(x, 25.0, 0.2, eta), x)
        assert area == pytest.approx(1.0, rel=0.02)


def test_lorentz_polarisation_rises_towards_low_angle():
    values = lorentz_polarisation(np.array([10.0, 40.0, 80.0]))
    assert values[0] > values[1] > values[2]


def test_march_dollase_is_neutral_at_r_equal_one():
    crystal = find_phase("grafito_2H")
    hkl = np.array([[0, 0, 2], [1, 0, 0], [1, 1, 0]])
    assert np.allclose(march_dollase(hkl, (0, 0, 1), crystal, 1.0), 1.0)


def test_march_dollase_enhances_the_texture_axis():
    crystal = find_phase("grafito_2H")
    hkl = np.array([[0, 0, 2], [1, 1, 0]])
    values = march_dollase(hkl, (0, 0, 1), crystal, 0.6)
    assert values[0] > 1.0 > values[1]


def test_scherrer_refuses_a_resolution_limited_peak():
    assert instrumental_correction(0.05, 0.06) is None
    assert instrumental_correction(0.10, 0.06) == pytest.approx(0.04)
    # 0.9 x 1.5406 / (0.04 deg in rad x cos 13.25 deg) = 2040 A = 204 nm,
    # which is well above what a line width can actually resolve.
    assert scherrer(0.04, 26.5, CU) == pytest.approx(204.0, rel=0.02)


# -- Kalpha2 ----------------------------------------------------------


def test_kalpha2_separation_grows_with_angle():
    offsets = kalpha2_offset(np.array([20.0, 50.0, 80.0]), 1.540598, 1.544426)
    assert 0.0 < offsets[0] < offsets[1] < offsets[2]
    assert offsets[2] == pytest.approx(0.21, abs=0.03)


def test_stripping_removes_most_of_the_satellite():
    crystal = find_phase("FeSe_tetragonal")
    pattern = simulate(crystal, two_theta_range=(30.0, 60.0), step=0.02)
    stripped = strip_kalpha2(pattern)
    index = int(np.argmin(np.abs(pattern.two_theta - 47.44)))
    assert stripped.intensity[index] < 0.2 * pattern.intensity[index]
    assert not stripped.has_doublet


def test_the_satellite_is_not_reported_as_an_unknown_phase():
    """Without either stripping or attribution, every reflection above 40°
    is found twice and the extras look like a phase that is not there."""
    pattern = make_xrd_demo("CNT_FeSe", seed=4)
    result = identify_phases(pattern)
    strongest = max(p.height for p in result.peaks)
    assert all(p.height < 0.15 * strongest for p in result.unexplained)


# -- peak detection ---------------------------------------------------


def test_the_detection_threshold_holds_on_pure_noise():
    """Calibration, not decoration: at a threshold of 8 this same detector
    found 44 peaks per pattern in noise."""
    angles = np.arange(10.0, 80.0, 0.02)
    found = []
    for seed in range(6):
        background = 300.0 + 200.0 * np.exp(-(angles - 10.0) / 25.0)
        counts = np.random.default_rng(500 + seed).poisson(background).astype(float)
        pattern = Pattern(angles, counts, counts=True, kalpha2_ratio=0.0)
        found.append(len(find_peaks(pattern)))
    assert np.mean(found) < 1.0


def test_every_graphite_reflection_is_found_and_nothing_else():
    pattern = simulate(find_phase("grafito_2H"), two_theta_range=(10.0, 80.0),
                       background=300.0, counts_at_max=20000.0, seed=2)
    peaks = find_peaks(pattern)
    assert 7 <= len(peaks) <= 11


# -- phase identification ---------------------------------------------


@pytest.mark.parametrize("kind", [name for name, _, _ in XRD_DEMOS])
def test_every_demo_mixture_is_identified(kind):
    pattern = make_xrd_demo(kind, seed=7)
    result = identify_phases(pattern)
    found = {match.crystal.name for match in result.accepted}
    assert set(pattern.metadata["phases"]) <= found, result.summary()


def test_the_two_fese_polymorphs_are_told_apart():
    """The question Raman leaves open: 101 at 28.6° against 32.2°."""
    pattern = make_xrd_demo("FeSe_dos_fases", seed=3)
    found = {m.crystal.name for m in identify_phases(pattern).accepted}
    assert {"FeSe_tetragonal", "FeSe_hexagonal"} <= found


def test_a_phase_that_is_absent_is_not_reported():
    pattern = simulate(find_phase("Si"), two_theta_range=(10.0, 80.0),
                       background=250.0, counts_at_max=12000.0, seed=8)
    found = {m.crystal.name for m in identify_phases(pattern).accepted}
    assert found == {"Si"}


def test_a_zero_shift_is_fitted_rather_than_defeating_the_match():
    pattern = make_xrd_demo("CNT_FeSe", seed=5)
    pattern.two_theta = pattern.two_theta + 0.12
    result = identify_phases(pattern)
    assert result.zero_shift == pytest.approx(0.12, abs=0.03)
    assert "FeSe_tetragonal" in {m.crystal.name for m in result.accepted}
    assert any("altura" in w for w in result.warnings)


def test_an_empty_pattern_says_so_instead_of_inventing_phases():
    angles = np.arange(10.0, 80.0, 0.02)
    flat = np.random.default_rng(0).poisson(300.0, angles.size).astype(float)
    result = identify_phases(Pattern(angles, flat, counts=True, kalpha2_ratio=0.0))
    assert not result.accepted
    assert result.warnings


# -- CIF --------------------------------------------------------------


def test_cif_round_trip(tmp_path):
    original = find_phase("FeSe_tetragonal")
    path = write_cif(original, tmp_path / "x.cif")
    back = read_cif(path)
    assert back.order == original.order
    assert back.atoms_per_cell == original.atoms_per_cell
    assert back.lattice.a == pytest.approx(original.lattice.a)


def test_cif_numbers_with_uncertainties_are_read():
    text = """data_test
_cell_length_a 3.7734(2)
_cell_length_b 3.7734(2)
_cell_length_c 5.5258(3)
loop_
_symmetry_equiv_pos_as_xyz
 'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
 Fe1 Fe 0.75 0.25 0.0
"""
    block = parse_cif(text)[0]
    from ramancarbon.xrd.cif import crystal_from_block

    crystal = crystal_from_block(block, "prueba")
    assert crystal.lattice.a == pytest.approx(3.7734)
    assert crystal.atoms_per_cell == 1


def test_a_cif_that_names_a_space_group_without_its_operations_says_so():
    text = """data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_symmetry_space_group_name_H-M 'F d -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
 Si1 Si 0.125 0.125 0.125
"""
    from ramancarbon.xrd.cif import crystal_from_block

    crystal = crystal_from_block(parse_cif(text)[0], "roto")
    assert "NO trae la lista de operaciones" in crystal.notes


def test_a_broken_cif_raises(tmp_path):
    path = tmp_path / "bad.cif"
    path.write_text("no soy un cif\n", encoding="utf-8")
    with pytest.raises(CIFError):
        read_cif(path)


# -- pattern IO -------------------------------------------------------


def test_pattern_rejects_impossible_input():
    with pytest.raises(PatternError):
        Pattern(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(PatternError):
        Pattern(np.arange(10.0), np.append(np.arange(9.0), np.nan))
    with pytest.raises(PatternError):
        Pattern(np.linspace(-5.0, 5.0, 10), np.ones(10))


def test_pattern_io_round_trip(tmp_path):
    pattern = make_xrd_demo("CNT_FeSe", seed=1)
    path = write_pattern(pattern, tmp_path / "p.xye")
    back = read_pattern(path)
    assert back.n == pattern.n
    assert back.sigma_origin == "measured"
    assert np.allclose(back.intensity, pattern.intensity)


def test_a_binary_export_says_what_to_do_instead(tmp_path):
    path = tmp_path / "d.raw"
    path.write_bytes(b"\x00\x01\x02")
    with pytest.raises(PatternIOError, match="binario"):
        read_pattern(path)


def test_covers_distinguishes_absent_from_unmeasured():
    pattern = make_xrd_demo("CNT_FeSe", two_theta_range=(20.0, 60.0), seed=1)
    assert pattern.covers(30.0, 50.0)
    assert not pattern.covers(70.0, 90.0)
