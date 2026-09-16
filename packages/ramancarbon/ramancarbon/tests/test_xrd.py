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
    # Added for the iron/nickel catalyst residues a CVD synthesis leaves.
    # Mackinawite is the one that shows what this check is for: written on
    # the wrong Wyckoff pair it came out with twelve atoms in the cell and
    # 11.7 g/cm3 against the mineral's 4.30.
    "Fe_gamma": 8.01, "FeP": 6.07, "Fe2P": 6.86,
    "FeS_mackinawita": 4.30, "FeS2_pirita": 5.01, "NiS_beta": 5.50,
    # The turbostratic carbons. Their density follows from the interlayer
    # spacing alone -- 2.26 g/cm3 at graphite's 3.356 A, falling as the
    # sheets separate -- so this check is a check on the cell, which is
    # the only thing about these three that carries information.
    "C_turbostratico_3.36": 2.259, "C_turbostratico_3.44": 2.206,
    "C_turbostratico_3.50": 2.169,
}

#: Strongest reflection, degrees 2theta with Cu Ka1.
STRONGEST = {
    "grafito_2H": 26.5, "FeSe_tetragonal": 28.6, "FeSe_hexagonal": 32.2,
    "Se_trigonal": 29.7, "Fe_alfa": 44.7, "Fe3C_cementita": 45.0,
    "Fe3O4_magnetita": 35.5, "Fe2O3_hematita": 33.2, "Si": 28.44,
    "MoS2_2H": 14.4, "WS2_2H": 14.4, "MoO2": 26.0,
    # Gamma iron's 111 sits one degree from alpha iron's 110 at 44.7,
    # which is the whole reason for carrying both.
    "Fe_gamma": 43.6, "FeP": 48.1, "Fe2P": 40.3,
    "FeS_mackinawita": 17.6, "NiS_beta": 45.5,
    # Bragg, nothing else: d = 3.36, 3.44, 3.50 A for the 002. Separating
    # a broad 002 into these three is arithmetic, and pinning the three
    # angles is pinning that arithmetic.
    "C_turbostratico_3.36": 26.5, "C_turbostratico_3.44": 25.9,
    "C_turbostratico_3.50": 25.4,
}

#: Pyrite is left out of STRONGEST on purpose. Its 311 at 56.3 deg and its
#: 200 at 33.0 come out within 4% of each other, and which one wins depends
#: on the thermal factors this calculation does not carry. Asserting an
#: order there would be pinning an artefact, so what is pinned instead is
#: that both are present and both are strong.
PYRITE_PAIR = (33.0, 56.3)


def test_the_library_loads():
    entries = load_library()
    assert len(entries) >= 20
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


def test_pyrite_has_both_of_its_strong_reflections():
    """Not which is strongest -- that is sample-dependent -- but that the
    pair the powder card names is there and dominant."""
    lines = reflections(find_phase("FeS2_pirita"), wavelength=CU,
                        two_theta_range=(5.0, 90.0))
    top = max(line.intensity for line in lines)
    strong = [line.two_theta for line in lines if line.intensity > 0.9 * top]
    for expected in PYRITE_PAIR:
        assert any(abs(angle - expected) < 0.6 for angle in strong), strong


def test_gamma_iron_is_resolvable_from_alpha_iron():
    """The pair this library exists to separate: austenite's 111 and
    ferrite's 110 are about one degree apart, so a fit that carries only
    one of them puts the difference into the peak width instead."""
    gamma = max(reflections(find_phase("Fe_gamma"), wavelength=CU,
                            two_theta_range=(5.0, 90.0)),
                key=lambda r: r.intensity)
    alpha = max(reflections(find_phase("Fe_alfa"), wavelength=CU,
                            two_theta_range=(5.0, 90.0)),
                key=lambda r: r.intensity)
    assert 0.8 < abs(alpha.two_theta - gamma.two_theta) < 1.6


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


# -- nanocrystalline patterns: broad peaks, noise, no 3D order ---------


def _pseudo_voigt(x, centre, fwhm, height):
    sigma = fwhm / 2.3548
    gamma = fwhm / 2.0
    gauss = np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + ((x - centre) / gamma) ** 2)
    return height * (0.4 * gauss + 0.6 * lorentz)


def _cvd_pattern(seed=0, noise=0.35, d002=3.44):
    """A CVD carbon on an iron catalyst, the sample this package is for.

    Broad turbostratic 002, nanocrystalline iron, a sloping background and
    real noise. Built self-consistently from Bragg so the 002 and the 004
    come from the same interlayer spacing.
    """
    from ramancarbon.xrd.pattern import Pattern

    wavelength = 1.5406

    def angle(d):
        return 2.0 * math.degrees(math.asin(wavelength / (2.0 * d)))

    rng = np.random.default_rng(seed)
    x = np.arange(10.0, 90.0, 0.02)
    y = _pseudo_voigt(x, angle(d002), 4.0, 11.0)
    y += _pseudo_voigt(x, angle(d002 / 2.0), 5.0, 0.55)
    y += _pseudo_voigt(x, 43.4, 2.5, 1.0)
    for centre, fwhm, height in ((44.67, 0.9, 1.6), (65.02, 1.1, 0.5),
                                 (82.33, 1.3, 0.4)):
        y += _pseudo_voigt(x, centre, fwhm, height)
    y += 3.0 + 4.8 * np.exp(-(x - 10.0) / 12.0)
    y += rng.normal(0.0, noise, x.size)
    return Pattern(two_theta=x, intensity=y, wavelength=wavelength,
                   name="cvd", counts=False)


def test_peak_significance_uses_prominence_not_height():
    """A ripple riding on a broad hump sits at the hump's intensity.

    Scoring it by height hands it the hump's significance, and on a
    nanocrystalline pattern that turns one 002 into a dozen peaks of
    0.02-0.09 degrees FWHM -- every one of them 'significant', every one
    unexplained. Which is exactly what a real CVD pattern did.
    """
    from ramancarbon.xrd.search import find_peaks

    pattern = _cvd_pattern(noise=0.35)
    peaks = find_peaks(pattern)
    assert len(peaks) <= 6, [round(p.two_theta, 2) for p in peaks]
    narrow = [p for p in peaks if p.fwhm and p.fwhm < 0.15]
    assert not narrow, "noise ripples are being reported as peaks"
    broad = max(peaks, key=lambda p: p.height)
    assert broad.fwhm > 2.0 and 24.0 < broad.two_theta < 27.0


def test_the_matching_window_follows_the_peak_width():
    """A fixed 0.12 degree window cannot match a peak three degrees wide.

    Turbostratic carbon at d = 3.44 A puts its 002 at 25.9 degrees where
    graphite's is at 26.5: a shift of 0.6 that the old window could not
    reach, so every reflection under a broad peak read as unexplained.
    """
    from ramancarbon.xrd.search import MATCH_WINDOW, WIDTH_TOLERANCE, find_peaks, match_phase
    from ramancarbon.xrd.reference import library_crystals

    assert WIDTH_TOLERANCE > 0
    pattern = _cvd_pattern(noise=0.35)
    peaks = find_peaks(pattern)
    carbon = next(c for c in library_crystals()
                  if c.name == "C_turbostratico_3.44")
    match = match_phase(peaks, carbon, pattern)
    assert match.matched, "the 002 was not matched at all"
    assert match.window_used > MATCH_WINDOW * 3


def test_turbostratic_carbon_has_no_three_dimensional_reflections():
    """Random rotation between sheets destroys 3D coherence: only 00l and
    hk0 survive. Asking a CVD carbon for graphite's 101 and 112 asks for
    reflections the material cannot produce, which is why graphite scored
    below the acceptance threshold on every one of these patterns."""
    from ramancarbon.xrd.powder import reflections
    from ramancarbon.xrd.reference import library_crystals

    library = {c.name: c for c in library_crystals()}
    turbostratic = library["C_turbostratico_3.44"]
    assert turbostratic.stacking == "turbostratic"
    for reflection in reflections(turbostratic, two_theta_range=(10.0, 90.0)):
        h, k, ell = reflection.hkl
        assert ell == 0 or (h == 0 and k == 0), reflection.hkl

    graphite = library["grafito_2H"]
    assert graphite.stacking == "ordered"
    mixed = [r.hkl for r in reflections(graphite, two_theta_range=(10.0, 90.0))
             if r.hkl[2] != 0 and (r.hkl[0], r.hkl[1]) != (0, 0)]
    assert mixed, "graphite should keep its 3D reflections"


def test_a_cvd_pattern_identifies_its_carbon_and_its_catalyst():
    """The end-to-end case: the three fixes together on a realistic
    pattern. Before them this identified nothing and reported twenty-one
    unexplained peaks."""
    from ramancarbon.xrd.search import find_peaks, identify_phases

    pattern = _cvd_pattern(noise=0.35)
    result = identify_phases(pattern, peaks=find_peaks(pattern))
    names = [m.crystal.name for m in result.accepted]
    assert any(n.startswith("C_turbostratico") for n in names), names
    assert "Fe_alfa" in names, names


def test_one_matched_line_cannot_carry_a_whole_phase():
    """The guard on the 'too weak to detect' excuse.

    Without it a phase matching ONE weak peak declared every other
    reflection undetectable -- the scale fitted to that peak makes them so
    -- and scored perfect coverage. It accepted MoSe2 on a single line in
    a sample with no molybdenum. What separates that from a genuine
    one-line phase is how much of the phase's intensity was seen: 88 % for
    a turbostratic carbon's 002, 30 % for MoSe2's strongest.
    """
    from ramancarbon.xrd.search import find_peaks, identify_phases, match_phase
    from ramancarbon.xrd.reference import library_crystals

    pattern = _cvd_pattern(noise=0.10)
    peaks = find_peaks(pattern)
    library = {c.name: c for c in library_crystals()}

    impostor = match_phase(peaks, library["MoSe2_2H"], pattern)
    if impostor.matched:
        assert impostor.intensity_coverage < 0.5
        assert impostor.score < 0.45

    carbon = match_phase(peaks, library["C_turbostratico_3.44"], pattern)
    assert carbon.intensity_coverage > 0.8

    names = [m.crystal.name for m in identify_phases(pattern, peaks=peaks).accepted]
    assert "MoSe2_2H" not in names


def test_sibling_cells_of_the_same_compound_are_reported_as_a_tie():
    """A 002 three degrees wide cannot choose between d = 3.44 and 3.50.
    Naming the winner alone would turn a coin toss into a measurement."""
    from ramancarbon.xrd.search import find_peaks, identify_phases

    pattern = _cvd_pattern(noise=0.35)
    result = identify_phases(pattern, peaks=find_peaks(pattern))
    if any(m.crystal.name.startswith("C_turbo") for m in result.accepted):
        assert any("empata" in w for w in result.warnings), result.warnings


def test_smoothing_does_not_get_to_pretend_the_data_are_better():
    """Savitzky-Golay removes point-to-point scatter by construction, so
    the noise estimate collapses with it, so the detection threshold
    collapses too. Unguarded, one true peak became a hundred and forty and
    two absent phases were accepted."""
    from ramancarbon.xrd.preprocess import savitzky_golay
    from ramancarbon.xrd.search import find_peaks

    pattern = _cvd_pattern(noise=1.2)
    smoothed = savitzky_golay(pattern, window=21, order=3)

    raw_scatter = float(np.median(np.abs(np.diff(pattern.intensity, n=2))))
    smooth_scatter = float(np.median(np.abs(np.diff(smoothed.intensity, n=2))))
    assert smooth_scatter < raw_scatter / 3.0, "the smoothing did nothing"

    assert smoothed.noise_estimate() == pytest.approx(
        pattern.noise_estimate(), rel=0.05)
    assert len(find_peaks(smoothed)) < 10
    assert "NO refines" in smoothed.metadata["smoothed"]


def _starved_pattern(seed=0, peak_counts=9.0, background=3.2, d002=3.44):
    """A CVD pattern with *counting* statistics, a few counts deep.

    The pattern the user brought in: ten to twenty counts at the top of
    the graphite 002, a sloping background of two or three, and Poisson
    noise on every point. Everything the detector does is a ratio against
    sqrt(N), and at N of three that denominator is itself noisy, which is
    why this case fails in ways the Gaussian-noise pattern above does not.
    """
    from ramancarbon.xrd.pattern import Pattern

    wavelength = 1.5406

    def angle(d):
        return 2.0 * math.degrees(math.asin(wavelength / (2.0 * d)))

    rng = np.random.default_rng(seed)
    x = np.arange(10.0, 90.0, 0.02)
    y = _pseudo_voigt(x, angle(d002), 4.5, peak_counts)
    y += _pseudo_voigt(x, angle(d002 / 2.0), 5.5, 0.05 * peak_counts)
    y += _pseudo_voigt(x, 44.67, 1.0, 0.14 * peak_counts)
    y += _pseudo_voigt(x, 65.02, 1.2, 0.05 * peak_counts)
    y += background + 3.0 * np.exp(-(x - 10.0) / 15.0)
    y = rng.poisson(np.maximum(y, 0.01)).astype(float)
    return Pattern(two_theta=x, intensity=y, wavelength=wavelength,
                   name="starved", counts=True)


def test_the_background_cutoff_stays_outside_the_broadest_reflection():
    """A fixed lam is a fixed cutoff in POINTS, and a point is worth a
    different number of degrees on every diffractometer. At the old fixed
    1e6 the half-power cutoff sat at 3.97 degrees on a 0.02-degree step —
    inside the widest reflection the search will report — so the
    background followed the turbostratic 002 and took a quarter of its
    height away."""
    from ramancarbon.core.baseline import cutoff_for_lambda
    from ramancarbon.xrd.search import (BROAD_SCALES_DEG,
                                        CUTOFF_REFLECTION_WIDTHS,
                                        baseline_lambda_for)

    for step in (0.005, 0.01, 0.02, 0.05):
        angles = np.arange(10.0, 90.0, step)
        pattern = Pattern(angles, np.ones(angles.size), wavelength=1.5406)
        lam = baseline_lambda_for(pattern)
        cutoff_deg = cutoff_for_lambda(lam) * pattern.step
        assert cutoff_deg >= CUTOFF_REFLECTION_WIDTHS * BROAD_SCALES_DEG[-1] * 0.99, (
            f"step {step}: cutoff {cutoff_deg:.2f} deg is inside the "
            "broadest reportable reflection")

    # And it is the STEP that sets it, not the number of points: the same
    # angular range sampled twice as finely needs a stiffness sixteen
    # times larger to keep the cutoff at the same number of degrees.
    coarse = Pattern(np.arange(10.0, 90.0, 0.02), np.ones(4000), wavelength=1.5406)
    fine = Pattern(np.arange(10.0, 90.0, 0.01), np.ones(8000), wavelength=1.5406)
    assert baseline_lambda_for(fine) == pytest.approx(
        16.0 * baseline_lambda_for(coarse), rel=0.02)


def test_a_photon_starved_pattern_still_finds_its_broad_002():
    """The whole point. With the old fixed background, the point-wise
    sigma read off one sample and the doublet stripped before the broad
    pass, this pattern returned ZERO peaks and identified nothing, while
    the 002 hump is plainly visible on screen."""
    from ramancarbon.xrd.search import find_peaks, identify_phases

    pattern = _starved_pattern()
    peaks = find_peaks(pattern)
    assert peaks, "no peak found on a pattern whose 002 is visible by eye"

    broad = [p for p in peaks if (p.fwhm or 0.0) > 1.0]
    assert broad, f"the 002 was not found as a broad peak: {peaks}"
    assert min(abs(p.d - 3.44) for p in broad) < 0.06, (
        f"the interlayer spacing is wrong: {[round(p.d, 3) for p in broad]}")

    result = identify_phases(pattern, peaks=peaks)
    assert any(m.crystal.name.startswith("C_turbostratico")
               for m in result.accepted), [m.crystal.name for m in result.accepted]


def test_the_broad_pass_sees_the_pattern_as_measured():
    """Rachinger stripping corrects a splitting of a few hundredths of a
    degree. It cannot help a reflection several degrees wide, and it is
    not free: it subtracts a shifted copy of the data, which on a
    photon-starved pattern costs the broad 002 about a fifth of its
    prominence and correlates its neighbours."""
    from ramancarbon.xrd.preprocess import strip_kalpha2
    from ramancarbon.xrd.search import find_peaks

    pattern = _starved_pattern()
    assert pattern.has_doublet

    as_measured = find_peaks(pattern)
    pre_stripped = find_peaks(strip_kalpha2(pattern), strip_doublet=False)

    def broad_significance(peaks):
        wide = [p for p in peaks if (p.fwhm or 0.0) > 1.0]
        return max((p.significance for p in wide), default=0.0)

    assert broad_significance(as_measured) > broad_significance(pre_stripped)

    # And the narrow pass still gets the stripped pattern, or every
    # reflection above about 40 degrees is found twice.
    sharp = simulate(find_phase("Fe_alfa"), two_theta_range=(30.0, 90.0),
                     background=300.0, counts_at_max=40000.0, seed=4)
    assert len(find_peaks(sharp)) <= 2 * len(find_peaks(sharp, strip_doublet=True))


def test_the_noise_of_a_broad_feature_is_measured_over_its_width():
    """sqrt(N) of ONE sample is a lottery when N is three: the same
    reflection scores 12 or 25 depending on which Poisson draw landed on
    its maximum. Averaging the variance over the feature's own width
    keeps the point-wise philosophy — the noise on a 10 000-count peak is
    still a hundred — while estimating it from the points the statistic
    actually used."""
    from ramancarbon.xrd.search import _window_noise

    rng = np.random.default_rng(3)
    counts = rng.poisson(np.full(4000, 4.0)).astype(float)
    point = np.maximum(np.sqrt(counts), 1e-9)
    windowed = _window_noise(point, 125.0)

    assert windowed.std() < point.std() / 3.0, "the window did not steady it"
    assert windowed.mean() == pytest.approx(2.0, rel=0.1), "and it is still sqrt(N)"

    # Point-wise, not global: a bright peak still carries its own noise.
    counts[2000:2100] = 10000.0
    windowed = _window_noise(np.sqrt(counts), 51.0)
    assert windowed[2050] > 50.0 * windowed[500]


def test_a_reflection_predicted_at_the_threshold_cannot_veto_a_phase():
    """A reflection predicted at exactly the detection threshold is found
    about half the time, so its absence decides nothing — and the
    prediction is a known over-estimate, because it scales a clean height
    by a calculated intensity and ignores what the detector loses to a
    neighbour raising the local minima. On the CVD pattern the
    turbostratic 004 is predicted at 23 and observed at 14, and the
    carbon was being rejected for the absence of a reflection this same
    code could not have found."""
    from ramancarbon.xrd.search import (DETECTION_MARGIN, find_peaks,
                                        identify_phases)

    assert DETECTION_MARGIN > 1.0

    pattern = _cvd_pattern(noise=0.35)
    result = identify_phases(pattern, peaks=find_peaks(pattern))
    names = [m.crystal.name for m in result.accepted]
    assert any(n.startswith("C_turbostratico") for n in names), names

    # But it is not a way in: a phase still has to account for most of
    # its own calculated intensity, which is what stopped MoSe2 being
    # accepted in a sample with no molybdenum.
    carbon = next(m for m in result.accepted
                  if m.crystal.name.startswith("C_turbostratico"))
    assert carbon.intensity_coverage > 0.5


def test_a_near_miss_is_reported_with_its_number():
    """Silence is the least useful answer a peak search can give. If the
    best candidate came close, say so with its angle, its width and its
    significance, and hand the decision back — rather than quietly
    lowering a threshold that is calibrated against pure noise."""
    from ramancarbon.xrd.search import (find_peaks, identify_phases,
                                        strongest_below_threshold)

    pattern = _starved_pattern(seed=11, peak_counts=3.0)
    peaks = find_peaks(pattern)
    if peaks:
        pytest.skip("this pattern is detectable; the warning is for when it is not")

    near = strongest_below_threshold(pattern)
    assert near is not None
    angle, width, significance = near
    assert 20.0 < angle < 32.0, angle
    assert width > 0.0 and significance > 0.0

    result = identify_phases(pattern, peaks=peaks)
    assert result.warnings
    message = result.warnings[0]
    assert f"{angle:.2f}" in message
    assert "umbral" in message


def test_the_threshold_still_holds_on_pure_noise_with_the_broad_pass():
    """The detection threshold was recalibrated after the broad pass and
    the new background rule, NOT lowered to make the CVD pattern work:
    twenty pure-Poisson patterns at two background levels still give well
    under one false peak each at 18."""
    angles = np.arange(10.0, 90.0, 0.02)
    for level in (40.0, 5.0):
        found = []
        for seed in range(8):
            rng = np.random.default_rng(900 + seed)
            background = level * (0.6 + 2.0 * np.exp(-(angles - 10.0) / 20.0))
            counts = rng.poisson(background).astype(float)
            pattern = Pattern(angles, counts, wavelength=1.5406, counts=True)
            found.append(len(find_peaks(pattern)))
        assert np.mean(found) < 1.0, (level, found)
