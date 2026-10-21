"""Rietveld refinement: round trips, constraints and honest limits."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.examples.demo_data import make_xrd_demo
from ramancarbon.xrd.powder import Profile, reflections, simulate
from ramancarbon.xrd.reference import find_phase
from ramancarbon.xrd.report import analyse_pattern, theoretical_pattern
from ramancarbon.xrd.rietveld import (
    GOF_LIMIT,
    SITE_MIN_SCATTERING,
    SITE_SHIFT_LIMIT,
    STAGES,
    PhaseModel,
    RefinementError,
    _site_scattering_shares,
    _sites_worth_refining,
    auto_refine,
    build_parameters,
    calculate_pattern,
    chebyshev_background,
    free_kinds,
    r_factors,
    refine,
)
from ramancarbon.xrd.structure import SITE_SYMMETRY_TOLERANCE, StructureError


@pytest.fixture(scope="module")
def two_phase():
    """A pattern whose true answer is known exactly."""
    return simulate(
        [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")],
        two_theta_range=(15.0, 72.0),
        step=0.025,
        scales=[1.0, 0.5],
        profile=Profile(u=0.010, v=-0.003, w=0.008, eta0=0.60),
        background=300.0,
        counts_at_max=12000.0,
        seed=3,
        name="dos_fases",
    )


@pytest.fixture(scope="module")
def refined(two_phase):
    return auto_refine(
        two_phase, [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")]
    )


# -- the round trip ---------------------------------------------------


def test_the_refinement_recovers_the_pattern_it_was_given(refined):
    assert refined.converged
    assert refined.gof == pytest.approx(1.0, abs=0.15)
    assert refined.r_wp < 0.08


def test_the_refinement_recovers_the_lattice_parameters(refined):
    truth = {
        "FeSe_tetragonal": (3.7734, 5.5258),
        "grafito_2H": (2.4640, 6.7110),
    }
    for phase in refined.phases:
        lattice = phase.current_crystal().lattice
        a, c = truth[phase.crystal.name]
        assert lattice.a == pytest.approx(a, rel=2e-4)
        assert lattice.c == pytest.approx(c, rel=2e-4)


def test_the_refinement_recovers_the_profile(refined):
    for phase in refined.phases:
        assert phase.profile.w == pytest.approx(0.008, abs=0.003)
        assert phase.profile.eta0 == pytest.approx(0.60, abs=0.15)


def test_symmetry_ties_a_and_b_in_a_hexagonal_cell(refined):
    """Refining them separately returned 2.46359 and 2.46451 — two numbers
    differing by four standard errors that symmetry says are one."""
    for phase in refined.phases:
        lattice = phase.current_crystal().lattice
        assert lattice.a == lattice.b
    names = [p.name for p in refined.parameters if p.kind == "lattice"]
    assert not any(name.startswith("b[") for name in names)


def test_weight_fractions_sum_to_one(refined):
    fractions = refined.weight_fractions()
    assert all(value is not None for value in fractions.values())
    assert sum(fractions.values()) == pytest.approx(1.0)


# -- the machinery ----------------------------------------------------


def test_a_chebyshev_background_of_one_term_is_a_constant():
    angles = np.linspace(10.0, 80.0, 100)
    assert np.allclose(chebyshev_background(angles, [42.0]), 42.0)


def test_r_factors_are_zero_for_a_perfect_fit():
    observed = np.array([100.0, 200.0, 300.0])
    weights = 1.0 / observed
    r_p, r_wp, _, gof = r_factors(observed, observed.copy(), weights, 1)
    assert r_p == 0.0 and r_wp == 0.0 and gof == 0.0


def test_refining_nothing_is_an_error(two_phase):
    phases = [PhaseModel(crystal=find_phase("grafito_2H"))]
    parameters = build_parameters(phases)
    with pytest.raises(RefinementError):
        refine(two_phase, phases, parameters=parameters)


def test_parameters_can_be_freed_one_group_at_a_time(two_phase):
    phases = [PhaseModel(crystal=find_phase("grafito_2H"))]
    parameters = build_parameters(phases)
    free_kinds(parameters, ("scale",))
    assert sum(1 for p in parameters if p.free) == 1
    result = refine(two_phase, phases, parameters=parameters)
    assert result.free_parameters == 1


def test_manual_and_automatic_reach_the_same_place(two_phase):
    """The staged protocol is a route to the minimum, not a different one.

    The comparison only means anything when both sides free the SAME
    parameters. An earlier version freed seven groups by hand and compared
    against the nine the staged schedule ends with, then asserted they
    agreed to 0.01 in Rwp — which they did on one version of SciPy and not
    on the next, because a fit with fewer degrees of freedom simply cannot
    reach as low an Rwp. That is arithmetic, not a bug in either path.
    """
    crystals = [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")]
    # Stopped before the coordinate stage, so that both sides free the
    # same parameters. Texture is excluded on both sides too, because
    # auto_refine skips it when no axis is given. Coordinates are
    # excluded for a stronger reason: the two routes genuinely do NOT
    # reach the same place once those are in, which is the staged
    # protocol's whole thesis and is measured in the test below rather
    # than being a defect in either path.
    staged = auto_refine(two_phase, crystals, stages=STAGES[:-1])

    every_group = tuple(
        kind for _, kinds in STAGES for kind in kinds
        if kind not in ("preferred", "site")
    )
    phases = [PhaseModel(crystal=c) for c in crystals]
    parameters = build_parameters(phases)
    for parameter in parameters:
        if parameter.name == "fondo_c0":
            parameter.value = float(np.percentile(two_phase.intensity, 5))
    free_kinds(parameters, every_group)
    manual = refine(two_phase, phases, parameters=parameters)

    assert manual.free_parameters == staged.free_parameters
    assert manual.r_wp == pytest.approx(staged.r_wp, abs=0.005)


def test_freeing_coordinates_from_scratch_lands_somewhere_worse(two_phase):
    """Why the coordinate stage goes last, with the same parameters freed.

    This is the staged protocol's claim made sharp. Up to the widths and
    the shape, the two routes agree to one part in ten thousand -- the
    test above asserts exactly that. Add the atomic coordinates to the
    set and freeing them all at once from the starting structure reaches
    **Rwp 10.55 %** against the staged **4.87 %**, on the same pattern,
    with the same twenty-five free parameters. The all-at-once fit is not
    short of degrees of freedom; it is in a different and worse basin,
    and coordinates are the parameters that dig one, because there are
    enough of them to absorb anything the earlier stages have not yet
    settled.
    """
    crystals = [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")]
    staged = auto_refine(two_phase, crystals)

    every_group = tuple(
        kind for _, kinds in STAGES for kind in kinds if kind != "preferred"
    )
    phases = [PhaseModel(crystal=c) for c in crystals]
    parameters = build_parameters(phases)
    for parameter in parameters:
        if parameter.name == "fondo_c0":
            parameter.value = float(np.percentile(two_phase.intensity, 5))
    free_kinds(parameters, every_group)
    at_once = refine(two_phase, phases, parameters=parameters)

    assert at_once.free_parameters == staged.free_parameters
    assert at_once.r_wp > 1.5 * staged.r_wp

    # And the staged route's answer for the coordinate is the right one:
    # this pattern was built at the published z, so the refined shift
    # should come back at zero rather than at something that merely fits.
    assert staged.phases[0].site_shifts[0] == pytest.approx(0.0, abs=0.002)


def test_a_partial_parameter_set_cannot_beat_the_full_one(two_phase):
    """Sanity on the comparison above: fewer free parameters, higher Rwp."""
    crystals = [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")]
    phases = [PhaseModel(crystal=c) for c in crystals]
    parameters = build_parameters(phases)
    for parameter in parameters:
        if parameter.name == "fondo_c0":
            parameter.value = float(np.percentile(two_phase.intensity, 5))
    free_kinds(parameters, ("scale", "background"))
    partial = refine(two_phase, phases, parameters=parameters)
    assert partial.r_wp >= auto_refine(two_phase, crystals).r_wp


def test_uncertainties_are_reported_for_free_parameters(refined):
    free = [p for p in refined.parameters if p.free]
    assert free
    assert all(p.error is not None and p.error >= 0.0 for p in free)


# -- honesty ----------------------------------------------------------


def test_a_missing_phase_shows_up_as_a_bad_fit(two_phase):
    """Rietveld does not discover phases; leaving one out must not look
    like a good fit that happens to be missing something."""
    partial = auto_refine(two_phase, [find_phase("grafito_2H")])
    assert partial.gof > GOF_LIMIT
    assert any("no describe los datos" in w for w in partial.warnings)


def test_the_weight_fraction_caveat_is_always_stated(refined):
    assert any("CRISTALINA E IDENTIFICADA" in w for w in refined.warnings)


def test_a_resolution_limited_phase_refuses_to_give_a_size(refined):
    sizes = refined.crystallite_sizes(instrument_fwhm=0.06)
    assert all(value is None for value in sizes.values())


def test_a_broadened_phase_gives_a_size_back():
    """A 15 nm crystallite broadens its peaks well past the instrument, and
    the size must come back close to the one that was put in."""
    crystal = find_phase("grafito_2H")
    pattern = simulate(
        crystal,
        two_theta_range=(15.0, 75.0),
        profile=Profile(u=0.02, v=0.0, w=0.35, eta0=0.8),
        background=200.0,
        counts_at_max=9000.0,
        seed=11,
    )
    result = auto_refine(pattern, [crystal])
    size = result.crystallite_sizes(instrument_fwhm=0.06)[crystal.name]
    assert size is not None and 10.0 < size < 40.0


def test_texture_is_refined_when_an_axis_is_given():
    pattern = make_xrd_demo("MoS2_texturado", seed=2)
    result = auto_refine(pattern, [find_phase("MoS2_2H")], preferred_axis=(0, 0, 1))
    phase = result.phases[0]
    assert phase.preferred_r < 0.9, "la textura del demo es fuerte y va hacia r<1"
    assert result.gof < 3.0


def test_without_a_texture_axis_no_texture_parameter_is_freed():
    pattern = make_xrd_demo("MoS2_texturado", seed=2)
    result = auto_refine(pattern, [find_phase("MoS2_2H")])
    assert all(not p.free for p in result.parameters if p.kind == "preferred")


# -- the orchestrator -------------------------------------------------


def test_analyse_pattern_identifies_then_refines():
    pattern = make_xrd_demo("CNT_FeSe", seed=6)
    result = analyse_pattern(pattern)
    assert result.refinement is not None
    assert result.calculated is not None
    assert result.difference is not None
    assert len(result.difference) == pattern.n
    found = {m.crystal.name for m in result.search.accepted}
    assert set(pattern.metadata["phases"]) <= found


def test_the_report_puts_the_difference_curve_before_the_r_factors():
    pattern = make_xrd_demo("CNT_FeSe", seed=6)
    text = analyse_pattern(pattern).report()
    assert "Mira PRIMERO la curva diferencia" in text
    # "Rwp" also appears in the prose above; compare against the R-factor
    # block itself.
    assert text.index("Diferencia:") < text.index("Rwp  =")


def test_the_report_row_carries_the_headline_numbers():
    row = analyse_pattern(make_xrd_demo("CNT_FeSe", seed=6)).to_dict()
    assert row["fases"]
    assert row["GOF"] > 0.0
    assert any(key.startswith("w_") for key in row)


def test_refusing_to_refine_without_phases_says_why():
    angles = np.arange(10.0, 80.0, 0.02)
    flat = np.random.default_rng(1).poisson(300.0, angles.size).astype(float)
    from ramancarbon.xrd.pattern import Pattern

    result = analyse_pattern(Pattern(angles, flat, counts=True, kalpha2_ratio=0.0))
    assert result.refinement is None
    assert any("no lo descubre" in w for w in result.warnings)


def test_theoretical_pattern_returns_a_comparable_curve():
    pattern = make_xrd_demo("CNT_FeSe", seed=6)
    calculated, difference, refinement = theoretical_pattern(
        pattern, [find_phase("FeSe_tetragonal"), find_phase("grafito_2H"),
                  find_phase("Se_trigonal")]
    )
    assert calculated.shape == pattern.intensity.shape
    assert np.allclose(difference, pattern.intensity - calculated)
    assert refinement.r_wp < 0.1


# -- what the refinement reports about itself --------------------------


def test_the_refinement_counts_its_own_iterations():
    """A refinement that returns instantly is either converged or never
    started, and from outside the two look identical. The count is what
    separates them."""
    from ramancarbon.examples.demo_data import xrd_demo_spectra
    from ramancarbon.xrd.report import analyse_pattern

    result = analyse_pattern(xrd_demo_spectra(seed=3)[0])
    refinement = result.refinement
    assert refinement is not None
    assert refinement.n_evaluations > 10
    assert "evaluaciones" in refinement.summary()


def test_reduced_chi_squared_is_the_goodness_of_fit_squared():
    """They are the same statement; both are quoted because different
    communities read different ones."""
    from ramancarbon.examples.demo_data import xrd_demo_spectra
    from ramancarbon.xrd.report import analyse_pattern

    refinement = analyse_pattern(xrd_demo_spectra(seed=3)[0]).refinement
    assert refinement.chi_squared == pytest.approx(refinement.gof ** 2)
    assert "χ² reducida" in refinement.summary()


def test_progress_is_reported_from_inside_the_fit_not_only_between_stages():
    """Between stages is too coarse: a single stage of a many-phase
    refinement can take most of the time, and the window goes quiet for
    all of it."""
    from ramancarbon.examples.demo_data import xrd_demo_spectra
    from ramancarbon.xrd.report import analyse_pattern
    from ramancarbon.xrd.rietveld import PhaseModel, auto_refine

    pattern = xrd_demo_spectra(seed=3)[0]
    search = analyse_pattern(pattern, refine=False).search
    models = [PhaseModel(crystal=m.crystal) for m in search.accepted]

    lines: list[str] = []
    auto_refine(pattern, models, callback=lines.append)
    assert len(lines) > len(("escala", "cero", "celda"))
    assert any("iteración" in line for line in lines)
    # Every line names the stage it came from.
    assert all(line.startswith("[") for line in lines)


def test_a_phase_can_be_put_into_the_model_and_taken_out_again():
    """The case every refinement meets: a systematic bump in the
    difference curve that belongs to a phase not in the model. Until now
    the only way to add one was to re-run the identification and hope."""
    from ramancarbon.examples.demo_data import xrd_demo_spectra
    from ramancarbon.gui.xrd_state import XRDSession

    session = XRDSession()
    session.add_pattern(xrd_demo_spectra(seed=3)[0])
    session.current = 0
    session.analyse_current(refine_after=False)
    session.prepare_manual()
    before = session.model_phase_names()
    assert before

    assert session.add_phase_to_model("Fe_alfa")
    assert "Fe_alfa" in session.model_phase_names()
    assert not session.add_phase_to_model("Fe_alfa")       # already there
    assert not session.add_phase_to_model("no_existe")

    assert session.remove_phase_from_model("Fe_alfa")
    assert session.model_phase_names() == before

    # The last phase cannot go: a refinement with no phases fits the
    # background and calls it a structure.
    for name in before[1:]:
        session.remove_phase_from_model(name)
    assert not session.remove_phase_from_model(session.model_phase_names()[0])


def test_the_metrics_table_carries_the_numbers_and_the_fractions():
    from ramancarbon.examples.demo_data import xrd_demo_spectra
    from ramancarbon.gui.xrd_state import XRDSession

    session = XRDSession()
    session.add_pattern(xrd_demo_spectra(seed=3)[0])
    session.current = 0
    session.analyse_current(refine_after=True)
    rows = dict(session.refinement_metrics())
    for key in ("Rp", "Rwp", "Rexp", "GOF", "χ² reducida", "evaluaciones"):
        assert key in rows, rows
    fractions = [v for k, v in rows.items() if "% peso" in v]
    assert fractions, rows


# -- atomic coordinates: the relative intensities inside a phase -------


class TestSiteFreedoms:
    """Which coordinates symmetry leaves free, read off the operations."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            # Se on 2c of P4/nmm: z free, and the one coordinate the
            # bundled CIF's own note says the intensities turn on.
            ("FeSe_tetragonal", 1),
            # NiAs type: both sites are fully fixed special positions.
            ("FeSe_hexagonal", 0),
            # Pnma cementite: Fe1 and C1 on the mirror (x, z), Fe2 general.
            ("Fe3C_cementita", 7),
            # Body-centred cubic iron: one atom at the origin.
            ("Fe_alfa", 0),
            # Graphite is fixed by a and c alone, which is why no Rietveld
            # of a graphitic carbon has ever had a coordinate to refine.
            ("grafito_2H", 0),
        ],
    )
    def test_the_count_matches_the_wyckoff_position(self, name, expected):
        assert find_phase(name).free_coordinates == expected

    def test_the_free_direction_of_tetragonal_fese_is_z(self):
        basis = find_phase("FeSe_tetragonal").site_freedoms()[1]
        assert basis.shape == (3, 1)
        assert np.allclose(np.abs(basis[:, 0]), [0.0, 0.0, 1.0])

    def test_a_tolerance_finer_than_the_cif_rounding_invents_a_freedom(self):
        """The reason :data:`SITE_SYMMETRY_TOLERANCE` is loose.

        A CIF writes 1/3 as 0.3333, so the three-fold at (1/3, 2/3, 1/4)
        misses the site by 1e-4. Demanding 1e-6 drops it from the
        stabiliser and graphite's second carbon comes back free along the
        basal direction that very three-fold forbids.
        """
        crystal = find_phase("grafito_2H")
        site = crystal.sites[1]
        fract = np.asarray(site.fract, dtype=float)
        strict = [
            operation for operation in crystal.operations
            if np.all(np.abs(
                np.mod(operation.R @ fract + operation.t - fract + 0.5, 1.0)
                - 0.5) < 1e-6)
        ]
        loose = [
            operation for operation in crystal.operations
            if np.all(np.abs(
                np.mod(operation.R @ fract + operation.t - fract + 0.5, 1.0)
                - 0.5) < SITE_SYMMETRY_TOLERANCE)
        ]
        assert len(strict) < len(loose)
        # And the structure really is at 1/3, not at 0.3333: with exact
        # thirds the strict test keeps the whole stabiliser.
        exact = np.array([1.0 / 3.0, 2.0 / 3.0, 0.25])
        recovered = [
            operation for operation in crystal.operations
            if np.all(np.abs(
                np.mod(operation.R @ exact + operation.t - exact + 0.5, 1.0)
                - 0.5) < 1e-6)
        ]
        assert len(recovered) == len(loose)

    def test_displacing_keeps_the_atom_count_over_the_whole_bound(self):
        """A site that merges with its own images loses half its scattering."""
        for name in ("FeSe_tetragonal", "Fe3C_cementita"):
            crystal = find_phase(name)
            total = crystal.free_coordinates
            reference = crystal.atoms_per_cell
            for position in range(total):
                for sign in (-1.0, 1.0):
                    shifts = [0.0] * total
                    shifts[position] = sign * SITE_SHIFT_LIMIT
                    assert crystal.displaced(shifts).atoms_per_cell == reference

    def test_a_shift_of_the_wrong_length_is_refused(self):
        with pytest.raises(StructureError):
            find_phase("FeSe_tetragonal").displaced([0.01, 0.02])

    def test_a_fixed_structure_takes_an_empty_shift_and_nothing_else(self):
        crystal = find_phase("Fe_alfa")
        assert crystal.displaced([]).sites == crystal.sites
        with pytest.raises(StructureError):
            crystal.displaced([0.01])


class TestCoordinatesMoveTheRelativeIntensities:
    """The point of the whole exercise, stated as a measurement."""

    def test_the_selenium_z_rewrites_the_001_against_the_101(self):
        crystal = find_phase("FeSe_tetragonal")
        ratios = []
        for shift in (-SITE_SHIFT_LIMIT, 0.0, SITE_SHIFT_LIMIT):
            lines = {
                tuple(sorted(map(abs, r.hkl), reverse=True)): r.intensity
                for r in reflections(crystal.displaced([shift]),
                                     wavelength=1.540593,
                                     two_theta_range=(10.0, 90.0))
            }
            ratios.append(lines[(1, 0, 0)] / lines[(1, 1, 0)])
        # Monotonic and by more than a factor of five end to end: this is
        # not a correction, it is the dominant control on the pattern.
        assert ratios[0] > ratios[1] > ratios[2]
        assert ratios[0] / ratios[2] > 5.0

    def test_scale_and_width_cannot_do_it(self):
        """Why a coordinate parameter was needed and no other would do.

        The scale multiplies every reflection of a phase by the same
        number and the width moves intensity between a peak's top and its
        flanks; neither can change the ratio of two integrated
        intensities. Only the structure factor can, and the structure
        factor is where the atoms are.
        """
        crystal = find_phase("FeSe_tetragonal")
        def ratio(c):
            lines = {
                tuple(sorted(map(abs, r.hkl), reverse=True)): r.intensity
                for r in reflections(c, wavelength=1.540593,
                                     two_theta_range=(10.0, 90.0))
            }
            return lines[(1, 0, 0)] / lines[(1, 1, 0)]
        # A different cell, a different scale: the ratio barely moves.
        stretched = crystal.with_lattice(crystal.lattice.scaled((1.0, 1.0, 1.01)))
        assert ratio(stretched) == pytest.approx(ratio(crystal), rel=0.25)
        # The coordinate moves it by a factor of several.
        assert ratio(crystal.displaced([SITE_SHIFT_LIMIT])) < 0.5 * ratio(crystal)


class TestTheCoordinateStage:
    """It runs last, and it does not run for everything."""

    def test_the_stage_exists_and_is_the_last_one(self):
        assert STAGES[-1][0] == "coordenadas"
        assert "site" in STAGES[-1][1]
        # And in no earlier one: coordinates are the parameters that will
        # absorb anything, so nothing else may be settling while they move.
        assert not any("site" in kinds for _, kinds in STAGES[:-1])

    def test_a_phase_with_fixed_symmetry_gets_no_site_parameter(self):
        parameters = build_parameters([PhaseModel(crystal=find_phase("Fe_alfa"))])
        assert not [p for p in parameters if p.kind == "site"]

    def test_the_parameters_are_named_the_way_the_cif_reads(self):
        parameters = build_parameters(
            [PhaseModel(crystal=find_phase("FeSe_tetragonal"))])
        sites = [p.name for p in parameters if p.kind == "site"]
        assert sites == ["Se1.z[FeSe_tetragonal]"]

    def test_the_cache_key_separates_two_sets_of_coordinates(self):
        """A reflection list computed at other coordinates is the wrong list.

        Unlike the March-Dollase r, which is a multiplicative correction
        on a list that does not depend on it, a site shift acts inside
        |F|. If it were left out of the key the second pattern would come
        back identical to the first and the Jacobian column would be zero.
        """
        crystal = find_phase("FeSe_tetragonal")
        angles = np.linspace(14.0, 60.0, 600)
        cache: dict = {}
        plain = calculate_pattern(
            angles, [PhaseModel(crystal=crystal)], wavelength=1.540593,
            reflection_cache=cache)
        moved = calculate_pattern(
            angles,
            [PhaseModel(crystal=crystal, site_shifts=(SITE_SHIFT_LIMIT,))],
            wavelength=1.540593, reflection_cache=cache)
        assert not np.allclose(plain, moved)

    def test_a_light_atom_beside_a_heavy_one_is_held_fixed(self):
        """Cementite's carbon carries 1.7 % of the scattering power."""
        shares = _site_scattering_shares(find_phase("Fe3C_cementita"))
        carbon = shares[2]
        assert carbon < SITE_MIN_SCATTERING
        assert carbon == pytest.approx(0.017, abs=0.005)
        # Its two coordinates are the two the stage must leave alone, so
        # of cementite's seven only five may be freed.
        assert find_phase("Fe3C_cementita").free_coordinates == 7

    def test_a_phase_too_faint_to_see_keeps_its_published_coordinates(self):
        """The gate is the share of the intensity the model explains."""
        models = [
            PhaseModel(crystal=find_phase("FeSe_tetragonal"), scale=1.0),
            PhaseModel(crystal=find_phase("Fe3C_cementita"), scale=1e-9),
        ]
        class _Result:
            phases = models
            @staticmethod
            def phase_contributions():
                return {"FeSe_tetragonal": 0.999, "Fe3C_cementita": 0.001}
        carried, notes = _sites_worth_refining(_Result(), models)
        assert 0 in carried and 1 not in carried
        assert any("0.1 %" in note and "Fe3C" in note for note in notes)

    def test_nothing_is_skipped_in_silence(self):
        """Every reason a coordinate was not refined is reported."""
        models = [
            PhaseModel(crystal=find_phase("Fe_alfa")),
            PhaseModel(crystal=find_phase("Fe3C_cementita")),
        ]
        class _Result:
            phases = models
            @staticmethod
            def phase_contributions():
                return {"Fe_alfa": 0.5, "Fe3C_cementita": 0.5}
        _, notes = _sites_worth_refining(_Result(), models)
        joined = " ".join(notes)
        assert "Fe_alfa" in joined and "simetría las fija" in joined
        assert "C1" in joined and "ligeros" in joined

    def test_the_refinement_recovers_a_coordinate_it_was_given(self):
        """The round trip: build a pattern at a moved z, get the z back."""
        crystal = find_phase("FeSe_tetragonal")
        truth = 0.03
        pattern = simulate(
            [crystal.displaced([truth])],
            two_theta_range=(14.0, 75.0), step=0.02,
            profile=Profile(u=0.004, v=-0.001, w=0.010, eta0=0.5),
            background=50.0, counts_at_max=200000.0, seed=11,
            name="z_movida",
        )
        result = auto_refine(pattern, [PhaseModel(crystal=crystal)])
        assert result.phases[0].site_shifts
        assert result.phases[0].site_shifts[0] == pytest.approx(truth, abs=0.006)
