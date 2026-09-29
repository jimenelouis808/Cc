"""Three things the XPS section could not say or could not be told.

**Whether the asymmetric shape is actually asymmetric.** The C 1s of
graphite is asymmetric because the conduction electrons screen the core
hole, and the database says so. But the singularity index α is a fitted
parameter with a lower bound of zero, and a narrow window over a Shirley
background parks it there -- the shape goes back to symmetric, its area
comes out short by whatever the tail was carrying, and the missing area
reappears as an extra sp3 component near 285.5 eV. The database calls
that the most common error in carbon XPS, and nothing reported it.

**Being able to do something about it.** α could not be set or held, so
the advice "fix it at the value your group uses" had nowhere to go.

**Where the composition came from.** It was always the fitted regions;
the survey route existed in the library and nothing called it. The two
disagree by 10-30 % and neither the number nor the window said which it
was.
"""

from __future__ import annotations

import pytest

from ramancarbon.examples.demo_data import make_xps_demo
from ramancarbon.gui.xps_state import XPSSession
from ramancarbon.xps.checks import MIN_MEANINGFUL_ASYMMETRY, bound_checks
from ramancarbon.xps.fitting import fit_region
from ramancarbon.xps.presets import state_model

NARROW = dict(states=["C-C sp2", "C-C sp3", "C-N", "C-O", "C-O-C"],
              window=(283.0, 290.0))


@pytest.fixture(scope="module")
def spectra():
    return {s.name.split("_")[-1]: s for s in make_xps_demo(seed=1)}


def _alpha(component) -> float:
    names = list(component.extra_names or ())
    return float(component.extra[names.index("asymmetry")])


def _session(spectra) -> XPSSession:
    session = XPSSession()
    session.spectra.extend(spectra.values())
    session.shifted = list(session.spectra)
    return session


class TestTheDatabaseSaysAsymmetricAndTheModelObeys:
    def test_the_sp2_state_is_declared_asymmetric(self):
        from ramancarbon.xps.tables import load_xps_database

        state = next(s for s in load_xps_database().states_for("C 1s")
                     if s.key == "C-C sp2")
        assert state.asymmetric is True

    def test_it_gets_a_doniach_sunjic_profile(self, spectra):
        model = state_model(spectra["C1s"], "C 1s", **NARROW)
        sp2 = next(c for c in model.components if c.state == "C-C sp2")
        assert sp2.profile.startswith("ds")
        assert "asymmetry" in sp2.extra_names


class TestAnAlphaOfZeroIsReported:
    def test_a_narrow_window_drives_it_to_the_bound(self, spectra):
        # Not hypothetical: this is the configuration the fault appeared
        # in, and it is an ordinary thing to ask for.
        result = fit_region(spectra["C1s"],
                            state_model(spectra["C1s"], "C 1s", **NARROW))
        sp2 = next(c for c in result.components if c.state == "C-C sp2")
        assert _alpha(sp2) <= MIN_MEANINGFUL_ASYMMETRY

    def test_and_the_checks_now_say_so(self, spectra):
        result = fit_region(spectra["C1s"],
                            state_model(spectra["C1s"], "C 1s", **NARROW))
        said = [text for _, text in bound_checks(result, "C 1s")]
        assert any("ASIMÉTRICA" in text and "α" in text for text in said), said

    def test_a_sound_fit_raises_no_such_note(self, spectra):
        # The default model over the whole region fits it properly, so
        # the check must stay quiet or it is noise.
        result = fit_region(spectra["C1s"],
                            state_model(spectra["C1s"], "C 1s"))
        sp2 = next(c for c in result.components if c.state == "C-C sp2")
        assert _alpha(sp2) > MIN_MEANINGFUL_ASYMMETRY
        said = [text for _, text in bound_checks(result, "C 1s")]
        assert not [t for t in said if "ASIMÉTRICA" in t]

    def test_a_symmetric_component_is_not_asked_about(self, spectra):
        result = fit_region(spectra["C1s"],
                            state_model(spectra["C1s"], "C 1s", **NARROW))
        symmetric = [c for c in result.components
                     if "asymmetry" not in (c.extra_names or ())]
        assert symmetric
        said = " ".join(text for _, text in bound_checks(result, "C 1s"))
        for component in symmetric:
            assert f"«{component.label}» se ajustó con forma" not in said


class TestAlphaCanBeSetAndHeld:
    def test_setting_it_moves_the_fitted_value(self, spectra):
        session = _session(spectra)
        session.set_window("C 1s", (283.0, 290.0))
        session.choice_for("C 1s").states = list(NARROW["states"])
        first = session.fit("C 1s")
        sp2 = next(c for c in first.components if c.state == "C-C sp2")
        assert _alpha(sp2) <= MIN_MEANINGFUL_ASYMMETRY

        session.set_component("C 1s", sp2.label, asymmetry=0.10,
                              fixed=("asymmetry",))
        second = session.fit("C 1s")
        held = next(c for c in second.components if c.state == "C-C sp2")
        assert _alpha(held) == pytest.approx(0.10)
        assert "asymmetry" in held.fixed
        assert not [t for _, t in bound_checks(second, "C 1s")
                    if "ASIMÉTRICA" in t]

    def test_a_symmetric_profile_says_why_it_cannot(self, spectra):
        session = _session(spectra)
        session.set_window("C 1s", (283.0, 290.0))
        session.choice_for("C 1s").states = list(NARROW["states"])
        result = session.fit("C 1s")
        plain = next(c for c in result.components
                     if "asymmetry" not in (c.extra_names or ()))
        session.set_component("C 1s", plain.label, asymmetry=0.1)
        session.fit("C 1s")
        assert any("no tiene asimetría" in text
                   for _, text in session.messages)


class TestTheCompositionSaysWhereItCameFrom:
    def test_the_default_is_the_fitted_regions(self, spectra):
        session = _session(spectra)
        session.run_survey()
        session.fit("C 1s")
        assert session.quantify() is not None
        assert session.composition_source == "regiones"

    def test_the_survey_route_works_without_any_fit(self, spectra):
        # That is its reason for existing: a first look before spending
        # an hour fitting five regions, and the elements no region was
        # recorded for.
        session = _session(spectra)
        session.run_survey()
        assert not session.fits
        composition = session.quantify(source="survey")
        assert composition is not None
        assert session.composition_source == "survey"
        assert {a.element for a in composition.abundances}

    def test_the_survey_route_says_it_is_not_for_publishing(self, spectra):
        session = _session(spectra)
        session.run_survey()
        session.quantify(source="survey")
        assert any("SURVEY" in text and "publicar" in text
                   for _, text in session.messages)

    def test_an_unknown_source_is_refused(self, spectra):
        with pytest.raises(ValueError):
            _session(spectra).quantify(source="adivinar")

    def test_the_two_routes_do_not_agree(self, spectra):
        # The reason the label matters. If they ever came out equal this
        # whole distinction would be pedantry; they do not.
        session = _session(spectra)
        session.run_survey()
        session.fit("C 1s")
        session.fit("O 1s")
        fitted = session.quantify()
        integrated = session.quantify(source="survey")
        assert fitted is not None and integrated is not None
        by_fit = {a.element: a.atomic_percent for a in fitted.abundances}
        by_survey = {a.element: a.atomic_percent for a in integrated.abundances}
        shared = set(by_fit) & set(by_survey)
        assert shared
        assert any(abs(by_fit[e] - by_survey[e]) > 1.0 for e in shared)


class TestTheElementSetCanBeEdited:
    def test_a_declared_element_is_kept_apart_from_a_detected_one(self,
                                                                  spectra):
        session = _session(spectra)
        session.run_survey()
        detected = session.detected_elements()
        assert "Si" not in detected
        assert session.add_element("Si")
        assert "Si" in session.quantified_elements()
        assert session.detected_elements() == detected
        assert ("Si", "declarado a mano") in session.element_rows()

    def test_declaring_one_says_the_evidence_is_not_there(self, spectra):
        session = _session(spectra)
        session.run_survey()
        session.add_element("Si")
        assert any("DECLARADO" in text for _, text in session.messages)

    def test_an_element_the_database_does_not_know_is_refused(self, spectra):
        session = _session(spectra)
        assert not session.add_element("Xx")
        assert "Xx" not in session.quantified_elements()

    def test_removing_one_leaves_its_peaks_unexplained(self, spectra):
        session = _session(spectra)
        session.run_survey()
        assert "Fe" in session.detected_elements()
        assert session.remove_element("Fe")
        assert "Fe" not in session.quantified_elements()
        assert "Fe" in session.detected_elements()
        assert any("sin explicar" in text for _, text in session.messages)

    def test_reset_goes_back_to_the_evidence(self, spectra):
        session = _session(spectra)
        session.run_survey()
        before = session.quantified_elements()
        session.add_element("Si")
        session.remove_element("Fe")
        assert session.quantified_elements() != before
        session.reset_elements()
        assert session.quantified_elements() == before
