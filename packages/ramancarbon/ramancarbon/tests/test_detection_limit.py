"""How much of a phase could be there and not have been seen.

A phase report says which lines were found. The question it leaves open is
what the missing ones are worth, and that question has a number. These
tests hold that number to the only standard that makes it meaningful: it
has to agree with the detector it claims to describe. A limit computed by
inverting a formula is a second implementation of the peak finder, and a
limit that does not match the peak finder invites exactly the conclusion
it cannot support.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.core.spectrum import Spectrum


def flat_spectrum(noise: float = 3.0, seed: int = 11,
                  bands: tuple[tuple[float, float, float], ...] = ()) -> Spectrum:
    """A baseline-free trace with whatever bands are asked for."""
    shift = np.linspace(100.0, 1800.0, 1300)
    signal = np.zeros_like(shift)
    for centre, fwhm, height in bands:
        signal = signal + height / (1.0 + ((shift - centre) / (fwhm / 2.0)) ** 2)
    rng = np.random.default_rng(seed)
    signal = signal + rng.normal(0.0, noise, signal.shape)
    return Spectrum(shift=shift, intensity=signal, laser_nm=532.0, name="prueba")


class TestTheLimitAgreesWithTheDetector:
    """The whole design. A limit the peak finder disagrees with is worse
    than no limit at all."""

    def test_a_band_just_above_the_limit_is_found(self):
        from ramancarbon.analysis.detection_limit import band_limit
        from ramancarbon.core.peaks import find_peaks

        spectrum = flat_spectrum()
        limit = band_limit(spectrum, 600.0, fwhm=15.0)
        assert limit is not None and limit > 0
        from ramancarbon.analysis.detection_limit import _inject

        louder = _inject(spectrum, 600.0, 15.0, limit * 1.3)
        assert any(abs(p.position - 600.0) <= 8.0
                   for p in find_peaks(louder, window=(540.0, 660.0)))

    def test_a_band_just_below_the_limit_is_not(self):
        from ramancarbon.analysis.detection_limit import _inject, band_limit
        from ramancarbon.core.peaks import find_peaks

        spectrum = flat_spectrum()
        limit = band_limit(spectrum, 600.0, fwhm=15.0)
        quieter = _inject(spectrum, 600.0, 15.0, limit * 0.7)
        assert not any(abs(p.position - 600.0) <= 8.0
                       for p in find_peaks(quieter, window=(540.0, 660.0)))

    def test_more_noise_means_a_higher_limit(self):
        from ramancarbon.analysis.detection_limit import band_limit

        quiet = band_limit(flat_spectrum(noise=2.0), 600.0, fwhm=15.0)
        loud = band_limit(flat_spectrum(noise=12.0), 600.0, fwhm=15.0)
        assert loud > 2.0 * quiet

    def test_a_broad_band_is_harder_to_see_than_a_narrow_one(self):
        """Which is why the assumed width is an argument and not a
        constant, and why it is printed with the answer."""
        from ramancarbon.analysis.detection_limit import band_limit

        narrow = band_limit(flat_spectrum(), 600.0, fwhm=8.0)
        broad = band_limit(flat_spectrum(), 600.0, fwhm=80.0)
        assert broad > narrow


class TestWhatCannotBeAnswered:
    def test_a_line_outside_the_measured_range_is_not_called_absent(self):
        """The most misleading thing this module could possibly say.

        A line at 3200 cm⁻¹ of a spectrum that stops at 1800 is not
        missing. It was never looked for.
        """
        from ramancarbon.analysis.detection_limit import detection_limit
        from ramancarbon.analysis.phases import Phase, PhaseBand

        phase = Phase(
            key="lejos", label="Fase lejana", formula="X", family="f",
            crystal_system="cubic", space_group=None,
            bands=(PhaseBand(3200.0, (3190.0, 3210.0), "A1g", 1.0),),
            strong=(3200.0,), discriminating=(), confidence="low",
            source="prueba", xrd_hint={}, notes="",
        )
        result = detection_limit(flat_spectrum(), phase)
        band = result.bands[0]
        assert band.limit_height is None
        assert "fuera del intervalo medido" in band.reason
        assert not band.detected

    def test_a_raman_silent_phase_says_so_instead_of_bounding_nothing(self):
        """Alpha iron has no first-order Raman modes. The diffraction
        refinement of the same sample puts it at 25 % by weight, and the
        right answer is not "absent" but "this technique cannot see it"."""
        from ramancarbon.analysis.detection_limit import detection_limit

        result = detection_limit(flat_spectrum(), "Fe_alpha")
        assert result.silent
        assert result.bands == []
        assert "no puede ver" in result.verdict()

    def test_an_unknown_phase_key_names_what_is_available(self):
        from ramancarbon.analysis.detection_limit import detection_limit

        with pytest.raises(ValueError, match="no está en la base"):
            detection_limit(flat_spectrum(), "no_existe_esta_fase")


class TestABandInTheWayIsNotNoise:
    """A weak limit has two causes and they call for opposite actions.

    High noise means count longer. A neighbouring band means no amount of
    counting helps and that line is not usable on this sample.
    """

    def test_a_neighbour_is_named(self):
        from ramancarbon.analysis.detection_limit import detection_limit
        from ramancarbon.analysis.phases import Phase, PhaseBand

        # A phase whose line sits 25 cm-1 from a tall band in the data.
        spectrum = flat_spectrum(bands=((215.0, 18.0, 200.0),))
        phase = Phase(
            key="vecina", label="Fase vecina", formula="X", family="f",
            crystal_system="cubic", space_group=None,
            bands=(PhaseBand(240.0, (235.0, 245.0), "A1g", 1.0),),
            strong=(240.0,), discriminating=(), confidence="low",
            source="prueba", xrd_hint={}, notes="",
        )
        band = detection_limit(spectrum, phase).bands[0]
        assert band.obscured_by == pytest.approx(215.0, abs=3.0)

    def test_an_obscured_line_does_not_count_as_the_tightest_limit(self):
        """Because its limit measures the neighbour, not the sensitivity."""
        from ramancarbon.analysis.detection_limit import detection_limit
        from ramancarbon.analysis.phases import Phase, PhaseBand

        spectrum = flat_spectrum(bands=((215.0, 18.0, 200.0),))
        phase = Phase(
            key="dos", label="Dos líneas", formula="X", family="f",
            crystal_system="cubic", space_group=None,
            bands=(PhaseBand(240.0, (235.0, 245.0), "A1g", 1.0),
                   PhaseBand(900.0, (895.0, 905.0), "Eg", 1.0)),
            strong=(240.0,), discriminating=(), confidence="low",
            source="prueba", xrd_hint={}, notes="",
        )
        best = detection_limit(spectrum, phase).best_limit
        assert best is not None
        assert best.position == pytest.approx(900.0)

    def test_a_tiny_neighbour_does_not_obscure_anything(self):
        from ramancarbon.analysis.detection_limit import detection_limit
        from ramancarbon.analysis.phases import Phase, PhaseBand

        # Far from any band: nothing to be obscured by.
        spectrum = flat_spectrum()
        phase = Phase(
            key="sola", label="Sola", formula="X", family="f",
            crystal_system="cubic", space_group=None,
            bands=(PhaseBand(900.0, (895.0, 905.0), "A1g", 1.0),),
            strong=(900.0,), discriminating=(), confidence="low",
            source="prueba", xrd_hint={}, notes="",
        )
        assert detection_limit(spectrum, phase).bands[0].obscured_by is None


class TestTheMarginThatSavedCementite:
    """The exclusion rule, and the case that set its threshold.

    Predicting one band's height from another's needs the catalogue's
    relative intensities to hold at the excitation used, and Raman
    relative intensities move with resonance, orientation and whatever
    the band sits on. At a margin of one this check ruled out cementite
    on a real spectrum — two lines plainly present at 215 and 282 cm⁻¹,
    16.4 % by weight in the Rietveld refinement of the same sample — from
    a missing 685 cm⁻¹ line short by a factor of three.
    """

    def _phase(self, key="ensayo"):
        from ramancarbon.analysis.phases import Phase, PhaseBand

        return Phase(
            key=key, label="Fase de ensayo", formula="X", family="f",
            crystal_system="cubic", space_group=None,
            bands=(PhaseBand(400.0, (395.0, 405.0), "A1g", 1.0),
                   PhaseBand(900.0, (895.0, 905.0), "Eg", 0.5)),
            strong=(400.0,), discriminating=(), confidence="low",
            source="prueba", xrd_hint={}, notes="",
        )

    def test_a_factor_of_three_does_not_exclude(self):
        from ramancarbon.analysis.detection_limit import detection_limit

        # 400 present at 120; the 900 would be at 60 and is missing, with
        # a limit around 20 in this noise: a factor of about three.
        spectrum = flat_spectrum(noise=6.0, bands=((400.0, 15.0, 120.0),))
        result = detection_limit(spectrum, self._phase())
        anchor = result.anchor
        assert anchor is not None and anchor.position == pytest.approx(400.0)
        missing = next(b for b in result.bands if b.position == 900.0)
        factor = (0.5 * anchor.measured_height) / missing.limit_height
        assert 1.5 < factor < 5.0, f"el caso de prueba ya no es marginal: {factor}"
        assert result.excluded_by == []
        assert "muy improbable" not in result.verdict()
        assert "ni confirmada ni descartada" in result.verdict()

    def test_a_large_factor_does_exclude(self):
        from ramancarbon.analysis.detection_limit import detection_limit

        # Same phase, much taller anchor and much quieter trace: the
        # missing line is now short by more than the margin.
        spectrum = flat_spectrum(noise=2.0, bands=((400.0, 15.0, 900.0),))
        result = detection_limit(spectrum, self._phase())
        assert result.excluded_by
        assert "muy improbable" in result.verdict()

    def test_the_wording_never_claims_more_than_the_premise_allows(self):
        from ramancarbon.analysis.detection_limit import detection_limit

        spectrum = flat_spectrum(noise=2.0, bands=((400.0, 15.0, 900.0),))
        verdict = detection_limit(spectrum, self._phase()).verdict()
        assert "intensidades relativas del catálogo" in verdict
        assert "DESCARTADA" not in verdict

    def test_two_lines_present_is_compatible_not_excluded(self):
        from ramancarbon.analysis.detection_limit import detection_limit

        spectrum = flat_spectrum(noise=4.0,
                                 bands=((400.0, 15.0, 200.0),
                                        (900.0, 15.0, 100.0)))
        result = detection_limit(spectrum, self._phase())
        assert "compatible" in result.verdict()


class TestThroughTheReport:
    def test_the_analysis_carries_the_limits(self):
        from ramancarbon.analysis.report import analyse, build_report

        shift = np.linspace(120.0, 3000.0, 1100)
        signal = (
            60.0
            + 320.0 / (1.0 + ((shift - 1348.0) / 75.0) ** 2)
            + 280.0 / (1.0 + ((shift - 1588.0) / 32.0) ** 2)
            + 120.0 / (1.0 + ((shift - 393.0) / 18.0) ** 2)
        )
        rng = np.random.default_rng(3)
        spectrum = Spectrum(shift=shift, intensity=signal + rng.normal(0, 3, shift.shape),
                            laser_nm=532.0, name="con fase")
        result = analyse(spectrum)
        assert result.absence, "el barrido dejó preguntas abiertas y no se acotó ninguna"
        assert "QUÉ SE HABRÍA VISTO" in build_report(result)

    def test_a_failure_to_bound_does_not_lose_the_analysis(self, monkeypatch):
        """A limit refines the report. It is never a reason to lose it."""
        from ramancarbon.analysis import detection_limit as module
        from ramancarbon.analysis.report import analyse

        def explode(*args, **kwargs):
            raise RuntimeError("fallo a propósito")

        monkeypatch.setattr(module, "absence_limits", explode)
        shift = np.linspace(120.0, 3000.0, 900)
        signal = (60.0
                  + 320.0 / (1.0 + ((shift - 1348.0) / 75.0) ** 2)
                  + 280.0 / (1.0 + ((shift - 1588.0) / 32.0) ** 2))
        spectrum = Spectrum(shift=shift, intensity=signal, laser_nm=532.0,
                            name="roto")
        result = analyse(spectrum)
        assert result.fit is not None
        assert result.absence == []
        assert any("acotar las fases ausentes" in w for w in result.warnings)


class TestWhySeparatesFromHowMuch:
    """A weak limit is a number; whether it is about sensitivity or about
    a neighbour is a diagnosis, and the two need different evidence.

    The first two attempts at that diagnosis both failed on real data.
    Asking whether a detected peak sits within a fixed distance left a
    limit of 171 counts unflagged with a band of 184 on top of it,
    because a peak's reported height and an injected band's height are
    not measured the same way. Asking whether the local maximum exceeds
    the limit flagged everything, because on a baseline-corrected carbon
    spectrum there is always something above a small number: the
    excess-over-median ratios were 0.86 for a line genuinely buried under
    the 215 cm⁻¹ band and 2.15 for one in clear air, the wrong way round.

    What does separate them is the limit itself, compared against the
    limit the same noise would impose with nothing else in the window.
    """

    def test_a_line_under_a_strong_band_is_blamed_on_the_band(self):
        from ramancarbon.analysis.detection_limit import (
            OBSCURED_RATIO,
            band_limit,
            noise_floor_limit,
        )

        spectrum = flat_spectrum(noise=4.0, bands=((215.0, 20.0, 200.0),))
        limit = band_limit(spectrum, 196.0, fwhm=15.0)
        floor = noise_floor_limit(spectrum, 196.0, fwhm=15.0)
        assert limit > OBSCURED_RATIO * floor

    def test_a_line_in_clear_air_is_not(self):
        from ramancarbon.analysis.detection_limit import (
            OBSCURED_RATIO,
            band_limit,
            noise_floor_limit,
        )

        spectrum = flat_spectrum(noise=4.0, bands=((215.0, 20.0, 200.0),))
        limit = band_limit(spectrum, 900.0, fwhm=15.0)
        floor = noise_floor_limit(spectrum, 900.0, fwhm=15.0)
        assert limit <= OBSCURED_RATIO * floor

    def test_the_noise_floor_tracks_the_noise(self):
        from ramancarbon.analysis.detection_limit import noise_floor_limit

        quiet = noise_floor_limit(flat_spectrum(noise=2.0), 900.0, fwhm=15.0)
        loud = noise_floor_limit(flat_spectrum(noise=10.0), 900.0, fwhm=15.0)
        assert loud > 2.0 * quiet

    def test_it_is_reproducible(self):
        """Seeded, so the same spectrum does not give two answers."""
        from ramancarbon.analysis.detection_limit import noise_floor_limit

        spectrum = flat_spectrum()
        first = noise_floor_limit(spectrum, 900.0, fwhm=15.0)
        second = noise_floor_limit(spectrum, 900.0, fwhm=15.0)
        assert first == second
