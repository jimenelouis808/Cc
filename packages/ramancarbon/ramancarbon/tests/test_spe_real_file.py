"""A real seven-region PHI file the reader refused to open.

The `.spe` reader did not COMPUTE where the values start -- it searched,
trying every four-byte offset and accepting a layout only if exactly one
of them read as a spectrum. On this file that tie cannot be broken: the
measurement is one long run of smooth, positive floats, so sliding the
window by a single sample still looks like a spectrum, many offsets
"fit", and the reader refused a file whose layout is fixed by its own
arithmetic.

PHI writes the region sub-headers first and the values last, so the
offset is a subtraction. Here: 17258 payload bytes, 4142 declared points,
4142 x 4 = 16568, values at 690 -- which is seven 96-byte sub-headers
plus an 18-byte gap.

The check that it is right is not that the arithmetic closes. It is that
every region lands where its chemistry puts it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ramancarbon.gui.xps_state import XPSSession
from ramancarbon.xps.io import read_xps

SPE = Path(__file__).resolve().parent / "data" / "versaprobe_7_regiones.spe"

#: ``region -> (points, binding energy of the maximum, tolerance)``. The
#: energies are what the sample IS: sp2 carbon, pyrrolic/graphitic
#: nitrogen, oxide oxygen, Fe(III) and a selenide. Read the block one
#: sample out of step and they all move together; read it at the wrong
#: offset and they are noise.
EXPECTED = {
    "C 1s": (441, 284.70, 0.3),
    "N 1s": (129, 400.75, 0.5),
    "O 1s": (561, 530.10, 0.4),
    "Fe 2p3": (801, 711.35, 0.5),
    "Se 3d": (421, 55.30, 0.4),
}


@pytest.fixture(scope="module")
def spectra():
    return {item.region: item for item in read_xps(SPE)}


class TestTheFileOpensAtAll:
    def test_it_yields_every_declared_region(self, spectra):
        # PHI writes "C1s"; the rest of the package speaks "C 1s", and
        # the reader translates so nothing downstream has to guess.
        assert set(spectra) == {"Su1s", "C 1s", "N 1s", "O 1s", "Fe 2p3",
                                "Se 3p", "Se 3d"}

    def test_one_of_them_is_the_survey_and_the_rest_are_not(self, spectra):
        assert spectra["Su1s"].is_survey
        assert not any(item.is_survey for name, item in spectra.items()
                       if name != "Su1s")

    def test_the_survey_spans_what_the_header_says(self, spectra):
        low, high = spectra["Su1s"].range
        assert low == pytest.approx(0.0, abs=1.0)
        assert high == pytest.approx(1350.0, abs=1.0)

    def test_the_anode_comes_from_the_header(self, spectra):
        # Al Ka monochromated. Without it the Auger lines cannot be
        # placed, because they are in KINETIC energy.
        assert spectra["C 1s"].photon_energy == pytest.approx(1486.6)


class TestTheBlockWasDecodedAndNotGuessed:
    @pytest.mark.parametrize("region", sorted(EXPECTED))
    def test_the_region_has_its_declared_length(self, spectra, region):
        assert len(spectra[region].counts) == EXPECTED[region][0]

    @pytest.mark.parametrize("region", sorted(EXPECTED))
    def test_its_maximum_sits_where_the_chemistry_puts_it(self, spectra,
                                                          region):
        _, energy, tolerance = EXPECTED[region]
        item = spectra[region]
        peak = float(item.binding_energy[int(np.argmax(item.counts))])
        assert peak == pytest.approx(energy, abs=tolerance), (
            f"{region}: máximo en {peak:.2f} eV, se esperaba {energy}")

    def test_every_count_is_finite_and_positive(self, spectra):
        # Big-endian reads the same bytes as NaNs and negatives, which is
        # what makes the byte order decidable rather than a preference.
        for item in spectra.values():
            assert np.all(np.isfinite(item.counts))
            assert float(item.counts.min()) > 0.0

    def test_the_survey_is_the_most_intense_thing_in_the_file(self, spectra):
        # A survey integrates over a far wider window than any region.
        assert spectra["Su1s"].counts.max() > max(
            item.counts.max() for name, item in spectra.items()
            if name != "Su1s")


class TestTheSectionLoadsIt:
    def test_the_session_takes_the_survey_and_the_regions_apart(self):
        session = XPSSession()
        assert session.load(SPE) == 7
        assert [item.region for item in session.surveys] == ["Su1s"]
        assert [session.region_label(item) for item in session.regions] == [
            "C 1s", "N 1s", "O 1s", "Fe 2p3/2", "Se 3p", "Se 3d5/2"]

    def test_no_error_is_logged(self):
        session = XPSSession()
        session.load(SPE)
        assert not [text for level, text in session.messages
                    if level == "error"]

    def test_the_survey_identifies_the_sample(self):
        # The point of the file: it is carbon with nitrogen, oxygen, iron
        # and selenium in it, and the identification has to say so.
        session = XPSSession()
        session.load(SPE)
        session.run_survey()
        found = set(session.detected_elements())
        assert {"C", "O", "Fe", "Se"} <= found, found


class TestTheRegionNameIsTranslated:
    """PHI writes ``C1s``; the state tables speak ``C 1s``.

    Nothing translated, so a file could be decoded correctly and still
    arrive with regions nobody recognised: ``region_for_line("C1s")``
    matched no state, the chemical-state list came back empty, and the
    survey's region evidence asked the database for an element called
    ``"C1s"`` -- which raised out of a loop whose whole job is to skip
    what it cannot use, taking the identification down with it.
    """

    @pytest.mark.parametrize("phi,canonical", [
        ("C1s", "C 1s"),
        ("N1s", "N 1s"),
        ("Fe2p3", "Fe 2p3"),
        ("Se3d", "Se 3d"),
    ])
    def test_it_splits_the_symbol_from_the_orbital(self, phi, canonical):
        from ramancarbon.xps.io import normalise_region

        assert normalise_region(phi) == canonical

    def test_it_leaves_the_orbital_as_a_prefix(self):
        # "Fe 2p3" is not spelled out to "Fe 2p3/2" here: the database
        # already resolves a prefix to the component that is fitted, and
        # two places inventing the same completion is how they come to
        # disagree.
        from ramancarbon.xps.tables import load_xps_database

        assert load_xps_database().region_for_line("Fe 2p3") == "Fe 2p3/2"
        assert load_xps_database().region_for_line("Se 3d") == "Se 3d5/2"

    def test_a_survey_is_not_given_an_element_that_does_not_exist(self):
        # PHI's own name for the survey parses as element-plus-orbital.
        # "Su" is not an element, so the name stays as written.
        from ramancarbon.xps.io import normalise_region

        assert normalise_region("Su1s") == "Su1s"
        assert normalise_region("Survey") == "Survey"

    def test_a_name_already_in_the_canonical_form_is_untouched(self):
        from ramancarbon.xps.io import normalise_region

        assert normalise_region("C 1s") == "C 1s"


class TestTheOffsetIsComputedRatherThanSearched:
    def test_the_values_are_the_last_block_of_the_payload(self):
        import re

        raw = SPE.read_bytes()
        head = raw[:raw.find(b"EOFH")].decode("latin-1")
        payload = raw[raw.find(b"EOFH") + 4:]
        points = [int(m) for m in re.findall(
            r"^SpectralRegDef:\s*\d+\s+\d+\s+\S+\s+\d+\s+(\d+)\s", head,
            re.M)]
        assert sum(points) == 4142
        # float32, so the arithmetic closes exactly and the offset is a
        # subtraction rather than one of 173 candidates.
        assert len(payload) - 4 * sum(points) == 690

    def test_the_prefix_carries_one_type_tag_per_region(self):
        from ramancarbon.xps.io import SPE_TYPE_TAG

        raw = SPE.read_bytes()
        payload = raw[raw.find(b"EOFH") + 4:]
        assert payload[:690].count(SPE_TYPE_TAG) == 7
