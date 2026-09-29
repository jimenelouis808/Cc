"""A .ras must keep its header through the reader the window calls.

``read_ras`` was careful and complete, and nothing reached it. The
window loads through :func:`ramancarbon.xrd.io.read_pattern`, which had
branches for .xrdml and .uxd and let everything else fall to the generic
text reader -- and a .ras *does* parse there, because its measurement
block is three plain columns. So the file opened, the numbers were
right, nothing looked wrong, and every d-spacing sat on a default
1.540598 Å instead of the 1.540593 the tube actually ran at.

A part in three thousand is a tenth of a degree at 80° 2θ. That is wider
than the peak-matching window, so it does not shift a fit -- it changes
which phase the search says the sample is.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.gui.xrd_state import XRDSession
from ramancarbon.xrd.io import read_pattern

HEADER = '''*RAS_DATA_START
*RAS_HEADER_START
*DISP_UNIT_Y "cps"
*FILE_OPERATOR "Administrator"
*FILE_SAMPLE "S1 I Se 900"
*HW_XG_WAVE_LENGTH_ALPHA1 "1.540593"
*HW_XG_WAVE_LENGTH_ALPHA2 "1.544414"
*MEAS_SCAN_AXIS_X "Theta/2-Theta"
*MEAS_SCAN_SPEED "4.269267"
*MEAS_SCAN_START_TIME "06/11/26 13:59:11"
*MEAS_SCAN_STEP "0.0100000000"
*MEAS_SCAN_UNIT_Y "{unit}"
*RAS_HEADER_END
*RAS_INT_START
'''


def write_ras(tmp_path, rows=None, unit="counts", name="m.ras"):
    rows = rows or [(10.0 + 0.01 * i, 600.0 + i, 1.0) for i in range(60)]
    body = "".join(f"{a:.4f} {b:.4f} {c:.4f}\r\n" for a, b, c in rows)
    path = tmp_path / name
    path.write_bytes((HEADER.format(unit=unit) + body
                      + "*RAS_INT_END\r\n*RAS_DATA_END\r\n").encode("latin-1"))
    return path


class TestTheHeaderSurvivesTheReaderTheWindowCalls:
    def test_the_wavelength_is_the_recorded_one(self, tmp_path):
        pattern = read_pattern(write_ras(tmp_path))
        assert pattern.wavelength == pytest.approx(1.540593)
        assert pattern.metadata["wavelength_recorded"] == pytest.approx(
            1.540593)

    def test_the_default_is_a_different_number(self):
        # Why the above is worth a test rather than a shrug.
        from ramancarbon.xrd.scattering import wavelength_for

        assert wavelength_for("Cu", "ka1") != pytest.approx(1.540593, abs=1e-7)

    def test_the_satellite_step_dwell_and_unit_come_along(self, tmp_path):
        meta = read_pattern(write_ras(tmp_path)).metadata
        assert meta["wavelength_alpha2"] == pytest.approx(1.544414)
        assert meta["step"] == pytest.approx(0.01)
        assert meta["dwell"] == pytest.approx(60.0 * 0.01 / 4.269267)
        assert meta["unit"] == "counts"
        assert meta["format"] == "ras"

    def test_the_pattern_is_named_after_the_sample(self, tmp_path):
        # Not after the file, which on this instrument is a timestamp.
        assert read_pattern(write_ras(tmp_path)).name == "S1 I Se 900"

    def test_an_explicit_wavelength_still_wins(self, tmp_path):
        # The file is sometimes wrong, and overriding it must stay
        # possible -- that is the whole reason the argument exists.
        pattern = read_pattern(write_ras(tmp_path), wavelength=1.6)
        assert pattern.wavelength == pytest.approx(1.6)
        assert pattern.metadata["wavelength_overridden"] is True

    def test_the_attenuator_is_undone(self, tmp_path):
        # When the filter is in, the recorded counts are what got
        # through; a factor that changes mid-scan leaves a step in the
        # pattern exactly where the instrument switched it.
        rows = [(10.0 + 0.01 * i, 600.0, 1.0 if i < 30 else 10.0)
                for i in range(60)]
        pattern = read_pattern(write_ras(tmp_path, rows))
        assert pattern.intensity[0] == pytest.approx(600.0)
        assert pattern.intensity[-1] == pytest.approx(6000.0)

    def test_the_numbers_are_the_same_either_way(self, tmp_path):
        # The generic reader was not wrong about the data, only about
        # everything around it. This pins that down.
        from ramancarbon.xrd.ras import read_ras

        path = write_ras(tmp_path)
        assert np.allclose(read_pattern(path).intensity,
                           read_ras(path)["intensity"])


class TestCountsPerSecondIsConvertedNotShrugged:
    """sigma = sqrt(N) is a statement about COUNTS.

    Applied to a rate it understates the variance by the dwell time, and
    nothing downstream objects: the refinement converges, chi-squared
    comes out small, and every error bar is wrong by the same constant.
    The header knows the time per step, so the file can be put back into
    counts rather than merely complained about.
    """

    DWELL = 60.0 * 0.01 / 4.269267

    def test_a_cps_file_is_multiplied_by_the_dwell(self, tmp_path):
        rate = read_pattern(write_ras(tmp_path, unit="cps"))
        assert rate.intensity[0] == pytest.approx(600.0 * self.DWELL)
        assert rate.metadata["cps_to_counts"] == pytest.approx(self.DWELL)
        assert rate.metadata["unit"] == "counts"
        assert rate.counts is True

    def test_a_counts_file_is_left_alone(self, tmp_path):
        assert "cps_to_counts" not in read_pattern(
            write_ras(tmp_path, unit="counts")).metadata

    def test_without_a_dwell_it_says_these_are_not_counts(self, tmp_path):
        # Guessing a dwell would be inventing the one number that makes
        # the weighting right or wrong.
        path = write_ras(tmp_path, unit="cps")
        text = path.read_bytes().decode("latin-1").replace(
            '*MEAS_SCAN_SPEED "4.269267"', '*MEAS_SCAN_SPEED "0"')
        path.write_bytes(text.encode("latin-1"))
        pattern = read_pattern(path)
        assert pattern.counts is False
        assert pattern.metadata["counts_unavailable"] is True
        assert pattern.intensity[0] == pytest.approx(600.0)


class TestTheSessionSaysWhatItRead:
    def test_it_reports_the_header(self, tmp_path):
        session = XRDSession()
        assert session.load([write_ras(tmp_path)]) == 1
        said = " ".join(text for _, text in session.messages)
        assert "1.540593" in said
        assert "0.01" in said

    def test_a_converted_cps_file_needs_no_warning(self, tmp_path):
        # It was converted, so sqrt(N) is now true of it.
        session = XRDSession()
        session.counts = True
        session.load([write_ras(tmp_path, unit="cps")])
        assert not [t for level, t in session.messages if level == "warn"]

    def test_a_counts_file_raises_no_such_warning(self, tmp_path):
        session = XRDSession()
        session.counts = True
        session.load([write_ras(tmp_path, unit="counts")])
        assert not [t for level, t in session.messages if level == "warn"]
