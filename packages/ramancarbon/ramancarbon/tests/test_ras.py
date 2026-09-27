"""The Rigaku .ras reader, on a file shaped like the instrument's."""
from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.xrd.ras import RASError, read_ras

CABECERA = '''*RAS_DATA_START
*RAS_HEADER_START
*DISP_UNIT_Y "cps"
*FILE_OPERATOR "Administrator"
*FILE_SAMPLE " "
*HW_XG_WAVE_LENGTH_ALPHA1 "1.540593"
*HW_XG_WAVE_LENGTH_ALPHA2 "1.544414"
*MEAS_SCAN_AXIS_X "Theta/2-Theta"
*MEAS_SCAN_SPEED "4.269267"
*MEAS_SCAN_START_TIME "06/11/26 13:59:11"
*MEAS_SCAN_STEP "0.0100000000"
*MEAS_SCAN_UNIT_Y "counts"
*RAS_HEADER_END
*RAS_INT_START
'''


def _escribir(tmp_path, filas, nombre="m.ras"):
    path = tmp_path / nombre
    cuerpo = "".join(f"{a:.4f} {b:.4f} {c:.4f}\r\n" for a, b, c in filas)
    path.write_bytes((CABECERA + cuerpo + "*RAS_INT_END\r\n"
                      "*RAS_DATA_END\r\n").encode("latin-1"))
    return path


def _filas(n=60, factor=1.0):
    return [(10.0 + 0.01 * i, 600.0 + i, factor) for i in range(n)]


class TestLoQueElExportPierde:
    def test_reads_the_wavelength_the_machine_recorded(self, tmp_path):
        """1.540593, not the 1.540598 a default substitutes. A part in
        three thousand is a tenth of a degree at 80 deg 2-theta, which is
        wider than the peak-matching window."""
        parsed = read_ras(_escribir(tmp_path, _filas()))
        assert parsed["wavelength"] == pytest.approx(1.540593)
        assert parsed["wavelength_alpha2"] == pytest.approx(1.544414)

    def test_the_dwell_comes_from_step_and_speed(self, tmp_path):
        """Counts per second is what the display shows; counts is what
        the statistics need, and only the header has the time."""
        parsed = read_ras(_escribir(tmp_path, _filas()))
        assert parsed["dwell"] == pytest.approx(60.0 * 0.01 / 4.269267)

    def test_it_keeps_the_counts_as_measured(self, tmp_path):
        parsed = read_ras(_escribir(tmp_path, _filas()))
        assert parsed["intensity"][0] == pytest.approx(600.0)
        assert parsed["unit"] == "counts"

    def test_the_whole_header_is_carried_through(self, tmp_path):
        """A field this reader has no use for may be exactly what
        explains a measurement later."""
        parsed = read_ras(_escribir(tmp_path, _filas()))
        assert parsed["header"]["FILE_OPERATOR"] == "Administrator"
        assert parsed["header"]["MEAS_SCAN_AXIS_X"] == "Theta/2-Theta"


class TestAtenuador:
    def test_a_constant_factor_of_one_changes_nothing(self, tmp_path):
        parsed = read_ras(_escribir(tmp_path, _filas(factor=1.0)))
        assert parsed["intensity"][0] == pytest.approx(600.0)

    def test_a_varying_factor_is_applied(self, tmp_path):
        """The attenuator is a filter in the beam: the recorded counts
        are what got through. Ignoring a factor that changes mid-scan
        leaves a step exactly where the instrument switched it."""
        filas = _filas(40, 1.0) + [(10.4 + 0.01 * i, 700.0, 10.0)
                                   for i in range(20)]
        parsed = read_ras(_escribir(tmp_path, filas))
        assert parsed["intensity"][0] == pytest.approx(600.0)
        assert parsed["intensity"][-1] == pytest.approx(7000.0)


class TestRechazos:
    def test_a_file_with_no_data_block_says_so(self, tmp_path):
        path = tmp_path / "no.ras"
        path.write_text("10.0 600.0\n10.01 610.0\n")
        with pytest.raises(RASError, match="RAS_INT_START"):
            read_ras(path)

    def test_a_block_too_short_to_be_a_pattern_is_refused(self, tmp_path):
        with pytest.raises(RASError, match="filas legibles"):
            read_ras(_escribir(tmp_path, _filas(3)))


class TestPorElCargadorGeneral:
    def test_load_recognises_it_and_keeps_the_metadata(self, tmp_path):
        from ramancarbon.dataio import load

        cargado = load(_escribir(tmp_path, _filas()))
        assert cargado.kind == "xrd"
        assert cargado.detection.fmt == "ras"
        pattern = cargado.data
        assert pattern.wavelength == pytest.approx(1.540593)
        assert pattern.metadata["formato"] == "ras"
        assert pattern.metadata["dwell"] > 0
        assert np.asarray(pattern.intensity)[0] == pytest.approx(600.0)
