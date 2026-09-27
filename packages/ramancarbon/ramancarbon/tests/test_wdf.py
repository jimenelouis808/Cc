"""Renishaw ``.wdf``, the file the instrument writes.

A 532 nm spectrum of carbon on FeSe arrived twice: as the ``.wdf`` WiRE
saved and as the two-column text WiRE exports. The numbers were the same
to 5e-7 — the rounding in ``%f`` — and everything around them was gone
from the export. These tests fix both halves of that: that the binary is
read at all, and that reading it recovers the excitation, the grating and
the objective the text had dropped.

The fixtures build a ``.wdf`` byte by byte rather than shipping one. A
checked-in binary from somebody's instrument cannot be adjusted to
exercise the failure paths — a truncated DATA, a map where a spectrum was
expected, a property set that does not close — and those are the paths
worth testing, because the format has no published specification and the
whole defence against a misparse is that the layout has to add up.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest


def _pset(entries: list[tuple[str, int, object]], names: dict[int, str]) -> bytes:
    """A WiRE property set: scalars and strings, then the key names."""
    body = b""
    for kind, key, value in entries:
        head = kind.encode("latin-1") + b"\x00" + struct.pack("<H", key)
        if kind == "i":
            body += head + struct.pack("<i", int(value))       # type: ignore[arg-type]
        elif kind == "r":
            body += head + struct.pack("<f", float(value))     # type: ignore[arg-type]
        elif kind == "u":
            raw = str(value).encode("utf-8")
            body += head + struct.pack("<I", len(raw)) + raw
        elif kind == "p":
            raw = bytes(value)                                 # type: ignore[arg-type]
            body += head + struct.pack("<I", len(raw)) + raw
        else:                                                  # pragma: no cover
            raise AssertionError(kind)
    for key, name in names.items():
        raw = name.encode("utf-8")
        body += b"k\x00" + struct.pack("<H", key) + struct.pack("<I", len(raw)) + raw
    return body


def _block(identifier: bytes, payload: bytes, uid: int = 0) -> bytes:
    return identifier + struct.pack("<I", uid) \
        + struct.pack("<Q", 16 + len(payload)) + payload


def _pset_block(identifier: bytes, payload: bytes) -> bytes:
    return _block(identifier, b"PSET" + struct.pack("<I", len(payload)) + payload)


FILETIME_START = 134255163780672885
FILETIME_END = 134255164515764930


def write_wdf(path, shift, intensity, *, laser_cm1: float = 18797.0,
              spectra: int = 1, x_units: int = 1, truncate_data: int = 0,
              break_pset: bool = False, title: str = "prueba"):
    """A minimal but complete ``.wdf``."""
    points = len(shift)
    header = bytearray(512)
    header[0:4] = b"WDF1"
    struct.pack_into("<I", header, 4, 1)
    struct.pack_into("<Q", header, 8, 512)
    struct.pack_into("<I", header, 0x3C, points)
    struct.pack_into("<Q", header, 0x40, spectra)
    struct.pack_into("<Q", header, 0x48, spectra)
    struct.pack_into("<I", header, 0x50, 3)               # accumulations
    struct.pack_into("<I", header, 0x54, 1)
    struct.pack_into("<I", header, 0x58, points)
    header[0x60:0x64] = b"WiRE"
    struct.pack_into("<4H", header, 0x78, 5, 2, 0, 10411)
    struct.pack_into("<I", header, 0x80, 2)               # continuous scan
    struct.pack_into("<I", header, 0x84, 1)
    struct.pack_into("<Q", header, 0x88, FILETIME_START)
    struct.pack_into("<Q", header, 0x90, FILETIME_END)
    struct.pack_into("<I", header, 0x98, 6)
    struct.pack_into("<f", header, 0x9C, laser_cm1)
    header[0xD0:0xD0 + 5] = b"Raman"
    raw_title = title.encode("latin-1")
    header[0xF0:0xF0 + len(raw_title)] = raw_title

    values = list(np.asarray(intensity, dtype=float).ravel())
    if truncate_data:
        values = values[:-truncate_data]
    data = _block(b"DATA", struct.pack(f"<{len(values)}f", *values), uid=0)
    xlst = _block(b"XLST", struct.pack("<II", 1, x_units)
                  + struct.pack(f"<{points}f", *shift))
    ylst = _block(b"YLST", struct.pack("<II", 4, 16) + struct.pack("<f", 0.0))

    origins = struct.pack("<I", 2)
    for kind, units, label, payload in (
        (11, 24, b"Time", struct.pack("<Q", FILETIME_END)),
        (17, 0, b"Flags", struct.pack("<Q", 0)),
    ):
        origins += struct.pack("<II", kind, units) + label.ljust(16, b"\x00") \
            + payload * spectra
    orgn = _block(b"ORGN", origins)

    lasers = _pset([("r", 1, laser_cm1)], {1: "Wavenumber"})
    configuration = _pset_block(b"WXCS", _pset(
        [("p", 1, _pset([("p", 2, lasers)], {2: "532 nm edge"}))],
        {1: "Lasers"},
    ))
    system = _pset([("u", 1, "1800 l/mm (vis)"), ("u", 2, "Regular")],
                   {1: "Grating", 2: "FocusMode"})
    microscope = _pset([("u", 1, "x50")], {1: "Objective"})
    state_body = _pset(
        [("u", 1, "InVia"), ("p", 2, system), ("p", 3, microscope)],
        {1: "Instrument type", 2: "System Configuration", 3: "Microscope"},
    )
    if break_pset:
        # A length that overruns its own block: the parse must refuse it
        # and the spectrum must survive the refusal.
        state_body = b"u\x00\x01\x00" + struct.pack("<I", 9999) + b"InVia"
    state = _pset_block(b"WXIS", state_body)
    zero = _pset_block(b"ZLDC", _pset(
        [("p", 1, _pset([("i", 2, 1726)], {2: "Value"})), ("u", 3, "ZeroLevel")],
        {1: "#403", 3: "Type"},
    ))
    text = _block(b"TEXT", b"A single scan measurement.\x00")

    path.write_bytes(bytes(header) + data + ylst + xlst + orgn + text
                     + configuration + state + zero)
    return path


@pytest.fixture
def spectrum_axes():
    shift = np.linspace(3000.807617, 99.855469, 2119)
    signal = (100.0
              + 380.0 * np.exp(-0.5 * ((shift - 1340.7) / 50.0) ** 2)
              + 375.0 * np.exp(-0.5 * ((shift - 1585.3) / 28.0) ** 2)
              + 170.0 * np.exp(-0.5 * ((shift - 282.5) / 12.0) ** 2))
    return shift, signal


class TestReading:
    def test_the_axis_comes_back_ascending(self, tmp_path, spectrum_axes):
        from ramancarbon.dataio.wdf import read_wdf

        shift, signal = spectrum_axes
        parsed = read_wdf(write_wdf(tmp_path / "a.wdf", shift, signal))
        assert np.all(np.diff(parsed["x"]) > 0)
        # Sorting the axis has to carry the intensities with it, or the
        # 282 cm-1 band ends up at 3000.
        assert parsed["y"][int(np.argmin(np.abs(parsed["x"] - 282.5)))] > 250.0

    def test_the_numbers_are_the_ones_written(self, tmp_path, spectrum_axes):
        from ramancarbon.dataio.wdf import read_wdf

        shift, signal = spectrum_axes
        parsed = read_wdf(write_wdf(tmp_path / "a.wdf", shift, signal))
        order = np.argsort(shift)
        assert np.allclose(parsed["x"], shift[order], atol=1e-3)
        assert np.allclose(parsed["y"], signal[order], atol=1e-3)

    def test_the_excitation_is_recovered_from_the_file(self, tmp_path,
                                                       spectrum_axes):
        from ramancarbon.dataio.wdf import read_wdf

        shift, signal = spectrum_axes
        parsed = read_wdf(write_wdf(tmp_path / "a.wdf", shift, signal))
        assert parsed["laser_nm"] == pytest.approx(532.0, abs=0.01)
        assert parsed["laser_cm1"] == pytest.approx(18797.0)

    def test_a_633_nm_file_is_not_read_as_532(self, tmp_path, spectrum_axes):
        """The whole point of reading the binary.

        Two files from the same instrument differ only in this number, and
        every ratio quoted per wavelength downstream depends on it.
        """
        from ramancarbon.dataio.wdf import read_wdf

        shift, signal = spectrum_axes
        parsed = read_wdf(write_wdf(tmp_path / "b.wdf", shift, signal,
                                    laser_cm1=1.0e7 / 632.8))
        assert parsed["laser_nm"] == pytest.approx(632.8, abs=0.01)

    def test_the_settings_the_text_export_drops(self, tmp_path, spectrum_axes):
        from ramancarbon.dataio.wdf import read_wdf

        shift, signal = spectrum_axes
        metadata = read_wdf(write_wdf(tmp_path / "a.wdf", shift, signal))["metadata"]
        assert metadata["red_de_difraccion"] == "1800 l/mm (vis)"
        assert metadata["objetivo"] == "x50"
        assert metadata["equipo"] == "InVia"
        assert metadata["acumulaciones"] == 3
        assert metadata["tipo_de_barrido"] == "continuo"
        assert metadata["duracion_s"] == pytest.approx(73.509, abs=0.01)
        assert metadata["inicio"].startswith("2026-")

    def test_the_subtracted_zero_level_is_reported(self, tmp_path, spectrum_axes):
        """Because it says the intensities are not photon counts.

        A Renishaw trace has had a bias of order a thousand ADU taken off
        it, so sigma = sqrt(I) is not valid for it. This package estimates
        sigma locally from the trace instead; the number here is the
        evidence that it has to, and it belongs in a methods section.
        """
        from ramancarbon.dataio.wdf import read_wdf

        shift, signal = spectrum_axes
        parsed = read_wdf(write_wdf(tmp_path / "a.wdf", shift, signal))
        assert parsed["metadata"]["nivel_cero_restado"] == pytest.approx(1726.0)

    def test_the_instant_of_the_measurement(self, tmp_path, spectrum_axes):
        from ramancarbon.dataio.wdf import read_wdf

        shift, signal = spectrum_axes
        parsed = read_wdf(write_wdf(tmp_path / "a.wdf", shift, signal))
        assert parsed["metadata"]["origen"]["Time"].startswith("2026-06-09")


class TestRefusals:
    def test_a_file_that_is_not_a_wdf(self, tmp_path):
        from ramancarbon.dataio.wdf import WdfError, read_wdf

        path = tmp_path / "no.wdf"
        path.write_bytes(b"\x00" * 900)
        with pytest.raises(WdfError, match="WDF1"):
            read_wdf(path)

    def test_a_truncated_data_block_is_not_padded(self, tmp_path, spectrum_axes):
        """Reading 2118 of 2119 points and not saying so is the bad outcome.

        The point count is declared in the header, so a short DATA block
        is arithmetic, not a judgement call.
        """
        from ramancarbon.dataio.wdf import WdfError, read_wdf

        shift, signal = spectrum_axes
        path = write_wdf(tmp_path / "corto.wdf", shift, signal, truncate_data=4)
        with pytest.raises(WdfError, match="valores"):
            read_wdf(path)

    def test_a_map_is_not_returned_as_one_spectrum(self, tmp_path):
        from ramancarbon.dataio.wdf import WdfError, load_wdf_spectrum

        shift = np.linspace(100.0, 2000.0, 64)
        stack = np.tile(100.0 + shift / 10.0, 5)
        path = write_wdf(tmp_path / "mapa.wdf", shift, stack, spectra=5)
        with pytest.raises(WdfError, match="mapa|espectros"):
            load_wdf_spectrum(path)

    def test_a_property_set_that_does_not_close_costs_metadata_only(
            self, tmp_path, spectrum_axes):
        """The layout is reverse engineered, so it is checked, not trusted.

        XLST and DATA are enough for the spectrum; the property sets are
        extra. A refused property set must therefore leave the numbers
        intact and say what it refused.
        """
        from ramancarbon.dataio.wdf import read_wdf

        shift, signal = spectrum_axes
        path = write_wdf(tmp_path / "roto.wdf", shift, signal, break_pset=True)
        parsed = read_wdf(path)
        assert parsed["y"].size == shift.size
        assert any("WXIS" in problem for problem in parsed["problems"])
        assert "objetivo" not in parsed["metadata"]


class TestThroughTheFrontDoor:
    def test_detect_recognises_it_without_the_extension(self, tmp_path,
                                                        spectrum_axes):
        """Vendors write binary into .txt and the suite reads by content."""
        from ramancarbon.dataio.detect import detect

        shift, signal = spectrum_axes
        path = write_wdf(tmp_path / "sinextension.dat", shift, signal)
        found = detect(path)
        assert found.kind == "raman"
        assert found.fmt == "wdf"

    def test_a_wdf_map_is_detected_as_a_map(self, tmp_path):
        from ramancarbon.dataio.detect import detect

        shift = np.linspace(100.0, 2000.0, 64)
        stack = np.tile(100.0 + shift / 10.0, 7)
        found = detect(write_wdf(tmp_path / "mapa.wdf", shift, stack, spectra=7))
        assert found.kind == "mapa"

    def test_load_returns_a_spectrum_carrying_the_laser(self, tmp_path,
                                                       spectrum_axes):
        from ramancarbon.dataio import load

        shift, signal = spectrum_axes
        loaded = load(write_wdf(tmp_path / "a.wdf", shift, signal))
        assert loaded.kind == "raman"
        assert loaded.data.laser_nm == pytest.approx(532.0, abs=0.01)
        assert loaded.data.intensity.size == shift.size

    def test_an_override_keeps_the_file_s_own_value_visible(self, tmp_path,
                                                           spectrum_axes):
        """A spectrum measured at 532 and analysed as 633 is wrong by the
        ratio of the wavelengths, and only the person who typed the
        override knows which is right. So the disagreement is recorded."""
        from ramancarbon.dataio import load

        shift, signal = spectrum_axes
        loaded = load(write_wdf(tmp_path / "a.wdf", shift, signal),
                      laser_nm=632.8)
        assert loaded.data.laser_nm == pytest.approx(632.8)
        assert loaded.data.metadata["laser_del_archivo_nm"] == pytest.approx(
            532.0, abs=0.01)
