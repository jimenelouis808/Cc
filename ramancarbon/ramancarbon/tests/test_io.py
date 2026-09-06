"""File reading: delimiters, decimal commas, headers, laser detection."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.core.io import (
    SpectrumReadError,
    detect_laser_nm,
    parse_spectrum_text,
    read_spectrum,
    wavelength_to_shift,
    write_spectrum,
)


@pytest.mark.parametrize(
    "text, expected_n",
    [
        ("100\t1\n101\t2\n102\t3\n", 3),
        ("100,1\n101,2\n102,3\n", 3),
        ("100;1\n101;2\n102;3\n", 3),
        ("  100   1\n  101   2\n  102   3\n", 3),
    ],
)
def test_delimiters_are_sniffed(text, expected_n):
    assert parse_spectrum_text(text).shift.size == expected_n


def test_decimal_comma_locale():
    """A Spanish-locale export writes 1580,25 and must not raise."""
    s = parse_spectrum_text("1580,25;1024,5\n1581,25;1000,1\n1582,25;990,0\n")
    assert s.shift[0] == pytest.approx(1580.25)
    assert s.intensity[0] == pytest.approx(1024.5)


def test_descending_axis_is_sorted():
    s = parse_spectrum_text("3200 1\n3199 2\n3198 3\n")
    assert s.shift[0] == 3198.0


@pytest.mark.parametrize(
    "header, expected",
    [
        ("Laser: 532 nm", 532.0),
        ("Excitation wavelength = 632.8 nm", 633.0),
        ("Longitud de onda del láser (nm);532,0", 532.0),
        ("#Laser=785", 785.0),
        ("Wavelength/nm 514.5", 514.5),
    ],
)
def test_laser_recovered_from_header(header, expected):
    assert detect_laser_nm(header) == pytest.approx(expected)


def test_grating_is_not_mistaken_for_the_laser():
    """'1800 l/mm' and 'slit 100 nm' must not be read as an excitation."""
    assert detect_laser_nm("Grating: 1800 l/mm, slit 100 nm") is None


def test_explicit_laser_overrides_the_header():
    s = parse_spectrum_text("Laser: 532 nm\n100 1\n101 2\n", laser_nm=633.0)
    assert s.laser_nm == 633.0


def test_multi_column_export():
    s = parse_spectrum_text("100,5,1\n101,6,2\n102,7,3\n", intensity_column=2)
    assert list(s.intensity) == [1.0, 2.0, 3.0]
    assert s.metadata["n_columns"] == 3


def test_wavelength_axis_is_flagged_not_converted():
    """Converting needs the laser; guessing it would be worse than warning."""
    s = parse_spectrum_text("\n".join(f"{535 + i * 0.1:.1f} {i}" for i in range(50)))
    assert "axis_warning" in s.metadata


def test_wavelength_to_shift():
    # 532 nm laser, scattered at 580 nm -> 1e7 (1/532 - 1/580)
    assert wavelength_to_shift([580.0], 532.0)[0] == pytest.approx(1555.6, abs=0.5)


def test_unreadable_file_raises_a_useful_error():
    with pytest.raises(SpectrumReadError, match="two-column"):
        parse_spectrum_text("esto no es un espectro\nni esto tampoco\n")


def test_round_trip_preserves_laser_and_history(tmp_path):
    from ramancarbon.core.spectrum import Spectrum

    s = Spectrum(np.arange(100.0, 200.0), np.arange(100.0), laser_nm=532.0, name="x")
    s.history.append("baseline(asls)")
    path = write_spectrum(s, tmp_path / "out.txt")
    back = read_spectrum(path)
    assert back.laser_nm == 532.0
    assert np.allclose(back.shift, s.shift)
    assert "baseline(asls)" in back.metadata["header"]


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        read_spectrum("/no/existe.txt")
