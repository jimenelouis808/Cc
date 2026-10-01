"""The user's own EC-Lab files, which the reader refused to open.

Three failures, found by handing it a real CV, a real GCPL and a real
PEIS written by a newer EC-Lab than the one this reader was built
against:

1. The module header grew eight bytes. Read with the old offsets, the
   0xFFFFFFFF marker that now stands before the length WAS the length,
   so every module claimed 4294967295 bytes and every file was refused
   as truncated.
2. The column count is a 16-bit field, so the identifier list starts one
   byte later than the reader looked. Read a byte early, every
   identifier came back multiplied by 256 — 1, 2, 3 arriving as 256,
   512, 768 — and the file was refused for "unknown identifiers" that
   were simply misread.
3. A Nyquist export carries no frequency and a Bode export carries no
   magnitude, so neither is an impedance spectrum alone. Together they
   are.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.echem.biologic import (
    ID_OFFSETS,
    MODULE_HEADER_LONG,
    MODULE_HEADER_SHORT,
    MODULE_LENGTH_MARKER,
    read_mpr,
)
from ramancarbon.echem.io import PHASE_TOLERANCE_DEG, EchemIOError, read_eis_pair

DATA = __import__("pathlib").Path(__file__).parent / "data"


def test_the_newer_module_header_is_read() -> None:
    """Every module of the real PEIS closes to the byte."""
    data = read_mpr(DATA / "eclab_peis.mpr")
    assert data is not None


def test_the_marker_cannot_be_mistaken_for_a_length() -> None:
    """Four gigabytes is not a module, which is what makes the test safe."""
    assert MODULE_LENGTH_MARKER == 0xFFFFFFFF
    assert MODULE_HEADER_LONG - MODULE_HEADER_SHORT == 8


def test_the_identifier_offset_is_read_off_the_file() -> None:
    assert ID_OFFSETS == (5, 6)


def test_the_real_peis_decodes_its_columns() -> None:
    """The binary table comes back with the names EC-Lab wrote."""
    data = read_mpr(DATA / "eclab_peis.mpr")
    assert data.n_points == 58
    for column in ("freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm", "Phase(Z)/deg"):
        assert column in data.columns, column


def test_the_real_peis_opens_as_an_impedance() -> None:
    """The path the user actually takes: hand the .mpr to the reader."""
    from ramancarbon.echem.io import read_eis

    spectrum = read_eis(DATA / "eclab_peis.mpr")
    assert spectrum.frequency.size == 58
    assert spectrum.frequency.max() > 1e5
    assert spectrum.frequency.min() < 1.0
    # Capacitive over most of the range, in the physical sign convention.
    assert np.median(spectrum.z.imag) < 0.0


def test_the_mpr_and_the_text_exports_agree() -> None:
    """They are the same measurement, so they had better come back equal.

    This is the check the module's own doctrine asks for and could never
    run before: a binary layout deduced by reverse engineering, contrasted
    against the ASCII export of the same run.
    """
    from ramancarbon.echem.io import read_eis

    binary = read_eis(DATA / "eclab_peis.mpr")
    text = read_eis_pair(DATA / "eclab_nyquist.txt", DATA / "eclab_bode.txt")
    assert binary.frequency.size == text.frequency.size
    assert np.allclose(binary.frequency, text.frequency, rtol=1e-4)
    assert np.allclose(binary.z.real, text.z.real, rtol=1e-4)
    assert np.allclose(binary.z.imag, text.z.imag, rtol=1e-4)


def test_the_nyquist_and_bode_exports_make_one_spectrum() -> None:
    spectrum = read_eis_pair(DATA / "eclab_nyquist.txt", DATA / "eclab_bode.txt")
    assert spectrum.frequency.size == 58
    assert spectrum.frequency.max() == pytest.approx(2.0e5, rel=1e-3)
    assert spectrum.frequency.min() == pytest.approx(0.0999, rel=1e-2)
    assert spectrum.z.real.min() == pytest.approx(49.8, abs=0.5)
    # Z'' negative where capacitive: the Nyquist file writes -Im(Z) and
    # that negation happens on the way in, once.
    assert spectrum.z.imag.min() < -2000.0
    assert np.median(spectrum.z.imag) < 0.0


def test_the_two_exports_are_checked_against_each_other() -> None:
    """The phase is written twice; agreement is the proof they pair.

    On the user's own files the worst row disagrees by 0.0000 degrees.
    """
    spectrum = read_eis_pair(DATA / "eclab_nyquist.txt", DATA / "eclab_bode.txt")
    assert spectrum.metadata["acuerdo_de_fase_grados"] < PHASE_TOLERANCE_DEG


def test_files_of_different_lengths_are_refused(tmp_path) -> None:
    """Same number of lines is not the same measurement; fewer is obvious."""
    short = tmp_path / "corto.txt"
    lines = (DATA / "eclab_bode.txt").read_text().splitlines()
    short.write_text("\n".join(lines[:20]) + "\n")
    with pytest.raises(EchemIOError, match="no son la misma medida"):
        read_eis_pair(DATA / "eclab_nyquist.txt", short)


def test_shuffled_rows_are_refused(tmp_path) -> None:
    """The failure the cross-check exists for.

    Two files with the same number of rows fuse into a spectrum that
    looks perfectly ordinary and belongs to no experiment. Reversing the
    frequency column is the cheapest way to produce exactly that.
    """
    lines = (DATA / "eclab_bode.txt").read_text().splitlines()
    header, body = lines[0], [line for line in lines[1:] if line.strip()]
    shuffled = tmp_path / "revuelto.txt"
    shuffled.write_text("\n".join([header, *reversed(body)]) + "\n")
    with pytest.raises(EchemIOError, match="no describen la misma medida"):
        read_eis_pair(DATA / "eclab_nyquist.txt", shuffled)


def test_a_nyquist_alone_still_says_what_is_missing() -> None:
    """It has no frequency, and everything downstream is a function of it."""
    from ramancarbon.echem.io import read_eis

    with pytest.raises(EchemIOError, match="tres columnas"):
        read_eis(DATA / "eclab_nyquist.txt")
