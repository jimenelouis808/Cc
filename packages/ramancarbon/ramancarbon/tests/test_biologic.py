"""EC-Lab files: the .mps settings and the .mpr binary.

The .mpr reader is held to the standard the .spe reader is held to: the
file says how many points it has, how many columns and which quantity
each one is, so the record size and the table's offset are COMPUTED and
then checked to close against the file's length. A layout that does not
close is refused.

There is no real .mpr in this repository, so the fixtures here build one
to the documented layout and read it back. That tests the decoder — the
arithmetic, the offset search, the dtype mapping, the refusals — and it
explicitly does not test the one thing that cannot be tested without
Bio-Logic's own specification: whether a given column identifier really
means what the community table says it means.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from ramancarbon.echem.biologic import (
    MPR_COLUMNS,
    BioLogicError,
    read_eclab,
    read_mpr,
    read_mps,
)
from ramancarbon.echem.curve import ChargeDischarge, Electrode, Impedance, Voltammogram


# -- fixtures: build an .mpr to the documented layout --------------------

def _module(short: str, long_name: str, payload: bytes, version: int = 2) -> bytes:
    return (
        b"MODULE"
        + short.encode("latin-1").ljust(10, b"\x00")
        + long_name.encode("latin-1").ljust(25, b"\x00")
        + struct.pack("<I", len(payload))
        + struct.pack("<I", version)
        + b"01/01/26"
        + payload
    )


def _data_module(columns: dict[int, np.ndarray],
                 header: int = 0x195) -> bytes:
    """Pack named columns into a ``VMP data`` payload."""
    ids = list(columns)
    n_points = len(next(iter(columns.values())))
    record = np.dtype({
        "names": [f"c{i}" for i in ids],
        "formats": [MPR_COLUMNS[i][1] for i in ids],
    })
    table = np.zeros(n_points, dtype=record)
    for identifier, values in columns.items():
        table[f"c{identifier}"] = values
    head = struct.pack("<I", n_points) + bytes([len(ids)])
    head += b"".join(struct.pack("<H", i) for i in ids)
    head += b"\x00" * max(0, header - len(head))
    return head + table.tobytes()


def _mpr(columns: dict[int, np.ndarray], header: int = 0x195,
         settings: bytes = b"Cyclic Voltammetry\x00") -> bytes:
    return (
        b"BIO-LOGIC MODULAR FILE\x1a".ljust(52, b"\x00")
        + _module("VMP Set", "VMP settings", settings)
        + _module("VMP data", "VMP data", _data_module(columns, header))
    )


def _cv_columns(n: int = 400):
    time = np.linspace(0.0, 60.0, n)
    ramp = np.linspace(0.0, 0.6, n // 2)
    potential = np.concatenate([ramp, ramp[::-1]])
    current = 2.0 * np.gradient(potential, time)      # mA
    return {4: time, 6: potential.astype("f4"), 8: current.astype("f4")}


# -- the decoder ---------------------------------------------------------

def test_an_mpr_round_trips_through_the_decoder(tmp_path):
    columns = _cv_columns()
    path = tmp_path / "medida.mpr"
    path.write_bytes(_mpr(columns))

    data = read_mpr(path)
    assert data.n_points == 400
    assert set(data.columns) == {"time/s", "Ewe/V", "I/mA"}
    assert data.columns["time/s"] == pytest.approx(columns[4])
    assert data.columns["Ewe/V"] == pytest.approx(columns[6], abs=1e-6)
    assert "VMP data" in data.modules

    # The provenance note travels with the numbers, every time.
    assert any("ingeniería inversa" in w for w in data.warnings)


@pytest.mark.parametrize("header", [0x195, 0x196, 0x200, 2048])
def test_the_table_offset_is_computed_not_assumed(tmp_path, header):
    """It differs between format versions, and hardcoding one of them is
    the guess this module exists not to make. The file says how many
    points and how wide each is, so where the table starts is arithmetic."""
    columns = _cv_columns(120)
    path = tmp_path / f"h{header}.mpr"
    path.write_bytes(_mpr(columns, header=header))
    data = read_mpr(path)
    assert data.n_points == 120
    assert data.columns["time/s"] == pytest.approx(columns[4])


def test_a_layout_that_does_not_close_is_refused(tmp_path):
    """The failure mode to want. A record size that is wrong by one byte
    makes every column after the first garbage, and garbage that still
    looks like data is worse than no data."""
    columns = _cv_columns(100)
    raw = bytearray(_mpr(columns))
    # Claim far more points than the table holds.
    index = raw.find(b"MODULE", 52)
    index = raw.find(b"MODULE", index + 1)
    payload_at = index + 6 + 51
    struct.pack_into("<I", raw, payload_at, 5000)

    path = tmp_path / "corto.mpr"
    path.write_bytes(bytes(raw))
    with pytest.raises(BioLogicError, match="no cuadra"):
        read_mpr(path)


def test_an_unknown_column_identifier_is_refused_by_name(tmp_path):
    """Skipping it would shift every column behind it and the result
    would still look like data."""
    columns = _cv_columns(60)
    raw = bytearray(_mpr(columns))
    index = raw.find(b"MODULE", 52)
    index = raw.find(b"MODULE", index + 1)
    payload_at = index + 6 + 51
    struct.pack_into("<H", raw, payload_at + 5, 60000)

    path = tmp_path / "raro.mpr"
    path.write_bytes(bytes(raw))
    with pytest.raises(BioLogicError, match="60000"):
        read_mpr(path)


def test_something_that_is_not_an_mpr_says_so(tmp_path):
    path = tmp_path / "no.mpr"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(200))
    with pytest.raises(BioLogicError, match="BIO-LOGIC MODULAR FILE"):
        read_mpr(path)

    truncated = tmp_path / "corto2.mpr"
    body = _mpr(_cv_columns(50))
    truncated.write_bytes(body[:len(body) // 2])
    with pytest.raises(BioLogicError):
        read_mpr(truncated)


def test_duplicate_column_names_do_not_collide(tmp_path):
    """<I>/mA appears under two identifiers, and numpy needs unique
    field names."""
    n = 80
    columns = {
        4: np.linspace(0.0, 10.0, n),
        11: np.linspace(0.0, 1.0, n),            # <I>/mA, double
        76: np.linspace(0.0, 1.0, n).astype("f4"),  # <I>/mA, float
        6: np.linspace(0.0, 0.5, n).astype("f4"),
    }
    path = tmp_path / "dup.mpr"
    path.write_bytes(_mpr(columns))
    data = read_mpr(path)
    assert "<I>/mA" in data.columns
    assert "<I>/mA (2)" in data.columns


# -- turning columns into curves -----------------------------------------

def test_a_voltammogram_needs_a_scan_rate_and_says_where_it_came_from(tmp_path):
    columns = _cv_columns()
    path = tmp_path / "cv.mpr"
    path.write_bytes(_mpr(columns))

    curve, warnings = read_eclab(path, kind="cv")
    assert isinstance(curve, Voltammogram)
    assert curve.scan_rate == pytest.approx(0.02, rel=0.1)
    assert any("dE/dt" in w or "velocidad" in w for w in warnings)
    assert any("no hay un .mps" in w for w in warnings)

    # An explicit rate wins and no longer needs the guess.
    curve, _ = read_eclab(path, kind="cv", scan_rate=0.05)
    assert curve.scan_rate == 0.05


def test_impedance_keeps_the_physical_sign_of_z(tmp_path):
    """EC-Lab's column is -Im(Z), the Nyquist DRAWING convention. This
    package stores Z'' with its physical sign — negative for a capacitive
    response — and negating on the way in is what keeps every circuit fit
    downstream from silently changing sign."""
    n = 40
    frequency = np.logspace(5, -1, n).astype("f4")
    real = np.linspace(2.0, 40.0, n).astype("f4")
    minus_imaginary = np.linspace(0.5, 25.0, n).astype("f4")
    columns = {4: np.linspace(0, 100, n), 32: frequency,
               37: real, 38: minus_imaginary}
    path = tmp_path / "eis.mpr"
    path.write_bytes(_mpr(columns))

    curve, _ = read_eclab(path)
    assert isinstance(curve, Impedance)
    assert np.all(curve.z.imag < 0), "Z'' must be negative for a capacitor"
    assert curve.z.real[0] == pytest.approx(real.max(), rel=1e-3) or \
        curve.z.real[-1] == pytest.approx(real.max(), rel=1e-3)


def test_a_charge_discharge_curve_comes_back_in_amps(tmp_path):
    """EC-Lab writes milliamps and this package uses SI. A factor of a
    thousand here is a factor of a thousand in every capacitance."""
    n = 200
    time = np.linspace(0.0, 120.0, n)
    potential = np.abs(np.linspace(-0.6, 0.6, n)).astype("f4")
    current = np.where(np.arange(n) < n // 2, 1.0, -1.0).astype("f4")  # mA
    path = tmp_path / "gcd.mpr"
    path.write_bytes(_mpr({4: time, 6: potential, 8: current}))

    curve, _ = read_eclab(path, kind="gcd",
                          electrode=Electrode(mass_mg=2.0, area_cm2=1.0))
    assert isinstance(curve, ChargeDischarge)
    assert np.max(np.abs(curve.current)) == pytest.approx(1e-3, rel=1e-6)


# -- the settings file ---------------------------------------------------

_MPS = """EC-LAB SETTING FILE

Number of linked techniques : 1

Filename : C:\\Users\\lab\\muestra.mps

Device : SP-150
Electrode material : carbono
Mass of active material : 2.140 mg
Electrode surface area : 1.000 cm2
Reference electrode : Ag/AgCl 3M

Cyclic Voltammetry

Ei (V)              0.000
E1 (V)              0.600
dE/dt               20.000
dE/dt unit          mV/s
nc cycles           3
"""


def test_the_settings_file_is_text_and_carries_the_scan_rate(tmp_path):
    """Without it the caller has to type the scan rate in, and a scan
    rate typed wrong gives a capacitance wrong in exactly that proportion
    with nothing downstream to catch it."""
    path = tmp_path / "muestra.mps"
    path.write_text(_MPS, encoding="latin-1")

    settings = read_mps(path)
    assert settings.technique == "cv"
    assert settings.technique_name == "Cyclic Voltammetry"
    assert settings.scan_rate_v_per_s == pytest.approx(0.02)
    assert settings.window_v == pytest.approx((0.0, 0.6))
    assert settings.cycles == 3
    assert settings.mass_mg == pytest.approx(2.14)
    assert settings.area_cm2 == pytest.approx(1.0)
    assert "20" in settings.describe() and "mV/s" in settings.describe()


def test_the_settings_beside_the_data_are_picked_up(tmp_path):
    """EC-Lab writes them as a pair and people copy them as a pair."""
    (tmp_path / "par.mps").write_text(_MPS, encoding="latin-1")
    (tmp_path / "par.mpr").write_bytes(_mpr(_cv_columns()))

    curve, warnings = read_eclab(tmp_path / "par.mpr")
    assert isinstance(curve, Voltammogram)
    assert curve.scan_rate == pytest.approx(0.02), "taken from the .mps"
    assert not any("no hay un .mps" in w for w in warnings)
    assert "ajustes" in curve.metadata


def test_a_file_that_is_not_an_mps_says_so(tmp_path):
    path = tmp_path / "otro.mps"
    path.write_text("no soy un archivo de EC-Lab\n", encoding="latin-1")
    with pytest.raises(BioLogicError, match="EC-LAB SETTING FILE"):
        read_mps(path)


# -- reaching the rest of the package ------------------------------------

def test_the_ordinary_readers_open_an_mpr(tmp_path):
    """By CONTENT, not by extension: instruments write binary inside .txt
    and text inside .spe, so the suffix decides nothing anywhere here."""
    from ramancarbon.echem.io import read_cv, read_eis, read_gcd

    (tmp_path / "cv.mpr").write_bytes(_mpr(_cv_columns()))
    curve = read_cv(tmp_path / "cv.mpr")
    assert isinstance(curve, Voltammogram)
    assert curve.potential.size == 400
    assert curve.metadata["avisos"], "the provenance note has to survive"

    n = 60
    frequency = np.logspace(5, -1, n).astype("f4")
    (tmp_path / "eis.mpr").write_bytes(_mpr({
        4: np.linspace(0, 100, n), 32: frequency,
        37: np.linspace(2.0, 40.0, n).astype("f4"),
        38: np.linspace(0.5, 25.0, n).astype("f4"),
    }))
    spectrum = read_eis(tmp_path / "eis.mpr")
    assert isinstance(spectrum, Impedance)
    assert np.all(spectrum.z.imag < 0)

    m = 200
    (tmp_path / "gcd.mpr").write_bytes(_mpr({
        4: np.linspace(0.0, 120.0, m),
        6: np.abs(np.linspace(-0.6, 0.6, m)).astype("f4"),
        8: np.where(np.arange(m) < m // 2, 1.0, -1.0).astype("f4"),
    }))
    assert isinstance(read_gcd(tmp_path / "gcd.mpr"), ChargeDischarge)

    # A file named .mpr that is not one still fails as an echem error,
    # not as a stray BioLogicError from two layers down.
    from ramancarbon.echem.io import EchemIOError

    (tmp_path / "fake.mpr").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(400))
    with pytest.raises(EchemIOError):
        read_cv(tmp_path / "fake.mpr")


def test_the_detector_knows_both_ec_lab_files(tmp_path):
    """An .mps holds no measurement at all, so "no numeric table found"
    is true and unhelpful: what it needs is the name of the file that
    does hold one."""
    from ramancarbon.dataio.detect import detect

    (tmp_path / "x.mpr").write_bytes(_mpr(_cv_columns()))
    binary = detect(tmp_path / "x.mpr")
    assert binary.kind == "echem" and binary.fmt == "mpr"
    assert "ingeniería inversa" in (binary.advice or "")

    (tmp_path / "x.mps").write_text(_MPS, encoding="latin-1")
    settings = detect(tmp_path / "x.mps")
    assert settings.kind == "ajustes" and settings.fmt == "mps"
    assert ".mpr" in (settings.advice or "")
