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
    MPR_FLAG_COLUMNS,
    BioLogicError,
    read_eclab,
    read_mpr,
    read_mps,
    read_mpt,
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
                 header: int = 0x195,
                 flags: dict[int, np.ndarray] | None = None) -> bytes:
    """Pack named columns into a ``VMP data`` payload.

    ``flags`` are the identifiers that live as BITS of one shared byte,
    which is what the real files do and what the reader has to
    reproduce: counted as a byte each the record comes out too wide and
    nothing closes.
    """
    flags = flags or {}
    ids = list(flags) + list(columns)
    n_points = len(next(iter((flags or columns).values())))
    names = (["_flags"] if flags else []) + [f"c{i}" for i in columns]
    formats = (["u1"] if flags else []) + [MPR_COLUMNS[i][1] for i in columns]
    record = np.dtype({"names": names, "formats": formats})
    table = np.zeros(n_points, dtype=record)
    for identifier, values in columns.items():
        table[f"c{identifier}"] = values
    if flags:
        packed = np.zeros(n_points, dtype=np.uint8)
        for identifier, values in flags.items():
            _, mask, shift = MPR_FLAG_COLUMNS[identifier]
            packed |= ((np.asarray(values, dtype=np.uint8) << shift) & mask)
        table["_flags"] = packed
    head = struct.pack("<I", n_points) + bytes([len(ids)])
    head += b"".join(struct.pack("<H", i) for i in ids)
    head += b"\x00" * max(0, header - len(head))
    return head + table.tobytes()


def _mpr(columns: dict[int, np.ndarray], header: int = 0x195,
         settings: bytes = b"Cyclic Voltammetry\x00",
         flags: dict[int, np.ndarray] | None = None) -> bytes:
    return (
        b"BIO-LOGIC MODULAR FILE\x1a".ljust(52, b"\x00")
        + _module("VMP Set", "VMP settings", settings)
        + _module("VMP data", "VMP data",
                  _data_module(columns, header, flags))
    )


#: An EC-Lab text export, as the instrument writes it: the identifier,
#: the declared header length, the technique, the parameter table, the
#: column names and the tab-separated numbers.
def _mpt(rows: list[list[str]], names: list[str],
         technique: str = "Cyclic Voltammetry",
         parameters: tuple[tuple[str, str], ...] = (("dE/dt", "50.000"),
                                                    ("dE/dt unit", "mV/s")),
         declared: int | None = None,
         identifier: str = "EC-Lab ASCII FILE") -> str:
    header = [identifier, "Nb header lines : {count}", "", technique, ""]
    header += [f"{name:<20}{value:<20}" for name, value in parameters]
    header.append("")
    count = declared if declared is not None else len(header) + 1
    header[1] = header[1].format(count=count)
    body = ["\t".join(names)] + ["\t".join(row) for row in rows]
    return "\r\n".join(header + body) + "\r\n"


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


# -- the flag byte, which is six columns in one -------------------------

def test_six_of_the_columns_are_bits_of_one_byte(tmp_path):
    """The thing that made the user's own files unreadable.

    Mode, sweep direction, error and the three counters are not six
    columns, they are bits of a single byte. Counted as a byte each the
    record came out five bytes too wide, the arithmetic did not close,
    and because three of the six were missing from the table as well the
    file was refused for «unknown identifiers» — two wrongs hiding each
    other. Established against the ASCII export of the same run.
    """
    n = 60
    flags = {
        1: np.full(n, 2),                       # mode, two bits wide
        2: np.tile([0, 1], n // 2),             # ox/red
        3: np.zeros(n, dtype=int),              # error
        21: np.concatenate([[1], np.zeros(n - 1, dtype=int)]),
        65: np.zeros(n, dtype=int),
    }
    columns = _cv_columns(n)
    path = tmp_path / "banderas.mpr"
    path.write_bytes(_mpr(columns, flags=flags))

    data = read_mpr(path)
    assert data.n_points == n
    for identifier, values in flags.items():
        name = MPR_FLAG_COLUMNS[identifier][0]
        assert name in data.columns, f"falta la bandera {name}"
        assert data.columns[name] == pytest.approx(values)
    # And the real columns are still where they belong, which is the
    # point: a flag counted as a whole byte shifts every one of them.
    assert data.columns["time/s"] == pytest.approx(columns[4])
    assert data.columns["Ewe/V"] == pytest.approx(columns[6], abs=1e-6)


def test_the_flag_byte_counts_once_towards_the_record(tmp_path):
    from ramancarbon.echem.biologic import _record_width

    assert _record_width([1, 2, 3, 21, 65]) == 1
    assert _record_width([1, 2, 3, 21, 65, 4]) == 9
    assert _record_width([4, 6]) == 12


# -- the .mpt, which is the reproducible route --------------------------

_MPT_NAMES = ["mode", "ox/red", "error", "time/s", "Ewe/V", "<I>/mA",
              "(Q-Qo)/C"]


def _mpt_rows(n: int = 40) -> list[list[str]]:
    rows = []
    for index in range(n):
        time = 0.02 * index
        potential = -0.2 + 0.001 * index
        rows.append(["2", "1", "0", f"{time:.6E}", f"{potential:.6E}",
                     f"{1e-3 * index:.6E}", f"{1e-4 * index:.6E}"])
    return rows


def test_an_mpt_is_read_with_nothing_reverse_engineered(tmp_path):
    """Named, delimited columns with the unit inside the name.

    This is the route the module has recommended from the start, and the
    reason it matters is in the header: the scan rate is DECLARED, so a
    voltammogram does not need it typed in and nothing is deduced from
    |dE/dt|.
    """
    path = tmp_path / "medida.mpt"
    path.write_text(_mpt(_mpt_rows(), _MPT_NAMES), encoding="latin-1")

    data = read_mpt(path)
    assert data.n_points == 40
    assert data.columns["Ewe/V"][0] == pytest.approx(-0.2)
    assert data.columns["time/s"][-1] == pytest.approx(0.78)
    # The charge column keeps EC-Lab's own name, unit included: nothing
    # downstream reads it, and converting it would make the same
    # experiment arrive differently from its own .mpr.
    assert data.columns["(Q-Qo)/C"][10] == pytest.approx(1e-3)
    settings = data.settings["ajustes"]
    assert settings.technique == "cv"
    assert settings.scan_rate_v_per_s == pytest.approx(0.05)
    assert any("reproducible" in w for w in data.warnings)

    curve, notes = read_eclab(path)
    assert isinstance(curve, Voltammogram)
    assert curve.scan_rate == pytest.approx(0.05)
    assert "aviso_velocidad" not in curve.metadata, (
        "con la velocidad declarada en la cabecera no hay nada que deducir")
    assert notes


def test_a_declared_header_length_that_is_wrong_is_refused(tmp_path):
    """One line off reads the column names as data, or the first
    measurement as a name, and both look like data afterwards.

    The right count is found from the file the fixture writes rather than
    written down here, so the test keeps testing the off-by-one and not
    the fixture's own shape.
    """
    rows = _mpt_rows()
    text = _mpt(rows, _MPT_NAMES)
    lines = text.splitlines()
    correct = lines.index("\t".join(_MPT_NAMES)) + 1

    good = tmp_path / "bien.mpt"
    good.write_text(_mpt(rows, _MPT_NAMES, declared=correct),
                    encoding="latin-1")
    assert read_mpt(good).n_points == len(rows)

    for wrong in (correct - 1, correct + 1, 999):
        path = tmp_path / f"mal{wrong}.mpt"
        path.write_text(_mpt(rows, _MPT_NAMES, declared=wrong),
                        encoding="latin-1")
        with pytest.raises(BioLogicError):
            read_mpt(path)


def test_a_truncated_row_is_refused_by_its_number(tmp_path):
    rows = _mpt_rows()
    rows[7] = rows[7][:4]
    path = tmp_path / "truncada.mpt"
    path.write_text(_mpt(rows, _MPT_NAMES), encoding="latin-1")
    with pytest.raises(BioLogicError, match="campos"):
        read_mpt(path)


def test_a_column_this_module_does_not_know_costs_only_itself(tmp_path):
    """The asymmetry with the .mpr, and it is not an inconsistency.

    In the binary the columns are positional, so an unknown identifier
    shifts everything behind it and the file has to be refused. In the
    text they are named and delimited, so an unknown one is kept under
    its own name, reported, and nothing moves.
    """
    names = _MPT_NAMES + ["Quelonio/parsec"]
    rows = [row + ["1.0"] for row in _mpt_rows()]
    path = tmp_path / "rara.mpt"
    path.write_text(_mpt(rows, names), encoding="latin-1")

    data = read_mpt(path)
    assert data.columns["Ewe/V"][0] == pytest.approx(-0.2)
    assert "Quelonio/parsec" in data.columns
    assert any("Quelonio/parsec" in w for w in data.warnings)


def test_a_comma_decimal_mark_is_read(tmp_path):
    """The delimiter is a tab, so a comma inside a field cannot be
    anything but a decimal mark. That is not true of a comma-delimited
    export, which is why this is safe here and nowhere else."""
    rows = [[cell.replace(".", ",") for cell in row] for row in _mpt_rows()]
    path = tmp_path / "coma.mpt"
    path.write_text(_mpt(rows, _MPT_NAMES), encoding="latin-1")
    data = read_mpt(path)
    assert data.columns["Ewe/V"][0] == pytest.approx(-0.2)


def test_something_that_is_not_an_mpt_says_so(tmp_path):
    path = tmp_path / "otra.mpt"
    path.write_text("tiempo\tpotencial\n0\t0\n1\t1\n", encoding="latin-1")
    with pytest.raises(BioLogicError, match="EC-LAB ASCII FILE"):
        read_mpt(path)


def test_the_detector_names_the_technique_the_header_declares(tmp_path):
    """The header says what it is, so the kind is not inferred from the
    shape of the columns — which read the user's galvanostatic file as a
    Raman MAP and his voltammogram as «desconocido»."""
    from ramancarbon.dataio.detect import detect

    path = tmp_path / "cv.mpt"
    path.write_text(_mpt(_mpt_rows(), _MPT_NAMES), encoding="latin-1")
    found = detect(path)
    assert found.kind == "cv" and found.fmt == "mpt"
    assert found.confidence == "alta"

    gcpl = tmp_path / "gcpl.mpt"
    gcpl.write_text(
        _mpt(_mpt_rows(), _MPT_NAMES,
             technique="Galvanostatic Cycling with Potential Limitation"),
        encoding="latin-1")
    assert detect(gcpl).kind == "gcd"


def test_a_galvanostatic_file_with_no_measured_current_still_reads(tmp_path):
    """The current is the quantity being CONTROLLED, so there is no
    measured current column at all: EC-Lab keeps it in `control/V/mA`,
    whose unit depends on the mode. The user's GCPL .mpr is exactly that
    — fifteen columns, Ewe/V among them, nothing `to_curve` would call a
    current — so it decoded cleanly and then came back «faltan columnas
    de potencial o de corriente».

    What is taken is the SETPOINT, and the note has to say so: its sign
    separates charge from discharge, which is what the curve needs, but
    its magnitude is what was asked for and not what was delivered.
    """
    n = 120
    time = np.linspace(0.0, 600.0, n)
    potential = np.concatenate([np.linspace(0.0, 1.2, n // 2),
                               np.linspace(1.2, 0.0, n - n // 2)])
    setpoint = np.concatenate([np.full(n // 2, 0.05),
                               np.full(n - n // 2, -0.05)])     # mA
    path = tmp_path / "gcpl.mpr"
    path.write_bytes(_mpr(
        {4: time, 6: potential.astype("f4"), 5: setpoint.astype("f4")},
        settings=b"Galvanostatic Cycling with Potential Limitation\x00",
        flags={1: np.ones(n, dtype=int)},               # galvanostatic
    ))

    curve, notes = read_eclab(path)
    assert isinstance(curve, ChargeDischarge)
    assert curve.current == pytest.approx(setpoint * 1e-3)
    assert np.any(curve.current > 0) and np.any(curve.current < 0), (
        "el signo es lo que separa carga de descarga")
    assert any("consigna" in note for note in notes)


def test_the_control_column_is_not_taken_as_current_when_the_mode_says_volts(
        tmp_path):
    """`control/V/mA` is volts when the potential is the controlled
    quantity, and reading those as milliamps would be a silent factor of
    a thousand on something that is not even a current."""
    n = 80
    time = np.linspace(0.0, 60.0, n)
    potential = np.linspace(0.0, 0.6, n)
    path = tmp_path / "potenciostatico.mpr"
    path.write_bytes(_mpr(
        {4: time, 6: potential.astype("f4"), 5: potential.astype("f4")},
        flags={1: np.full(n, 2)},                       # potentiostatic
    ))
    with pytest.raises(BioLogicError, match="corriente"):
        read_eclab(path)
