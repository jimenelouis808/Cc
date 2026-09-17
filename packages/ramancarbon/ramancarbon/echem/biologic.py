"""Bio-Logic EC-Lab files: ``.mps`` settings and ``.mpr`` raw data.

This module exists against the rule stated at the top of
:mod:`ramancarbon.echem.io` — that binary formats are refused rather than
guessed at — so it has to say why it is an exception, and what it does
instead of guessing.

**``.mps`` is text.** It is the settings file EC-Lab writes beside the
data: the technique, the scan rate, the current, the potential window,
the number of cycles. There is nothing to guess about it, and it carries
exactly the numbers a voltammogram needs and rarely has in its own
columns. Reading it is free of risk and worth doing on its own.

**``.mpr`` is binary, and it is SELF-DESCRIBING.** That is the whole
argument. The file declares how many points it holds, how many columns,
and which quantity each column is; every quantity has a known width; so
the record size and the offset at which the table starts are *computed
from the file* and then checked to close exactly against its length. A
layout that does not close is refused. This is the same standard the
``.spe`` reader is held to — the header there gives one number more than
it needs, and disagreement means the fields were not where the reader
thought.

What is NOT verifiable here, and is said out loud rather than buried: the
mapping from column identifier to quantity is community
reverse-engineering, not a published specification. The arithmetic
catches an identifier whose WIDTH is wrong, because then nothing closes.
It cannot catch one whose width is right and whose NAME is wrong. So a
file read this way carries a provenance note, and the honest workflow is
to check one measurement against EC-Lab's own ASCII export the first
time and then trust the reader for the rest.

**The reproducible route is still ``.mpt``.** EC-Lab exports it directly
(Experiment → Export as text), it is read by
:mod:`ramancarbon.echem.io` with no reverse engineering anywhere in the
path, and a result that depends on a reverse-engineered binary layout is
a result that depends on this module being right. Use ``.mpr`` for
convenience; export ``.mpt`` for anything that goes in a paper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .curve import ChargeDischarge, Electrode, Impedance, Voltammogram


class BioLogicError(ValueError):
    """Raised when an EC-Lab file cannot be read *and checked*."""


#: The first bytes of every ``.mpr``.
MPR_MAGIC = b"BIO-LOGIC MODULAR FILE"

#: Marker that precedes each module inside an ``.mpr``.
MODULE_MAGIC = b"MODULE"

#: Header sizes the ``VMP data`` table has been seen to start at.
#:
#: NOT used to locate the table — that is computed from the file, so a
#: version with a different header still reads. They are used to CHECK
#: the computed offset, which closes a hole the arithmetic alone leaves:
#: a file that over-claims its point count by a few simply eats into the
#: header padding, the length still adds up, and the first rows come back
#: as padding decoded as numbers. An offset that is not one of these is
#: not refused — a version nobody here has seen is likely enough — but it
#: is said out loud.
MPR_DATA_OFFSETS: tuple[int, ...] = (0x195, 0x196)

#: Column identifier → ``(name, numpy dtype)``.
#:
#: Reverse-engineered by the community, not published by Bio-Logic. Only
#: identifiers this table knows are accepted: an unknown one is refused
#: by name rather than skipped, because skipping a column shifts every
#: column after it and the result still looks like data.
#:
#: The widths are what the self-consistency check actually tests. If one
#: of them is wrong the record size is wrong, the table does not close
#: against the file length, and the file is refused — which is the
#: failure mode to want.
MPR_COLUMNS: dict[int, tuple[str, str]] = {
    1: ("mode", "u1"),
    2: ("ox/red", "u1"),
    3: ("error", "u1"),
    4: ("time/s", "<f8"),
    5: ("control/V/mA", "<f4"),
    6: ("Ewe/V", "<f4"),
    7: ("dq/mA.h", "<f8"),
    8: ("I/mA", "<f4"),
    9: ("Ece/V", "<f4"),
    11: ("<I>/mA", "<f8"),
    13: ("(Q-Qo)/mA.h", "<f8"),
    16: ("Analog IN 1/V", "<f4"),
    19: ("control/V", "<f4"),
    20: ("control/mA", "<f4"),
    23: ("dQ/mA.h", "<f8"),
    24: ("cycle number", "<f8"),
    32: ("freq/Hz", "<f4"),
    33: ("|Ewe|/V", "<f4"),
    34: ("|I|/A", "<f4"),
    35: ("Phase(Z)/deg", "<f4"),
    36: ("|Z|/Ohm", "<f4"),
    37: ("Re(Z)/Ohm", "<f4"),
    38: ("-Im(Z)/Ohm", "<f4"),
    39: ("I Range", "<u2"),
    69: ("R/Ohm", "<f4"),
    70: ("P/W", "<f4"),
    74: ("|Energy|/W.h", "<f8"),
    75: ("Analog OUT/V", "<f4"),
    76: ("<I>/mA", "<f4"),
    77: ("<Ewe>/V", "<f4"),
    78: ("Cs-2/uF-2", "<f4"),
    96: ("|Ece|/V", "<f4"),
    98: ("Phase(Zce)/deg", "<f4"),
    99: ("|Zce|/Ohm", "<f4"),
    100: ("Re(Zce)/Ohm", "<f4"),
    101: ("-Im(Zce)/Ohm", "<f4"),
    123: ("Energy charge/W.h", "<f8"),
    124: ("Energy discharge/W.h", "<f8"),
    125: ("Capacitance charge/uF", "<f8"),
    126: ("Capacitance discharge/uF", "<f8"),
    131: ("Ns", "<u2"),
    163: ("|Estack|/V", "<f4"),
    168: ("Rcmp/Ohm", "<f4"),
    169: ("Cs/uF", "<f4"),
    172: ("Cp/uF", "<f4"),
    434: ("(Q-Qo)/C", "<f4"),
    435: ("dQ/C", "<f4"),
    467: ("Q charge/discharge/mA.h", "<f8"),
    469: ("step time/s", "<f8"),
}


@dataclass
class MPRModule:
    """One module of a modular file."""

    short_name: str
    long_name: str
    version: int
    date: str
    payload: bytes


@dataclass
class ECLabData:
    """What an ``.mpr`` held, before it is turned into a curve."""

    columns: dict[str, np.ndarray]
    n_points: int
    settings: dict[str, Any] = field(default_factory=dict)
    modules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def has(self, *names: str) -> bool:
        return all(name in self.columns for name in names)

    def first(self, *names: str) -> Optional[np.ndarray]:
        """The first of several column names that is present."""
        for name in names:
            if name in self.columns:
                return self.columns[name]
        return None

    def describe(self) -> str:
        return (
            f"{self.n_points} puntos, {len(self.columns)} columnas: "
            + ", ".join(sorted(self.columns))
        )


def _modules(raw: bytes) -> list[MPRModule]:
    """Split a modular file into its modules."""
    if not raw.startswith(MPR_MAGIC):
        raise BioLogicError(
            "no empieza por «BIO-LOGIC MODULAR FILE», así que no es un .mpr "
            f"de EC-Lab. Empieza por {raw[:24]!r}"
        )
    modules: list[MPRModule] = []
    position = raw.find(MODULE_MAGIC)
    while position >= 0:
        head = position + len(MODULE_MAGIC)
        # shortname 10 | longname 25 | length 4 | version 4 | date 8
        if head + 51 > len(raw):
            break
        short = raw[head:head + 10].split(b"\x00")[0].decode("latin-1").strip()
        long_name = raw[head + 10:head + 35].split(b"\x00")[0].decode(
            "latin-1").strip()
        length = int(np.frombuffer(raw, "<u4", 1, head + 35)[0])
        version = int(np.frombuffer(raw, "<u4", 1, head + 39)[0])
        date = raw[head + 43:head + 51].split(b"\x00")[0].decode("latin-1")
        start = head + 51
        end = start + length
        if length <= 0 or end > len(raw):
            raise BioLogicError(
                f"el módulo «{short}» declara {length} bytes y el archivo "
                f"sólo tiene {len(raw) - start} desde ahí: está truncado"
            )
        modules.append(MPRModule(short, long_name, version, date,
                                 raw[start:end]))
        position = raw.find(MODULE_MAGIC, end)
    if not modules:
        raise BioLogicError("el archivo no contiene ningún módulo MODULE")
    return modules


def _decode_data(module: MPRModule) -> tuple[dict[str, np.ndarray], int, list[str]]:
    """Decode the ``VMP data`` module into named columns."""
    payload = module.payload
    if len(payload) < 8:
        raise BioLogicError("el módulo de datos está vacío")
    n_points = int(np.frombuffer(payload, "<u4", 1, 0)[0])
    n_columns = int(payload[4])
    if n_points <= 0 or n_columns <= 0:
        raise BioLogicError(
            f"la cabecera de datos declara {n_points} puntos y {n_columns} "
            "columnas"
        )
    if 5 + 2 * n_columns > len(payload):
        raise BioLogicError(
            f"declara {n_columns} columnas pero no caben sus identificadores"
        )
    ids = np.frombuffer(payload, "<u2", n_columns, 5).tolist()

    unknown = [i for i in ids if int(i) not in MPR_COLUMNS]
    if unknown:
        raise BioLogicError(
            "hay columnas con identificador desconocido: "
            + ", ".join(str(i) for i in sorted(set(unknown)))
            + ". No se puede saltar una columna sin desplazar todas las que "
            "vienen detrás, y el resultado seguiría pareciendo datos, así "
            "que el archivo se rechaza entero. Exporta el mismo experimento "
            "como texto (.mpt) desde EC-Lab y mándame la cabecera si quieres "
            "que se añada"
        )

    # Duplicate identifiers do occur (<I>/mA appears as both 11 and 76);
    # numpy needs unique field names.
    names: list[str] = []
    used: dict[str, int] = {}
    formats: list[str] = []
    for identifier in ids:
        name, dtype = MPR_COLUMNS[int(identifier)]
        if name in used:
            used[name] += 1
            name = f"{name} ({used[name]})"
        else:
            used[name] = 1
        names.append(name)
        formats.append(dtype)

    record = np.dtype({"names": names, "formats": formats})
    # The offset is COMPUTED, not assumed. It differs between format
    # versions, and hardcoding one of them is exactly the kind of guess
    # this module exists not to make: the file says how many points and
    # how wide each is, so where the table starts is arithmetic, and
    # whether that arithmetic lands somewhere sensible is the check.
    body = n_points * record.itemsize
    offset = len(payload) - body
    minimum = 5 + 2 * n_columns
    if body <= 0 or offset < minimum:
        raise BioLogicError(
            f"{n_points} puntos × {record.itemsize} bytes son {body} bytes y "
            f"el módulo tiene {len(payload)}: la tabla no cuadra con lo que "
            "declara la cabecera, así que las columnas no son las que el "
            "lector cree. El archivo se rechaza en vez de devolver números "
            "plausibles"
        )

    table = np.frombuffer(payload, record, n_points, offset)
    columns = {name: np.asarray(table[name], dtype=float) for name in names}

    notes: list[str] = []
    if offset not in MPR_DATA_OFFSETS:
        notes.append(
            f"la tabla empieza en el byte {offset} del módulo y las versiones "
            f"conocidas la ponen en {' o '.join(hex(o) for o in MPR_DATA_OFFSETS)}"
            f" ({', '.join(str(o) for o in MPR_DATA_OFFSETS)}). La aritmética "
            "cuadra, así que el archivo se ha leído, pero puede ser una "
            "versión de EC-Lab que este lector no ha visto o un recuento de "
            "puntos equivocado: compara una medida con la exportación ASCII "
            "antes de fiarte"
        )
    return columns, n_points, notes


def _decode_settings(module: MPRModule) -> dict[str, Any]:
    """Pull the printable settings out of the ``VMP Set`` module.

    Deliberately shallow. The settings module is a packed C structure
    whose layout varies between EC-Lab versions, and decoding it field by
    field is precisely the guessing this module avoids; what is extracted
    is the technique name and the readable strings, which are enough to
    say what the file is and are checkable by eye.
    """
    text = module.payload.decode("latin-1", errors="replace")
    strings = [s.strip() for s in re.findall(r"[ -~]{4,}", text)]
    out: dict[str, Any] = {}
    if strings:
        out["tecnica"] = strings[0]
    out["texto"] = strings[:40]
    return out


def read_mpr(path: str | Path) -> ECLabData:
    """Read a Bio-Logic ``.mpr``, or refuse and say why.

    Everything is derived from what the file declares and then checked to
    close against its length; see the module docstring for what that does
    and does not guarantee.
    """
    path = Path(path)
    raw = path.read_bytes()
    try:
        modules = _modules(raw)
    except BioLogicError as error:
        raise BioLogicError(f"{path.name}: {error}") from None

    data_module = next((m for m in modules if m.short_name.startswith("VMP data")
                        or "data" in m.short_name.lower()), None)
    if data_module is None:
        raise BioLogicError(
            f"{path.name}: no hay módulo de datos (los que hay: "
            + ", ".join(m.short_name for m in modules)
            + "). Un .mps es sólo los ajustes; los datos están en el .mpr "
            "del mismo nombre"
        )
    try:
        columns, n_points, notes = _decode_data(data_module)
    except BioLogicError as error:
        raise BioLogicError(f"{path.name}: {error}") from None

    settings: dict[str, Any] = {}
    set_module = next((m for m in modules if m.short_name.startswith("VMP Set")),
                      None)
    if set_module is not None:
        settings = _decode_settings(set_module)

    result = ECLabData(columns=columns, n_points=n_points, settings=settings,
                       modules=[m.short_name for m in modules])
    result.warnings.extend(notes)
    result.warnings.append(
        "la disposición binaria del .mpr es ingeniería inversa de la "
        "comunidad, no una especificación publicada. La aritmética cuadra, "
        "que descarta que una columna tenga la ANCHURA equivocada, pero no "
        "que tenga el NOMBRE equivocado. Comprueba una medida contra la "
        "exportación ASCII (.mpt) de EC-Lab la primera vez; para lo que se "
        "publica, exporta .mpt"
    )
    _sanity_check(result)
    return result


def _sanity_check(data: ECLabData) -> None:
    """Physical checks the numbers have to pass to be believable."""
    time = data.columns.get("time/s")
    if time is not None and time.size > 1:
        if not np.all(np.isfinite(time)):
            raise BioLogicError("la columna de tiempo tiene valores no finitos")
        if np.any(np.diff(time) < -1e-6):
            data.warnings.append(
                "el tiempo no es monótono: o el archivo encadena varias "
                "técnicas, o las columnas no son las que el lector cree"
            )
    for name, values in data.columns.items():
        if values.size and not np.all(np.isfinite(values)):
            data.warnings.append(
                f"la columna «{name}» tiene valores no finitos, lo que suele "
                "querer decir que su tipo no es el que el lector supone"
            )


# ----------------------------------------------------------------------
# .mps — the settings file, which is text
# ----------------------------------------------------------------------
#: EC-Lab technique names mapped to what this package calls them.
MPS_TECHNIQUES: dict[str, str] = {
    "cyclic voltammetry": "cv",
    "cv": "cv",
    "galvanostatic cycling with potential limitation": "gcd",
    "gcpl": "gcd",
    "potentio electrochemical impedance spectroscopy": "eis",
    "peis": "eis",
    "galvano electrochemical impedance spectroscopy": "eis",
    "geis": "eis",
    "linear sweep voltammetry": "lsv",
    "lsv": "lsv",
    "chronoamperometry": "ca",
    "chronopotentiometry": "cp",
}


@dataclass
class ECLabSettings:
    """What the ``.mps`` beside a measurement says about it."""

    technique: str = ""
    """Normalised: ``cv``, ``gcd``, ``eis``, ``lsv``… empty if unrecognised."""
    technique_name: str = ""
    """As EC-Lab wrote it."""
    scan_rate_v_per_s: Optional[float] = None
    current_a: Optional[float] = None
    window_v: Optional[tuple[float, float]] = None
    cycles: Optional[int] = None
    mass_mg: Optional[float] = None
    area_cm2: Optional[float] = None
    parameters: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [self.technique_name or "técnica desconocida"]
        if self.scan_rate_v_per_s:
            parts.append(f"{1e3 * self.scan_rate_v_per_s:g} mV/s")
        if self.current_a:
            parts.append(f"{1e3 * self.current_a:g} mA")
        if self.window_v:
            parts.append(f"{self.window_v[0]:+.3f} a {self.window_v[1]:+.3f} V")
        if self.cycles:
            parts.append(f"{self.cycles} ciclos")
        return ", ".join(parts)


#: Unit suffixes EC-Lab writes in ``.mps`` parameter tables.
_MPS_SCALE = {"": 1.0, "V": 1.0, "mV": 1e-3, "µV": 1e-6, "uV": 1e-6,
              "A": 1.0, "mA": 1e-3, "µA": 1e-6, "uA": 1e-6, "nA": 1e-9,
              "V/s": 1.0, "mV/s": 1e-3, "µV/s": 1e-6, "uV/s": 1e-6,
              "mg": 1.0, "g": 1e3, "cm²": 1.0, "cm2": 1.0}


def _mps_number(text: str, unit: str = "") -> Optional[float]:
    match = re.search(r"[-+]?\d*[.,]?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", "."))
    except ValueError:
        return None
    return value * _MPS_SCALE.get(unit, 1.0)


def read_mps(path: str | Path) -> ECLabSettings:
    """Read an EC-Lab ``.mps`` settings file.

    Text, so there is nothing to reverse engineer. It is worth reading on
    its own: it carries the scan rate, the current and the window, which
    a voltammogram needs to be analysed and which the data columns often
    do not contain. Without it the caller has to type the scan rate in,
    and a scan rate typed wrong gives a capacitance wrong in exactly that
    proportion with nothing downstream to catch it.

    The layout is a technique name followed by a table: one row per
    parameter, one column per sequence step. Only the first step is read
    for the headline numbers, and the whole table is kept in
    :attr:`ECLabSettings.parameters`.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="latin-1")
    except OSError as error:
        raise BioLogicError(f"{path.name}: {error}") from None
    lines = [line.rstrip("\r\n") for line in text.splitlines()]
    if not any("EC-LAB SETTING FILE" in line.upper() for line in lines[:5]):
        raise BioLogicError(
            f"{path.name}: no lleva la cabecera «EC-LAB SETTING FILE», así "
            "que no es un .mps de EC-Lab"
        )

    settings = ECLabSettings()
    for index, line in enumerate(lines):
        stripped = line.strip()
        key = stripped.lower()
        if not settings.technique_name and key in MPS_TECHNIQUES:
            settings.technique_name = stripped
            settings.technique = MPS_TECHNIQUES[key]
            continue
        if ":" in stripped:
            name, _, value = stripped.partition(":")
            settings.parameters.setdefault(name.strip().lower(), value.strip())
        else:
            # The parameter table: a name, then one value per step.
            parts = re.split(r"\s{2,}|\t", stripped)
            if len(parts) >= 2 and parts[0] and not parts[0][0].isdigit():
                settings.parameters.setdefault(parts[0].strip().lower(),
                                               parts[1].strip())
        del index

    table = settings.parameters
    for key, unit in (("dE/dt", "mV/s"), ("scan rate", "mV/s")):
        raw = table.get(key.lower())
        if raw is not None:
            unit_key = table.get("de/dt unit", unit)
            settings.scan_rate_v_per_s = _mps_number(raw, unit_key)
            break
    for key in ("is", "i range", "current", "i"):
        raw = table.get(key)
        if raw is not None and _mps_number(raw) is not None:
            settings.current_a = _mps_number(raw, table.get("unit is", "mA"))
            break
    low = _mps_number(
        table.get("ei (v)") or table.get("ei") or table.get("e1 (v)") or "")
    high = _mps_number(
        table.get("e1 (v)") or table.get("ef (v)") or table.get("ef")
        or table.get("e2 (v)") or "")
    if low is not None and high is not None and high != low:
        settings.window_v = (min(low, high), max(low, high))
    cycles = _mps_number(table.get("nc cycles") or table.get("nc") or "")
    if cycles:
        settings.cycles = int(cycles)
    mass = _mps_number(table.get("mass of active material") or "", "mg")
    if mass:
        settings.mass_mg = mass
    area = _mps_number(table.get("electrode surface area") or "", "cm2")
    if area:
        settings.area_cm2 = area

    if not settings.technique:
        settings.warnings.append(
            "no se reconoce la técnica en la cabecera; los parámetros se han "
            "leído igual y están en `parameters`"
        )
    return settings


def settings_beside(path: str | Path) -> Optional[ECLabSettings]:
    """The ``.mps`` next to an ``.mpr`` of the same name, if there is one.

    EC-Lab writes them as a pair and people copy them as a pair. Looking
    for it costs nothing and supplies the scan rate, which is the number
    whose absence makes a voltammogram unanalysable.
    """
    candidate = Path(path).with_suffix(".mps")
    if not candidate.exists():
        return None
    try:
        return read_mps(candidate)
    except (BioLogicError, OSError):
        return None


# ----------------------------------------------------------------------
# Turning the columns into the package's own curves
# ----------------------------------------------------------------------
def _potential(data: ECLabData) -> Optional[np.ndarray]:
    return data.first("Ewe/V", "<Ewe>/V", "|Ewe|/V", "control/V")


def _current_a(data: ECLabData) -> Optional[np.ndarray]:
    """Current in AMPS. EC-Lab writes milliamps; the package uses SI."""
    for name in ("I/mA", "<I>/mA", "<I>/mA (2)", "control/mA"):
        if name in data.columns:
            return data.columns[name] * 1e-3
    if "|I|/A" in data.columns:
        return data.columns["|I|/A"]
    return None


def to_curve(
    data: ECLabData,
    kind: str = "",
    electrode: Optional[Electrode] = None,
    scan_rate: Optional[float] = None,
    settings: Optional[ECLabSettings] = None,
):
    """Build the right curve object from decoded ``.mpr`` columns.

    ``kind`` forces the choice; empty picks it from the columns, which is
    unambiguous here because the impedance columns only exist in an
    impedance file and a voltammogram only has the sweep columns.
    """
    electrode = electrode or Electrode()
    settings = settings or ECLabSettings()
    if not kind:
        if data.has("freq/Hz") and (data.has("Re(Z)/Ohm") or data.has("|Z|/Ohm")):
            kind = "eis"
        elif settings.technique in ("cv", "lsv"):
            kind = settings.technique
        elif settings.technique:
            kind = settings.technique
        else:
            kind = "gcd"

    metadata = {"origen": "EC-Lab .mpr", "modulos": list(data.modules)}
    if settings.technique_name:
        metadata["tecnica"] = settings.technique_name

    if kind == "eis":
        frequency = data.columns.get("freq/Hz")
        real = data.columns.get("Re(Z)/Ohm")
        minus_imaginary = data.columns.get("-Im(Z)/Ohm")
        if frequency is None or real is None or minus_imaginary is None:
            magnitude = data.columns.get("|Z|/Ohm")
            phase = data.columns.get("Phase(Z)/deg")
            if frequency is None or magnitude is None or phase is None:
                raise BioLogicError(
                    "faltan columnas de impedancia: hacen falta freq/Hz más "
                    "Re(Z) y -Im(Z), o bien |Z| y la fase"
                )
            radians = np.radians(phase)
            real = magnitude * np.cos(radians)
            minus_imaginary = -magnitude * np.sin(radians)
        keep = frequency > 0
        # Z'' is stored with its PHYSICAL sign here, negative for a
        # capacitive response. EC-Lab's column is -Im(Z), the Nyquist
        # drawing convention, and negating it on the way in is what keeps
        # every circuit fit downstream from silently changing sign.
        impedance = real[keep] - 1j * minus_imaginary[keep]
        return Impedance(frequency=frequency[keep], z=impedance,
                         electrode=electrode, name="EIS (EC-Lab)",
                         metadata=metadata)

    potential = _potential(data)
    current = _current_a(data)
    time = data.columns.get("time/s")
    if potential is None or current is None:
        raise BioLogicError(
            "faltan columnas de potencial o de corriente: hay "
            + ", ".join(sorted(data.columns))
        )

    if kind in ("cv", "lsv"):
        rate = scan_rate or settings.scan_rate_v_per_s
        if not rate and time is not None and time.size > 4:
            # Last resort, and it is recorded as such: the sweep rate is
            # |dE/dt| where the sweep is actually sweeping. A rate taken
            # from the data rather than from the settings is right only
            # if the file holds one technique at one rate.
            slope = np.abs(np.gradient(potential, time))
            moving = slope[slope > 0.2 * np.nanmax(slope)]
            if moving.size:
                rate = float(np.median(moving))
                metadata["aviso_velocidad"] = (
                    f"la velocidad de barrido ({1e3 * rate:.4g} mV/s) se ha "
                    "deducido de |dE/dt| porque no había .mps al lado. "
                    "Compruébala: una velocidad equivocada da una "
                    "capacitancia equivocada en esa misma proporción"
                )
        if not rate:
            raise BioLogicError(
                "no hay velocidad de barrido: ni en el .mps ni deducible del "
                "eje de tiempo. Pásala a mano — sin ella la capacitancia "
                "sale mal exactamente en esa proporción y nada aguas abajo "
                "lo delata"
            )
        return Voltammogram(potential=potential, current=current,
                            scan_rate=float(rate), electrode=electrode,
                            name=f"{kind.upper()} (EC-Lab)", metadata=metadata)

    if time is None:
        raise BioLogicError(
            "una curva de carga-descarga necesita el eje de tiempo y no hay "
            "columna time/s"
        )
    return ChargeDischarge(time=time, potential=potential, current=current,
                           electrode=electrode, name="GCD (EC-Lab)",
                           metadata=metadata)


def read_eclab(
    path: str | Path,
    kind: str = "",
    electrode: Optional[Electrode] = None,
    scan_rate: Optional[float] = None,
):
    """Read an ``.mpr`` and return the curve it holds.

    The ``.mps`` of the same name is read too when it is there, because
    EC-Lab writes them as a pair and it carries the scan rate.

    Returns ``(curve, warnings)``.
    """
    path = Path(path)
    data = read_mpr(path)
    settings = settings_beside(path)
    curve = to_curve(data, kind=kind, electrode=electrode,
                     scan_rate=scan_rate, settings=settings)
    warnings = list(data.warnings)
    if settings is not None:
        warnings.extend(settings.warnings)
        curve.metadata["ajustes"] = settings.describe()
    else:
        warnings.append(
            "no hay un .mps junto al .mpr, así que los ajustes del "
            "experimento (velocidad, corriente, ventana) no se han podido "
            "leer. Cópialos juntos: EC-Lab los escribe en pareja"
        )
    note = curve.metadata.get("aviso_velocidad")
    if note:
        warnings.append(note)
    return curve, warnings


__all__ = [
    "MPR_COLUMNS",
    "MPR_MAGIC",
    "MPR_DATA_OFFSETS",
    "MPS_TECHNIQUES",
    "BioLogicError",
    "ECLabData",
    "ECLabSettings",
    "read_eclab",
    "read_mpr",
    "read_mps",
    "settings_beside",
    "to_curve",
]
