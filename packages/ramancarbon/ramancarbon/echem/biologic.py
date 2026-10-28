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
from typing import Any, Optional, Sequence

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
#: Bytes before a module's payload in the layout this reader was first
#: written against: shortname 10, longname 25, length 4, version 4,
#: date 8.
MODULE_HEADER_SHORT = 51

#: And in the newer one, which inserts a marker and a reserved field:
#: shortname 10, longname 25, marker 4, length 4, reserved 4, version 4,
#: date 8.
MODULE_HEADER_LONG = 59

#: What stands where the length used to be in the newer layout.
#:
#: 0xFFFFFFFF, and it cannot be mistaken for a real length because no
#: module is four gigabytes -- which is exactly how the two layouts are
#: told apart without guessing a version number. Read with the old
#: offsets, this value WAS taken for a length, so every module declared
#: 4294967295 bytes and the file was refused as truncated.
MODULE_LENGTH_MARKER = 0xFFFFFFFF

#: Offsets inside the data module at which the column identifiers may
#: start: straight after the column count, or one pad byte later.
#:
#: Which one a file uses is decided BY THE FILE -- both are tried and
#: each has to produce known identifiers whose record width closes
#: against the module length. Newer EC-Lab uses the second, and read at
#: the first every identifier arrives multiplied by 256.
ID_OFFSETS = (5, 6)

#: header padding, the length still adds up, and the first rows come back
#: as padding decoded as numbers. An offset that is not one of these is
#: not refused — a version nobody here has seen is likely enough — but it
#: is said out loud.
MPR_DATA_OFFSETS: tuple[int, ...] = (0x195, 0x196, 0x3EF)

#: Identifiers that are BIT FIELDS of one shared byte, as
#: ``(name, mask, shift)``.
#:
#: This is not a detail, it is what made the user's own files unreadable.
#: Six of the columns EC-Lab lists — the mode, the sweep direction, the
#: error flag and the three counters — are not six columns: they are bits
#: of a single byte at the front of each record. Counted as a byte each,
#: the record came out five bytes too wide, the arithmetic did not close,
#: and because three of the six were missing from the table as well the
#: file was refused for «unknown identifiers» — two wrongs hiding each
#: other.
#:
#: Established against the ASCII export of the SAME run, which is the
#: flow this module has recommended from the start: the user's
#: ``CV_multpit_01_CV_C01`` as ``.mpr`` and as ``.mpt``. With the record
#: at 39 bytes and the table at byte 1007 of the module, every one of
#: these six comes back EQUAL to its column in the text file, and
#: ``time/s``, ``control/V``, ``Ewe/V``, ``<I>/mA``, ``cycle number``,
#: ``(Q-Qo)/C`` and ``I Range`` match theirs to float precision.
MPR_FLAG_COLUMNS: dict[int, tuple[str, int, int]] = {
    1: ("mode", 0x03, 0),
    2: ("ox/red", 0x04, 2),
    3: ("error", 0x08, 3),
    21: ("control changes", 0x10, 4),
    31: ("Ns changes", 0x20, 5),
    65: ("counter inc.", 0x80, 7),
}

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
    # Read off the user's GCPL file: with the six flags packed into one
    # byte the record closes at 49 bytes only with four more, the table
    # lands at the same byte 1007 as the CV file, and the column comes
    # back as a counter that rises 0, 0, 0, 1, 1, 1, … to 2001 — which is
    # what a half cycle is, and it is also where the ASCII export of that
    # sequence puts «half cycle» in the column order. Not compared value
    # by value against a text export, because the one the user has is the
    # NEXT technique of the same sequence: its time axis starts at 37102 s,
    # exactly where this file's ends.
    468: ("half cycle", "<u4"),
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
        if head + MODULE_HEADER_SHORT > len(raw):
            break
        short = raw[head:head + 10].split(b"\x00")[0].decode("latin-1").strip()
        long_name = raw[head + 10:head + 35].split(b"\x00")[0].decode(
            "latin-1").strip()
        # Two header layouts, told apart by the file itself.
        #
        # The one this reader was written against is
        #     shortname 10 | longname 25 | length 4 | version 4 | date 8
        # and newer EC-Lab writes eight bytes more:
        #     ... | 0xFFFFFFFF 4 | length 4 | reserved 4 | version 4 | date 8
        #
        # Reading the new layout with the old offsets takes the marker
        # for the length, so every module claimed 4294967295 bytes and
        # the file was refused as truncated. That is what the user's
        # three real files did -- a CV, a GCPL and a PEIS, none of which
        # would open.
        #
        # The marker is what distinguishes them, and it cannot be a
        # length: no module is four gigabytes. The arithmetic still has
        # to close against the file afterwards, which is the check this
        # module exists to make, and on those three files it closes to
        # the byte on every module.
        marker = int(np.frombuffer(raw, "<u4", 1, head + 35)[0])
        extended = marker == MODULE_LENGTH_MARKER
        if extended:
            if head + MODULE_HEADER_LONG > len(raw):
                break
            length = int(np.frombuffer(raw, "<u4", 1, head + 39)[0])
            version = int(np.frombuffer(raw, "<u4", 1, head + 47)[0])
            header = MODULE_HEADER_LONG
        else:
            length = marker
            version = int(np.frombuffer(raw, "<u4", 1, head + 39)[0])
            header = MODULE_HEADER_SHORT
        date = raw[head + header - 8:head + header].split(
            b"\x00")[0].decode("latin-1")
        start = head + header
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


#: Field name the shared flag byte is decoded under before being
#: unpacked. Not a column: it never reaches the caller.
_FLAG_FIELD = "_flags"


def _known_id(identifier: int) -> bool:
    """Whether this column identifier has a name here, flags included."""
    return identifier in MPR_COLUMNS or identifier in MPR_FLAG_COLUMNS


def _record_fields(ids: Sequence[int]) -> tuple[list[str], list[str], list[int]]:
    """``(names, formats, flag ids)`` for one record.

    The flag identifiers share ONE byte, so the first of them to appear
    reserves it and the rest add nothing to the record's width. Counting
    them as a byte each is what made the record five bytes too wide on
    the user's files.

    Duplicate identifiers do occur (``<I>/mA`` appears as both 11 and
    76); numpy needs unique field names.
    """
    names: list[str] = []
    formats: list[str] = []
    flags: list[int] = []
    used: dict[str, int] = {}
    for identifier in (int(i) for i in ids):
        if identifier in MPR_FLAG_COLUMNS:
            if not flags:
                names.append(_FLAG_FIELD)
                formats.append("u1")
            flags.append(identifier)
            continue
        name, dtype = MPR_COLUMNS[identifier]
        if name in used:
            used[name] += 1
            name = f"{name} ({used[name]})"
        else:
            used[name] = 1
        names.append(name)
        formats.append(dtype)
    return names, formats, flags


def _record_width(ids: Sequence[int]) -> int:
    """Bytes per point, with the flag byte counted once."""
    _, formats, _ = _record_fields([i for i in ids if _known_id(int(i))])
    return sum(np.dtype(fmt).itemsize for fmt in formats)


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
    # Where the identifier list starts is READ OFF the file, not assumed.
    #
    # Newer EC-Lab writes the column count as a 16-bit field, so the
    # identifiers begin one byte later than this reader looked. Read a
    # byte early, every identifier comes back multiplied by 256 -- 1, 2,
    # 3 arriving as 256, 512, 768 -- because each little-endian pair is
    # split across two real ones. All three of the user's files did
    # exactly that and were refused for "unknown identifiers" that were
    # simply misread.
    #
    # Each candidate offset has to produce identifiers this reader knows.
    # That is a separate question from whether the record then closes
    # against the module length, and keeping them separate is what lets
    # the two failures say different things: identifiers it cannot name,
    # or a table whose arithmetic does not add up.
    readable: list[tuple[int, list[int]]] = []
    for id_offset in ID_OFFSETS:
        if id_offset + 2 * n_columns > len(payload):
            continue
        candidate = np.frombuffer(payload, "<u2", n_columns, id_offset).tolist()
        if all(_known_id(int(i)) for i in candidate):
            readable.append((id_offset, candidate))

    if not readable:
        # Report the reading that got FURTHEST, not the first one tried:
        # at the wrong offset every identifier comes back multiplied by
        # 256, and a list of those blames the file for a misalignment
        # that belongs to the reader.
        best_offset, best_ids, best_score = ID_OFFSETS[0], [], -1
        for id_offset in ID_OFFSETS:
            if id_offset + 2 * n_columns > len(payload):
                continue
            probe = np.frombuffer(payload, "<u2", n_columns, id_offset).tolist()
            score = sum(1 for i in probe if _known_id(int(i)))
            if score > best_score:
                best_offset, best_ids, best_score = id_offset, probe, score
        unknown = sorted({int(i) for i in best_ids if not _known_id(int(i))})
        width = _record_width(best_ids)
        available = (
            (len(payload) - best_offset - 2 * n_columns) / n_points
            if n_points else 0.0
        )
        raise BioLogicError(
            "columnas con identificador desconocido: "
            + ", ".join(str(i) for i in unknown)
            + f" ({best_score} de {n_columns} sí se reconocen). Las "
            f"reconocidas ya ocupan {width} bytes por punto y el módulo sólo "
            f"tiene {available:.1f}, así que ni siquiera se puede deducir "
            "cuánto miden las que faltan: cualquier anchura que se les "
            "suponga desplaza TODAS las columnas de detrás y el resultado "
            "seguiría pareciendo datos. Por eso el archivo se rechaza entero "
            "en vez de devolver números plausibles. Exporta el mismo "
            "experimento como texto (.mpt) desde EC-Lab —esa ruta no depende "
            "de ingeniería inversa— y mándame su cabecera si quieres que "
            "estos identificadores se añadan"
        )

    if len(readable) > 1:
        # Both offsets name columns this reader knows. Whichever record
        # then closes against the module decides; if both do, neither is
        # used, which is the rule the `.spe` reader applies to two equally
        # plausible layouts and for the same reason.
        closing = []
        for id_offset, candidate in readable:
            body = n_points * _record_width(candidate)
            if body > 0 and len(payload) - body >= id_offset + 2 * n_columns:
                closing.append((id_offset, candidate))
        if len(closing) > 1:
            raise BioLogicError(
                "los identificadores de columna encajan empezando en "
                + " y en ".join(str(o) for o, _ in closing)
                + ", y las dos lecturas cuadran igual de bien. Dos "
                "disposiciones igual de plausibles no son una lectura: el "
                "archivo se rechaza en vez de elegir una. Exporta el mismo "
                "experimento como texto (.mpt) desde EC-Lab"
            )
        readable = closing or readable[:1]
    id_offset, ids = readable[0]

    names, formats, flags = _record_fields(ids)
    record = np.dtype({"names": names, "formats": formats})
    # The offset is COMPUTED, not assumed. It differs between format
    # versions, and hardcoding one of them is exactly the kind of guess
    # this module exists not to make: the file says how many points and
    # how wide each is, so where the table starts is arithmetic, and
    # whether that arithmetic lands somewhere sensible is the check.
    body = n_points * record.itemsize
    offset = len(payload) - body
    minimum = id_offset + 2 * n_columns
    if body <= 0 or offset < minimum:
        raise BioLogicError(
            f"{n_points} puntos × {record.itemsize} bytes son {body} bytes y "
            f"el módulo tiene {len(payload)}: la tabla no cuadra con lo que "
            "declara la cabecera, así que las columnas no son las que el "
            "lector cree. El archivo se rechaza en vez de devolver números "
            "plausibles"
        )

    table = np.frombuffer(payload, record, n_points, offset)
    columns = {name: np.asarray(table[name], dtype=float)
               for name in names if name != _FLAG_FIELD}
    if flags:
        packed = np.asarray(table[_FLAG_FIELD], dtype=np.uint8)
        for identifier in flags:
            name, mask, shift = MPR_FLAG_COLUMNS[identifier]
            columns[name] = np.asarray((packed & mask) >> shift, dtype=float)

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
    return _settings_from_lines(lines)


def _settings_from_lines(lines: Sequence[str]) -> ECLabSettings:
    """The technique and the parameter table, from either file.

    An ``.mpt``'s header is the same thing an ``.mps`` is — the technique
    name and then one row per parameter — so it is read with the same
    code. That is the whole reason the ``.mpt`` route needs nothing
    guessed: the scan rate, the current, the window and the electrode
    travel inside the data file itself.
    """
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


#: What the first line of an EC-Lab text export says.
MPT_IDENTIFIER = "EC-LAB ASCII FILE"

#: How many header lines the file declares. This is the self-check, and
#: it is the same bar the ``.mpr`` and the PHI ``.spe`` are held to: the
#: file states one number more than it needs to, so if the declared count
#: does not land on the column-name row then the fields are not where the
#: reader thinks and going on would read numbers from the wrong columns.
_MPT_HEADER_COUNT = re.compile(r"nb\s+header\s+lines\s*:\s*(\d+)", re.I)

#: Columns whose unit is not the one the rest of this module READS,
#: mapped to the name it reads and the factor that gets there.
#:
#: Only those. EC-Lab's own spelling is kept for everything else —
#: including the charge columns, which come in coulombs on one technique
#: and in milliamp-hours on another — because the unit is inside the
#: name, nothing downstream reads them, and converting one of them would
#: make the same experiment arrive under a different name depending on
#: which of its two files was opened. What is normalised is exactly what
#: :func:`_potential`, :func:`_current_a` and :func:`to_curve` look up by
#: name, where a factor of a thousand would travel silently into every
#: capacitance.
MPT_UNIT_ALIASES: dict[str, tuple[str, float]] = {
    "i/a": ("I/mA", 1e3),
    "<i>/a": ("<I>/mA", 1e3),
    "i/\u00b5a": ("I/mA", 1e-3),
    "i/ua": ("I/mA", 1e-3),
    "<i>/\u00b5a": ("<I>/mA", 1e-3),
    "<i>/ua": ("<I>/mA", 1e-3),
    "ewe/mv": ("Ewe/V", 1e-3),
    "time/ms": ("time/s", 1e-3),
    "time/h": ("time/s", 3600.0),
}


#: Column names this module knows, for the «unknown column» note. The
#: binary table's own names plus the ones only the text export writes.
_MPT_KNOWN: frozenset[str] = frozenset(
    [name.lower() for name, _ in MPR_COLUMNS.values()]
    + [name.lower() for name, _ in MPT_UNIT_ALIASES.values()]
    + list(MPT_UNIT_ALIASES)
    + [
        "mode", "ox/red", "error", "control changes", "counter inc.",
        "ns changes", "ns", "time/s", "ewe/v", "ece/v", "<ewe>/v", "<ece>/v",
        "i/ma", "<i>/ma", "control/v", "control/ma", "control/v/ma",
        "dq/ma.h", "(q-qo)/ma.h", "i range", "half cycle", "cycle number",
        "q charge/discharge/ma.h", "q discharge/ma.h", "q charge/ma.h",
        "capacity/ma.h", "efficiency/%", "energy charge/w.h",
        "energy discharge/w.h", "capacitance charge/\u00b5f",
        "capacitance discharge/\u00b5f", "x", "p/w", "pwe/w", "rwe/ohm",
        "r/ohm", "freq/hz", "re(z)/ohm", "-im(z)/ohm", "|z|/ohm",
        "phase(z)/deg", "|ewe|/v", "|i|/a", "cs/\u00b5f", "cp/\u00b5f",
        "analog in 1/v", "analog in 2/v", "step time/s", "dqm/ma.h/g",
        "q-qo/c", "nb cycle",
    ]
)


def read_mpt(path: str | Path) -> ECLabData:
    """Read an EC-Lab ``.mpt`` — the ASCII export, header and all.

    This is the route this module has recommended from the start and the
    only one that depends on nothing reverse-engineered: the columns are
    delimited and NAMED, with their unit inside the name, and the header
    is the same technique-plus-parameter-table an ``.mps`` holds. So the
    scan rate, the current, the window, the mass and the electrode area
    come out of the data file itself, and nothing has to be typed in or
    deduced from |dE/dt|.

    Two things are checked rather than assumed:

    * **The declared header length.** ``Nb header lines : 58`` has to land
      exactly on the row of column names, with a numeric table under it.
      If it does not, the file is refused: a header length that is wrong
      by one reads the column names as data or the first measurement as a
      name, and both look like data afterwards.
    * **The row width.** Every data row has to have as many fields as
      there are names. A short row is a truncated file and a long one is
      a delimiter inside a value.

    One thing is deliberately NOT fatal here, unlike in the ``.mpr``: a
    column whose name this module does not know is kept under its own
    name and reported. In the binary the columns are positional, so an
    unknown identifier shifts everything behind it and the whole file has
    to be refused; in the text they are named and delimited, so an
    unknown one costs exactly itself.

    Returns
    -------
    ECLabData
        The same object :func:`read_mpr` returns, with the header's
        settings under ``settings["ajustes"]``.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="latin-1")
    except OSError as error:
        raise BioLogicError(f"{path.name}: {error}") from None
    lines = [line.rstrip("\r\n") for line in text.splitlines()]
    if not lines or MPT_IDENTIFIER not in lines[0].upper():
        first = (lines[0][:40] if lines else "")
        raise BioLogicError(
            f"{path.name}: la primera línea no es «{MPT_IDENTIFIER}» "
            f"({first!r}), así que no es una exportación de texto de EC-Lab. "
            "En EC-Lab: «Export as text» sobre el experimento"
        )
    declared = None
    for line in lines[:10]:
        match = _MPT_HEADER_COUNT.search(line)
        if match:
            declared = int(match.group(1))
            break
    if declared is None:
        raise BioLogicError(
            f"{path.name}: no declara «Nb header lines», que es lo único que "
            "dice dónde empieza la tabla. Vuelve a exportarlo desde EC-Lab"
        )
    if declared < 1 or declared > len(lines):
        raise BioLogicError(
            f"{path.name}: declara {declared} líneas de cabecera y el archivo "
            f"tiene {len(lines)}"
        )

    names = [cell.strip() for cell in lines[declared - 1].split("\t")]
    while names and not names[-1]:
        names.pop()
    if len(names) < 2 or any(_looks_numeric(name) for name in names):
        raise BioLogicError(
            f"{path.name}: la línea {declared} tendría que ser la de los "
            f"nombres de columna y es {lines[declared - 1][:60]!r}. El "
            "archivo declara una cabecera de otra longitud que la que tiene, "
            "y leer la tabla desde donde dice daría números de otras columnas"
        )
    rows, width = _mpt_rows(lines[declared:], len(names), path.name)

    original = list(names)
    columns: dict[str, np.ndarray] = {}
    unknown: list[str] = []
    for index, name in enumerate(names[:width]):
        canonical, factor = MPT_UNIT_ALIASES.get(name.lower(), (name, 1.0))
        values = rows[:, index] * factor
        if canonical in columns:
            # EC-Lab repeats a name when a sequence has several steps.
            canonical = f"{canonical} (2)"
        columns[canonical] = values
        if canonical == name and name.lower() not in _MPT_KNOWN:
            unknown.append(name)

    settings = _settings_from_lines(lines[:declared - 1])
    warnings = [
        "leído del .mpt, la exportación de texto: las columnas vienen "
        "nombradas y con su unidad, así que aquí no hay nada deducido por "
        "ingeniería inversa. Es la ruta reproducible"
    ]
    if unknown:
        warnings.append(
            "columnas que este módulo no conoce, guardadas con su nombre tal "
            "cual y sin usar: " + ", ".join(unknown)
            + ". En un .mpt eso no desplaza nada, porque las columnas van "
              "nombradas y delimitadas"
        )
    return ECLabData(columns=columns, n_points=int(rows.shape[0]),
                     settings={"ajustes": settings,
                               "columnas_originales": original,
                               "lineas_cabecera": declared},
                     modules=[settings.technique_name or "mpt"],
                     warnings=warnings)


def _looks_numeric(text: str) -> bool:
    try:
        float(text.replace(",", "."))
    except ValueError:
        return False
    return True


def _mpt_rows(lines: Sequence[str], columns: int, name: str
              ) -> tuple[np.ndarray, int]:
    """The numeric table under the column names, width checked.

    The decimal mark is whatever the machine that exported it used: the
    delimiter is a tab, so a comma inside a field cannot be anything but
    a decimal mark and reading it as one is unambiguous. That is not true
    of a comma-delimited export, which is why this is safe here and
    nowhere else.
    """
    rows: list[list[float]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split("\t")]
        while cells and not cells[-1]:
            cells.pop()
        if len(cells) < columns:
            raise BioLogicError(
                f"{name}: la fila {number} de la tabla tiene {len(cells)} "
                f"campos y la cabecera declara {columns} nombres. Un archivo "
                "truncado se lee igual de bien que uno entero y da otros "
                "números"
            )
        try:
            rows.append([float(cell.replace(",", ".")) for cell in cells[:columns]])
        except ValueError:
            raise BioLogicError(
                f"{name}: la fila {number} de la tabla no es numérica "
                f"({line[:60]!r}). Comprueba «Nb header lines»: si no cae "
                "exactamente en la fila de nombres, lo que sigue no es la "
                "tabla"
            ) from None
    if len(rows) < 2:
        raise BioLogicError(
            f"{name}: debajo de la cabecera hay {len(rows)} fila(s) de datos")
    return np.asarray(rows, dtype=float), columns


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


#: What the ``mode`` flag means. Read off the user's own pair of files:
#: his voltammogram is 2 throughout and his galvanostatic cycling is 1
#: throughout, which is the community reading as well.
MPR_MODE_GALVANOSTATIC = 1
MPR_MODE_POTENTIOSTATIC = 2


def _current_a(data: ECLabData) -> Optional[np.ndarray]:
    """Current in AMPS. EC-Lab writes milliamps; the package uses SI."""
    for name in ("I/mA", "<I>/mA", "<I>/mA (2)", "control/mA"):
        if name in data.columns:
            return data.columns[name] * 1e-3
    if "|I|/A" in data.columns:
        return data.columns["|I|/A"]
    # A galvanostatic file may hold NO measured current column at all:
    # the current is the quantity being controlled, so EC-Lab keeps it in
    # `control/V/mA`, whose unit depends on the mode — volts when the
    # potential is controlled, milliamps when the current is. The user's
    # GCPL .mpr is exactly that: fifteen columns, Ewe/V among them, and
    # nothing `to_curve` would call a current, so the file decoded
    # cleanly and then came back «faltan columnas de potencial o de
    # corriente».
    #
    # Taking it is right for a charge-discharge curve and it is NOT a
    # measurement: it is the setpoint. The sign is what separates the
    # branches, which is what the curve needs it for, and the note says
    # where it came from, because a setpoint read as a measurement hides
    # whatever the instrument could not deliver.
    control = data.columns.get("control/V/mA")
    mode = data.columns.get("mode")
    if control is not None and mode is not None and mode.size:
        if np.all(np.asarray(mode, dtype=int) == MPR_MODE_GALVANOSTATIC):
            data.warnings.append(
                "este archivo no trae columna de corriente medida: la "
                "corriente sale de «control/V/mA», que en modo "
                "galvanostático son los miliamperios CONSIGNADOS. El signo "
                "—que es lo que separa carga de descarga— es correcto; la "
                "magnitud es la consigna, no lo que el equipo entregó"
            )
            return control * 1e-3
    return None


#: Fraction of the potential span a swing has to cover to count as a
#: reversal, for :func:`technique_from_data`.
#:
#: The same threshold and the same reason as the generic format
#: detector: without one, a clean galvanostatic curve has "several
#: hundred reversals" because every noise wiggle reverses the sign of
#: the first difference.
REVERSAL_FRACTION = 0.05


def technique_from_data(data: ECLabData) -> tuple[str, str]:
    """Which technique an ``.mpr`` holds, read off the file itself.

    Returns ``(kind, why)``.

    This exists because the `.mps` is not always there. EC-Lab writes the
    pair and people copy one of them, and when the settings file is
    missing the technique was not deduced — it defaulted to ``gcd``. The
    user's five cyclic voltammograms therefore came back as
    charge–discharge curves, which puts every CV analysis out of reach:
    no capacitance by either convention, no Dunn, no storage mechanism.
    Defaulting is the wrong move when the file says so itself, and this
    one says so three times over.

    In order of how strongly each says it:

    1. **The ``mode`` flag.** EC-Lab writes 1 for galvanostatic and 2 for
       potentiostatic in every record. It is the instrument's own
       statement of which variable it controlled, and it is not a guess.
    2. **Which control column exists.** ``control/V`` is volts and
       ``control/V/mA`` carries a unit that depends on the mode; a file
       with the first is controlling potential.
    3. **The shape**, which is the package's standing rule for deciding a
       format — a voltammogram sweeps back over itself and nothing else
       does. Used to separate a cyclic sweep from a linear one, and as
       the answer when the flags are missing or mixed.
    """
    mode = data.columns.get("mode")
    reversals = _count_reversals(data.columns.get("Ewe/V"))
    cyclic = reversals >= 2

    if mode is not None and mode.size:
        values = np.unique(np.asarray(mode, dtype=int))
        if values.size == 1 and values[0] == MPR_MODE_POTENTIOSTATIC:
            kind = "cv" if cyclic else "lsv"
            return kind, (
                f"la bandera «mode» vale {MPR_MODE_POTENTIOSTATIC} "
                "(potenciostático) en todo el registro, así que lo "
                f"controlado es el potencial; y el barrido tiene "
                f"{reversals} inversion(es), así que es "
                + ("cíclico" if cyclic else "lineal")
            )
        if values.size == 1 and values[0] == MPR_MODE_GALVANOSTATIC:
            return "gcd", (
                "la bandera «mode» vale 1 (galvanostático) en todo el "
                "registro: lo controlado es la corriente"
            )

    if "control/V" in data.columns and "control/V/mA" not in data.columns:
        return ("cv" if cyclic else "lsv"), (
            "la columna de control es «control/V», en voltios, así que lo "
            f"controlado es el potencial ({reversals} inversion(es))"
        )

    if cyclic:
        return "cv", (
            f"el potencial vuelve sobre sí mismo {reversals} veces, que es "
            "lo que hace un voltamperograma y no hace ninguna otra medida"
        )
    return "gcd", "nada en el archivo declara la técnica; se supone GCD"


def _count_reversals(potential: Optional[np.ndarray]) -> int:
    """How many times the potential turns round, ignoring noise."""
    if potential is None or potential.size < 3:
        return 0
    values = np.asarray(potential, dtype=float)
    span = float(values.max() - values.min())
    if span <= 0.0:
        return 0
    # Walk the series keeping only moves bigger than the threshold, so a
    # reversal is a real turn and not a wiggle.
    step = REVERSAL_FRACTION * span
    direction = 0
    anchor = values[0]
    turns = 0
    for value in values[1:]:
        if abs(value - anchor) < step:
            continue
        way = 1 if value > anchor else -1
        if direction and way != direction:
            turns += 1
        direction = way
        anchor = value
    return turns


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
    deduced = ""
    if not kind:
        if data.has("freq/Hz") and (data.has("Re(Z)/Ohm") or data.has("|Z|/Ohm")):
            kind = "eis"
        elif settings.technique:
            kind = settings.technique
        else:
            # No .mps, so nothing DECLARED the technique. Reading it off
            # the file beats defaulting to gcd, which silently turned
            # every voltammogram measured without its settings file into
            # a charge-discharge curve.
            kind, deduced = technique_from_data(data)

    metadata = {"origen": "EC-Lab .mpr", "modulos": list(data.modules)}
    if settings.technique_name:
        metadata["tecnica"] = settings.technique_name
    if deduced:
        # Said out loud: a technique the program worked out is a
        # different kind of fact from one the settings file declared,
        # and which of the two it was decides how much to trust what
        # comes after.
        metadata["tecnica_deducida"] = f"{kind} — {deduced}"

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
            # |dE/dt| where the sweep is actually sweeping. This is a
            # MEASUREMENT, not the guess the package refuses to make: a
            # supposed scan rate gives a capacitance wrong in exactly
            # that proportion, but one read off the potential and the
            # clock is as good as the instrument's own clock, and it
            # comes with its own quality check.
            #
            # Filtered against the MEDIAN and not against the maximum.
            # The maximum is set by whatever single point is steepest —
            # in the user's 5 mV/s file the first three records carry
            # the 155 uA charging transient from the step out of the
            # previous experiment, and `0.2 * max` then discards the
            # entire real sweep and keeps the artefact.
            slope = np.abs(np.gradient(potential, time))
            slope = slope[np.isfinite(slope) & (slope > 0)]
            if slope.size:
                middle = float(np.median(slope))
                moving = slope[(slope > 0.3 * middle) & (slope < 3.0 * middle)]
            else:
                moving = slope
            if moving.size:
                rate = float(np.median(moving))
                spread = (100.0 * float(np.std(moving)) / rate
                          if rate else float("inf"))
                metadata["velocidad_medida_mv_s"] = 1e3 * rate
                metadata["velocidad_dispersion_pct"] = spread
                note = (
                    f"la velocidad de barrido ({1e3 * rate:.4g} mV/s) está "
                    "MEDIDA de |dE/dt|, no declarada: no había .mps al lado. "
                    f"La dispersión sobre el barrido es del {spread:.1f} %"
                )
                # A rate is one rate only if it holds. Above this the
                # file is more than one segment, and a single median is
                # then the average of two experiments.
                if spread > 25.0:
                    note += (
                        ". Eso es mucho: el archivo probablemente lleva más "
                        "de un segmento o más de una velocidad, y una sola "
                        "mediana es entonces el promedio de dos experimentos"
                    )
                metadata["aviso_velocidad"] = note
        if not rate:
            raise BioLogicError(
                "no hay velocidad de barrido: ni en el .mps ni deducible del "
                "eje de tiempo. Pásala a mano — sin ella la capacitancia "
                "sale mal exactamente en esa proporción y nada aguas abajo "
                "lo delata"
            )
        # The cycle column is in the file and was being dropped, so
        # `Voltammogram.cycles()` fell back to counting sweep reversals —
        # a heuristic, where the instrument's own counter was sitting in
        # column 24. It matters beyond tidiness: the reversal detector
        # returns the trailing partial cycle as if it were a cycle, and
        # a partial cycle integrated over a third of the window is a
        # capacitance of nothing.
        cycle = data.columns.get("cycle number")
        return Voltammogram(potential=potential, current=current,
                            scan_rate=float(rate), electrode=electrode,
                            cycle=cycle, time=time,
                            name=f"{kind.upper()} (EC-Lab)", metadata=metadata)

    if time is None:
        raise BioLogicError(
            "una curva de carga-descarga necesita el eje de tiempo y no hay "
            "columna time/s"
        )
    return ChargeDischarge(time=time, potential=potential, current=current,
                           electrode=electrode, name="GCD (EC-Lab)",
                           metadata=metadata)


def is_eclab_text(path: str | Path) -> bool:
    """Whether this is an EC-Lab ``.mpt``, by its first line.

    By content and not by extension, the same rule the rest of the
    package follows.
    """
    try:
        with Path(path).open("rb") as handle:
            first = handle.readline(200).decode("latin-1", "replace")
    except OSError:
        return False
    return MPT_IDENTIFIER in first.upper()


def read_eclab(
    path: str | Path,
    kind: str = "",
    electrode: Optional[Electrode] = None,
    scan_rate: Optional[float] = None,
):
    """Read an ``.mpr`` or an ``.mpt`` and return the curve it holds.

    Which of the two is decided by the content. For an ``.mpr`` the
    ``.mps`` of the same name is read too when it is there, because
    EC-Lab writes them as a pair and it carries the scan rate; an
    ``.mpt`` needs no companion, because its own header holds the same
    parameter table.

    Returns ``(curve, warnings)``.
    """
    path = Path(path)
    if is_eclab_text(path):
        data = read_mpt(path)
        settings = data.settings.get("ajustes")
        curve = to_curve(data, kind=kind, electrode=electrode,
                         scan_rate=scan_rate, settings=settings)
        curve.metadata["origen"] = "EC-Lab .mpt"
        warnings = list(data.warnings)
        if settings is not None:
            warnings.extend(settings.warnings)
            curve.metadata["ajustes"] = settings.describe()
        note = curve.metadata.get("aviso_velocidad")
        if note:
            warnings.append(note)
        return curve, warnings
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
    "ID_OFFSETS",
    "MPR_DATA_OFFSETS",
    "MPR_FLAG_COLUMNS",
    "MPR_MODE_GALVANOSTATIC",
    "MPR_MODE_POTENTIOSTATIC",
    "MPS_TECHNIQUES",
    "BioLogicError",
    "ECLabData",
    "ECLabSettings",
    "MPT_IDENTIFIER",
    "MPT_UNIT_ALIASES",
    "is_eclab_text",
    "read_eclab",
    "read_mpr",
    "read_mpt",
    "read_mps",
    "settings_beside",
    "to_curve",
]
