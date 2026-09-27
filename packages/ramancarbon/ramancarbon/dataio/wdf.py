"""Renishaw WiRE ``.wdf``: the file the instrument wrote, not its export.

A Renishaw spectrum reaches this package twice. Once as two columns of
text, because WiRE offers "export to TXT" and that is what gets emailed,
and once as the ``.wdf`` the instrument saved. The numbers in the two are
the same — checked on a real pair, the largest disagreement between the
text export and the binary was 5e-7 cm⁻¹, which is the rounding in
``%f``. What the text loses is everything around the numbers: the laser
the machine was actually using, the grating, the objective, how many
accumulations went into the trace, when it was measured, and the fact
that a zero level was subtracted from it.

That matters for the same reason it mattered for the Rigaku ``.ras``. A
Raman shift axis is meaningless without the excitation, and I_D/I_G,
L_a from Tuinstra-Koenig and every dispersion argument in the literature
are quoted *per wavelength*; a spectrum that arrives without its laser
gets the package's default, which is right until the day somebody
measures at 633 nm. Reading the file the instrument wrote means the
excitation is the instrument's, not a default.

The layout is a chain of blocks, each ``id`` (4 chars), ``uid`` (uint32),
``size`` (uint64), payload. That much is public. The property sets
(``PSET``) inside the instrument blocks are community reverse
engineering, and this module treats them the way the package treats the
Bio-Logic ``.mpr``: the layout is computed from the file and checked to
close against the declared length, and a parse that does not close is
reported rather than half-used. The spectrum itself does not depend on
them — ``XLST`` and ``DATA`` are enough for that — so a refused property
set costs metadata and never data.

References
----------
The block structure and the type and unit enumerations follow the
community description of the format used by ``py-wdf-reader``
(https://github.com/alchem0x2A/py-wdf-reader) and the Gwyddion WDF
module. Renishaw publishes no specification.
"""

from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

#: The signature of the file, and the size of the fixed header it opens.
MAGIC = b"WDF1"
HEADER_SIZE = 512

#: The unit codes the format uses, as far as this package needs them.
UNITS: dict[int, str] = {
    0: "arbitrario", 1: "cm-1", 2: "cm-1", 3: "nm", 4: "eV", 5: "um",
    6: "cuentas", 7: "electrones", 8: "mm", 9: "m", 10: "K", 11: "Pa",
    12: "s", 13: "ms", 14: "h", 15: "d", 16: "px", 17: "intensidad",
    18: "intensidad relativa", 19: "grados", 20: "rad", 21: "C",
    22: "F", 23: "K/min", 24: "filetime", 25: "us",
}

#: The quantity codes, for the axis lists and the origin lists.
QUANTITIES: dict[int, str] = {
    0: "arbitraria", 1: "espectral", 2: "intensidad", 3: "x", 4: "y",
    5: "z", 6: "r", 7: "theta", 8: "phi", 9: "temperatura",
    10: "presion", 11: "tiempo", 12: "derivada", 13: "polarizacion",
    14: "seguimiento de foco", 15: "velocidad de rampa", 16: "suma de control",
    17: "banderas", 18: "tiempo transcurrido", 19: "frecuencia",
}

#: The scan types WiRE records. Only the first three hold one spectrum.
SCAN_TYPES: dict[int, str] = {
    0: "desconocido", 1: "estatico", 2: "continuo", 3: "por pasos",
    4: "multipasos discontinuo", 5: "multipasos continuo",
    6: "multipasos por pasos", 7: "interpolado", 8: "multiespectro",
}

#: Sizes and ``struct`` codes of the scalar property types.
_SCALARS: dict[str, tuple[int, str]] = {
    "?": (1, "?"), "c": (1, "b"), "s": (2, "h"), "i": (4, "i"),
    "w": (8, "q"), "r": (4, "f"), "q": (8, "d"), "t": (8, "Q"),
}

#: Epoch of the Windows FILETIME the format stores its instants in.
#:
#: Tagged UTC, which is what a FILETIME is by definition. Leaving it naive
#: reads as local time to everything downstream, and a methods section
#: that says a spectrum was taken at 22:06 without saying in which zone
#: has lost the only part of the timestamp anyone would check.
_FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


class WdfError(ValueError):
    """Raised when the file is not a ``.wdf`` or does not hold together."""


def _filetime(value: int) -> Optional[str]:
    """A Windows FILETIME as ISO 8601, or ``None`` for an unset one.

    Zero is what WiRE writes for "not recorded", and 1601-01-01 in a
    report reads as a real date the instrument never saw.
    """
    if value <= 0:
        return None
    try:
        moment = _FILETIME_EPOCH + timedelta(microseconds=value / 10)
    except OverflowError:
        return None
    return moment.isoformat(sep=" ", timespec="seconds")


def _string(raw: bytes) -> str:
    return raw.split(b"\x00")[0].decode("latin-1").strip()


def _parse_pset(buf: bytes, start: int, end: int) -> dict[str, Any]:
    """One WiRE property set, as a dictionary.

    Entries are ``type`` (one char), ``flags``, ``key`` (uint16), value.
    The names are not a table at the head of the set: each one is its own
    entry, of type ``k``, carrying the name of the key it is filed
    under — which is why the names are collected first and applied
    afterwards, and why an unnamed key keeps its number instead of being
    dropped. A key whose name never arrives is still data.
    """
    collected: list[tuple[int, Any]] = []
    names: dict[int, str] = {}
    offset = start
    while offset + 4 <= end:
        kind = chr(buf[offset])
        key = struct.unpack_from("<H", buf, offset + 2)[0]
        offset += 4
        if kind in _SCALARS:
            width, code = _SCALARS[kind]
            if offset + width > end:
                raise WdfError(f"un valor {kind!r} se sale del bloque")
            value = struct.unpack_from("<" + code, buf, offset)[0]
            offset += width
            collected.append((key, _filetime(value) if kind == "t" else value))
        elif kind in "ubpk":
            if offset + 4 > end:
                raise WdfError(f"una longitud {kind!r} se sale del bloque")
            length = struct.unpack_from("<I", buf, offset)[0]
            offset += 4
            stop = offset + length
            if stop > end:
                raise WdfError(
                    f"un valor {kind!r} declara {length} bytes y sólo quedan "
                    f"{end - offset}"
                )
            if kind == "u":
                collected.append((key, buf[offset:stop].decode("utf-8", "replace")))
            elif kind == "b":
                collected.append((key, buf[offset:stop]))
            elif kind == "p":
                collected.append((key, _parse_pset(buf, offset, stop)))
            else:
                names[key] = buf[offset:stop].decode("utf-8", "replace")
            offset = stop
        else:
            raise WdfError(
                f"tipo de propiedad desconocido {kind!r} (0x{buf[offset - 4]:02x}) "
                f"en el byte {offset - 4}"
            )
    if offset != end:
        raise WdfError(f"el conjunto de propiedades cierra en {offset}, no en {end}")
    return {names.get(key, f"#{key}"): value for key, value in collected}


def _blocks(raw: bytes) -> list[tuple[str, int, int, int]]:
    """The block chain, as ``(id, uid, offset, size)``.

    Walked rather than searched. A ``.wdf`` has no table of contents: the
    blocks are found by adding each one's size to the offset of the last,
    so a wrong size is not a missing block but a chain that stops, and
    that is worth saying out loud instead of silently returning what came
    before it.
    """
    found: list[tuple[str, int, int, int]] = []
    offset = 0
    while offset + 16 <= len(raw):
        identifier = raw[offset:offset + 4].decode("latin-1")
        uid, = struct.unpack_from("<I", raw, offset + 4)
        size, = struct.unpack_from("<Q", raw, offset + 8)
        if size < 16 or offset + size > len(raw):
            raise WdfError(
                f"el bloque «{identifier}» en el byte {offset} declara {size} "
                f"bytes y el archivo tiene {len(raw)}: la cadena de bloques "
                "está rota o el archivo está truncado"
            )
        found.append((identifier, uid, offset, size))
        offset += size
    return found


def read_wdf(path: str | Path) -> dict[str, Any]:
    """Read a Renishaw ``.wdf``.

    Returns
    -------
    dict
        ``x`` and ``y`` as arrays with ``x`` ascending, ``x_units``,
        ``laser_nm``, ``n_spectra``, ``title``, and the instrument's own
        settings under ``instrument``. ``problems`` lists what could not
        be read; it is a list and not an exception because a property set
        that will not parse is no reason to withhold the spectrum.

    Raises
    ------
    WdfError
        When the file is not a ``.wdf``, or when the axis and the data do
        not agree on how many points there are.
    """
    p = Path(path)
    raw = p.read_bytes()
    if raw[:4] != MAGIC:
        raise WdfError(
            f"{p.name}: no empieza por «WDF1», así que no es un archivo de "
            "WiRE (¿un .wxd antiguo, o un .wdf renombrado?)"
        )
    if len(raw) < HEADER_SIZE:
        raise WdfError(f"{p.name}: son {len(raw)} bytes, menos que la cabecera")

    def u32(offset: int) -> int:
        return struct.unpack_from("<I", raw, offset)[0]

    def u64(offset: int) -> int:
        return struct.unpack_from("<Q", raw, offset)[0]

    points = u32(0x3C)
    capacity = u64(0x40)
    collected = u64(0x48)
    accumulations = u32(0x50)
    x_count = u32(0x58)
    laser_wavenumber = struct.unpack_from("<f", raw, 0x9C)[0]
    scan_type = u32(0x80)
    problems: list[str] = []

    blocks = _blocks(raw)
    by_id: dict[str, tuple[int, int]] = {}
    for identifier, _uid, offset, size in blocks:
        by_id.setdefault(identifier, (offset, size))

    if "XLST" not in by_id or "DATA" not in by_id:
        raise WdfError(
            f"{p.name}: falta {'XLST' if 'XLST' not in by_id else 'DATA'}; "
            "sin eje o sin datos no hay espectro"
        )

    x_offset, x_size = by_id["XLST"]
    x_quantity, x_units = u32(x_offset + 16), u32(x_offset + 20)
    expected = 24 + 4 * x_count
    if x_size != expected:
        raise WdfError(
            f"{p.name}: XLST ocupa {x_size} bytes y para {x_count} puntos "
            f"debería ocupar {expected}"
        )
    x = np.asarray(struct.unpack_from(f"<{x_count}f", raw, x_offset + 24), dtype=float)

    data_offset, data_size = by_id["DATA"]
    values = (data_size - 16) // 4
    if values < points:
        raise WdfError(
            f"{p.name}: DATA lleva {values} valores y la cabecera declara "
            f"{points} puntos por espectro"
        )
    y_all = np.asarray(struct.unpack_from(f"<{values}f", raw, data_offset + 16),
                       dtype=float)

    spectra = max(int(collected), 1)
    if x_count != points:
        problems.append(
            f"el eje tiene {x_count} puntos y la cabecera declara {points}"
        )
    if values != points * spectra:
        problems.append(
            f"DATA lleva {values} valores, que no son {points} x {spectra}"
        )

    instrument: dict[str, Any] = {}
    for identifier, label in (("WXCS", "configuracion"), ("WXIS", "estado"),
                              ("WXDM", "medida"), ("ZLDC", "nivel cero")):
        if identifier not in by_id:
            continue
        offset, size = by_id[identifier]
        if raw[offset + 16:offset + 20] not in (b"PSET", b"PSETC"):
            continue
        length = u32(offset + 20)
        try:
            instrument[label] = _parse_pset(raw, offset + 24, offset + 24 + length)
        except WdfError as error:
            problems.append(f"{identifier}: {error}")

    origin: dict[str, Any] = {}
    if "ORGN" in by_id:
        offset, size = by_id["ORGN"]
        try:
            origin = _read_origins(raw, offset, size, spectra)
        except WdfError as error:
            problems.append(f"ORGN: {error}")

    laser_nm: Optional[float] = None
    if laser_wavenumber > 0:
        laser_nm = 1.0e7 / float(laser_wavenumber)

    units = UNITS.get(x_units, f"codigo {x_units}")
    if units == "nm" and laser_nm:
        from ..core.io import wavelength_to_shift

        x = np.asarray(wavelength_to_shift(x, laser_nm), dtype=float)
        problems.append(
            f"el eje venía en nm y se ha convertido a cm-1 con el láser del "
            f"propio archivo ({laser_nm:.2f} nm)"
        )
        units = "cm-1"

    title = _string(raw[0xF0:0xF0 + 160])
    comment = ""
    if "TEXT" in by_id:
        offset, size = by_id["TEXT"]
        comment = _string(raw[offset + 16:offset + size])

    metadata: dict[str, Any] = {
        "formato": "wdf",
        "version_wire": ".".join(
            str(value) for value in struct.unpack_from("<4H", raw, 0x78)
        ),
        "aplicacion": _string(raw[0x60:0x60 + 24]),
        "usuario": _string(raw[0xD0:0xD0 + 32]),
        "titulo": title,
        "comentario": comment,
        "tipo_de_barrido": SCAN_TYPES.get(scan_type, f"codigo {scan_type}"),
        "acumulaciones": int(accumulations),
        "espectros": spectra,
        "capacidad": int(capacity),
        "puntos": int(points),
        "eje": QUANTITIES.get(x_quantity, f"codigo {x_quantity}"),
        "inicio": _filetime(u64(0x88)),
        "fin": _filetime(u64(0x90)),
    }
    start, stop = u64(0x88), u64(0x90)
    if start > 0 and stop > start:
        metadata["duracion_s"] = round((stop - start) / 1.0e7, 3)
    zero = _zero_level(instrument.get("nivel cero"))
    if zero is not None:
        metadata["nivel_cero_restado"] = zero
    settings = _collection(instrument)
    metadata.update(settings)
    if origin:
        metadata["origen"] = origin

    y = y_all[:points] if spectra == 1 else y_all
    if spectra == 1:
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]

    return {
        "x": x,
        "y": y,
        "x_units": units,
        "laser_nm": laser_nm,
        "laser_cm1": float(laser_wavenumber) if laser_wavenumber > 0 else None,
        "n_spectra": spectra,
        "title": title,
        "instrument": instrument,
        "metadata": metadata,
        "problems": problems,
    }


def _read_origins(raw: bytes, offset: int, size: int,
                  spectra: int) -> dict[str, Any]:
    """The per-spectrum origin lists: when, where, and the checksum.

    Its length is fully determined — a count, then one fixed-size record
    per list — so it is checked to close. For a single spectrum the
    interesting one is the instant; for a map it is X and Y, and this is
    where the mapping reader will find them.
    """
    count = struct.unpack_from("<I", raw, offset + 16)[0]
    record = 24 + 8 * spectra
    expected = 16 + 4 + count * record
    if expected != size:
        raise WdfError(
            f"declara {count} listas para {spectra} espectros, que son "
            f"{expected} bytes, y el bloque ocupa {size}"
        )
    result: dict[str, Any] = {}
    cursor = offset + 20
    for _ in range(count):
        kind = struct.unpack_from("<I", raw, cursor)[0] & 0x7FFFFFFF
        units = struct.unpack_from("<I", raw, cursor + 4)[0]
        label = _string(raw[cursor + 8:cursor + 24]) or QUANTITIES.get(
            kind, f"codigo {kind}")
        body = cursor + 24
        if units == 24:                                   # FILETIME
            values: list[Any] = [
                _filetime(struct.unpack_from("<Q", raw, body + 8 * i)[0])
                for i in range(spectra)
            ]
        elif kind in (16, 17):                            # checksum, flags
            values = [struct.unpack_from("<Q", raw, body + 8 * i)[0]
                      for i in range(spectra)]
        else:
            values = [struct.unpack_from("<d", raw, body + 8 * i)[0]
                      for i in range(spectra)]
        result[label] = values[0] if spectra == 1 else values
        cursor += record
    return result


def _zero_level(block: Optional[dict[str, Any]]) -> Optional[float]:
    """The zero level WiRE subtracted, when it recorded one.

    Worth carrying because it says what the intensities *are*. A
    Renishaw trace is bias-subtracted ADU, not photon counts, so σ = √I
    would be wrong for it — this package estimates σ locally from the
    trace instead, and the number here is the evidence that it has to.
    """
    if not isinstance(block, dict):
        return None
    for value in block.values():
        if isinstance(value, dict) and "Value" in value:
            try:
                return float(value["Value"])
            except (TypeError, ValueError):
                return None
    return None


def _collection(instrument: dict[str, Any]) -> dict[str, Any]:
    """Grating, objective and laser, pulled out of the property sets.

    Only the entries WiRE itself names are taken. The acquisition block
    stores exposure and laser power under numbered keys whose names live
    in WiRE's own code and not in the file, and guessing which number is
    the exposure would put a wrong number of seconds in a methods
    section — so they are left where they are, inside ``instrument``, for
    anyone who wants to dig.
    """
    found: dict[str, Any] = {}
    state = instrument.get("estado")
    if isinstance(state, dict):
        if isinstance(state.get("Instrument type"), str):
            found["equipo"] = state["Instrument type"]
        configuration = state.get("System Configuration")
        if isinstance(configuration, dict):
            if isinstance(configuration.get("Grating"), str):
                found["red_de_difraccion"] = configuration["Grating"]
            if isinstance(configuration.get("FocusMode"), str):
                found["modo"] = configuration["FocusMode"]
        microscope = state.get("Microscope")
        if isinstance(microscope, dict):
            for value in microscope.values():
                if isinstance(value, str) and value.lower().startswith("x"):
                    found["objetivo"] = value
                    break
        slits = state.get("Slits")
        if isinstance(slits, dict) and isinstance(slits.get("Opening"), str):
            found["rendija"] = slits["Opening"]
        if isinstance(state.get("ND Transmission %"), str):
            found["transmision_nd_pct"] = state["ND Transmission %"]
    configuration = instrument.get("configuracion")
    if isinstance(configuration, dict):
        lasers = configuration.get("Lasers")
        if isinstance(lasers, dict):
            for name, value in lasers.items():
                if isinstance(value, dict) and "Wavenumber" in value:
                    found["laser"] = name
                    break
    return found


def load_wdf_spectrum(path: str | Path, laser_nm: Optional[float] = None):
    """A ``.wdf`` as a :class:`~ramancarbon.core.spectrum.Spectrum`.

    The laser is the instrument's unless one is passed, and passing one
    that disagrees is recorded rather than resolved: a spectrum measured
    at 532 nm and analysed as 633 nm gives a crystallite size wrong by
    the ratio of the wavelengths, and the person who typed the override
    is the only one who knows which is right.
    """
    from ..core.spectrum import Spectrum

    parsed = read_wdf(path)
    if parsed["n_spectra"] > 1:
        raise WdfError(
            f"{Path(path).name}: lleva {parsed['n_spectra']} espectros "
            "(es un mapa o una serie), no uno. Ábrelo con "
            "ramancarbon.mapping, o exporta el espectro que te interese"
        )
    if parsed["x_units"] not in ("cm-1", "arbitrario"):
        raise WdfError(
            f"{Path(path).name}: el eje está en {parsed['x_units']} y no se "
            "ha podido convertir a desplazamiento Raman"
        )
    metadata = dict(parsed["metadata"])
    metadata["path"] = str(path)
    if parsed["problems"]:
        metadata["avisos"] = list(parsed["problems"])
    chosen = parsed["laser_nm"]
    if laser_nm:
        if chosen and abs(float(laser_nm) - chosen) > 1.0:
            metadata["laser_del_archivo_nm"] = round(chosen, 3)
        chosen = float(laser_nm)
    return Spectrum(
        shift=parsed["x"], intensity=parsed["y"], laser_nm=chosen,
        name=parsed["title"] or Path(path).stem,
        metadata=metadata,
    )


__all__ = [
    "MAGIC",
    "QUANTITIES",
    "SCAN_TYPES",
    "UNITS",
    "WdfError",
    "load_wdf_spectrum",
    "read_wdf",
]
