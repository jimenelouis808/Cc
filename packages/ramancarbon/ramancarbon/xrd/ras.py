"""Rigaku .ras files, as the SmartLab writes them.

A .ras is plain text: a header of ``*KEY "value"`` lines, then the
measurement between ``*RAS_INT_START`` and ``*RAS_INT_END`` as three
columns -- angle, intensity, attenuation factor.

Reading it rather than the exported two-column text is worth the module
for three things the export throws away.

**The counts.** An export is often normalised. One from this
instrument arrived divided by 543, running 1.00 to 2.84 where the
original runs 544 to 1543. Nothing in the pattern's shape changes, but
Poisson weighting does: `sigma = sqrt(N)` is a statement about counts,
and applied to normalised numbers it understates the variance by the
normalising factor.

**The wavelength.** Cu K-alpha-1 is 1.540593 A and K-alpha-2 is
1.544414 A, both recorded in the header. Assuming a single 1.5406 A puts
every d-spacing out by a part in three thousand, which at 80 deg 2-theta
is a tenth of a degree -- larger than the peak-matching window.

**The dwell time.** Counts per second is what the display shows; counts
is what the statistics need. The two differ by the time per step, which
is the step divided by the scan speed, and only the header has it.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

#: The header keys this reader understands. Everything else is carried
#: through untouched, because a header field this module has no use for
#: may be exactly what someone later needs to explain a measurement.
_HEADER = re.compile(r'^\*([A-Z0-9_\-]+)\s+"(.*)"\s*$')

_START = "*RAS_INT_START"
_END = "*RAS_INT_END"


class RASError(ValueError):
    """A .ras file that cannot be read, and why."""


def _number(header: dict[str, str], key: str) -> float | None:
    raw = header.get(key)
    if raw is None:
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None


def read_ras(path: str | Path) -> dict:
    """Read one .ras file.

    Returns
    -------
    dict
        ``two_theta`` and ``intensity`` arrays, ``attenuation`` as
        measured, and the metadata worth keeping: ``wavelength`` (K-alpha-1),
        ``wavelength_alpha2``, ``step``, ``dwell`` in seconds, ``unit``,
        ``sample``, ``operator``, ``started`` and the whole ``header``.
    """
    p = Path(path)
    text = p.read_bytes().decode("latin-1")
    if _START not in text:
        raise RASError(
            f"{p.name}: no contiene un bloque {_START}; ¿es realmente un "
            "archivo .ras de Rigaku?")

    header: dict[str, str] = {}
    rows: list[tuple[float, float, float]] = []
    inside = False
    for line in text.splitlines():
        line = line.rstrip("\r")
        if line.startswith(_START):
            inside = True
            continue
        if line.startswith(_END):
            inside = False
            continue
        if inside:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                angle = float(parts[0])
                counts = float(parts[1])
                factor = float(parts[2]) if len(parts) > 2 else 1.0
            except ValueError:
                continue
            rows.append((angle, counts, factor))
        else:
            found = _HEADER.match(line)
            if found:
                header[found.group(1)] = found.group(2)

    if len(rows) < 8:
        raise RASError(
            f"{p.name}: el bloque de datos tiene {len(rows)} filas legibles, "
            "que no alcanzan para un difractograma")

    data = np.asarray(rows, dtype=float)
    two_theta, counts, attenuation = data[:, 0], data[:, 1], data[:, 2]

    # The attenuator is a physical filter in the beam: when it is in,
    # the recorded counts are what got through, and the true intensity
    # is that times the factor. Ignoring a varying factor leaves a step
    # in the pattern exactly where the instrument switched it.
    if not np.allclose(attenuation, 1.0):
        counts = counts * attenuation

    step = _number(header, "MEAS_SCAN_STEP")
    speed = _number(header, "MEAS_SCAN_SPEED")          # deg/min
    dwell = None
    if step and speed and speed > 0:
        dwell = 60.0 * step / speed

    return {
        "two_theta": two_theta,
        "intensity": counts,
        "attenuation": attenuation,
        "wavelength": _number(header, "HW_XG_WAVE_LENGTH_ALPHA1"),
        "wavelength_alpha2": _number(header, "HW_XG_WAVE_LENGTH_ALPHA2"),
        "step": step,
        "scan_speed_deg_per_min": speed,
        "dwell": dwell,
        "unit": header.get("MEAS_SCAN_UNIT_Y") or header.get("DISP_UNIT_Y"),
        "sample": (header.get("FILE_SAMPLE") or "").strip(),
        "operator": (header.get("FILE_OPERATOR") or "").strip(),
        "started": header.get("MEAS_SCAN_START_TIME"),
        "axis": header.get("MEAS_SCAN_AXIS_X"),
        "header": header,
    }


def load_ras_pattern(path: str | Path):
    """Read a .ras straight into a :class:`~ramancarbon.xrd.pattern.Pattern`."""
    from .pattern import Pattern

    parsed = read_ras(path)
    pattern = Pattern(two_theta=parsed["two_theta"],
                      intensity=parsed["intensity"])
    for field in ("wavelength", "wavelength_alpha2", "dwell", "unit",
                  "sample", "operator", "started"):
        value = parsed.get(field)
        if value is None:
            continue
        try:
            setattr(pattern, field, value)
        except (AttributeError, TypeError):
            break
    return pattern


ASC_HEADER = re.compile(r"^\*([A-Z0-9_]+)\s*=\s*(.*?)\s*$")


def read_asc(path: str | Path) -> dict:
    """Read a Rigaku ``.asc`` export.

    The same instrument as the ``.ras``, written differently: the header
    is ``*KEY =  value`` and the counts come packed several to a line,
    comma separated, with **no angle column at all** -- the angle is
    implied by ``*START``, ``*STEP`` and the position in the stream.

    That packing is why the file has to be read rather than fed to a
    generic table parser. Four counts per line look exactly like four
    columns of data, and the generic reader duly reported "x from 305 to
    1044 in 2000 points, 4 columns" and refused the file. Nothing about
    the numbers says they are not columns; only the header does.
    """
    p = Path(path)
    text = p.read_bytes().decode("latin-1")
    header: dict[str, str] = {}
    counts: list[float] = []
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("*BEGIN"):
            inside = True
            continue
        if stripped.startswith(("*END", "*EOF")):
            inside = False
            continue
        found = ASC_HEADER.match(stripped)
        if found:
            # Keys repeat inside a group (*START, *STEP...); the last one
            # wins, which is the group actually being read.
            header[found.group(1)] = found.group(2)
            continue
        if inside and stripped:
            for piece in stripped.split(","):
                piece = piece.strip()
                if not piece:
                    continue
                try:
                    counts.append(float(piece))
                except ValueError:
                    pass

    if len(counts) < 8:
        raise RASError(
            f"{p.name}: sólo se leyeron {len(counts)} cuentas entre *BEGIN y "
            "*END; ¿es realmente un .asc de Rigaku?")

    start = _number(header, "START")
    step = _number(header, "STEP")
    declared = _number(header, "COUNT")
    if start is None or step is None or step <= 0:
        raise RASError(
            f"{p.name}: el cabecero no trae *START y *STEP utilizables, y sin "
            "ellos no hay eje: el archivo no guarda los ángulos")
    if declared and abs(declared - len(counts)) > 0.5:
        # Worth saying rather than silently trusting one of the two: a
        # short read gives a pattern that ends early and looks fine.
        raise RASError(
            f"{p.name}: el cabecero declara {declared:.0f} puntos y se "
            f"leyeron {len(counts)}")

    intensity = np.asarray(counts, dtype=float)
    two_theta = start + step * np.arange(intensity.size, dtype=float)

    speed = _number(header, "SPEED")
    dwell = speed if (header.get("SPEED_DIM", "").startswith("sec")
                      and speed) else None
    return {
        "two_theta": two_theta,
        "intensity": intensity,
        "attenuation": np.ones_like(intensity),
        "wavelength": _number(header, "WAVE_LENGTH1"),
        "wavelength_alpha2": _number(header, "WAVE_LENGTH2"),
        "step": step,
        "scan_speed_deg_per_min": None,
        "dwell": dwell,
        "unit": header.get("YUNIT"),
        "sample": (header.get("SAMPLE") or "").strip(),
        "operator": "",
        "started": header.get("DATE"),
        "axis": header.get("SCAN_AXIS"),
        "header": header,
    }


__all__ = ["RASError", "load_ras_pattern", "read_asc", "read_ras"]
