"""Readers for what potentiostats export.

Every format handled here is text, for the same reason as on the
diffraction side: guessing at a binary layout is the quietest way to
misread data. The three that matter in practice are all text anyway —
Bio-Logic ``.mpt``, Gamry ``.DTA`` and CH Instruments ``.txt`` — and each
carries its own header with the units, which is worth reading rather than
assuming.

**Units are the whole problem.** Potentiostats export volts or millivolts,
amps or milliamps or microamps, and seconds or milliseconds, and which they
use depends on the software's display settings at the time. A capacitance
computed from milliamps taken for amps is out by a thousand. So the column
units are read from the header where there is one, and where there is not
the caller must say — there is no guessing from magnitudes, because a
0.001 could be an amp or a milliamp with equal ease.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .curve import ChargeDischarge, Electrode, Impedance, Voltammogram

#: Multipliers to SI, by the unit strings these files use.
UNIT_SCALE: dict[str, float] = {
    "v": 1.0, "mv": 1e-3, "uv": 1e-6, "µv": 1e-6,
    "a": 1.0, "ma": 1e-3, "ua": 1e-6, "µa": 1e-6, "na": 1e-9,
    "s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0,
    "ohm": 1.0, "mohm": 1e-3, "kohm": 1e3, "ω": 1.0,
    "hz": 1.0, "khz": 1e3, "mhz": 1e6,
}

#: Column names, lowercased, that mean each quantity.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "potential": ("ewe", "e", "potential", "v", "voltage", "vf", "potencial",
                  "working electrode potential"),
    "current": ("i", "current", "im", "corriente", "<i>", "i/ma", "current/a"),
    "time": ("time", "t", "tiempo", "time/s", "elapsed time"),
    "cycle": ("cycle", "cycle number", "ciclo", "half cycle"),
    "frequency": ("freq", "frequency", "f", "frecuencia", "freq/hz"),
    "z_real": ("zre", "z'", "re(z)", "zreal", "z_real", "zr"),
    "z_imag": ("zim", "z''", "-z''", "im(z)", "zimag", "z_imag", "zi"),
}


class EchemIOError(ValueError):
    """Raised when an electrochemistry file cannot be read."""


def _split_unit(header: str) -> tuple[str, float]:
    """``"Ewe/V"`` → ``("ewe", 1.0)``; ``"<I>/mA"`` → ``("i", 1e-3)``."""
    text = header.strip().lower()
    scale = 1.0
    for separator in ("/", " (", "("):
        if separator in text:
            name, _, unit = text.partition(separator)
            unit = unit.strip(" )").strip()
            if unit in UNIT_SCALE:
                return name.strip(), UNIT_SCALE[unit]
            text = name
            break
    return text.strip(), scale


def _identify(headers: Sequence[str]) -> dict[str, tuple[int, float]]:
    """Map quantity → (column index, unit scale)."""
    found: dict[str, tuple[int, float]] = {}
    for index, header in enumerate(headers):
        name, scale = _split_unit(header)
        for quantity, aliases in COLUMN_ALIASES.items():
            if quantity in found:
                continue
            if name in aliases or any(name.startswith(a) for a in aliases if len(a) > 2):
                found[quantity] = (index, scale)
                break
    return found


def _read_table(path: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Pull a header row and a numeric block out of a text export."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    metadata: dict[str, Any] = {}
    header: list[str] = []
    rows: list[list[float]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = re.split(r"[\t,;]|\s{2,}", stripped)
        parts = [p for p in parts if p]
        numeric = []
        ok = True
        for part in parts:
            try:
                numeric.append(float(part.replace(",", ".")))
            except ValueError:
                ok = False
                break
        if ok and len(numeric) >= 2:
            rows.append(numeric)
            continue
        if rows:
            continue  # trailing prose
        if len(parts) >= 2 and any(
            any(alias in p.lower() for aliases in COLUMN_ALIASES.values()
                for alias in aliases)
            for p in parts
        ):
            header = parts
    # "key: value" lines above the table carry the scan rate and the units.
    for line in lines[: max(len(lines) - len(rows), 0)]:
        # Comment markers are stripped first: the scan rate is normally
        # written as "# scan rate: 20 mV/s", and a metadata scan that
        # insists the key start with a letter never sees it.
        line = line.lstrip("#!*; \t")
        match = re.match(r"^\s*([A-Za-z][\w .%/()-]*)\s*[:=]\s*(.+?)\s*$", line)
        if match:
            metadata[match.group(1).strip().lower()] = match.group(2).strip()

    if len(rows) < 5:
        raise EchemIOError(
            f"{path.name}: no se han encontrado al menos 5 filas de datos "
            "numéricos. Comprueba que es un archivo de texto exportado del "
            "potenciostato y no un formato binario"
        )
    width = min(len(r) for r in rows)
    table = np.array([r[:width] for r in rows], dtype=float)
    return header[:width], table, metadata


def read_cv(
    path: str | Path,
    scan_rate: Optional[float] = None,
    electrode: Optional[Electrode] = None,
    potential_scale: float = 1.0,
    current_scale: float = 1.0,
) -> Voltammogram:
    """Read a cyclic voltammogram.

    Parameters
    ----------
    scan_rate:
        In V/s. Taken from the file header when it states one; otherwise it
        must be given, because a capacitance cannot be computed without it
        and inferring it from the data assumes the time column is right.
    potential_scale, current_scale:
        Multipliers to volts and amps, for files with no unit headers.
        There is no guessing: a column of numbers around 0.001 is equally
        plausibly amps or milliamps.
    """
    location = Path(path)
    header, table, metadata = _read_table(location)
    columns = _identify(header) if header else {}

    if "potential" in columns and "current" in columns:
        p_index, p_scale = columns["potential"]
        i_index, i_scale = columns["current"]
    else:
        p_index, p_scale = 0, 1.0
        i_index, i_scale = 1, 1.0
    potential = table[:, p_index] * p_scale * potential_scale
    current = table[:, i_index] * i_scale * current_scale

    cycle = None
    if "cycle" in columns:
        cycle = table[:, columns["cycle"][0]]

    rate = scan_rate
    if rate is None:
        for key, value in metadata.items():
            if "scan rate" in key or "velocidad" in key or key.strip() == "dE/dt":
                match = re.search(r"([\d.eE+-]+)\s*(mv|v)?", value.lower())
                if match:
                    rate = float(match.group(1))
                    if match.group(2) == "mv" or "mv" in key:
                        rate *= 1e-3
                break
    if rate is None:
        raise EchemIOError(
            f"{location.name}: hace falta la velocidad de barrido y el archivo "
            "no la declara. Pásala con scan_rate=, en V/s"
        )

    used = electrode or Electrode()
    declared = metadata.get("reference")
    warning = None
    if declared and declared != used.reference:
        # Not overridden silently: a potential scale is the user's to
        # declare, and quietly using the file's would be just as wrong when
        # the file is the one that is stale. But a mismatch here is 1.04 V
        # on every overpotential downstream, so it cannot pass unmentioned.
        warning = (
            f"el archivo declara la referencia {declared!r} y se ha pedido "
            f"{used.reference!r}. Una de las dos está mal, y la diferencia va "
            "entera a cualquier sobrepotencial que salga después"
        )
    return Voltammogram(
        potential=potential,
        current=current,
        scan_rate=float(rate),
        electrode=used,
        cycle=cycle,
        name=location.stem,
        metadata={
            "path": str(location),
            **metadata,
            **({"aviso_referencia": warning} if warning else {}),
        },
    )


def read_gcd(
    path: str | Path,
    electrode: Optional[Electrode] = None,
    current: Optional[float] = None,
    potential_scale: float = 1.0,
    current_scale: float = 1.0,
) -> ChargeDischarge:
    """Read a galvanostatic charge–discharge file.

    ``current`` sets a constant current in amps for the files that record
    only time and potential. Its **sign matters**: the branches are split
    by it, so a file with a single positive value is read as one long
    charge. When a current column exists it is used instead.
    """
    location = Path(path)
    header, table, metadata = _read_table(location)
    columns = _identify(header) if header else {}

    t_index, t_scale = columns.get("time", (0, 1.0))
    p_index, p_scale = columns.get("potential", (1, 1.0))
    time = table[:, t_index] * t_scale
    potential = table[:, p_index] * p_scale * potential_scale

    if "current" in columns:
        i_index, i_scale = columns["current"]
        current_values = table[:, i_index] * i_scale * current_scale
    elif current is not None:
        # Alternate the sign at every turning point of the potential.
        slope = np.gradient(potential)
        current_values = np.where(slope >= 0, abs(current), -abs(current))
    else:
        raise EchemIOError(
            f"{location.name}: no hay columna de corriente y no se ha dado "
            "current=. Sin ella no se pueden separar carga y descarga"
        )

    cycle = table[:, columns["cycle"][0]] if "cycle" in columns else None
    return ChargeDischarge(
        time=time,
        potential=potential,
        current=current_values,
        electrode=electrode or Electrode(),
        cycle=cycle,
        name=location.stem,
        metadata={"path": str(location), **metadata},
    )


def read_eis(
    path: str | Path,
    electrode: Optional[Electrode] = None,
    imaginary_is_negated: Optional[bool] = None,
) -> Impedance:
    """Read an impedance spectrum.

    ``imaginary_is_negated`` handles the sign convention. Many exports
    write ``-Z''`` because that is what a Nyquist plot shows, and reading it
    as ``Z''`` flips the spectrum into the inductive half-plane where every
    circuit fit fails. Left as ``None`` the sign is inferred: a physically
    ordinary electrode is capacitive over most of its range, so whichever
    convention makes the majority of the imaginary parts negative is the
    one taken — and the choice is recorded in the metadata.
    """
    location = Path(path)
    header, table, metadata = _read_table(location)
    columns = _identify(header) if header else {}

    f_index, f_scale = columns.get("frequency", (0, 1.0))
    r_index, r_scale = columns.get("z_real", (1, 1.0))
    i_index, i_scale = columns.get("z_imag", (2, 1.0))
    if table.shape[1] < 3:
        raise EchemIOError(
            f"{location.name}: un espectro de impedancia necesita al menos "
            "tres columnas (frecuencia, Z' y Z'')"
        )
    frequency = table[:, f_index] * f_scale
    real = table[:, r_index] * r_scale
    imaginary = table[:, i_index] * i_scale

    negated = imaginary_is_negated
    if negated is None:
        header_text = " ".join(header).lower()
        if "-z" in header_text:
            negated = True
        else:
            negated = float(np.mean(imaginary)) > 0.0
    if negated:
        imaginary = -imaginary

    return Impedance(
        frequency=frequency,
        z=real + 1j * imaginary,
        electrode=electrode or Electrode(),
        name=location.stem,
        metadata={
            "path": str(location),
            "imaginario_invertido": bool(negated),
            **metadata,
        },
    )


def write_cv(curve: Voltammogram, path: str | Path) -> Path:
    """Write a voltammogram as a labelled two-column text file."""
    destination = Path(path)
    electrode = curve.electrode
    lines = [
        f"# {curve.name}",
        f"# scan rate: {1e3 * curve.scan_rate:g} mV/s",
        f"# reference: {electrode.reference}",
    ]
    if electrode.ph is not None:
        lines.append(f"# pH: {electrode.ph:g}")
    if electrode.mass_mg:
        lines.append(f"# active mass: {electrode.mass_mg:g} mg")
    if electrode.area_cm2:
        lines.append(f"# area: {electrode.area_cm2:g} cm2")
    lines.append("Ewe/V\tI/A")
    lines.extend(f"{v:.6f}\t{i:.9g}" for v, i in zip(curve.potential, curve.current))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def write_gcd(curve: ChargeDischarge, path: str | Path) -> Path:
    """Write a charge–discharge curve as three columns."""
    destination = Path(path)
    lines = [f"# {curve.name}", "time/s\tEwe/V\tI/A"]
    lines.extend(
        f"{t:.6f}\t{v:.6f}\t{i:.9g}"
        for t, v, i in zip(curve.time, curve.potential, curve.current)
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def write_eis(spectrum: Impedance, path: str | Path) -> Path:
    """Write an impedance spectrum as three columns, Z'' with its own sign."""
    destination = Path(path)
    lines = [f"# {spectrum.name}", "freq/Hz\tZre/ohm\tZim/ohm"]
    lines.extend(
        f"{f:.6g}\t{z.real:.6g}\t{z.imag:.6g}"
        for f, z in zip(spectrum.frequency, spectrum.z)
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


__all__ = [
    "COLUMN_ALIASES",
    "UNIT_SCALE",
    "EchemIOError",
    "read_cv",
    "read_eis",
    "read_gcd",
    "write_cv",
    "write_eis",
    "write_gcd",
]
