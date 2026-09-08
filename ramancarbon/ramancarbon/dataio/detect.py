"""What is this file? — asked of the data, not of the extension.

The suite reads four kinds of measurement and users have all four in
folders called ``datos`` with everything named ``.txt``. Sorting them by
extension does not work: ``.txt``, ``.csv``, ``.dat`` and ``.asc`` are used
by every instrument in the building.

So the decision is made from the numbers. A Raman spectrum, a
diffractogram, a voltammogram, a charge–discharge curve and an impedance
spectrum have shapes that are not alike:

============ ====================================================
Raman        x rises monotonically over hundreds to thousands of
             cm⁻¹, y is positive, one value per x.
DRX          x rises monotonically inside 0–160°, in fine even
             steps, y looks like counts.
CV           x is **not** monotonic — it goes up and comes back.
             That single fact separates it from everything else.
GCD          x is time, rising; y is a potential that goes up and
             down inside a volt or two.
EIS          three columns, frequency spanning decades, and a
             column that is mostly negative (Z″).
============ ====================================================

Where two readings are possible, both are reported. A pattern measured
from 10 to 80° and a Raman spectrum of a low-frequency region measured
from 10 to 80 cm⁻¹ are the same numbers; nothing in the file distinguishes
them, and saying so is better than choosing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

#: Extensions that name their instrument unambiguously.
KNOWN_SUFFIXES: dict[str, str] = {
    ".xrdml": "xrd", ".uxd": "xrd", ".xye": "xrd", ".qam": "xrd",
    ".xy": "xrd", ".cif": "cif",
    ".jdx": "jcamp", ".dx": "jcamp", ".jcm": "jcamp",
    ".mpt": "echem", ".mpr": "echem-binario", ".nox": "echem",
    ".spc": "raman-binario", ".wdf": "raman-binario", ".sp": "raman-binario",
    ".spa": "raman-binario", ".ngs": "raman-binario",
    ".raw": "xrd-binario", ".brml": "xrd-binario",
    ".rcproj": "proyecto",
}

#: Binary formats, with the export that replaces them.
BINARY_ADVICE: dict[str, str] = {
    "raman-binario": "expórtalo desde el programa del equipo como texto de dos "
                     "columnas o como JCAMP-DX (.jdx)",
    "xrd-binario": "expórtalo como .xy, .xye o .xrdml desde el programa del "
                   "difractómetro",
    "echem-binario": "expórtalo como texto (.mpt en EC-Lab, «Export as text»)",
}


@dataclass
class Detection:
    """What a file appears to hold, and how sure that is."""

    kind: str
    """``raman``, ``xrd``, ``cv``, ``gcd``, ``eis``, ``cif``, ``proyecto``,
    ``binario`` or ``desconocido``."""
    fmt: str = "texto"
    """The encoding: ``texto``, ``jcamp``, ``xrdml``, ``uxd``, ``cif``…"""
    confidence: str = "media"
    reasons: list[str] = field(default_factory=list)
    """Why. Shown to the user, because a wrong guess has to be arguable."""
    alternatives: list[str] = field(default_factory=list)
    """Other readings the data also fit."""
    columns: int = 0
    header: dict[str, Any] = field(default_factory=dict)
    advice: str = ""
    """What to do when the file cannot be read at all."""

    def __str__(self) -> str:
        text = f"{self.kind} ({self.fmt}, confianza {self.confidence})"
        if self.alternatives:
            text += f"; también podría ser: {', '.join(self.alternatives)}"
        return text


_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_DELIMITERS = ("\t", ";", ",", None)


def _read_text(path: Path, limit: int = 60_000) -> tuple[list[str], bytes]:
    """The file as lines, up to a few megabytes.

    Reading only the first few hundred lines is faster and wrong: the
    decision turns on where the x axis *ends*, and the first 400 points of
    a Raman spectrum span 90 to 490 cm⁻¹ — which is inside the range of a
    diffractogram.
    """
    raw = path.read_bytes()
    head = raw[:8_000_000]
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        text = head.decode("latin-1")
    return text.splitlines()[:limit], raw[:512]


def _looks_binary(sample: bytes) -> bool:
    """A NUL byte in the first half-kilobyte means it is not text.

    Checked rather than assumed from the extension, because vendors write
    binary into ``.txt`` and text into ``.raw``.
    """
    return b"\x00" in sample


def _table(lines: list[str]) -> Optional[np.ndarray]:
    """The numeric block of a text file, as an array, or ``None``."""
    best: Optional[np.ndarray] = None
    for delimiter in _DELIMITERS:
        rows: list[list[float]] = []
        width = 0
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped[0] in "#%!'\"" or stripped.startswith("##"):
                continue
            parts = stripped.split(delimiter) if delimiter else stripped.split()
            values: list[float] = []
            for part in parts:
                part = part.strip().replace(",", ".") if delimiter != "," else part.strip()
                try:
                    values.append(float(part))
                except ValueError:
                    values = []
                    break
            if len(values) >= 2:
                if width and len(values) != width:
                    continue
                width = len(values)
                rows.append(values)
        if len(rows) >= 8:
            candidate = np.asarray(rows, dtype=float)
            if best is None or candidate.size > best.size:
                best = candidate
    return best


def _monotone(x: np.ndarray) -> bool:
    step = np.diff(x)
    return bool(np.all(step > 0) or np.all(step < 0))


def _turning_points(values: np.ndarray, fraction: float = 0.05) -> int:
    """How many times a column really reverses direction.

    A voltammogram comes back on itself and nothing else in the suite
    does, which makes this the most discriminating number in the module —
    but only if noise does not count. Counting sign changes of the first
    difference gives a clean diffraction pattern several hundred
    "reversals", so a reversal is counted only once the column has moved
    back by ``fraction`` of its own range.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return 0
    span = float(np.nanmax(values) - np.nanmin(values))
    if span <= 0:
        return 0
    threshold = fraction * span
    reversals = 0
    direction = 0
    lowest = highest = extreme = float(values[0])
    for value in values[1:]:
        value = float(value)
        if direction == 0:
            # Which way the column is going has to be established before a
            # change of direction can be counted; otherwise the initial
            # rise of any curve reads as a reversal.
            highest, lowest = max(highest, value), min(lowest, value)
            if value - lowest > threshold:
                direction, extreme = 1, value
            elif highest - value > threshold:
                direction, extreme = -1, value
        elif direction > 0:
            if value > extreme:
                extreme = value
            elif extreme - value > threshold:
                reversals += 1
                direction, extreme = -1, value
        else:
            if value < extreme:
                extreme = value
            elif value - extreme > threshold:
                reversals += 1
                direction, extreme = 1, value
    return reversals


def _ascending(x: np.ndarray) -> bool:
    """Non-decreasing, and actually going somewhere.

    Not strict: a charge-discharge file records time to a hundredth of a
    second and repeats a timestamp now and then, and demanding a strictly
    increasing axis threw those out.
    """
    step = np.diff(x)
    return bool(np.all(step >= -1e-12) and np.count_nonzero(step > 0) > 0.5 * step.size)


def detect(path: str | Path) -> Detection:
    """Say what a file holds.

    Never raises for an unreadable file: an unknown file is a result, and
    a batch import that stops on the first stray README is worse than one
    that reports it.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    named = KNOWN_SUFFIXES.get(suffix, "")

    if not p.is_file():
        return Detection(kind="desconocido", confidence="alta",
                         reasons=[f"no existe: {p}"])

    lines, sample = _read_text(p)

    if named.endswith("-binario") or (_looks_binary(sample) and named != "proyecto"):
        family = named if named.endswith("-binario") else "binario"
        return Detection(
            kind="binario", fmt=suffix.lstrip(".") or "binario", confidence="alta",
            reasons=["el archivo es binario, no texto"],
            advice=BINARY_ADVICE.get(
                family,
                "este formato es binario y hace falta la biblioteca del "
                "fabricante; expórtalo como texto desde el programa del equipo",
            ),
        )

    if named == "proyecto":
        return Detection(kind="proyecto", fmt="rcproj", confidence="alta",
                         reasons=["extensión de proyecto de la suite"])

    text_head = "\n".join(lines[:80])

    if named == "cif" or "_cell_length_a" in text_head:
        return Detection(kind="cif", fmt="cif", confidence="alta",
                         reasons=["contiene ítems cristalográficos (_cell_length_a)"])

    if text_head.lstrip().startswith("##TITLE") or "##XYDATA" in text_head \
            or "##XYPOINTS" in text_head or named == "jcamp":
        return _detect_jcamp(p, text_head)

    if "<xrdMeasurement" in text_head or suffix == ".xrdml":
        return Detection(kind="xrd", fmt="xrdml", confidence="alta",
                         reasons=["XML de PANalytical"])
    if "_2THETACOUNTS" in text_head.upper() or suffix == ".uxd":
        return Detection(kind="xrd", fmt="uxd", confidence="alta",
                         reasons=["cabecera Bruker UXD"])

    table = _table(lines)
    if table is None:
        return Detection(
            kind="desconocido", confidence="alta",
            reasons=["no se encontró ningún bloque de al menos ocho filas "
                     "con dos o más columnas numéricas"],
            advice="comprueba que el archivo sea una tabla de texto",
        )

    detection = _from_shape(table, header_text=text_head)
    detection.columns = int(table.shape[1])
    if named == "xrd" and detection.kind != "xrd":
        detection.alternatives.append(detection.kind)
        detection.kind = "xrd"
        detection.reasons.append(f"la extensión {suffix} es de difracción")
    return detection


def _detect_jcamp(path: Path, head: str) -> Detection:
    """A JCAMP file says what it is; the label is trusted, and checked."""
    from .jcamp import parse_jcamp

    match = re.search(r"##DATA[\s_-]*TYPE\s*=\s*(.+)", head, re.IGNORECASE)
    declared = (match.group(1).strip().upper() if match else "")
    kind = "raman"
    reasons = [f"JCAMP-DX, ##DATA TYPE={declared or 'sin declarar'}"]
    confidence = "alta" if declared else "media"
    if "XRD" in declared or "DIFFRAC" in declared or "X-RAY" in declared:
        kind = "xrd"
    elif "INFRARED" in declared or "IR " in declared:
        kind = "raman"
        reasons.append("es un espectro infrarrojo: se lee, pero el análisis "
                       "de este programa es de Raman")
        confidence = "baja"
    try:
        parsed = parse_jcamp(path.read_text(encoding="utf-8", errors="replace"))
        if parsed["x_units"] == "deg" and kind == "raman":
            kind, confidence = "xrd", "alta"
            reasons.append("##XUNITS está en grados")
    except Exception as error:                       # noqa: BLE001
        return Detection(kind=kind, fmt="jcamp", confidence="baja",
                         reasons=reasons + [f"no se pudo leer: {error}"])
    return Detection(kind=kind, fmt="jcamp", confidence=confidence, reasons=reasons)


def _from_shape(table: np.ndarray, header_text: str = "") -> Detection:
    """The decision itself, from the columns' own shape."""
    x = table[:, 0]
    y = table[:, 1]
    columns = table.shape[1]
    span = float(np.nanmax(x) - np.nanmin(x))
    low, high = float(np.nanmin(x)), float(np.nanmax(x))
    reasons: list[str] = []
    alternatives: list[str] = []

    turns = _turning_points(x)
    if turns >= 1 and span < 10.0:
        # A voltammogram, and nothing else in the suite, comes back on
        # itself. The window check keeps a noisy Raman axis from qualifying.
        return Detection(
            kind="cv", confidence="alta" if turns >= 2 else "media",
            reasons=[f"la primera columna cambia de sentido {turns} "
                     f"{'vez' if turns == 1 else 'veces'} en una ventana de "
                     f"{span:.2f} V: es un barrido de ida y vuelta"],
        )

    if columns >= 3:
        third = table[:, 2]
        decades = (np.log10(max(high, 1e-12)) - np.log10(max(low, 1e-12))
                   if low > 0 else 0.0)
        negative = float(np.mean(third < 0))
        if decades >= 2 and (negative > 0.5 or np.mean(y > 0) > 0.9):
            return Detection(
                kind="eis", confidence="alta",
                reasons=[f"tres columnas y la frecuencia abarca {decades:.1f} "
                         "décadas"],
            )

    if _ascending(x):
        reversals = _turning_points(y)
        starts_at_zero = low <= 0.05 * max(span, 1e-9)
        bounded = float(np.nanmax(np.abs(y))) < 20.0
        if starts_at_zero and bounded and 1 <= reversals <= 20:
            # Time against potential: a charge-discharge curve. The
            # reversal count is what keeps a normalised diffractogram out —
            # that has hundreds of them, a triangular wave has one per
            # cycle.
            return Detection(
                kind="gcd", confidence="alta" if reversals >= 2 else "media",
                reasons=[f"x crece desde cero (tiempo) hasta {high:.4g} y la "
                         f"segunda columna sube y baja {reversals} "
                         f"{'vez' if reversals == 1 else 'veces'} dentro de "
                         f"{float(np.nanmax(np.abs(y))):.2f} V"],
            )

        if 0.0 <= low and high <= 165.0 and span > 5.0:
            counts = bool(np.allclose(y, np.round(y), atol=1e-6)) and np.nanmin(y) >= 0
            reasons.append(f"x crece de {low:.2f} a {high:.2f}, dentro del "
                           "intervalo de 2θ")
            if counts:
                reasons.append("y son cuentas enteras no negativas")
            confidence = "alta" if counts else "media"
            if not counts and span > 40.0 and high > 60.0:
                alternatives.append("raman (región de baja frecuencia)")
            return Detection(kind="xrd", confidence=confidence,
                             reasons=reasons, alternatives=alternatives)

        if -2000.0 <= low and high <= 6000.0 and span > 150.0:
            reasons.append(f"x crece de {low:.0f} a {high:.0f} cm⁻¹, con un "
                           f"recorrido de {span:.0f}")
            if re.search(r"cm-?\s*-?1|cm\^-1|raman", header_text, re.IGNORECASE):
                reasons.append("la cabecera menciona cm⁻¹ o Raman")
                return Detection(kind="raman", confidence="alta", reasons=reasons)
            return Detection(kind="raman", confidence="media", reasons=reasons)

    return Detection(
        kind="desconocido", confidence="baja",
        reasons=[f"x va de {low:.4g} a {high:.4g} en {table.shape[0]} puntos "
                 f"y {columns} columnas; no encaja con ninguna de las medidas "
                 "que lee el programa"],
        advice="indica el tipo explícitamente al importar",
    )


def scan(folder: str | Path, pattern: str = "*") -> dict[str, list[Path]]:
    """Sort a whole folder by what its files hold.

    The point of a folder scan is the *unknown* bucket. It is where the
    file with the wrong extension, the truncated export and the vendor
    binary end up, and those are the ones that need a person.
    """
    directory = Path(folder)
    result: dict[str, list[Path]] = {}
    for item in sorted(directory.glob(pattern)):
        if not item.is_file():
            continue
        result.setdefault(detect(item).kind, []).append(item)
    return result


__all__ = [
    "BINARY_ADVICE",
    "KNOWN_SUFFIXES",
    "Detection",
    "detect",
    "scan",
]
