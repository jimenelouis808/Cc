"""Readers for the files diffractometers actually produce.

Every format here is text, which is the criterion for inclusion: a binary
export (Bruker ``.raw``, ``.brml``) needs the vendor's own library and
guessing at its layout is how data get silently misread. When one of those
turns up the reader says which format it is and what to export instead,
rather than trying.

What is handled:

``.xy`` ``.dat`` ``.txt`` ``.asc`` ``.csv``
    Two or three whitespace- or comma-separated columns, with any number
    of comment lines. The workhorse.
``.xye`` ``.qam``
    Three columns where the third is a measured e.s.d. Those are used
    directly instead of √N, which matters: a pattern that has been
    background-subtracted or scaled no longer has Poisson statistics and
    assuming it does mis-weights the whole refinement.
``.xrdml``
    PANalytical's XML export. Common, entirely text, and it carries the
    wavelength and the step explicitly — which is worth having, since a
    wavelength guessed wrong shifts every refined lattice parameter by the
    same fraction and nothing in the fit complains.
``.uxd``
    Bruker's ASCII export.

The wavelength is taken from the file when the file states it and from the
``anode`` argument otherwise. It is never inferred from the data.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

from .pattern import Pattern, PatternError
from .scattering import ANODES, wavelength_for

#: Extensions whose third column is a measured uncertainty.
ESD_SUFFIXES = (".xye", ".qam")

#: Binary formats, named so the error message can be useful.
BINARY_SUFFIXES = {
    ".raw": "Bruker RAW",
    ".brml": "Bruker BRML (ZIP+XML)",
    ".rd": "Philips RD",
    ".sad": "Siemens SAD",
    ".dif": "binario de difractómetro",
}

_NUMBER_LINE = re.compile(r"^[\s]*[-+]?[\d.]")


class PatternIOError(ValueError):
    """Raised when a diffractogram file cannot be read."""


def _columns(text: str) -> tuple[np.ndarray, ...]:
    """Pull numeric columns out of a text file, ignoring headers."""
    rows: list[list[float]] = []
    width = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or not _NUMBER_LINE.match(stripped):
            continue
        if stripped[0] in "#!*;'\"":
            continue
        parts = re.split(r"[,;\s]+", stripped)
        try:
            values = [float(p) for p in parts if p not in ("", None)]
        except ValueError:
            continue
        if len(values) < 2:
            continue
        rows.append(values)
        width = max(width, len(values))
    if len(rows) < 5:
        raise PatternIOError(
            "no se han encontrado al menos 5 filas de dos columnas numéricas. "
            "¿Es realmente un difractograma en texto?"
        )
    usable = min(len(r) for r in rows)
    array = np.array([r[:usable] for r in rows], dtype=float)
    return tuple(array[:, i] for i in range(array.shape[1]))


def _read_xrdml(path: Path) -> Pattern:
    """PANalytical XRDML: an XML file with a start, an end and a count list."""
    tree = ET.parse(path)
    root = tree.getroot()
    namespace = root.tag.split("}")[0] + "}" if "}" in root.tag else ""

    def find(tag: str):
        return root.iter(namespace + tag)

    intensities: Optional[np.ndarray] = None
    for element in find("intensities"):
        intensities = np.array(element.text.split(), dtype=float)
        break
    if intensities is None:
        for element in find("counts"):
            intensities = np.array(element.text.split(), dtype=float)
            break
    if intensities is None:
        raise PatternIOError(f"{path.name}: no contiene una lista de intensidades")

    start = end = None
    for element in find("positions"):
        if element.get("axis") not in ("2Theta", None):
            continue
        for child in element:
            tag = child.tag.split("}")[-1]
            if tag == "startPosition":
                start = float(child.text)
            elif tag == "endPosition":
                end = float(child.text)
        if start is not None and end is not None:
            break
    if start is None or end is None:
        raise PatternIOError(f"{path.name}: no se encuentra el rango de 2θ")

    wavelength = None
    for element in find("kAlpha1"):
        wavelength = float(element.text)
        break
    anode = "Cu"
    for element in find("usedWavelength"):
        anode = element.get("intended") or anode
        break

    axis = np.linspace(start, end, intensities.size)
    return Pattern(
        two_theta=axis,
        intensity=intensities,
        wavelength=wavelength or wavelength_for("Cu", "ka1"),
        name=path.stem,
        anode=anode if anode in ANODES else "Cu",
        counts=True,
        metadata={"format": "xrdml", "path": str(path)},
    )


def _read_uxd(path: Path) -> Pattern:
    """Bruker UXD: key/value header lines then the counts."""
    text = path.read_text(encoding="utf-8", errors="replace")
    header = {}
    for line in text.splitlines():
        if line.startswith("_"):
            key, _, value = line[1:].partition("=")
            header[key.strip().upper()] = value.strip()
    columns = _columns(text)
    wavelength = float(header.get("WL1", header.get("WAVELENGTH", 0)) or 0) or None
    return Pattern(
        two_theta=columns[0],
        intensity=columns[1],
        wavelength=wavelength or wavelength_for("Cu", "ka1"),
        name=path.stem,
        counts=True,
        metadata={"format": "uxd", "path": str(path), **header},
    )


def read_pattern(
    path: str | Path,
    wavelength: Optional[float] = None,
    anode: str = "Cu",
    line: str = "ka1",
    counts: bool = True,
    kalpha2_ratio: float = 0.5,
) -> Pattern:
    """Read a diffractogram.

    Parameters
    ----------
    path:
        The file. The format is chosen from the extension and, for text
        files, from the number of columns.
    wavelength:
        In Å. Overrides whatever the file says, for the case where the
        file is wrong — which happens, and which shifts every refined
        lattice parameter by the same relative amount without anything in
        the fit objecting.
    anode, line:
        Used when neither ``wavelength`` nor the file supplies one. The
        default is Kα₁, with the satellite carried separately in
        ``kalpha2_ratio``.
    kalpha2_ratio:
        Intensity of the Kα₂ satellite relative to Kα₁. Set it to 0 for
        monochromated or synchrotron data: leaving it at 0.5 invents a
        satellite the data do not contain, and a profile fit then widens
        every peak to cover it.
    counts:
        Whether the intensities are raw counts. If the pattern has already
        been background-subtracted, smoothed or scaled, pass ``False``:
        √N is then not its uncertainty and using it mis-weights the
        refinement.

    Returns
    -------
    Pattern
    """
    location = Path(path)
    if not location.is_file():
        raise PatternIOError(f"no existe el archivo {location}")
    suffix = location.suffix.lower()

    if suffix in BINARY_SUFFIXES:
        raise PatternIOError(
            f"{location.name} está en formato {BINARY_SUFFIXES[suffix]}, que es "
            "binario y propietario. Expórtalo desde el software del equipo "
            "como texto de dos columnas (.xy, .dat o .txt) o como .xrdml y "
            "vuelve a intentarlo. Adivinar la estructura de un binario es la "
            "forma más silenciosa de leer mal unos datos"
        )

    if suffix == ".xrdml":
        pattern = _read_xrdml(location)
    elif suffix == ".uxd":
        pattern = _read_uxd(location)
    else:
        text = location.read_text(encoding="utf-8", errors="replace")
        columns = _columns(text)
        errors = None
        if len(columns) >= 3 and suffix in ESD_SUFFIXES:
            errors = columns[2]
        try:
            pattern = Pattern(
                two_theta=columns[0],
                intensity=columns[1],
                sigma=errors,
                wavelength=wavelength or wavelength_for(anode, line),
                name=location.stem,
                anode=anode,
                counts=counts,
                kalpha2_ratio=kalpha2_ratio,
                metadata={"format": suffix.lstrip(".") or "texto", "path": str(location)},
            )
        except PatternError as exc:
            raise PatternIOError(f"{location.name}: {exc}") from exc

    if wavelength is not None:
        pattern.wavelength = float(wavelength)
        pattern.metadata["wavelength_overridden"] = True
    pattern.counts = counts
    pattern.kalpha2_ratio = float(kalpha2_ratio)
    return pattern


def write_pattern(pattern: Pattern, path: str | Path) -> Path:
    """Write a pattern as two or three columns of text."""
    destination = Path(path)
    lines = [
        f"# {pattern.name}",
        f"# lambda = {pattern.wavelength:.6f} A ({pattern.anode})",
        f"# sigma  = {pattern.sigma_origin}",
        "# 2theta  intensidad  sigma",
    ]
    for angle, value, error in zip(pattern.two_theta, pattern.intensity, pattern.sigma):
        lines.append(f"{angle:10.5f} {value:14.4f} {error:12.4f}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def read_many(paths, **kwargs) -> list[Pattern]:
    """Read several patterns, keeping the order given."""
    return [read_pattern(p, **kwargs) for p in paths]


__all__ = [
    "BINARY_SUFFIXES",
    "ESD_SUFFIXES",
    "PatternIOError",
    "read_many",
    "read_pattern",
    "write_pattern",
]
