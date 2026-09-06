"""Reading spectra out of whatever the spectrometer wrote.

There is no universal Raman file format. What every vendor's ASCII export
has in common is two numeric columns; what they disagree about is the
delimiter, the decimal separator, the number and syntax of header lines,
whether the abscissa ascends or descends, and whether the abscissa is a
Raman shift at all or an absolute wavelength.

So the reader here sniffs rather than assumes:

* delimiter — tab, comma, semicolon or whitespace, chosen by whichever
  parses the most rows consistently;
* decimal comma — detected and converted, because a Spanish- or
  German-locale export writes ``1580,25`` and a naive float() call turns a
  perfectly good spectrum into an exception;
* header — any leading lines that do not parse as numbers are kept in
  ``metadata["header"]`` and mined for the laser wavelength;
* direction — a descending abscissa is sorted ascending by
  :class:`~ramancarbon.core.spectrum.Spectrum` itself;
* abscissa units — an axis whose values look like nanometres (roughly
  200–2000, monotonic, and inconsistent with a Raman shift) is flagged, not
  silently converted, because converting needs the laser wavelength and
  guessing it would be worse than asking.

Vendor-specific *binary* formats (Renishaw ``.wxd``, Thermo ``.spa``,
Bruker ``.opus``) are not supported and are not going to be guessed at;
export them as text from the vendor software.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .spectrum import Spectrum

#: Extensions the file dialogs offer and :func:`read_spectrum` accepts.
TEXT_SUFFIXES = (".txt", ".csv", ".tsv", ".dat", ".asc", ".spc.txt", ".prn", ".xy")

#: Laser lines commonly used for carbon materials, in nm. Used to snap a
#: value recovered from a header to a sane number and to populate the GUI.
COMMON_LASERS = (325.0, 442.0, 458.0, 473.0, 488.0, 514.5, 532.0, 561.0, 633.0, 660.0, 785.0, 830.0, 1064.0)

#: Words that introduce an excitation wavelength in a vendor header, in the
#: languages these files actually turn up in.
_LASER_WORDS = r"laser|l[áa]ser|excitation|excitaci[óo]n|anregung|wavelength|longitud de onda"

_LASER_PATTERNS = (
    # "Laser: 532 nm", "Excitation wavelength = 632.8 nm"
    re.compile(rf"(?:{_LASER_WORDS})\D{{0,40}}?(\d{{3,4}}(?:[.,]\d+)?)\s*nm", re.I),
    # "Longitud de onda del láser (nm);532,0" — the unit precedes the value.
    re.compile(rf"(?:{_LASER_WORDS})\D{{0,40}}?(\d{{3,4}}(?:[.,]\d+)?)", re.I),
    # Last resort: a bare "NNN nm" anywhere in the header.
    re.compile(r"(\d{3,4}(?:[.,]\d+)?)\s*nm", re.I),
)

_DELIMITERS = ("\t", ";", ",", None)  # None means "any run of whitespace"


class SpectrumReadError(ValueError):
    """Raised when a file cannot be interpreted as a spectrum."""


def _to_float(token: str) -> Optional[float]:
    """Parse a number, tolerating a decimal comma and a Fortran ``D`` exponent."""
    t = token.strip().strip('"').strip("'")
    if not t:
        return None
    t = t.replace("D", "E").replace("d", "e")
    try:
        return float(t)
    except ValueError:
        pass
    # Decimal comma: "1580,25" -> 1580.25, but "1,580.25" -> 1580.25 too.
    if "," in t:
        if t.count(",") == 1 and "." not in t:
            try:
                return float(t.replace(",", "."))
            except ValueError:
                return None
        try:
            return float(t.replace(",", ""))
        except ValueError:
            return None
    return None


def _split(line: str, delimiter: Optional[str]) -> list[str]:
    if delimiter is None:
        return line.split()
    return line.split(delimiter)


def _score_delimiter(lines: Sequence[str], delimiter: Optional[str]) -> tuple[int, int]:
    """How many lines parse to >= 2 numbers with this delimiter, and how wide."""
    good = 0
    width = 0
    for line in lines:
        parts = _split(line, delimiter)
        values = [_to_float(p) for p in parts]
        numeric = [v for v in values if v is not None]
        if len(numeric) >= 2 and len(numeric) == len(values):
            good += 1
            width = max(width, len(numeric))
    return good, width


def detect_laser_nm(text: str) -> Optional[float]:
    """Recover an excitation wavelength from header text.

    Looks for an explicit ``laser``/``excitation``/``wavelength`` key first
    and only then for a bare ``NNN nm``, so that a line reading
    ``Grating: 1800 l/mm, slit 100 nm`` does not get mistaken for the laser.
    A recovered value within 1.5 nm of a standard line is snapped to it;
    anything outside 200–1100 nm is discarded.

    Parameters
    ----------
    text:
        The header, or the whole file.

    Returns
    -------
    float or None
        Wavelength in nm.
    """
    for pattern in _LASER_PATTERNS:
        for match in pattern.finditer(text):
            value = _to_float(match.group(1))
            if value is None or not 200.0 <= value <= 1100.0:
                continue
            for standard in COMMON_LASERS:
                if abs(value - standard) <= 1.5:
                    return standard
            return value
    return None


def parse_spectrum_text(
    text: str,
    name: str = "spectrum",
    laser_nm: Optional[float] = None,
    intensity_column: int = 1,
) -> Spectrum:
    """Parse two-column spectral data out of a string.

    Parameters
    ----------
    text:
        File contents.
    name:
        Name for the resulting spectrum.
    laser_nm:
        Excitation wavelength. When ``None``, one is looked for in the
        header via :func:`detect_laser_nm`.
    intensity_column:
        Which column holds the intensity, 0-based, counting the abscissa as
        column 0. Multi-column exports (a map, or raw/dark/corrected
        triples) put the corrected intensity in different places.

    Returns
    -------
    Spectrum

    Raises
    ------
    SpectrumReadError
        If fewer than two data rows can be parsed under any delimiter.
    """
    raw_lines = [ln for ln in text.splitlines() if ln.strip()]
    if not raw_lines:
        raise SpectrumReadError("the file is empty")

    sample = raw_lines[: min(len(raw_lines), 400)]
    best: Optional[str] = None
    best_score = (0, 0)
    for delimiter in _DELIMITERS:
        score = _score_delimiter(sample, delimiter)
        if score > best_score:
            best_score, best = score, delimiter
    if best_score[0] < 2:
        raise SpectrumReadError(
            "no two-column numeric data found. Export the spectrum as ASCII "
            "(x, y) from the instrument software; binary vendor formats are "
            "not supported."
        )

    header: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    n_columns = best_score[1]
    column = intensity_column if 0 < intensity_column < n_columns else 1
    for line in raw_lines:
        parts = _split(line, best)
        values = [_to_float(p) for p in parts]
        if len(values) < 2 or any(v is None for v in values):
            if not xs:  # header lines only count before the data starts
                header.append(line.strip())
            continue
        if column >= len(values):
            continue
        xs.append(values[0])
        ys.append(values[column])

    if len(xs) < 2:
        raise SpectrumReadError("fewer than two data points could be parsed")

    header_text = "\n".join(header)
    if laser_nm is None:
        laser_nm = detect_laser_nm(header_text)

    metadata: dict = {"n_columns": n_columns, "intensity_column": column}
    if header:
        metadata["header"] = header_text
    if laser_nm is not None and "header" in metadata:
        metadata["laser_source"] = "header"

    x_arr = np.asarray(xs, dtype=float)
    if _looks_like_wavelength(x_arr):
        metadata["axis_warning"] = (
            "El eje X parece una longitud de onda absoluta en nm, no un "
            "desplazamiento Raman. Convierte con la longitud de onda del "
            "láser antes de analizar."
        )

    return Spectrum(
        shift=x_arr,
        intensity=np.asarray(ys, dtype=float),
        laser_nm=laser_nm,
        name=name,
        metadata=metadata,
    )


def _looks_like_wavelength(x: np.ndarray) -> bool:
    """Heuristic: does this abscissa look like nm rather than cm⁻¹?

    A Raman shift axis for carbon starts near zero and runs to ~3200. An
    absolute-wavelength axis for a 532 nm laser runs 535–650 nm. The tell
    is a *minimum* far above zero combined with a small total span.
    """
    lo, hi = float(np.min(x)), float(np.max(x))
    return lo > 200.0 and hi < 2000.0 and (hi - lo) < 400.0


def wavelength_to_shift(wavelength_nm: Sequence[float], laser_nm: float) -> np.ndarray:
    """Convert absolute wavelengths (nm) to Raman shift (cm⁻¹).

    ``Δν̃ = 1e7 (1/λ_laser − 1/λ)`` with both wavelengths in nm.

    Parameters
    ----------
    wavelength_nm:
        Scattered wavelengths in nm.
    laser_nm:
        Excitation wavelength in nm.

    Returns
    -------
    numpy.ndarray
        Raman shifts in cm⁻¹. Stokes scattering gives positive values.
    """
    w = np.asarray(wavelength_nm, dtype=float)
    if laser_nm <= 0:
        raise ValueError("laser wavelength must be positive")
    if np.any(w <= 0):
        raise ValueError("wavelengths must be positive")
    return 1e7 * (1.0 / float(laser_nm) - 1.0 / w)


def read_spectrum(
    path: str | Path,
    laser_nm: Optional[float] = None,
    intensity_column: int = 1,
    encoding: Optional[str] = None,
) -> Spectrum:
    """Read one spectrum from a text file.

    Parameters
    ----------
    path:
        File to read. Any text file with two numeric columns works,
        whatever the extension.
    laser_nm:
        Excitation wavelength, overriding whatever the header says. Pass it
        explicitly whenever you know it: an unknown laser blocks the
        dispersion corrections, the diameter estimates and the defect-size
        formulas.
    intensity_column:
        0-based index of the intensity column.
    encoding:
        Text encoding. When ``None``, UTF-8 is tried first and Latin-1 used
        as a fallback — vendor exports from Windows are routinely cp1252,
        and Latin-1 never fails, it only mangles accents in the header.

    Returns
    -------
    Spectrum

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    SpectrumReadError
        If the contents cannot be parsed.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such spectrum file: {p}")
    raw = p.read_bytes()
    if encoding:
        text = raw.decode(encoding, errors="replace")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    spectrum = parse_spectrum_text(
        text, name=p.stem, laser_nm=laser_nm, intensity_column=intensity_column
    )
    spectrum.metadata["path"] = str(p.resolve())
    return spectrum


def read_many(
    paths: Iterable[str | Path], laser_nm: Optional[float] = None
) -> tuple[list[Spectrum], list[tuple[Path, str]]]:
    """Read several files, collecting failures instead of aborting.

    Returns
    -------
    (list[Spectrum], list[(Path, str)])
        The spectra that loaded, and ``(path, message)`` for each that did
        not. A batch of 200 map spectra should not be lost because one of
        them is a stray README.
    """
    ok: list[Spectrum] = []
    failed: list[tuple[Path, str]] = []
    for item in paths:
        p = Path(item)
        try:
            ok.append(read_spectrum(p, laser_nm=laser_nm))
        except (OSError, ValueError) as exc:
            failed.append((p, str(exc)))
    return ok, failed


def write_spectrum(
    spectrum: Spectrum,
    path: str | Path,
    delimiter: str = "\t",
    comment: str = "#",
) -> Path:
    """Write a spectrum back out as ASCII with a provenance header.

    The header records the laser and the full preprocessing history, so a
    file exported from this package can be re-read without losing the
    context that makes its intensities interpretable.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{comment} ramancarbon export: {spectrum.name}",
        f"{comment} laser_nm: {spectrum.laser_nm if spectrum.laser_nm is not None else 'unknown'}",
    ]
    for step in spectrum.history:
        lines.append(f"{comment} processing: {step}")
    lines.append(f"{comment} RamanShift_cm-1{delimiter}Intensity_arb")
    for x, y in zip(spectrum.shift, spectrum.intensity):
        lines.append(f"{x:.4f}{delimiter}{y:.6g}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


__all__ = [
    "COMMON_LASERS",
    "SpectrumReadError",
    "TEXT_SUFFIXES",
    "detect_laser_nm",
    "parse_spectrum_text",
    "read_many",
    "read_spectrum",
    "wavelength_to_shift",
    "write_spectrum",
]
