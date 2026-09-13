"""Reading the map files instruments export.

Two layouts cover almost everything that comes out as text.

**Long (Renishaw WiRE, and most exports).** One row per point:
``x  y  shift  intensity``. Every spectrum's whole axis is repeated for
every pixel, so a 100 × 100 map of 1000 points is ten million rows and
about 300 MB — which is why this reader works on the columns rather than
grouping rows in Python.

**Wide (Horiba LabSpec, and grid exports).** The first row is the Raman
shift axis; each row after it is ``x  y`` followed by that pixel's whole
spectrum. Compact, and it states the axis once, which removes any question
about whether all pixels share it.

The layout is detected, not asked for. What is *not* guessed is the
spatial unit: coordinates come out in whatever the file used, and the
laser spot size — needed to say whether neighbouring pixels are
independent — is never inferred.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from .cube import MapError, RamanMap

#: A header line, or anything that is not a row of numbers.
_COMMENT = re.compile(r"^\s*(#|//|%|[A-Za-z])")


def _numbers(text: str) -> tuple[np.ndarray, list[str]]:
    """The numeric block of a text file plus the comment lines above it."""
    comments: list[str] = []
    rows: list[list[float]] = []
    width = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _COMMENT.match(stripped):
            comments.append(stripped)
            continue
        parts = re.split(r"[\t,;]|\s+", stripped)
        try:
            values = [float(p) for p in parts if p]
        except ValueError:
            comments.append(stripped)
            continue
        if len(values) < 2:
            continue
        if width and len(values) != width:
            continue
        width = len(values)
        rows.append(values)
    if not rows:
        raise MapError("el archivo no contiene ninguna fila de números")
    return np.asarray(rows, dtype=float), comments


def read_map(
    path: str | Path,
    laser_nm: Optional[float] = None,
    spot_um: Optional[float] = None,
    layout: Optional[str] = None,
) -> RamanMap:
    """Read a Raman map from a text file.

    Parameters
    ----------
    layout:
        ``"largo"`` or ``"ancho"``, forcing the reading. Left ``None``,
        the layout is detected from the shape of the numbers.

    Raises
    ------
    MapError
        If the file is not a map, or is a map with missing pixels that
        cannot be placed on a grid.
    """
    source = Path(path)
    if not source.is_file():
        raise MapError(f"no existe: {source}")
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    table, comments = _numbers(text)
    laser = laser_nm or _laser_from(comments)
    chosen = layout or _detect_layout(table)
    if chosen == "largo":
        cube = _from_long(table, source.stem, laser, spot_um)
    elif chosen == "ancho":
        cube = _from_wide(table, source.stem, laser, spot_um)
    else:
        raise MapError(f"disposición desconocida: {chosen!r}")
    cube.metadata["path"] = str(source.resolve())
    return cube


def _detect_layout(table: np.ndarray) -> str:
    """Long or wide, from the shape.

    A long file has four columns (or three for a line scan) and many rows;
    a wide file has as many columns as spectral points, which is hundreds.
    The threshold is not close: nobody exports a map with eight columns.
    """
    return "largo" if table.shape[1] <= 6 else "ancho"


def _from_long(table, name, laser, spot) -> RamanMap:
    """``x y shift intensity`` rows, worked on as columns."""
    if table.shape[1] == 3:
        # A line scan: position, shift, intensity.
        xs = table[:, 0]
        ys = np.zeros_like(xs)
        shifts, values = table[:, 1], table[:, 2]
    elif table.shape[1] >= 4:
        xs, ys, shifts, values = (table[:, 0], table[:, 1],
                                  table[:, 2], table[:, 3])
    else:
        raise MapError(
            f"un archivo largo necesita 3 o 4 columnas y trae {table.shape[1]}"
        )

    axis = np.unique(shifts)
    if axis.size < 2:
        raise MapError("solo hay un valor de desplazamiento: no es un mapa")
    unique_x = np.unique(xs)
    unique_y = np.unique(ys)
    expected = unique_x.size * unique_y.size * axis.size
    if values.size > expected:
        raise MapError(
            f"hay {values.size} valores para una rejilla de "
            f"{unique_y.size}×{unique_x.size}×{axis.size} = {expected}: "
            "el archivo tiene posiciones repetidas o no está en rejilla"
        )

    cube = np.full((unique_y.size, unique_x.size, axis.size), np.nan)
    rows = np.searchsorted(unique_y, ys)
    columns = np.searchsorted(unique_x, xs)
    channels = np.searchsorted(axis, shifts)
    cube[rows, columns, channels] = values
    return RamanMap(shift=axis, intensity=cube, x=unique_x, y=unique_y,
                    laser_nm=laser, spot_um=spot, name=name)


def _from_wide(table, name, laser, spot) -> RamanMap:
    """First row is the axis; each row after is ``x y`` plus a spectrum."""
    header = table[0]
    body = table[1:]
    if body.size == 0:
        raise MapError("el archivo solo tiene la fila del eje")

    leading = body.shape[1] - header.size
    if leading in (1, 2):
        # The axis row holds only the axis; the data rows have one or two
        # coordinate columns in front of it.
        axis = header
    elif header.size == body.shape[1]:
        # The axis row carries placeholders of its own where the
        # coordinates go, which is what LabSpec writes.
        leading, axis = 2, header[2:]
    else:
        raise MapError(
            f"la fila del eje tiene {header.size} valores y las de datos "
            f"{body.shape[1]}: no cuadran ni con una ni con dos columnas de "
            "coordenadas"
        )

    if leading == 2:
        xs, ys = body[:, 0], body[:, 1]
    else:
        xs, ys = body[:, 0], np.zeros(body.shape[0])
    spectra = body[:, leading:]
    if spectra.shape[1] != axis.size:
        raise MapError(
            f"cada fila trae {spectra.shape[1]} intensidades y el eje tiene "
            f"{axis.size} puntos"
        )

    unique_x, unique_y = np.unique(xs), np.unique(ys)
    cube = np.full((unique_y.size, unique_x.size, axis.size), np.nan)
    cube[np.searchsorted(unique_y, ys), np.searchsorted(unique_x, xs)] = spectra
    return RamanMap(shift=axis, intensity=cube, x=unique_x, y=unique_y,
                    laser_nm=laser, spot_um=spot, name=name)


def _laser_from(comments: list[str]) -> Optional[float]:
    for line in comments:
        match = re.search(r"(\d{3,4}(?:\.\d+)?)\s*nm", line, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if 200.0 <= value <= 1200.0:
                return value
    return None


def write_map(cube: RamanMap, path: str | Path, layout: str = "ancho") -> Path:
    """Write a map as text, in either layout.

    The wide layout is the default because it states the spectral axis
    once. A long file repeats it for every pixel, which is both larger and
    an invitation for the axes to disagree.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {cube.name}"]
    if cube.laser_nm:
        lines.append(f"# láser {cube.laser_nm:g} nm")
    if cube.spot_um:
        lines.append(f"# punto {cube.spot_um:g} um")

    if layout == "ancho":
        # The axis row carries two zeros where the coordinates go: a row
        # of numbers, so no reader has to decide whether "x" is a label or
        # a datum.
        lines.append("\t".join(["0", "0"] + [f"{v:.4f}" for v in cube.shift]))
        for row, y in enumerate(cube.y):
            for column, x in enumerate(cube.x):
                values = cube.intensity[row, column]
                lines.append("\t".join(
                    [f"{x:.5g}", f"{y:.5g}"]
                    + ["" if not np.isfinite(v) else f"{v:.6g}" for v in values]))
    elif layout == "largo":
        lines.append("\t".join(["x", "y", "desplazamiento", "intensidad"]))
        for row, y in enumerate(cube.y):
            for column, x in enumerate(cube.x):
                for shift, value in zip(cube.shift, cube.intensity[row, column]):
                    if np.isfinite(value):
                        lines.append(
                            f"{x:.5g}\t{y:.5g}\t{shift:.4f}\t{value:.6g}")
    else:
        raise MapError(f"disposición desconocida: {layout!r}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


__all__ = ["read_map", "write_map"]
