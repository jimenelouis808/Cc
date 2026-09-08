"""One function that opens anything the suite understands.

Four instruments mean four readers with four signatures, and a user with a
folder of measurements should not have to know which is which. :func:`load`
detects the file, calls the right reader and returns the right object; what
it will not do is *guess a required number*. A voltammogram without its
scan rate and a diffractogram without its wavelength are refused with the
argument that would fix them, because a capacitance computed from an
assumed scan rate is wrong by exactly the ratio of the assumption, and
nothing downstream can tell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .detect import Detection, detect


class LoadError(ValueError):
    """Raised when a file is recognised but cannot be turned into data."""


@dataclass
class Loaded:
    """What came out of a file, with what was decided along the way."""

    data: Any
    kind: str
    detection: Detection
    path: Path
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.path.name}: {self.kind}"


def load(
    path: str | Path,
    kind: Optional[str] = None,
    **options,
) -> Loaded:
    """Read one file of any supported kind.

    Parameters
    ----------
    path:
        The file.
    kind:
        Force the reading — ``raman``, ``xrd``, ``cv``, ``gcd``, ``eis``.
        Use it when :func:`~ramancarbon.dataio.detect.detect` gets it wrong
        or reports low confidence; the detection is still run and kept, so
        the disagreement is visible in the result.
    **options:
        Passed to the underlying reader (``laser_nm``, ``anode``,
        ``scan_rate``, ``electrode``…).

    Raises
    ------
    LoadError
        For a binary format, an unknown file, or a reader that refused.
    """
    p = Path(path)
    found = detect(p)
    chosen = kind or found.kind
    warnings: list[str] = []

    if kind and found.kind not in (kind, "desconocido") and found.confidence != "baja":
        warnings.append(
            f"se ha pedido leerlo como {kind} y el archivo parece {found.kind} "
            f"({found.reasons[0] if found.reasons else 'sin motivo'})"
        )
    elif found.confidence == "baja":
        warnings.append(
            f"la detección es de confianza baja: {'; '.join(found.reasons)}"
        )

    if chosen == "binario":
        raise LoadError(f"{p.name}: {found.advice}")
    if chosen == "mapa":
        raise LoadError(
            f"{p.name}: es un mapa Raman, no una sola medida. Ábrelo con "
            "ramancarbon.mapping.read_map, o con «ramancarbon mapa»"
        )
    if chosen in ("desconocido", "cif", "proyecto"):
        raise LoadError(
            f"{p.name}: no es una medida que se pueda cargar como serie "
            f"({chosen}). {found.advice or '; '.join(found.reasons)}"
        )

    reader = _READERS.get(chosen)
    if reader is None:
        raise LoadError(f"{p.name}: no hay lector para «{chosen}»")
    try:
        data = reader(p, found, _accepted(reader, options))
    except LoadError:
        raise
    except Exception as error:                       # noqa: BLE001
        raise LoadError(f"{p.name}: {error}") from error
    return Loaded(data=data, kind=chosen, detection=found, path=p, warnings=warnings)


#: Options each reader understands. A folder of mixed measurements is
#: read with one set of options, and ``laser_nm`` is meaningless to a
#: diffractogram: passing it through made every non-Raman file in a folder
#: fail with a TypeError about a keyword argument, which is a programming
#: error dressed up as a data problem.
_ACCEPTED: dict[str, frozenset[str]] = {
    "_load_raman": frozenset({"laser_nm", "intensity_column", "encoding"}),
    "_load_xrd": frozenset({"anode", "wavelength", "line"}),
    "_load_cv": frozenset({"scan_rate", "electrode", "potential_scale",
                           "current_scale"}),
    "_load_gcd": frozenset({"electrode", "current", "potential_scale",
                            "current_scale"}),
    "_load_eis": frozenset({"electrode", "impedance_scale"}),
}


def _accepted(reader, options: dict[str, Any]) -> dict[str, Any]:
    """The options this reader understands, dropping the rest silently."""
    allowed = _ACCEPTED.get(reader.__name__)
    if allowed is None:
        return options
    return {key: value for key, value in options.items() if key in allowed}


def _load_raman(path: Path, found: Detection, options: dict[str, Any]):
    from ..core.io import read_spectrum
    from ..core.spectrum import Spectrum

    if found.fmt == "jcamp":
        from .jcamp import read_jcamp

        parsed = read_jcamp(path)
        laser = options.get("laser_nm") or parsed["laser_nm"]
        x = parsed["x"]
        if parsed["x_units"] == "nm":
            if not laser:
                raise LoadError(
                    "el eje está en nanómetros y hace falta la longitud de "
                    "onda del láser para convertirlo a desplazamiento Raman; "
                    "pásala con laser_nm="
                )
            from ..core.io import wavelength_to_shift

            x = wavelength_to_shift(x, laser)
        return Spectrum(
            shift=x, intensity=parsed["y"], laser_nm=laser,
            name=parsed["title"] or path.stem,
            metadata={"path": str(path), "formato": "jcamp",
                      "x_units": parsed["x_units"], "y_units": parsed["y_units"]},
        )
    return read_spectrum(path, **options)


def _load_xrd(path: Path, found: Detection, options: dict[str, Any]):
    from ..xrd.io import read_pattern
    from ..xrd.pattern import Pattern

    if found.fmt == "jcamp":
        from .jcamp import read_jcamp

        parsed = read_jcamp(path)
        wavelength = options.get("wavelength")
        if wavelength is None:
            anode = options.get("anode", "Cu")
            from ..xrd.scattering import wavelength_for

            wavelength = wavelength_for(anode, "ka1")
        return Pattern(
            two_theta=parsed["x"], intensity=parsed["y"],
            wavelength=float(wavelength), name=parsed["title"] or path.stem,
            metadata={"path": str(path), "formato": "jcamp"},
        )
    return read_pattern(path, **options)


def _load_cv(path: Path, found: Detection, options: dict[str, Any]):
    from ..echem.io import read_cv

    return read_cv(path, **options)


def _load_gcd(path: Path, found: Detection, options: dict[str, Any]):
    from ..echem.io import read_gcd

    return read_gcd(path, **options)


def _load_eis(path: Path, found: Detection, options: dict[str, Any]):
    from ..echem.io import read_eis

    return read_eis(path, **options)


_READERS = {
    "raman": _load_raman,
    "xrd": _load_xrd,
    "cv": _load_cv,
    "gcd": _load_gcd,
    "eis": _load_eis,
}


def load_folder(
    folder: str | Path,
    pattern: str = "*",
    kind: Optional[str] = None,
    **options,
) -> tuple[list[Loaded], list[tuple[Path, str]]]:
    """Read every readable file in a folder.

    Returns
    -------
    (loaded, failures)
        ``failures`` carries ``(path, message)`` for each file that did not
        open. A folder import that aborts on the first stray README loses
        the other ninety-nine files, and the README is the normal case.
    """
    directory = Path(folder)
    if not directory.is_dir():
        raise LoadError(f"no es una carpeta: {directory}")
    loaded: list[Loaded] = []
    failures: list[tuple[Path, str]] = []
    for item in sorted(directory.glob(pattern)):
        if not item.is_file():
            continue
        try:
            loaded.append(load(item, kind=kind, **options))
        except Exception as error:                   # noqa: BLE001
            failures.append((item, str(error)))
    return loaded, failures


__all__ = ["LoadError", "Loaded", "load", "load_folder"]
