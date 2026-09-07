"""The reference structure library: bundled phases plus the user's own.

The Crystallography Open Database is the intended source of reference
structures, and the way to use it is the way it is meant to be used:
download the CIF of the phase you care about and drop it in a directory.
:func:`load_library` reads the bundled starter set and any directories you
name, and every entry it returns is a real crystal structure whose pattern
is computed rather than looked up.

There is no built-in online search, and that is a deliberate limitation
rather than an oversight. A phase-identification program that silently
queries a server gives different answers depending on the network, cannot
be reproduced a year later, and quietly stops working offline — which is
where a lot of instrument computers live. Downloading the CIFs you need
once and keeping them with the project is both reproducible and faster.

The bundled set is a starting point for the materials this package was
built around: nanostructured carbon, iron selenides and their neighbours,
the common dichalcogenides, and silicon as a calibration standard. It is
not a database and does not pretend to be one; a few dozen phases against
the COD's half a million. What it does carry, and a COD download does not,
is a ``confidence`` field saying how much the *coordinates* deserve
trust — which matters for intensities and not at all for positions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

from ..database.loader import DATA_DIR
from .cif import CIFError, read_cif
from .structure import Crystal

#: Directory of the CIFs shipped with the package.
BUNDLED_DIR = DATA_DIR / "cif"

_EXTRA = re.compile(
    r"_ramancarbon_(confidence|source)\s+'([^']*)'", re.IGNORECASE
)
_NOTES = re.compile(r"_ramancarbon_notes\s*\n;\s*\n(.*?)\n;", re.IGNORECASE | re.DOTALL)


@dataclass
class LibraryEntry:
    """One reference phase and where it came from."""

    crystal: Crystal
    path: Path
    bundled: bool

    @property
    def key(self) -> str:
        return self.crystal.name

    def __str__(self) -> str:
        origin = "incluida" if self.bundled else str(self.path.parent)
        return f"{self.crystal.name:22s} {self.crystal.formula:12s} [{origin}]"


def _read_with_extras(path: Path, bundled: bool) -> LibraryEntry:
    """Read a CIF, picking up this package's own extra tags if present."""
    crystal = read_cif(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    for tag, value in _EXTRA.findall(text):
        if tag.lower() == "confidence":
            crystal.confidence = value
        elif tag.lower() == "source" and value:
            crystal.source = value
    match = _NOTES.search(text)
    if match:
        note = " ".join(match.group(1).split())
        crystal.notes = (crystal.notes + " " + note).strip() if crystal.notes else note
    return LibraryEntry(crystal=crystal, path=path, bundled=bundled)


@lru_cache(maxsize=4)
def _load_directory(directory: str, bundled: bool) -> tuple[LibraryEntry, ...]:
    path = Path(directory)
    if not path.is_dir():
        return ()
    entries: list[LibraryEntry] = []
    for candidate in sorted(path.glob("*.cif")):
        try:
            entries.append(_read_with_extras(candidate, bundled))
        except (CIFError, ValueError):
            continue
    return tuple(entries)


def load_library(
    extra_directories: Optional[Sequence[str | Path]] = None,
    include_bundled: bool = True,
) -> list[LibraryEntry]:
    """Every available reference structure.

    Parameters
    ----------
    extra_directories:
        Directories of CIF files to add — your COD downloads. Later
        directories override earlier ones and both override the bundled
        set when names collide, so a downloaded structure of a phase that
        is also bundled wins. That is the right precedence: your own file
        is more specific than the starter set.
    include_bundled:
        Whether to include the phases shipped with the package.

    Returns
    -------
    list[LibraryEntry]
    """
    entries: dict[str, LibraryEntry] = {}
    if include_bundled:
        for entry in _load_directory(str(BUNDLED_DIR), True):
            entries[entry.key] = entry
    for directory in extra_directories or ():
        for entry in _load_directory(str(Path(directory).resolve()), False):
            entries[entry.key] = entry
    return list(entries.values())


def library_crystals(
    extra_directories: Optional[Sequence[str | Path]] = None,
    include_bundled: bool = True,
    only: Optional[Sequence[str]] = None,
) -> list[Crystal]:
    """The structures alone, optionally filtered by name."""
    entries = load_library(extra_directories, include_bundled)
    if only is not None:
        wanted = {name.lower() for name in only}
        entries = [e for e in entries if e.key.lower() in wanted]
    return [e.crystal for e in entries]


def find_phase(
    name: str,
    extra_directories: Optional[Sequence[str | Path]] = None,
) -> Optional[Crystal]:
    """One reference structure by name, case-insensitively."""
    for entry in load_library(extra_directories):
        if entry.key.lower() == name.lower():
            return entry.crystal
    return None


def describe_library(
    extra_directories: Optional[Sequence[str | Path]] = None,
) -> str:
    """A listing of what is available, for the CLI and the GUI."""
    entries = load_library(extra_directories)
    if not entries:
        return "La biblioteca de fases está vacía."
    lines = [f"{len(entries)} fase(s) de referencia:"]
    lines.extend("  " + str(e) for e in entries)
    lines.append("")
    lines.append(
        "Para añadir más: descarga el CIF de la fase desde la Crystallography "
        "Open Database (crystallography.net) y ponlo en un directorio propio; "
        "pásalo con --cif DIRECTORIO. No hay búsqueda en línea a propósito: un "
        "resultado que depende de la red no se reproduce."
    )
    return "\n".join(lines)


__all__ = [
    "BUNDLED_DIR",
    "LibraryEntry",
    "describe_library",
    "find_phase",
    "library_crystals",
    "load_library",
]
