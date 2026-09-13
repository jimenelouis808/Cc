"""Scanning a pseudopotential directory and matching it to a calculation.

:mod:`carbonforge.exports.pseudos` says which files a calculation *needs*.
This module looks at what you actually *have*, by reading the UPF headers
rather than trusting filenames, and pairs the two up.

The matching is requirement-driven: Raman needs norm-conserving, spin-orbit
needs fully-relativistic, and a file that fails either is reported as
unsuitable **with the reason**, not merely absent. That distinction matters —
"you have no carbon pseudopotential" and "your carbon pseudopotential is PAW,
which cannot do Raman" call for very different responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ase import Atoms

from .upf import PseudoInfo, read_upf_header

#: Where each family can be downloaded. Kept here so a missing file always
#: comes with somewhere to get it.
DOWNLOAD_SOURCES: dict[str, str] = {
    "NC": "http://www.pseudo-dojo.org/  (tabla 'nc-sr' estándar)",
    "NC-FR": "http://www.pseudo-dojo.org/  (tabla 'nc-fr', relativista)",
    "PAW": "https://pseudopotentials.quantum-espresso.org/legacy_tables",
    "PAW-FR": "https://pseudopotentials.quantum-espresso.org/legacy_tables"
              "  (archivos 'rel-')",
    "USPP": "https://pseudopotentials.quantum-espresso.org/legacy_tables",
    "SSSP": "https://www.materialscloud.org/discover/sssp/table/efficiency"
            "  (incluye cutoffs recomendados por elemento)",
}


@dataclass
class PseudoCatalog:
    """Everything usable found in one or more directories."""

    entries: list[PseudoInfo] = field(default_factory=list)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def elements(self) -> list[str]:
        """Elements covered by the catalogue."""
        return sorted({entry.element for entry in self.entries if entry.element})

    def for_element(
        self,
        element: str,
        family: Optional[str] = None,
        relativistic: Optional[bool] = None,
    ) -> list[PseudoInfo]:
        """Return the candidates for one element, optionally filtered.

        Parameters
        ----------
        element
            Chemical symbol.
        family
            ``"NC"``, ``"USPP"`` or ``"PAW"``. ``None`` accepts any.
        relativistic
            ``True`` demands a fully-relativistic file (needed for
            spin-orbit); ``False`` demands one that is not; ``None`` accepts
            either.
        """
        matches = [e for e in self.entries if e.element == element]
        if family is not None:
            matches = [e for e in matches if e.family == family]
        if relativistic is not None:
            matches = [e for e in matches if e.is_fully_relativistic == relativistic]
        # Prefer files that state a recommended cutoff: they carry more
        # information and usually come from a curated table.
        return sorted(
            matches,
            key=lambda e: (e.suggested_wfc_cutoff is None, e.path.name),
        )

    def summary(self) -> str:
        """Human-readable listing of what was found."""
        if not self.entries:
            lines = ["No se encontró ningún pseudopotencial legible."]
        else:
            lines = [f"{len(self.entries)} pseudopotenciales, "
                     f"{len(self.elements)} elemento(s):"]
            for element in self.elements:
                lines.append(f"\n  {element}:")
                for entry in self.for_element(element):
                    lines.append(f"    {entry.path.name}")
                    lines.append(f"      {entry.describe()}")
        if self.unreadable:
            lines.append(f"\n{len(self.unreadable)} archivo(s) ilegibles:")
            lines.extend(
                f"  {path.name}: {reason}" for path, reason in self.unreadable
            )
        return "\n".join(lines)


def scan_directory(
    directory: str | Path,
    recursive: bool = True,
) -> PseudoCatalog:
    """Read every UPF file under ``directory``.

    Parameters
    ----------
    directory
        Where to look.
    recursive
        Descend into subdirectories. Pseudopotential tables usually unpack
        into one folder per element or per family, so this defaults on.

    Returns
    -------
    PseudoCatalog
        Files that could not be parsed are collected separately rather than
        aborting the scan, so one bad download does not hide the rest.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory}: no es un directorio.")

    pattern = "**/*" if recursive else "*"
    catalog = PseudoCatalog()
    for path in sorted(directory.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in (".upf", ".psml", ".psf"):
            continue
        if path.suffix.lower() in (".psml", ".psf"):
            # SIESTA formats: recorded so the user sees them, but only UPF
            # headers are parsed here.
            catalog.unreadable.append(
                (path, "formato de SIESTA (.psml/.psf), no se analiza aquí")
            )
            continue
        try:
            catalog.entries.append(read_upf_header(path))
        except ValueError as exc:
            catalog.unreadable.append((path, str(exc).split(": ", 1)[-1]))
    return catalog


@dataclass
class MatchResult:
    """Outcome of matching a catalogue against a calculation's needs."""

    resolved: dict[str, PseudoInfo] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unsuitable: dict[str, list[tuple[PseudoInfo, str]]] = field(default_factory=dict)
    cutoff_hint: Optional[float] = None

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unsuitable

    def pseudopotential_map(self) -> dict[str, str]:
        """The ``{element: filename}`` mapping for :class:`QESettings`."""
        return {
            element: info.path.name for element, info in self.resolved.items()
        }

    def summary(self) -> str:
        lines: list[str] = []
        if self.resolved:
            lines.append("Encontrados y compatibles:")
            for element, info in sorted(self.resolved.items()):
                lines.append(f"  ✅ {element}: {info.path.name}")
                lines.append(f"       {info.describe()}")

        if self.unsuitable:
            lines.append("\nPresentes pero NO válidos para este cálculo:")
            for element, rejected in sorted(self.unsuitable.items()):
                for info, reason in rejected:
                    lines.append(f"  ❌ {element}: {info.path.name}")
                    lines.append(f"       {reason}")

        if self.missing:
            lines.append(f"\nNo hay ningún archivo para: {', '.join(self.missing)}")

        if self.cutoff_hint:
            lines.append(
                f"\nCutoff mínimo sugerido por los propios archivos: "
                f"{self.cutoff_hint:g} Ry."
            )
            lines.append(
                "  Es un mínimo del generador, no un valor convergido: "
                "sigue haciendo el barrido."
            )

        if self.ok and self.resolved:
            lines.append("\n✅ Todo listo para este cálculo.")
        return "\n".join(lines)


def match_requirements(
    catalog: PseudoCatalog,
    atoms: Atoms,
    needs_raman: bool = False,
    needs_soc: bool = False,
) -> MatchResult:
    """Pick a suitable pseudopotential per element, or explain why none fits.

    Parameters
    ----------
    catalog
        What is available, from :func:`scan_directory`.
    atoms
        The structure, whose elements must all be covered.
    needs_raman
        DFPT Raman: restricts to norm-conserving.
    needs_soc
        Spin-orbit: restricts to fully-relativistic.

    Returns
    -------
    MatchResult
        Reports "you have nothing for N" separately from "your N file is PAW
        and Raman cannot use it", because the fixes differ.
    """
    elements = sorted(set(atoms.get_chemical_symbols()))
    result = MatchResult()
    cutoffs: list[float] = []

    for element in elements:
        candidates = catalog.for_element(element)
        if not candidates:
            result.missing.append(element)
            continue

        rejected: list[tuple[PseudoInfo, str]] = []
        chosen: Optional[PseudoInfo] = None
        for info in candidates:
            if needs_raman and not info.supports_raman:
                rejected.append((
                    info,
                    f"es {info.family}; el Raman por DFPT en QE solo admite "
                    "norm-conserving.",
                ))
                continue
            if needs_soc and not info.is_fully_relativistic:
                rejected.append((
                    info,
                    f"es '{info.relativistic}'; el espín-órbita necesita un "
                    "pseudopotencial totalmente relativista, y con uno escalar "
                    "el desdoblamiento sale cero sin dar error.",
                ))
                continue
            chosen = info
            break

        if chosen is not None:
            result.resolved[element] = chosen
            if chosen.suggested_wfc_cutoff:
                cutoffs.append(chosen.suggested_wfc_cutoff)
        elif rejected:
            result.unsuitable[element] = rejected
        else:
            result.missing.append(element)

    if cutoffs:
        result.cutoff_hint = max(cutoffs)
    return result


def download_instructions(
    missing: Sequence[str],
    needs_raman: bool = False,
    needs_soc: bool = False,
) -> str:
    """Say exactly which table to download and why.

    No download happens here. Fetching multi-megabyte files from third-party
    servers is something to do knowingly, and the URLs change; naming the
    table and letting you fetch it is more durable than a hardcoded link that
    silently rots.
    """
    if not missing:
        return "No falta nada."

    if needs_raman and needs_soc:
        key, reason = "NC-FR", (
            "Raman exige norm-conserving y el espín-órbita exige totalmente "
            "relativista: la intersección son las tablas nc-fr de PseudoDojo."
        )
    elif needs_raman:
        key, reason = "NC", "El Raman por DFPT solo funciona con norm-conserving."
    elif needs_soc:
        key, reason = "PAW-FR", "El espín-órbita necesita los archivos 'rel-'."
    else:
        key, reason = "PAW", "Para un cálculo normal, PAW es eficiente y preciso."

    lines = [
        f"Faltan: {', '.join(missing)}",
        f"Motivo de la familia elegida: {reason}",
        "",
        f"Descarga desde:  {DOWNLOAD_SOURCES[key]}",
        "",
        "Si quieres además cutoffs recomendados por elemento, la tabla SSSP "
        "los lista:",
        f"  {DOWNLOAD_SOURCES['SSSP']}",
        "",
        "Deja los archivos en una carpeta y vuelve a escanearla con:",
        "  carbonforge pseudos --scan <carpeta>",
    ]
    return "\n".join(lines)
