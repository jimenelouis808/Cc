"""Importing structures from other programs, and diagnosing what arrives.

ASE reads some eighty formats, so getting the atoms in is the easy part. The
hard part is that files from other tools routinely arrive **incomplete in
ways that silently break a DFT calculation**:

* An XYZ file carries no cell and no periodicity at all. ASE hands back a
  zero cell, and an export built on it produces a nonsense input.
* A CIF from a database may have atoms sitting exactly on top of each other
  at special positions, or duplicated by a symmetry expansion.
* A slab from someone else's workflow may have far too little vacuum, so it
  interacts with its own periodic image.
* Coordinates may lie outside the cell, which is legal but confusing, and
  breaks the geometric assumptions of the functionalisation code.

:func:`import_structure` reads the file and reports every one of these.
:mod:`carbonforge.io.autofix` repairs the ones that can be repaired safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

from ..utils.constants import HARD_MIN_DISTANCE

#: Extensions ASE handles that are worth offering in a file dialog.
IMPORT_FORMATS: dict[str, str] = {
    ".xyz": "XYZ / extended XYZ",
    ".extxyz": "Extended XYZ",
    ".cif": "CIF (bases de datos cristalográficas)",
    ".vasp": "VASP POSCAR/CONTCAR",
    ".poscar": "VASP POSCAR",
    ".in": "Entrada de Quantum ESPRESSO",
    ".out": "Salida de Quantum ESPRESSO",
    ".pwo": "Salida de Quantum ESPRESSO",
    ".data": "Datos de LAMMPS",
    ".lammps": "Datos de LAMMPS",
    ".pdb": "Protein Data Bank",
    ".traj": "Trayectoria de ASE",
    ".json": "JSON de ASE",
    ".gen": "DFTB+",
    ".mol": "MDL Molfile",
    ".sdf": "SDF",
}

#: Files named like this are VASP structures regardless of extension.
_VASP_NAMES = {"poscar", "contcar"}


@dataclass
class ImportIssue:
    """One problem found in an imported structure.

    Attributes
    ----------
    code
        Stable identifier, used by the repair engine to decide what to do.
    severity
        ``"error"`` blocks export; ``"warning"`` is worth knowing.
    message
        What is wrong, in plain language.
    fixable
        Whether :mod:`carbonforge.io.autofix` can repair it safely.
    """

    code: str
    severity: str
    message: str
    fixable: bool = False


@dataclass
class ImportResult:
    """An imported structure plus what is wrong with it."""

    atoms: Atoms
    source: Path
    format_used: str = ""
    issues: list[ImportIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def fixable_issues(self) -> list[ImportIssue]:
        return [issue for issue in self.issues if issue.fixable]

    def summary(self) -> str:
        lines = [
            f"Importado: {self.source.name}"
            + (f"  (formato {self.format_used})" if self.format_used else ""),
            f"{len(self.atoms)} átomos, fórmula {self.atoms.get_chemical_formula()}",
            f"Periodicidad: {''.join(a for a, p in zip('xyz', self.atoms.get_pbc()) if p) or 'ninguna'}",
        ]
        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        if errors:
            lines.append("\nProblemas que impiden exportar:")
            lines.extend(
                f"  ❌ {i.message}" + ("  [reparable]" if i.fixable else "")
                for i in errors
            )
        if warnings:
            lines.append("\nAvisos:")
            lines.extend(
                f"  ⚠️  {i.message}" + ("  [reparable]" if i.fixable else "")
                for i in warnings
            )
        if self.fixable_issues:
            lines.append(
                f"\n{len(self.fixable_issues)} problema(s) se pueden reparar "
                "automáticamente."
            )
        elif self.ok and not warnings:
            lines.append("\n✅ La estructura llegó completa.")
        return "\n".join(lines)


def _guess_format(path: Path) -> Optional[str]:
    """Pick an ASE format for a path, when the extension is ambiguous."""
    if path.name.lower() in _VASP_NAMES or path.stem.lower() in _VASP_NAMES:
        return "vasp"
    suffix = path.suffix.lower()
    if suffix in (".out", ".pwo"):
        return "espresso-out"
    if suffix == ".in":
        return "espresso-in"
    if suffix in (".data", ".lammps"):
        return "lammps-data"
    return None  # let ASE work it out from the extension


def diagnose(atoms: Atoms) -> list[ImportIssue]:
    """Find the problems that make an imported structure unusable.

    Ordered roughly by how badly each one breaks a calculation.
    """
    issues: list[ImportIssue] = []

    if len(atoms) == 0:
        issues.append(ImportIssue(
            "empty", "error", "El archivo no contiene ningún átomo.",
        ))
        return issues

    cell = np.array(atoms.cell)
    volume = abs(np.linalg.det(cell))
    pbc = atoms.get_pbc()

    if volume < 1e-6:
        issues.append(ImportIssue(
            "no_cell", "error",
            "No trae celda (volumen cero). Los formatos como XYZ no la "
            "guardan. Sin celda no se puede exportar a DFT: hace falta una "
            "caja con vacío alrededor.",
            fixable=True,
        ))
    elif not any(pbc):
        issues.append(ImportIssue(
            "no_pbc", "warning",
            "Hay celda pero ninguna dirección está marcada como periódica. "
            "Si es una molécula o un fragmento finito está bien; si querías "
            "un cristal o una lámina, hay que marcarlo.",
            fixable=True,
        ))

    # Overlapping atoms: duplicates from a symmetry expansion, or genuinely
    # broken coordinates.
    if len(atoms) > 1:
        distances = atoms.get_all_distances(mic=any(pbc))
        np.fill_diagonal(distances, np.inf)
        smallest = float(distances.min())
        n_duplicates = int(np.sum(distances < 0.1) // 2)
        if n_duplicates:
            issues.append(ImportIssue(
                "duplicates", "error",
                f"{n_duplicates} par(es) de átomos prácticamente superpuestos "
                "(<0.1 Å). Suele venir de una expansión por simetría de un CIF "
                "que duplicó posiciones especiales.",
                fixable=True,
            ))
        elif smallest < HARD_MIN_DISTANCE:
            issues.append(ImportIssue(
                "overlap", "error",
                f"Dos átomos a {smallest:.3f} Å, por debajo del mínimo físico "
                f"de {HARD_MIN_DISTANCE} Å. Esto no se puede arreglar moviendo "
                "átomos a ciegas: revisa el archivo de origen.",
                fixable=False,
            ))

    # Vacuum, only meaningful once there is a cell.
    if volume > 1e-6:
        positions = atoms.get_positions()
        for axis in range(3):
            if pbc[axis]:
                continue
            length = float(cell[axis, axis])
            if length <= 0:
                continue
            span = float(np.ptp(positions[:, axis]))
            vacuum = length - span
            if vacuum < 8.0:
                issues.append(ImportIssue(
                    f"thin_vacuum_{axis}", "warning",
                    f"Solo {vacuum:.1f} Å de vacío en el eje {'xyz'[axis]}. "
                    "Con menos de ~10 Å la estructura interacciona con su "
                    "propia imagen periódica.",
                    fixable=True,
                ))

        # Atoms outside the cell are legal but break geometric assumptions.
        if any(pbc):
            fractional = atoms.get_scaled_positions(wrap=False)
            outside = int(np.sum((fractional < -0.01) | (fractional > 1.01)))
            if outside:
                issues.append(ImportIssue(
                    "outside_cell", "warning",
                    f"{outside} coordenada(s) fuera de la celda. Es legal, "
                    "pero confunde al análisis de vecinos y a la detección de "
                    "sitios de anclaje.",
                    fixable=True,
                ))

    # Elements this package has no data for.
    from ..utils.constants import COVALENT_RADII

    unknown = sorted(
        {s for s in atoms.get_chemical_symbols() if s not in COVALENT_RADII}
    )
    if unknown:
        issues.append(ImportIssue(
            "unknown_elements", "warning",
            f"Elementos sin datos en carbonforge: {', '.join(unknown)}. "
            "La detección de enlaces y la validación usarán valores por "
            "defecto y pueden equivocarse.",
            fixable=False,
        ))
    return issues


def import_structure(
    path: str | Path,
    index: int = -1,
) -> ImportResult:
    """Read a structure from a file and report what is wrong with it.

    Parameters
    ----------
    path
        The file. Format is inferred from the name; VASP files named
        ``POSCAR``/``CONTCAR`` and QE outputs are recognised specially,
        because their extensions are ambiguous or absent.
    index
        Which frame, for files holding several. ``-1`` takes the last, which
        for a relaxation output is the converged geometry.

    Returns
    -------
    ImportResult

    Raises
    ------
    ValueError
        If the file does not exist or cannot be parsed. The message names the
        formats that are understood, since a wrong extension is the usual
        cause.
    """
    from ase.io import read as ase_read

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: no existe o no es un archivo.")

    fmt = _guess_format(path)
    try:
        atoms = ase_read(str(path), index=index, format=fmt)
    except Exception as exc:
        known = ", ".join(sorted(IMPORT_FORMATS))
        raise ValueError(
            f"{path.name}: ASE no pudo leerlo ({exc}).\n"
            f"Extensiones reconocidas: {known}.\n"
            "Si la extensión no coincide con el contenido, renómbrala."
        ) from exc

    if isinstance(atoms, list):
        if not atoms:
            raise ValueError(f"{path.name}: no contiene ninguna estructura.")
        atoms = atoms[-1]

    atoms.info.setdefault("structure_type", "imported")
    atoms.info["imported_from"] = str(path)

    return ImportResult(
        atoms=atoms,
        source=path,
        format_used=fmt or path.suffix.lstrip("."),
        issues=diagnose(atoms),
    )
