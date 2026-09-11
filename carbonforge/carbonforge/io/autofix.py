"""Repairing imported structures.

Every repair here is one where the right answer is unambiguous: adding the
vacuum box an XYZ file never had, removing atoms that a symmetry expansion
duplicated on top of each other, wrapping coordinates back into the cell.

Deliberately **not** repaired: atoms that merely sit too close. Two carbons
at 0.6 Å could be a broken file, a units mix-up, or a genuine but unrelaxed
geometry, and nudging them apart would invent a structure the user never
had. That one is reported and left alone.

Every fix records what it did, so an imported-and-repaired structure carries
its own history into the exported input and the dataset metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from ase import Atoms

from ..utils.constants import DEFAULT_VACUUM_2D
from ..utils.geometry import ensure_vacuum
from .importer import ImportIssue, diagnose


@dataclass
class FixRecord:
    """One repair that was applied."""

    code: str
    description: str
    detail: str = ""


@dataclass
class FixResult:
    """The repaired structure and an account of what changed."""

    atoms: Atoms
    applied: list[FixRecord] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    remaining: list[ImportIssue] = field(default_factory=list)

    @property
    def fully_repaired(self) -> bool:
        return not any(issue.severity == "error" for issue in self.remaining)

    def summary(self) -> str:
        lines: list[str] = []
        if self.applied:
            lines.append("Reparaciones aplicadas:")
            for record in self.applied:
                lines.append(f"  ✔ {record.description}")
                if record.detail:
                    lines.append(f"      {record.detail}")
        else:
            lines.append("No hizo falta reparar nada.")

        if self.skipped:
            lines.append("\nNo reparado a propósito:")
            for code, reason in self.skipped:
                lines.append(f"  • {reason}")

        if self.remaining:
            errors = [i for i in self.remaining if i.severity == "error"]
            if errors:
                lines.append("\nSigue habiendo problemas que impiden exportar:")
                lines.extend(f"  ❌ {i.message}" for i in errors)
            else:
                lines.append("\n✅ Ya no hay errores. Quedan avisos menores.")
        else:
            lines.append("\n✅ Estructura limpia.")
        return "\n".join(lines)


def remove_duplicate_atoms(
    atoms: Atoms,
    tolerance: float = 0.1,
) -> tuple[Atoms, int]:
    """Drop atoms that sit on top of another one.

    Symmetry expansion of a CIF routinely emits the same atom several times
    at a special position. Keeping the first occurrence of each cluster is
    the standard resolution.

    Parameters
    ----------
    atoms
        Structure to clean.
    tolerance
        Separation below which two atoms are considered the same, in Å.

    Returns
    -------
    (ase.Atoms, int)
        The cleaned structure and how many atoms were removed.
    """
    if len(atoms) < 2:
        return atoms, 0

    distances = atoms.get_all_distances(mic=any(atoms.get_pbc()))
    np.fill_diagonal(distances, np.inf)

    drop: set[int] = set()
    for i in range(len(atoms)):
        if i in drop:
            continue
        for j in range(i + 1, len(atoms)):
            if j not in drop and distances[i, j] < tolerance:
                drop.add(j)

    if not drop:
        return atoms, 0
    keep = [i for i in range(len(atoms)) if i not in drop]
    cleaned = atoms[keep]
    cleaned.info = {**atoms.info}
    return cleaned, len(drop)


def add_missing_cell(
    atoms: Atoms,
    vacuum: float = DEFAULT_VACUUM_2D,
) -> Atoms:
    """Give a cell-less structure an orthorhombic box with vacuum.

    A molecule or fragment read from XYZ has no cell at all. The box is sized
    to the atoms plus ``vacuum`` on every side, and left **non-periodic**:
    guessing that the user wanted periodicity would be the more dangerous
    assumption.
    """
    out = atoms.copy()
    out.info = {**atoms.info}
    positions = out.get_positions()
    span = positions.max(axis=0) - positions.min(axis=0)
    out.set_cell(np.diag(span + vacuum))
    out.set_pbc(False)
    out.center()
    return out


def suggest_periodicity(atoms: Atoms) -> tuple[bool, bool, bool]:
    """Guess which axes are periodic, from where the atoms sit in the cell.

    An axis is called periodic when the atoms nearly fill it — the same test
    :func:`carbonforge.validation.checks.check_periodicity_coherence` uses in
    reverse. It is a suggestion: a genuinely thin slab is indistinguishable
    from a molecule in a tight box, so the GUI shows this and lets the user
    override it.
    """
    cell = np.diag(np.array(atoms.cell))
    positions = atoms.get_positions()
    guess: list[bool] = []
    for axis in range(3):
        if cell[axis] <= 0:
            guess.append(False)
            continue
        span = float(np.ptp(positions[:, axis]))
        # Filling more than 70 % of the axis means there is no vacuum gap
        # worth speaking of, so it is almost certainly periodic.
        guess.append(span / cell[axis] > 0.7)
    return tuple(guess)  # type: ignore[return-value]


def autofix(
    atoms: Atoms,
    issues: Optional[Sequence[ImportIssue]] = None,
    vacuum: float = DEFAULT_VACUUM_2D,
    infer_periodicity: bool = True,
) -> FixResult:
    """Apply every safe repair to an imported structure.

    Parameters
    ----------
    atoms
        Structure to repair (not mutated).
    issues
        Problems from :func:`~carbonforge.io.importer.diagnose`. Recomputed
        if omitted.
    vacuum
        Padding to use when adding a missing cell or thickening thin vacuum.
    infer_periodicity
        Guess which axes are periodic when the file did not say. Turn off to
        keep whatever the file declared.

    Returns
    -------
    FixResult
        The repaired structure, what was done, and what was deliberately left
        alone.
    """
    if issues is None:
        issues = diagnose(atoms)
    codes = {issue.code for issue in issues}

    out = atoms.copy()
    out.info = {**atoms.info}
    applied: list[FixRecord] = []
    skipped: list[tuple[str, str]] = []

    # 1. Duplicates first: everything downstream measures distances.
    if "duplicates" in codes:
        out, removed = remove_duplicate_atoms(out)
        if removed:
            applied.append(FixRecord(
                "duplicates",
                f"Eliminados {removed} átomo(s) duplicados.",
                "Estaban a menos de 0.1 Å de otro, típico de una expansión "
                "por simetría. Se conserva el primero de cada grupo.",
            ))

    # 2. A cell must exist before vacuum or wrapping mean anything.
    if "no_cell" in codes:
        out = add_missing_cell(out, vacuum=vacuum)
        applied.append(FixRecord(
            "no_cell",
            f"Añadida una caja ortorrómbica con {vacuum:g} Å de vacío.",
            "Se deja NO periódica: suponer periodicidad sería la hipótesis "
            "más arriesgada. Márcala tú si es una lámina o un cristal.",
        ))

    # 3. Periodicity, only when there is a real cell and nothing was declared.
    if "no_pbc" in codes and infer_periodicity:
        guess = suggest_periodicity(out)
        if any(guess):
            out.set_pbc(guess)
            axes = "".join(a for a, p in zip("xyz", guess) if p)
            applied.append(FixRecord(
                "no_pbc",
                f"Marcados como periódicos los ejes: {axes}.",
                "Deducido de que los átomos llenan esos ejes sin dejar hueco. "
                "Es una suposición: compruébala.",
            ))
        else:
            skipped.append((
                "no_pbc",
                "No se marcó ninguna dirección periódica: los átomos dejan "
                "hueco en las tres, así que parece un sistema finito.",
            ))

    # 4. Wrapping, once periodicity is known.
    if "outside_cell" in codes and any(out.get_pbc()):
        out.wrap()
        applied.append(FixRecord(
            "outside_cell",
            "Coordenadas replegadas dentro de la celda.",
            "No cambia la física: solo evita confundir al análisis de vecinos.",
        ))

    # 5. Vacuum last, since it depends on the final positions.
    thin = [c for c in codes if c.startswith("thin_vacuum_")]
    if thin:
        before = np.diag(np.array(out.cell)).copy()
        out = ensure_vacuum(out, min_vacuum=vacuum)
        after = np.diag(np.array(out.cell))
        grown = [
            f"{'xyz'[a]}: {before[a]:.1f} → {after[a]:.1f} Å"
            for a in range(3)
            if after[a] > before[a] + 1e-6
        ]
        if grown:
            applied.append(FixRecord(
                "thin_vacuum",
                f"Ampliado el vacío hasta {vacuum:g} Å.",
                "; ".join(grown),
            ))

    # Never repaired: moving atoms apart would invent a structure.
    if "overlap" in codes:
        skipped.append((
            "overlap",
            "Hay átomos demasiado cerca y NO se han movido: separarlos a "
            "ciegas inventaría una estructura que tú no tenías. Puede ser un "
            "archivo corrupto, un lío de unidades (¿bohr en vez de Å?) o una "
            "geometría real sin relajar.",
        ))
    if "unknown_elements" in codes:
        skipped.append((
            "unknown_elements",
            "Hay elementos que carbonforge no conoce; no se puede inventar "
            "su radio covalente.",
        ))

    if applied:
        out.info.setdefault("autofix", []).extend(
            {"code": r.code, "description": r.description} for r in applied
        )

    return FixResult(
        atoms=out,
        applied=applied,
        skipped=skipped,
        remaining=diagnose(out),
    )
