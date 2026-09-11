"""Working out which pseudopotentials a calculation needs, and checking them.

carbonforge writes pseudopotential *names* into its inputs but cannot ship
the files: they are large, licensed separately, and the right choice depends
on what you are computing. That leaves a gap where a run dies immediately
with "file not found", or worse, runs with a set that silently cannot do what
was asked.

This module closes it. Given a structure and what you intend to compute, it
says which family is required and why, produces the filenames, points at
where to download them, and verifies a directory actually contains them.

The two constraints that drive the choice are the ones
:mod:`carbonforge.validation.calculations` already enforces:

* **Raman (DFPT)** needs norm-conserving pseudopotentials. PAW and ultrasoft
  are not supported by ``ph.x`` for Raman tensors.
* **Spin-orbit coupling** needs fully-relativistic ones. A scalar set gives
  zero splitting with no error at all.

Needing both at once means norm-conserving *and* fully-relativistic, which is
a genuinely narrower set — PseudoDojo's ``nc-fr`` tables are the usual
source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ase import Atoms

#: Where each family can be downloaded.
SOURCES: dict[str, str] = {
    "PAW": "https://pseudopotentials.quantum-espresso.org/legacy_tables",
    "NC": "http://www.pseudo-dojo.org/  (o SG15: http://www.quantum-simulation.org/potentials/sg15_oncv/)",
    "NC-FR": "http://www.pseudo-dojo.org/  → tabla 'nc-fr' (fully relativistic)",
    "PAW-FR": "https://pseudopotentials.quantum-espresso.org/legacy_tables  → archivos 'rel-'",
}

#: Curated cutoff tables worth consulting instead of guessing.
CUTOFF_REFERENCE = (
    "https://www.materialscloud.org/discover/sssp/table/efficiency"
)

#: Rough starting cutoffs (Ry) by family. These are *starting points for a
#: convergence sweep*, not converged values, and not a substitute for the
#: per-element SSSP table. Norm-conserving pseudopotentials are harder and
#: need substantially more than PAW or ultrasoft.
TYPICAL_CUTOFFS: dict[str, tuple[float, float]] = {
    "PAW": (50.0, 60.0),
    "USPP": (40.0, 50.0),
    "NC": (80.0, 100.0),
}

_PSLIB_STEMS: dict[str, str] = {
    "C": "C.{rel}pbe-n-kjpaw_psl.1.0.0.UPF",
    "N": "N.{rel}pbe-n-kjpaw_psl.1.0.0.UPF",
    "B": "B.{rel}pbe-n-kjpaw_psl.1.0.0.UPF",
    "S": "S.{rel}pbe-n-kjpaw_psl.1.0.0.UPF",
    "P": "P.{rel}pbe-n-kjpaw_psl.1.0.0.UPF",
    "H": "H.{rel}pbe-kjpaw_psl.1.0.0.UPF",
    "O": "O.{rel}pbe-n-kjpaw_psl.1.0.0.UPF",
}


@dataclass
class PseudoRequirement:
    """One pseudopotential the calculation needs."""

    element: str
    filename: str
    family: str
    relativistic: bool
    reason: str

    @property
    def source(self) -> str:
        """Download location for this requirement's family."""
        key = self.family
        if self.relativistic:
            key = "NC-FR" if self.family == "NC" else "PAW-FR"
        return SOURCES.get(key, SOURCES["PAW"])


def requirements_for(
    atoms: Atoms,
    needs_raman: bool = False,
    needs_soc: bool = False,
) -> list[PseudoRequirement]:
    """Determine the pseudopotentials a calculation requires.

    Parameters
    ----------
    atoms
        Structure whose elements must be covered.
    needs_raman
        Whether DFPT Raman tensors will be computed. Forces norm-conserving.
    needs_soc
        Whether spin-orbit coupling is enabled. Forces fully-relativistic.

    Returns
    -------
    list[PseudoRequirement]
        One entry per distinct element, sorted by symbol.
    """
    elements = sorted(set(atoms.get_chemical_symbols()))

    if needs_raman:
        family = "NC"
        reason = "Raman por DFPT: ph.x no admite PAW ni ultrasoft."
    else:
        family = "PAW"
        reason = "Cálculo estándar: PAW es una elección eficiente y precisa."
    if needs_soc:
        reason += (
            " Espín-órbita: hacen falta pseudos totalmente relativistas; "
            "con los escalares el desdoblamiento sale cero sin dar error."
        )

    requirements: list[PseudoRequirement] = []
    for element in elements:
        if family == "NC":
            # PseudoDojo / SG15 naming; the FR tables use the same stem.
            filename = f"{element}_ONCV_PBE-1.2.upf"
            if needs_soc:
                filename = f"{element}_ONCV_PBE_FR-1.0.upf"
        else:
            stem = _PSLIB_STEMS.get(element, "{el}.{rel}pbe.UPF")
            filename = stem.format(rel="rel-" if needs_soc else "", el=element)
        requirements.append(
            PseudoRequirement(
                element=element,
                filename=filename,
                family=family,
                relativistic=needs_soc,
                reason=reason,
            )
        )
    return requirements


@dataclass
class PseudoCheck:
    """Outcome of verifying a pseudopotential directory."""

    directory: Path
    found: dict[str, Path]
    missing: list[PseudoRequirement]
    substitutes: dict[str, list[str]]

    @property
    def ok(self) -> bool:
        """True when every requirement was matched exactly."""
        return not self.missing

    def summary(self) -> str:
        """Human-readable report, in Spanish, for the CLI and GUI."""
        lines = [f"Directorio: {self.directory}"]
        if not self.directory.exists():
            lines.append("  ⚠️  No existe.")
        if self.found:
            lines.append("\nEncontrados:")
            lines.extend(
                f"  ✅ {element}: {path.name}"
                for element, path in sorted(self.found.items())
            )
        if self.missing:
            lines.append("\nFaltan:")
            for requirement in self.missing:
                lines.append(f"  ❌ {requirement.element}: {requirement.filename}")
                alternatives = self.substitutes.get(requirement.element)
                if alternatives:
                    shown = ", ".join(alternatives[:3])
                    lines.append(
                        f"      hay otros archivos de {requirement.element} "
                        f"en la carpeta ({shown}). Si quieres usarlos, pásalos "
                        "explícitamente en QESettings.pseudopotentials."
                    )
            families = {r.source for r in self.missing}
            lines.append("\nDescárgalos de:")
            lines.extend(f"  • {source}" for source in sorted(families))
        if self.ok and self.found:
            lines.append("\n✅ Todo listo.")
        return "\n".join(lines)


def check_directory(
    directory: str | Path,
    requirements: Sequence[PseudoRequirement],
) -> PseudoCheck:
    """Verify that ``directory`` contains every required pseudopotential.

    Matching is case-insensitive on the filename, since UPF files circulate
    with both ``.UPF`` and ``.upf`` extensions and QE accepts either.

    When an exact match is missing, any other file whose name starts with the
    element symbol is reported as a possible substitute — it might be a
    perfectly good pseudopotential from a different table, but choosing it is
    the user's call, not something to silently assume.
    """
    directory = Path(directory)
    present: dict[str, Path] = {}
    if directory.exists():
        for path in directory.iterdir():
            if path.is_file():
                present[path.name.lower()] = path

    found: dict[str, Path] = {}
    missing: list[PseudoRequirement] = []
    substitutes: dict[str, list[str]] = {}

    for requirement in requirements:
        match = present.get(requirement.filename.lower())
        if match is not None:
            found[requirement.element] = match
            continue
        missing.append(requirement)
        prefix = requirement.element.lower()
        candidates = [
            path.name
            for name, path in sorted(present.items())
            # "C" must not match "Ca..."; require a separator after the symbol.
            if name.startswith(prefix)
            and (len(name) > len(prefix) and not name[len(prefix)].isalpha())
        ]
        if candidates:
            substitutes[requirement.element] = candidates

    return PseudoCheck(
        directory=directory,
        found=found,
        missing=missing,
        substitutes=substitutes,
    )


def describe(requirements: Sequence[PseudoRequirement]) -> str:
    """Explain what is needed and where to get it."""
    if not requirements:
        return "No hay elementos que cubrir."

    family = requirements[0].family
    relativistic = requirements[0].relativistic
    lines = [
        f"Familia requerida: {family}"
        + (" totalmente relativista" if relativistic else ""),
        f"Motivo: {requirements[0].reason}",
        "",
        "Archivos:",
    ]
    lines.extend(f"  {r.element}:  {r.filename}" for r in requirements)

    lines += ["", "Dónde descargarlos:"]
    lines.extend(f"  • {source}" for source in sorted({r.source for r in requirements}))

    low, high = TYPICAL_CUTOFFS.get(family, (60.0, 80.0))
    lines += [
        "",
        f"Cutoff orientativo para empezar ({family}): {low:g}–{high:g} Ry.",
        "Esto es un punto de PARTIDA para un barrido de convergencia, no un",
        "valor convergido. Para valores por elemento, consulta la tabla SSSP:",
        f"  {CUTOFF_REFERENCE}",
        "Y converge con:  carbonforge converge <estructura> --parameter cutoff",
    ]
    return "\n".join(lines)


def pseudopotential_map(
    requirements: Sequence[PseudoRequirement],
) -> dict[str, str]:
    """Return the ``{element: filename}`` mapping for ``QESettings``."""
    return {r.element: r.filename for r in requirements}
