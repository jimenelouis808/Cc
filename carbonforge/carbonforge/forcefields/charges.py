"""Deriving partial charges from a DFT calculation.

The tabulated charges in :mod:`carbonforge.forcefields.parameters` are
representative literature values, and for doped carbon there is no
consensus set: Bader, Mulliken, Löwdin and RESP disagree, sometimes by a
factor of two, and the answer also moves with the functional and the
supercell. Fine for exploring trends; not fine for a number in a paper.

The way out is to derive the charges from a calculation on **the very
structure being simulated**. ``projwfc.x`` already prints Löwdin charges,
and carbonforge already runs ``projwfc.x`` for the projected density of
states, so the data is usually sitting there after a DOS run.

Löwdin charges have their own well-known bias — they depend on the atomic
basis used for the projection and tend to underestimate charge transfer
relative to Bader — so this reports what it read rather than claiming it is
the truth. The point is that they are *your* structure's numbers, computed
consistently, instead of someone else's supercell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

#: projwfc.x prints "Atom #  12: total charge =   4.2153, s, p, d ..."
_LOWDIN_LINE = re.compile(
    r"Atom\s*#\s*(\d+)\s*:\s*total charge\s*=\s*(-?\d+\.\d+)",
    re.IGNORECASE,
)

#: Valence electrons per element for the pseudopotentials this package
#: defaults to. The Löwdin *population* minus this gives the charge.
VALENCE_ELECTRONS: dict[str, float] = {
    "H": 1.0, "B": 3.0, "C": 4.0, "N": 5.0,
    "O": 6.0, "P": 5.0, "S": 6.0,
}


@dataclass
class DerivedCharges:
    """Partial charges read from a DFT run.

    Attributes
    ----------
    charges
        Per-atom charge in units of e, parallel to the structure.
    method
        Which partitioning scheme produced them.
    by_type
        Mean charge per force-field type, which is what a classical run
        needs — one charge per type, not per atom.
    """

    charges: np.ndarray
    method: str = "lowdin"
    by_type: dict[str, float] = field(default_factory=dict)

    @property
    def net_charge(self) -> float:
        return float(self.charges.sum())

    def summary(self) -> str:
        lines = [
            f"Cargas derivadas por {self.method} sobre {len(self.charges)} átomos.",
            f"Carga neta: {self.net_charge:+.4f} e",
        ]
        if abs(self.net_charge) > 0.05:
            lines.append(
                "  ⚠️  La suma debería ser ~0 en un sistema neutro. Una "
                "desviación grande suele indicar que el número de electrones "
                "de valencia asumido no coincide con el de tus "
                "pseudopotenciales."
            )
        if self.by_type:
            lines.append("\nMedia por tipo de campo de fuerza:")
            for name, value in sorted(self.by_type.items()):
                lines.append(f"  {name:12s} {value:+.4f} e")
            lines.append(
                "\nEstas son las que usaría la simulación clásica: un valor "
                "por tipo, no por átomo."
            )
        lines.append(
            "\nNota sobre el método: las cargas de Löwdin dependen de la base "
            "atómica\nusada en la proyección y tienden a subestimar la "
            "transferencia de carga frente\na Bader. Su ventaja aquí no es "
            "ser exactas, sino ser LAS DE TU estructura,\ncalculadas de forma "
            "consistente."
        )
        return "\n".join(lines)


def read_lowdin_charges(
    path: str | Path,
    atoms: Optional[Atoms] = None,
) -> DerivedCharges:
    """Read Löwdin charges from a ``projwfc.x`` output.

    Parameters
    ----------
    path
        The captured stdout of ``projwfc.x`` (``projwfc.out`` in the scripts
        carbonforge generates).
    atoms
        The structure, used to convert Löwdin *populations* into *charges* by
        subtracting each element's valence electron count. Without it the raw
        populations are returned instead.

    Returns
    -------
    DerivedCharges

    Raises
    ------
    ValueError
        When no Löwdin section is present — usually because ``projwfc.x``
        was run without reaching that stage, or the file is a different
        output entirely.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: no existe.")

    text = path.read_text(errors="replace")
    matches = _LOWDIN_LINE.findall(text)
    if not matches:
        raise ValueError(
            f"{path.name}: no contiene una sección de cargas de Löwdin. "
            "projwfc.x las imprime al final; comprueba que terminó."
        )

    populations = np.zeros(len(matches))
    for raw_index, raw_charge in matches:
        index = int(raw_index) - 1
        if 0 <= index < len(populations):
            populations[index] = float(raw_charge)

    if atoms is None:
        return DerivedCharges(charges=populations, method="lowdin (población)")

    if len(atoms) != len(populations):
        raise ValueError(
            f"{path.name} trae {len(populations)} átomos y la estructura "
            f"tiene {len(atoms)}. ¿Son el mismo cálculo?"
        )

    symbols = atoms.get_chemical_symbols()
    charges = np.array([
        VALENCE_ELECTRONS.get(symbol, 0.0) - population
        for symbol, population in zip(symbols, populations)
    ])
    return DerivedCharges(charges=charges, method="lowdin")


def charges_by_type(
    atoms: Atoms,
    charges: np.ndarray,
    types: Optional[list[str]] = None,
) -> dict[str, float]:
    """Average per-atom charges into one value per force-field type.

    A classical force field assigns a charge to a *type*, not an atom, so
    the per-atom DFT values have to be collapsed. The mean is the natural
    choice; a large spread within one type is a sign the typing is too
    coarse for this structure.
    """
    if types is None:
        from .typing import assign_types, refine_carboxyl_oxygens

        typing = assign_types(atoms)
        types = refine_carboxyl_oxygens(atoms, typing.types)

    if len(types) != len(charges):
        raise ValueError(
            f"{len(types)} tipos y {len(charges)} cargas: no cuadran."
        )

    grouped: dict[str, list[float]] = {}
    for name, charge in zip(types, charges):
        grouped.setdefault(name, []).append(float(charge))
    return {name: float(np.mean(values)) for name, values in grouped.items()}


def charge_spread(
    types: list[str],
    charges: np.ndarray,
) -> dict[str, float]:
    """Standard deviation of the charge within each type.

    A type whose atoms scatter by more than ~0.1 e is doing too much work:
    the environments it lumps together are not equivalent, and averaging them
    throws away something real. Worth checking before trusting the averages.
    """
    grouped: dict[str, list[float]] = {}
    for name, charge in zip(types, charges):
        grouped.setdefault(name, []).append(float(charge))
    return {
        name: float(np.std(values)) if len(values) > 1 else 0.0
        for name, values in grouped.items()
    }


def apply_derived_charges(
    atoms: Atoms,
    derived: DerivedCharges,
    types: Optional[list[str]] = None,
) -> tuple[dict[str, float], str]:
    """Turn DFT charges into per-type values, with a quality report.

    Returns
    -------
    (mapping, report)
        ``mapping`` is ``{type_name: charge}`` ready to override the
        tabulated defaults. ``report`` flags any type whose atoms disagree
        too much to be represented by a single number.
    """
    if types is None:
        from .typing import assign_types, refine_carboxyl_oxygens

        typing = assign_types(atoms)
        types = refine_carboxyl_oxygens(atoms, typing.types)

    means = charges_by_type(atoms, derived.charges, types)
    spreads = charge_spread(types, derived.charges)

    lines = ["Carga media por tipo, derivada de tu propio DFT:", ""]
    noisy: list[str] = []
    for name in sorted(means):
        spread = spreads.get(name, 0.0)
        lines.append(f"  {name:12s} {means[name]:+.4f} e   (σ = {spread:.4f})")
        if spread > 0.1:
            noisy.append(name)

    if noisy:
        lines += [
            "",
            f"⚠️  Estos tipos tienen mucha dispersión interna: "
            f"{', '.join(noisy)}",
            "    Un solo valor no los representa bien: los entornos que "
            "agrupan no son",
            "    equivalentes. Considera separarlos en tipos distintos, o "
            "acepta que la",
            "    media pierde algo real.",
        ]
    else:
        lines += [
            "",
            "✅ Cada tipo es internamente consistente (σ < 0.1 e), así que la "
            "media lo representa bien.",
        ]
    return means, "\n".join(lines)
