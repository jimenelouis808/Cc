"""Assembling an electric double-layer capacitor cell.

The geometry is a sandwich: electrode | electrolyte | electrode, periodic in
x and y, finite in z. Two methodological points decide whether the result
means anything.

**Constant potential, not constant charge.** In a real capacitor the
electrodes are held at fixed potential and their charge fluctuates in
response to the electrolyte. Fixing the charge instead — which is what a
plain MD run does — gives a different ensemble and a capacitance that can be
wrong by tens of percent, badly so at high potential. LAMMPS's ELECTRODE
package implements the constant-potential method, and
:func:`build_edlc_cell` sets it up.

**Slab correction.** The cell is periodic in x and y but not z, and it
carries a net dipole once the double layer forms. Ewald summation assumes 3D
periodicity, so without ``kspace_modify slab`` the calculation includes
spurious interactions between a slab and its images stacked along z. That
correction is not optional here.

The mirror symmetry also matters: the two electrodes are held at ±V/2 so the
cell is neutral overall and there is no net field across the periodic
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from ase import Atoms

from .electrolyte import ElectrolyteBox, ElectrolyteKind, build_electrolyte
from .parameters import get_params
from .typing import assign_types, refine_carboxyl_oxygens


@dataclass
class EDLCCell:
    """A complete electrode | electrolyte | electrode assembly.

    Attributes
    ----------
    atoms
        Everything, stacked along z.
    types
        Force-field type per atom.
    molecule_ids
        Molecule index per atom, for rigid-body constraints.
    groups
        Atom indices belonging to ``"bottom"``, ``"top"`` and
        ``"electrolyte"``.
    potential_v
        Potential difference applied between the electrodes, in volts.
    separation
        Electrode separation in Å.
    """

    atoms: Atoms
    types: list[str] = field(default_factory=list)
    molecule_ids: list[int] = field(default_factory=list)
    groups: dict[str, list[int]] = field(default_factory=dict)
    potential_v: float = 0.0
    separation: float = 0.0
    electrolyte_composition: dict[str, int] = field(default_factory=dict)

    @property
    def unique_types(self) -> list[str]:
        return sorted(set(self.types))

    def net_charge(self) -> float:
        """Total charge of everything except the electrodes.

        The electrodes' charges are not fixed — the constant-potential
        method solves for them each step — so only the electrolyte and the
        electrode's *chemical* partial charges enter here. A non-zero value
        means the electrolyte is not neutral, which will make the Ewald sum
        misbehave.
        """
        electrolyte = set(self.groups.get("electrolyte", []))
        return sum(
            get_params(self.types[i]).charge
            for i in range(len(self.types))
            if i in electrolyte
        )

    def summary(self) -> str:
        lines = [
            f"Celda EDLC: {len(self.atoms)} átomos",
            f"  electrodo inferior: {len(self.groups.get('bottom', []))} átomos",
            f"  electrolito:        {len(self.groups.get('electrolyte', []))} átomos",
            f"  electrodo superior: {len(self.groups.get('top', []))} átomos",
            f"  separación: {self.separation:.1f} Å",
            f"  potencial aplicado: {self.potential_v:.2f} V "
            f"(±{self.potential_v / 2:.2f} V por electrodo)",
        ]
        if self.electrolyte_composition:
            lines.append("\nElectrolito:")
            for species, count in sorted(self.electrolyte_composition.items()):
                lines.append(f"  {species}: {count}")

        charge = self.net_charge()
        lines.append(f"\nCarga neta del electrolito: {charge:+.4f} e")
        if abs(charge) > 0.01:
            lines.append(
                "  ⚠️  No es neutro. La suma de Ewald exige neutralidad: "
                "revisa la composición iónica."
            )
        return "\n".join(lines)


def build_edlc_cell(
    electrode: Atoms,
    separation: float = 40.0,
    electrolyte: ElectrolyteKind = "aqueous",
    potential_v: float = 1.0,
    vacuum_z: float = 20.0,
    mirror_top: bool = True,
    wall_gap: float = 3.0,
    electrolyte_kwargs: Optional[dict] = None,
) -> EDLCCell:
    """Stack an electrode, an electrolyte and a second electrode along z.

    Parameters
    ----------
    electrode
        The carbon electrode: a graphene sheet, a doped or functionalised
        one, whatever carbonforge built. Must be periodic in x and y, since
        the electrode has to tile the cross-section.
    separation
        Gap between the electrode surfaces, in Å. Needs to be wide enough
        that the two double layers do not overlap — for aqueous electrolyte
        that means at least ~30 Å, and the function warns below that.
    electrolyte
        ``"aqueous"``, ``"ionic_liquid"`` or ``"vacuum"``.
    potential_v
        Potential difference across the cell, in volts. Applied as ±V/2 so
        the cell stays neutral.
    vacuum_z
        Vacuum above and below the sandwich. The slab correction needs empty
        space to work in; ~20 Å is the usual choice.
    wall_gap
        Gap left between each electrode surface and the nearest electrolyte
        molecule, in Å. Without it the filler would place water directly on
        top of the electrode carbons. 3 Å is roughly where the first water
        layer sits against graphene, so the starting configuration begins
        near the right place rather than having to be pushed there.
    mirror_top
        Build the top electrode by mirroring the bottom one. Set ``False`` to
        get two identical (non-mirrored) electrodes, which is what you want
        when studying an asymmetric cell.
    electrolyte_kwargs
        Passed through to the electrolyte builder — ``salt``, ``molarity``,
        ``seed``.

    Returns
    -------
    EDLCCell

    Raises
    ------
    ValueError
        If the electrode is not periodic in the plane, or the separation is
        physically implausible.
    """
    pbc = electrode.get_pbc()
    if not (pbc[0] and pbc[1]):
        raise ValueError(
            "El electrodo debe ser periódico en x e y: tiene que teselar la "
            "sección transversal de la celda. Usa una lámina de grafeno, no "
            "un fragmento finito."
        )
    if separation <= 5.0:
        raise ValueError(
            f"Separación de {separation} Å: no cabe ni una capa de solvente."
        )

    electrode = electrode.copy()
    cell = np.array(electrode.cell)
    lx, ly = float(cell[0, 0]), float(cell[1, 1])

    positions = electrode.get_positions()
    thickness = float(np.ptp(positions[:, 2]))

    # Type the electrode from its chemistry, before it is duplicated.
    typing = assign_types(electrode)
    electrode_types = refine_carboxyl_oxygens(electrode, typing.types)

    # --- stack along z ---------------------------------------------------
    bottom = electrode.copy()
    bottom_positions = bottom.get_positions()
    bottom_positions[:, 2] -= bottom_positions[:, 2].min()
    bottom_positions[:, 2] += vacuum_z
    bottom.set_positions(bottom_positions)

    top = electrode.copy()
    top_positions = top.get_positions()
    top_positions[:, 2] -= top_positions[:, 2].min()
    if mirror_top:
        # Mirror so the functionalised face points into the electrolyte on
        # both sides, which is what a symmetric cell means.
        top_positions[:, 2] = thickness - top_positions[:, 2]
    top_positions[:, 2] += vacuum_z + thickness + separation
    top.set_positions(top_positions)

    # Inset from both electrode surfaces: the filler knows nothing about the
    # electrodes, so without this it would place molecules on top of them.
    if separation <= 2 * wall_gap:
        raise ValueError(
            f"Separación de {separation} Å con {wall_gap} Å de hueco a cada "
            "pared no deja sitio para el electrolito. Aumenta la separación."
        )
    electrolyte_z = (
        vacuum_z + thickness + wall_gap,
        vacuum_z + thickness + separation - wall_gap,
    )
    total_z = 2 * vacuum_z + 2 * thickness + separation

    filled = build_electrolyte(
        electrolyte,
        (lx, ly, total_z),
        electrolyte_z,
        **(electrolyte_kwargs or {}),
    )

    # --- combine ---------------------------------------------------------
    combined = bottom + filled.atoms + top
    combined.set_cell(np.diag([lx, ly, total_z]))
    # Periodic in the plane, finite along z: the slab geometry the
    # correction assumes.
    combined.set_pbc((True, True, False))
    combined.info = {
        "structure_type": "edlc_cell",
        "potential_v": potential_v,
        "separation": separation,
        "electrolyte": electrolyte,
    }

    n_bottom = len(bottom)
    n_electrolyte = len(filled.atoms)
    groups = {
        "bottom": list(range(n_bottom)),
        "electrolyte": list(range(n_bottom, n_bottom + n_electrolyte)),
        "top": list(range(n_bottom + n_electrolyte, len(combined))),
    }

    # Electrode atoms carry no molecule id (they are not rigid bodies); the
    # electrolyte molecule ids are offset past them.
    molecule_ids = (
        [0] * n_bottom
        + [m for m in filled.molecule_ids]
        + [0] * len(top)
    )
    types = electrode_types + filled.types + electrode_types

    return EDLCCell(
        atoms=combined,
        types=types,
        molecule_ids=molecule_ids,
        groups=groups,
        potential_v=potential_v,
        separation=separation,
        electrolyte_composition=filled.composition,
    )


def check_edlc_setup(cell: EDLCCell) -> list[str]:
    """Flag the setup mistakes that make an EDLC result meaningless.

    Returns a list of warnings, empty when nothing is obviously wrong.
    """
    warnings: list[str] = []

    if cell.separation < 30.0 and cell.electrolyte_composition:
        warnings.append(
            f"Separación de {cell.separation:.0f} Å: las dos dobles capas "
            "pueden solaparse y la capacitancia dejaría de ser la de una "
            "interfaz aislada. Para electrolito acuoso se suelen usar ≥30 Å, "
            "y más con líquidos iónicos, cuyas capas ordenadas son gruesas."
        )

    charge = cell.net_charge()
    if abs(charge) > 0.01:
        warnings.append(
            f"El electrolito tiene carga neta {charge:+.3f} e. La suma de "
            "Ewald exige un sistema neutro; con carga neta LAMMPS añade un "
            "fondo uniforme y la energía deja de ser interpretable."
        )

    if abs(cell.potential_v) > 4.0:
        warnings.append(
            f"{cell.potential_v:.1f} V es más de lo que aguanta cualquier "
            "electrolito real: el agua se electroliza en torno a 1.23 V y "
            "los líquidos iónicos rondan los 4 V. La simulación no modela "
            "esa descomposición, así que seguirá corriendo y dará un número "
            "sin sentido físico."
        )

    n_electrolyte = len(cell.groups.get("electrolyte", []))
    if n_electrolyte and n_electrolyte < 300:
        warnings.append(
            f"Solo {n_electrolyte} átomos de electrolito: muy poco para "
            "muestrear una doble capa. Amplía la sección transversal del "
            "electrodo o la separación."
        )
    return warnings
