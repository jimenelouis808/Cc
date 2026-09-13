"""Every structure has a unit cell — and what it becomes depends on it.

Plane-wave DFT codes and periodic viewers are three-dimensionally
periodic. There is no "molecule" setting anywhere in Quantum ESPRESSO,
VASP, VESTA or OVITO: a molecule is a molecule in a box big enough not
to see its own images. So "what is the unit cell of this?" always has an
answer, and the answer is decided by how many directions the structure
genuinely repeats in.

This prints one of each and the cell it becomes. Two results in it are
worth watching for, because both were bugs first:

1. **A periodic axis comes out untouched.** The tube's axial lattice
   vector is the physics; padding it would change the crystal rather
   than the box around it. Only the two directions it does not repeat in
   grow.

2. **Convergence is measured across vacuum only.** In a real crystal an
   atom bonds to its image -- the nanotube is 1.42 Å from itself along
   its own axis, the schwarzite 1.37 Å. Counting every image and
   reporting the minimum called both of those unconverged, which is
   exactly backwards: that contact *is* the structure. A bulk crystal
   reports no number at all, because it has nothing to converge.

Run with::

    python -m nanocarbon_lab.examples.ex09_unit_cells
"""

from __future__ import annotations

from ..builders import (
    build_cnt,
    build_fullerene,
    build_graphene_supercell,
    build_schwarzite,
)
from ..cell import cell_report, describe_periodicity, to_unit_cell
from ..tmd import build_tmd_monolayer


def show(label: str, atoms) -> None:
    """Convert one structure and print what its cell became."""
    before = describe_periodicity(atoms)
    converted = to_unit_cell(atoms)
    report = cell_report(converted)
    a, b, c = report["lengths"]
    padded = converted.info["unit_cell"]["vacuum_axes"]

    print(f"\n{label}")
    print(f"  {before} -> {report['periodicity']}, {len(converted)} atoms")
    print(f"  cell    {a:8.3f} x {b:8.3f} x {c:8.3f} Å   "
          f"volume {report['volume']:.0f} Å³")
    if padded:
        axes = ", ".join(f"{'xyz'[axis]} (+{value:g} Å each side)"
                         for axis, value in sorted(padded.items()))
        print(f"  padded  {axes}")
    else:
        print("  padded  nothing — it already repeats in all three")

    separation = report["image_separation"]
    if separation is None:
        print("  images  bulk crystal: no vacuum direction, nothing to converge")
    else:
        shown = ">= 20" if separation >= 20.0 else f"{separation:.2f}"
        verdict = "converged" if report["converged"] else "TOO CLOSE"
        print(f"  images  {shown} Å apart across vacuum — {verdict}")


def main() -> None:
    show("C60 — a molecule, so a molecule in a box",
         build_fullerene(family="C60", freq=1))
    show("(6,6) nanotube — repeats along its axis only",
         build_cnt(n=6, m=6, length=10))
    show("Graphene 2x2 — repeats in the plane, vacuum above and below",
         build_graphene_supercell(2, 2))
    show("MoS2 monolayer — a slab, and the vacuum has to be declared",
         build_tmd_monolayer("MoS2"))
    show("Gyroid schwarzite — already a unit cell, left alone",
         build_schwarzite(kind="gyroid", cell=36.0))

    print("\nThe tube's axial vector is unchanged above: it is a lattice")
    print("vector, not padding. And only the schwarzite reports no image")
    print("distance — it has no vacuum direction, so there is nothing a")
    print("bigger cell would converge.")


if __name__ == "__main__":
    main()
