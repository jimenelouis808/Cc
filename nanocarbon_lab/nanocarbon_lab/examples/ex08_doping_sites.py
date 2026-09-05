"""Where a dopant sits, and why the same fraction means two things.

Doping a nanocarbon is usually described by one number -- "3% nitrogen"
-- and on a flat sheet that is the whole story. On a curved structure it
is not. The pentagons carry the positive curvature, which pyramidalises
them, and pyramidalised carbons are the reactive ones: a fullerene's
chemistry happens at its pentagons and a capped tube's happens at its
cap. A nitrogen there is a different object from a nitrogen 40 Å away on
the cylinder, and "3% N" does not distinguish them.

So this prints the same request placed two ways and shows the two things
that fall out of it:

1. **The fraction is measured against different pools.** Under pentagon
   placement it counts the pentagon sites; a capped tube has 60 of them
   out of 240 atoms, so 10% pentagon doping is 2.5% of the structure.
   Both numbers are recorded, because either alone reads as the other.

2. **Placement needs no ring perception.** The mesh-based builders
   already record ``info["rings"]`` as the real atom indices per ring, so
   this reads them. A plain ``build_cnt`` sheet has no such metadata and
   refuses rather than re-deriving rings from coordinates -- which is
   what ``builders/fullerene_mesh.py`` exists to avoid.

It also shows the per-element ceilings doing their job: the same 10% is
unremarkable for nitrogen and meaningless for iron, and the difference is
chemistry rather than degree.

Run with::

    python -m nanocarbon_lab.examples.ex08_doping_sites
"""

from __future__ import annotations

import warnings

from ..builders import build_capped_cnt, build_cnt, build_fullerene
from ..dopants import (
    DOPANT_CHEMISTRY,
    describe,
    dope_random,
    dope_rings,
    ring_size_census,
)


def show_placement(atoms, label: str) -> None:
    """Dope one structure both ways and compare what 10% meant."""
    print(f"\n=== {label}: {len(atoms)} atoms ===")
    census = ring_size_census(atoms)
    print("  sites by ring size: "
          + ", ".join(f"{k}-ring: {v}" for k, v in census.items()))

    scattered = dope_random(atoms, "N", 0.10, seed=0)
    n_scattered = scattered.get_chemical_symbols().count("N")
    print(f"  random   10%  -> {n_scattered:3d} N, anywhere")

    pentagons = dope_rings(atoms, "N", ring_size=5, concentration=0.10, seed=0)
    n_pentagon = pentagons.get_chemical_symbols().count("N")
    print(f"  pentagon 10%  -> {n_pentagon:3d} N, all on five-membered rings")
    print(f"                   = {pentagons.info['doping_concentration']:.1%} of "
          f"the {pentagons.info['doping_sites_available']} pentagon sites")
    print(f"                   = {pentagons.info['doping_concentration_overall']:.1%} "
          f"of the whole structure")


def show_ceilings() -> None:
    """The same fraction against two elements' own limits."""
    print("\n=== 10% of what? ===")
    cap = build_capped_cnt(n_body_rings=6, freq=2)
    for element in ("N", "Fe"):
        chem = DOPANT_CHEMISTRY[element]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dope_random(cap, element, 0.10, seed=0)
        verdict = "warns" if caught else "no warning"
        print(f"  {element:2s} at 10%  ({chem.site:8s}, ceiling "
              f"{chem.max_fraction:.0%}): {verdict}")
    print()
    for element in ("N", "Fe"):
        print("  " + describe(element))


def main() -> None:
    show_placement(build_fullerene(family="C60", freq=1),
                   "C60 — every carbon is on a pentagon")
    show_placement(build_capped_cnt(n_body_rings=6, freq=2),
                   "Capped tube — pentagons only in the caps")

    # A plain tube never went through the dual, so it has no ring data.
    print("\n=== A structure with no ring metadata ===")
    try:
        dope_rings(build_cnt(n=6, m=6, length=8), "N", concentration=0.1)
    except ValueError as exc:
        print(f"  refused: {str(exc).split('.')[0]}.")

    show_ceilings()
    print("\nOn a flat sheet the two placements are the same request. On a")
    print("curved one they are different chemistry, and only one of them")
    print("puts the dopant where the structure is actually reactive.")


if __name__ == "__main__":
    main()
