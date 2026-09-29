#!/usr/bin/env python3
"""Build every exact-census structure and check it against its own law.

Nothing here is asserted: each row's budget comes from the skeleton
before anything is meshed, and the script prints the measured census
beside it. Run it and read the "law" column.

    python -m nanocarbon_lab.examples.verify_exact_structures
    python -m nanocarbon_lab.examples.verify_exact_structures --all
    python -m nanocarbon_lab.examples.verify_exact_structures --write out/

The laws, in one place:

* A **node of c arms** is a sphere with c holes, so ``chi = 2 - c`` and
  ``sum(6-n) = 6(2-c)``.
* Summed over a **graph**, that is ``sum_v 6(2 - deg v) = 12(V - E)`` --
  the same number ``supernetwork.SuperGraph.ring_budget`` reaches from
  ``chi = 2(V - E)``, by a different route.
* A **torus** -- a toroid, or one periodic cell of a coil -- has
  ``chi = 0``, so its pentagons and heptagons come out equal.
* A **minimal surface** saddles everywhere, so it carries no pentagon at
  all. The one exception is a *planar crossing*, whose two poles are a
  pillow over the crossing point and are genuinely positively curved.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

#: ``(label, mode, params, expected sum(6-n) or None, slow)``
CASES: list[tuple[str, str, dict, int | None, bool]] = [
    ("toroid, 6 knees", "toroid (knees)", {}, 0, False),
    ("coil, finite", "coil (knees)", {"turns": 1}, None, False),
    ("coil, periodic cell", "coil (knees, periodic)", {}, 0, False),
    ("junction Y", "junction (knees)", {"kind": "y"}, -6, False),
    ("junction X (planar)", "junction (knees)", {"kind": "x"}, -12, False),
    ("junction, diamond node", "junction (knees)",
     {"kind": "tetrahedral"}, -12, False),
    ("sheet, super-square", "supernetwork (knees)",
     {"net": "super-square"}, -12, False),
    ("sheet, super-graphene", "supernetwork (knees)",
     {"net": "super-graphene"}, -24, False),
    ("Schwarz P cell", "schwarzite (knees)", {"kind": "primitive"}, -24, True),
    ("Schwarz D cell", "schwarzite (knees)", {"kind": "diamond"}, -96, True),
    ("gyroid cell", "schwarzite (knees)", {"kind": "gyroid"}, -48, True),
]


def main(argv: list[str] | None = None) -> int:
    """Build each case, print its census against its law, return 0 if all met."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="Include the schwarzite cells (minutes, not "
                             "seconds).")
    parser.add_argument("--write", metavar="DIR",
                        help="Also write each structure as DIR/<name>.xyz.")
    parser.add_argument("--no-relax", action="store_true",
                        help="Skip the force field. The census is set by the "
                             "mesh, so only the geometry changes.")
    args = parser.parse_args(argv)

    from ase.io import write as ase_write

    from ..jobs import Job, build
    from ..validation.quality import sp2_quality

    out = Path(args.write) if args.write else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    header = (f"{'structure':24s} {'atoms':>6s}  {'census':30s} "
              f"{'sum(6-n)':>9s} {'law':>6s}  {'bonds (A)':13s} "
              f"{'verdict':9s} {'placed':>7s} {'s':>6s}")
    print(header)
    print("-" * len(header))

    failures = 0
    for label, mode, params, budget, slow in CASES:
        if slow and not args.all:
            print(f"{label:24s} {'--':>6s}  (skipped; pass --all)")
            continue
        started = time.time()
        arguments = dict(params)
        if args.no_relax:
            arguments["relax"] = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                atoms = build(Job(mode, arguments))
        except Exception as problem:                     # noqa: BLE001
            print(f"{label:24s} FAILED: {problem}")
            failures += 1
            continue
        info = atoms.info
        counts = info.get("ring_counts", {})
        census = " ".join(f"{s}:{c}" for s, c in sorted(counts.items()))
        deficit = sum((6 - s) * c for s, c in counts.items())
        geometry = info.get("geometry", {})
        verdict = sp2_quality(geometry)[0] if geometry else "?"
        placed = info.get("disclinations_placed")
        agrees = "--" if budget is None else ("ok" if deficit == budget
                                              else "WRONG")
        if agrees == "WRONG":
            failures += 1
        bonds = (f"{geometry.get('bond_min', 0):.3f}-"
                 f"{geometry.get('bond_max', 0):.3f}")
        shown = "--" if placed is None else f"{100 * placed:.0f}%"
        print(f"{label:24s} {len(atoms):6d}  {census:30s} {deficit:+9d} "
              f"{agrees:>6s}  {bonds:13s} {verdict:9s} {shown:>7s} "
              f"{time.time() - started:6.1f}")
        if out:
            stem = label.replace(", ", "_").replace(" ", "_")
            ase_write(out / f"{stem}.xyz", atoms)

    print()
    print("census    ring size : count, from the builder's own recorded rings")
    print("sum(6-n)  measured; 'law' says whether it equals the budget the")
    print("          skeleton fixes BEFORE anything is meshed")
    print("placed    fraction of disclinations on the curvature side they")
    print("          belong on, from fitting the surface over each ring --")
    print("          not read off the ring size")
    if out:
        print(f"\nWrote {len(list(out.glob('*.xyz')))} files to {out}/")
    print(f"\n{failures} law(s) not met." if failures else "\nAll laws met.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
