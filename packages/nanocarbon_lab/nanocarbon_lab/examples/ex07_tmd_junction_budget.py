"""Why an MX2 junction is easier to close than an MX2 schwarzite.

Example 6 ends on an obstruction: even mesh degrees make the net
bipartite on a sphere, but above genus 0 there are 2g more Z/2 classes
and the repair does not control which side of them it lands on, so a
36 Å Schwarz P cell keeps 2.2% homoelemental bonds however hard it is
repaired.

A capped tube junction has no such classes. Whatever its arm count, it
is a sphere with tubes pulled out of it -- genus 0, chi = 2 -- so the
2-colouring theorem applies exactly: even degrees are necessary *and*
sufficient, and ``parity="split"`` reaches zero homoelemental bonds every
time. That is why the junction builder defaults to ``split`` and the
schwarzite builder does not.

The ring budget flips sign with the genus, and this prints both halves
of it:

* ``sum(6 - n) = 6*chi = +12`` here, where a Schwarz P cell wants -24.
* With odd rings forbidden, +12 cannot be paid in pentagons. It is paid
  in **squares** at +2 each -- the same six-square budget that closes an
  MX2 sphere as the observed MoS2 nano-octahedron. The caps carry them;
  the crotch between the arms is a saddle and pays its negative curvature
  back in octagons and decagons.

So the census below is not an arbitrary mix of ring sizes. Every square
is positive curvature the caps needed, every octagon is negative
curvature the branch needed, and the two must come out at exactly +12.

Builds take a couple of minutes each. Run with::

    python -m nanocarbon_lab.examples.ex07_tmd_junction_budget
"""

from __future__ import annotations

from ..tmd.curved import build_tmd_junction, schwarzite_quality


def show(kind: str, tube_radius: float = 12.0, arm_length: float = 26.0) -> None:
    print(f"\n=== MoS2 {kind} junction, r = {tube_radius:.0f} Å, "
          f"arms {arm_length:.0f} Å ===")
    atoms = build_tmd_junction("MoS2", kind=kind, tube_radius=tube_radius,
                               arm_length=arm_length, parity="split")
    info = atoms.info

    rings = ", ".join(f"{k}:{v}" for k, v in sorted(info["ring_counts"].items()))
    deficit = info["ring_deficit"]
    expected = 6 * info["euler"]

    print(f"  {len(atoms)} atoms, X/M = {info['stoichiometry']:.4f}")
    print(f"  genus {info['genus']}, chi {info['euler']}")
    print(f"  rings {rings}")
    print(f"  sum(6-n) = {deficit}, 6·χ = {expected} "
          f"{'✓' if deficit == expected else '✗'}")
    print(f"  odd rings {info['odd_rings']}, "
          f"{info['parity_splits']} splits")
    print(f"  homoelemental bonds {info['antiphase_bonds']}/"
          f"{info['n_net_bonds']} ({info['antiphase_fraction']:.1%})")
    print(f"  metal coordination {info['graph_metal_coordination']}, "
          f"chalcogen {info['graph_chalcogen_coordination']}")
    verdict, reason = schwarzite_quality(atoms)
    print(f"  {verdict.upper()}: {reason}")


def main() -> None:
    # L bends one tube; Y branches three. The budget is +12 for both,
    # because the arm count changes the genus not at all.
    show("L", tube_radius=10.0, arm_length=20.0)
    show("Y")
    print("\nThe arm count does not touch the genus, so the budget is +12")
    print("for every junction -- and on a sphere, unlike on a minimal")
    print("surface, even rings are the whole story.")


if __name__ == "__main__":
    main()
