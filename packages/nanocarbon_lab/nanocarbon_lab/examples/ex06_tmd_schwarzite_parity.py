"""What the M/X parity repair does to an MX2 schwarzite, in one run.

Atoms sit at triangle centroids and bond across shared mesh edges, so the
net is the mesh's dual and **ring size == mesh vertex degree**. M and X
have to alternate around every ring, i.e. the net must be bipartite, and
a triangulation's faces are 2-colourable exactly when every vertex degree
is even. So "MoS2 schwarzite" is really "even-degree triangulation of a
triply periodic minimal surface".

Note what that does *not* say. Pentagons are forbidden, but negative
curvature never needed them: ``sum(6 - n) = 6*chi``, and an octagon pays
-2 just as a heptagon does while staying even. Schwarz P (chi = -4) wants
twelve octagons, and the chemistry permits that -- h-BN schwarzites in
the literature are built exactly this way. What pentagons rule out is the
*sphere*, which MX2 closes with six squares instead.

Running this prints the three repairs side by side and shows:

1. **Splitting reaches exactly zero odd rings.** An edge flip toggles
   four vertex degrees at once, so with odd vertices sparse it can only
   shuffle them; a split toggles exactly the two vertices opposite the
   edge, which is the weight-two move that annihilates a pair.

2. **Even degrees are necessary but not sufficient above genus 0.** The
   2-colouring theorem is a sphere result; at genus g there are 2g more
   Z/2 classes and even-degree meshes exist on both sides of them. So a
   perfectly bipartite cell is *reachable but not guaranteed* -- at 30 Å
   the split repair lands on zero homoelemental bonds with X/M exactly
   2.0000, and at 36 Å the same repair leaves 2.2%. What is left over is
   an inversion-domain boundary, the line defect seen in grown MoS2 and
   h-BN, so the builder reports its size rather than assuming it away.

3. **The repair costs geometry.** Each split inserts a vertex, and enough
   of them put more sites on the surface than its area holds at spacing
   ``a/sqrt(3)``. That is the whole reason ``parity`` is a parameter
   rather than a decision made inside the builder.

Run with::

    python -m nanocarbon_lab.examples.ex06_tmd_schwarzite_parity
"""

from __future__ import annotations

from ..tmd.curved import build_tmd_schwarzite, schwarzite_quality


def show(material: str, kind: str, cell: float) -> None:
    print(f"\n=== {material} on the {kind} surface, {cell:.0f} Å cell ===")
    print(f"  {'parity':7s} {'atoms':>6s} {'odd':>4s} {'M–M/X–X':>9s} "
          f"{'X/M':>7s} {'p95':>7s} {'worst':>7s}")
    for parity in ("none", "flip", "split"):
        atoms = build_tmd_schwarzite(material, kind, cell=cell, parity=parity)
        info = atoms.info
        print(f"  {parity:7s} {len(atoms):6d} {info['odd_rings']:4d} "
              f"{info['antiphase_fraction']:8.2%} "
              f"{info['stoichiometry']:7.4f} "
              f"{info['bond_deviation_p95']:6.1%} "
              f"{info['bond_deviation_max']:6.1%}")
        if parity == "split":
            deficit = info["ring_deficit"]
            expected = 6 * info["euler"]
            rings = ", ".join(f"{k}:{v}"
                              for k, v in sorted(info["ring_counts"].items()))
            print(f"          rings {rings}")
            print(f"          sum(6-n) = {deficit}, 6·χ = {expected} "
                  f"{'✓' if deficit == expected else '✗'}   "
                  f"genus {info['genus']}")
            print(f"          {schwarzite_quality(atoms)[0].upper()}")


def main() -> None:
    # 30 Å is where the split repair happens to land on a bipartite
    # triangulation; 36 Å is where it does not. Same surface, same
    # material, same repair -- the obstruction belongs to the mesh.
    show("MoS2", "primitive", 30.0)
    show("MoS2", "primitive", 36.0)
    print("\nEven rings are necessary, not sufficient: at genus 3 six more")
    print("Z/2 classes decide whether a cell can alternate perfectly, and")
    print("the repair does not control which side of them it lands on.")


if __name__ == "__main__":
    main()
