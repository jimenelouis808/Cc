"""Example 10 — a carbide core inside graphitic shells, from a ``.cif``.

The material this was written for: a metal or carbide particle wrapped in
a few graphitic layers, which is what arc discharge and CVD produce and
what a Raman D/G plus an XPS survey of such a sample is measuring.

Run: ``python -m nanocarbon_lab.examples.ex10_encapsulated_nanoparticle``

It writes its own ``.cif`` first so the example runs with no input file.
Point ``CRYSTAL`` at a real one -- yours, or a database download -- to
use it instead; nothing below cares where the crystal came from.

Four things worth watching in the output, each of which is a decision the
cut makes and the caller does not:

1. The **centring**. For a compound it decides whether the particle comes
   out cation- or anion-rich, and therefore whether it is charged.
2. **Atoms on the cut**. Non-zero means the facets landed on lattice
   planes and the atom count is sensitive to the last digit of the size.
3. The **achieved** gap, in three numbers. The wall of a closed cage is
   corrugated, so the mean, the fifth percentile and the minimum say
   different things.
4. ``sum(6-n)`` against Euler. An envelope is sphere-like whatever it
   wraps, so the budget is +12 and it is checked rather than assumed.
"""

from pathlib import Path

from ase.build import bulk
from ase.io import write

from nanocarbon_lab.exports import write_render_bundle
from nanocarbon_lab.nanoparticle import particle_quality
from nanocarbon_lab.nanoparticle.build import (
    build_nanoparticle,
    estimate_nanoparticle_atoms,
)

#: Replace with a path to your own crystal to use it instead.
CRYSTAL = Path("out/np/iron.cif")


def main() -> None:
    outdir = Path("out/np")
    outdir.mkdir(parents=True, exist_ok=True)
    if not CRYSTAL.exists():
        write(CRYSTAL, bulk("Fe", "bcc", a=2.866, cubic=True))
        print(f"wrote a bcc iron cell to {CRYSTAL} to run against")

    shape, params = "truncated octahedron", {"face_distance": 8.0}
    print(f"estimate before building: "
          f"{estimate_nanoparticle_atoms(CRYSTAL, shape, params, shells=2)} "
          "atoms")

    atoms, crystal, carve, coat = build_nanoparticle(
        CRYSTAL,
        shape=shape,
        shape_params=params,
        # The control that decides the composition of a compound. On an
        # element it only shifts which atom sits at the centre.
        centre="atom:Fe",
        shells=2,
        # 3.35 A is graphite's spacing and a van der Waals contact. Iron
        # forms a stable carbide and chemisorbs nearer 2.1; both are real
        # and they are different interfaces, so the builder warns either
        # way rather than choosing.
        gap=3.35,
        interlayer=3.35,
        # On the SHELL. The core is never substituted: a nitrogen where
        # an iron was is a different compound, not a dopant.
        dopant="N",
        dopant_conc=0.03,
        dopant_site="pentagon",
        graft="carboxyl",
        graft_coverage=0.03,
        seed=0,
    )

    print()
    print(crystal.summary())
    print()
    print(carve.summary())
    verdict, why = particle_quality(carve)
    print(f"  verdict: {verdict.upper()} -- {why}")
    print()
    if coat is not None:
        print(coat.summary())

    xyz, meta = write_render_bundle(atoms, outdir / "fe_at_c")
    print()
    print(f"{len(atoms)} atoms written to {xyz} and {meta}")
    start, stop = atoms.info["core_indices"]
    print(f"core is atoms [{start}, {stop}); the rest is the shell, "
          "its dopants and its groups")


if __name__ == "__main__":
    main()
