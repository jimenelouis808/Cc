"""Wrapping a nanoparticle in graphitic carbon.

A carbon-encapsulated particle is a real and common material -- arc
discharge and CVD both produce metal cores inside a few graphitic shells,
and the user's own Fe3C / alpha-Fe sample is one -- and it is also the one
place where this framework's two halves meet: a crystal core from
somebody's ``.cif``, and a closed carbon surface built the way every
other curved carbon in this package is built.

**The route is the existing one.** A graphitic shell around a particle is
a closed carbon surface, so it goes through exactly the pipeline the
junctions, schwarzites, networks and supernetworks go through: a
signed-distance field, marching cubes, isotropic remeshing to the
honeycomb's ring spacing, the dual, and a valence-force-field relaxation
(:func:`nanocarbon_lab.builders.junction._finish`, shared with those
builders rather than copied). Nothing about the ring statistics is
placed or assumed: the field's own topology fixes them, and the Euler
check inside that function verifies the census against the mesh's genus.
A sphere-like core therefore gets 12 pentagons' worth of deficit, a
toroidal one gets zero, and a hollow cage gets 12 per closed shell --
none of which this module has to know.

**The field is the core's own van der Waals envelope**, not the shape it
was cut to. ``min_i |r - r_i| - gap`` over the core atoms: the set of
points at least ``gap`` from every atom. Three things follow, and all
three are the reason to do it this way rather than offsetting the
carving shape:

* It works for a core from **anywhere** -- a relaxed particle, an
  imported cluster, a structure this framework did not build. There is no
  shape object to require.
* It follows the **atoms**, so a facet that lost its outermost plane to
  the loose-atom trim is coated where the atoms actually are.
* It **rounds every edge and corner at radius** ``gap``, automatically,
  because that is what the offset of a solid is. This is not an
  approximation to apologise for: a graphitic sheet cannot fold along a
  zero-radius crease, and a real encapsulated cube is round at the
  corners in every micrograph of one. The geometry and the physics agree
  here, which is rare enough to be worth saying.

**The gap is centre to centre, and what is reported is what was
achieved.** ``gap`` places the shell's surface that far from the nearest
core atom; the relaxation then pulls on the shell, so the finished
distance is measured and reported, never restated from the request. The
same rule the coil builder follows for its pitch. The wall is anchored to
its own surface during the relaxation (``wall_anchor``), which is on by
default here and off for a junction -- and for a physical reason rather
than a tuning one: a junction's wall has nothing behind it, while a
coated particle's wall has the core in the way, so a free relaxation that
lets it collapse inward is relaxing through solid matter.

**A one-sided constraint cannot set the radius, and that was measured
the hard way.** The constraint a core imposes looks one-sided -- the
carbon must not go inside, and is free everywhere else -- so the obvious
scheme is alternating projection: relax freely, find the atoms that ended
up too close, push those onto the envelope, anchor those only, relax
again. It was built and it does not work. Pushing one atom out pulls its
neighbours in, the loop oscillates, and asked to hold the whole wall at
0.8 of a 3.35 Å gap it ended with 133 atoms of 594 still violating, a
closest approach of 1.36 Å and 1.70 Å bonds -- worse on every count than
doing nothing. At a stiff enough anchor it has nothing to do, and at a
soft one it destroys what the anchor achieved (1.92 Å bonds). So it is
not here. A closed network's radius is already fixed by its ring count;
a repair loop cannot renegotiate it, and the anchor along the normal is
the constraint that respects that.

**3.35 Å is a van der Waals contact, not a chemical bond.** It is
graphite's interlayer spacing, and it is the right order for carbon
physisorbed on a noble metal or an oxide. The carbide formers -- Fe, Co,
Ni -- chemisorb, and their metal-carbon distance is around 2.1 Å, which
is a different structure with a different electronic answer. This builder
places a van der Waals shell by default and says so; a chemisorbed
interface is `gap=2.1` and the honest warning that it has not been
relaxed against any interface energy, because nothing here computes one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree

from ..builders import remesh as rm
from ..builders.junction import MAX_GRID_RESOLUTION, _finish
from ..dopants import codope, dope_random, dope_rings
from ..functionalize import functionalize
from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.rng import make_rng
from ..validation.quality import sp2_quality
from .shapes import Field

#: Graphite's interlayer spacing, Å. The default core-to-shell distance
#: and the default shell-to-shell one.
GRAPHITIC_GAP = 3.35

#: Elements whose carbides are stable, so that carbon on them
#: chemisorbs at roughly 2.1 Å rather than sitting at a van der Waals
#: contact. Used only to warn: the distance is the caller's to choose,
#: and this framework computes no interface energy to choose it with.
CARBIDE_FORMERS = frozenset({
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Zr", "Nb", "Mo", "Hf",
    "Ta", "W", "Si", "B", "Al",
})

#: Closest a shell atom may come to a core atom before the coating is
#: reported as touching, Å. A floor and not the target: the wall of a
#: closed carbon cage is genuinely corrugated, so a 3.35 Å gap means
#: 3.35 Å on average and less at the dips, and 1.8 Å is where a contact
#: stops being a contact and becomes a bond.
CORE_SHELL_FLOOR = 1.8

#: Target voxel as a fraction of the gap. The surface's finest feature is
#: its corner rounding, whose radius is the gap, so this is what keeps
#: marching cubes from faceting a corner into a prism.
VOXEL_FRACTION = 0.35


@dataclass
class CoatReport:
    """What the coating came out as, measured.

    Attributes
    ----------
    n_shells
        How many closed carbon shells were built.
    shell_atoms
        Carbon atoms per shell.
    requested_gap, achieved_gap_min, achieved_gap_p5, achieved_gap_mean
        What was asked for, and the shortest, fifth-percentile and mean
        core-to-shell distance after relaxation. Three numbers because
        one will not do: the wall of a closed cage is corrugated, so the
        **mean** says where the shell sits, the **minimum** is a single
        atom out of several hundred, and the **fifth percentile** is the
        one to compare against a van der Waals contact. A coating judged
        on its minimum alone reads as a failure whenever it is merely
        wavy.
    interlayer
        Requested shell-to-shell spacing, and ``achieved_interlayer``
        what came out.
    ring_counts
        Ring-size census per shell, from the mesh rather than perceived.
    ring_budget
        ``sum(6 - n)`` per shell and the value the mesh's Euler
        characteristic demands. They are equal or the build raised.
    verdict
        :func:`~nanocarbon_lab.validation.quality.sp2_quality` on each
        shell's geometry, unchanged -- the sp2 window is not widened
        here to make a coating pass it.
    bond_max, bond_p95
        Per shell, the longest bond and the 95th percentile. Both,
        because the verdict is set by the maximum and the maximum is one
        bond out of a thousand: a coating conforming to a faceted core
        carries its strain in a few bonds at the corners while the rest
        of the wall stays graphitic, and a single number cannot say that.
        A shell whose p95 is near 1.45 Å and whose max is 1.58 Å is a
        good wall with a strained corner, not a bad wall.
    dopant, dopant_fraction, graft, graft_count
        What chemistry was applied to the shell, and how much of it
        landed.
    warnings
        Everything worth saying before this structure is used.
    """

    n_shells: int
    shell_atoms: list[int]
    requested_gap: float
    achieved_gap_min: float
    achieved_gap_p5: float
    achieved_gap_mean: float
    interlayer: float
    achieved_interlayer: list[float]
    ring_counts: list[Mapping[int, int]]
    ring_budget: list[tuple[int, int]]
    verdict: list[str]
    bond_max: list[float] = dataclass_field(default_factory=list)
    bond_p95: list[float] = dataclass_field(default_factory=list)
    dopant: str | None = None
    dopant_fraction: float = 0.0
    graft: str | None = None
    graft_count: int = 0
    warnings: list[str] = dataclass_field(default_factory=list)

    def summary(self) -> str:
        """The report as text."""
        lines = [
            f"{self.n_shells} graphitic shell(s), "
            f"{sum(self.shell_atoms)} carbon atoms "
            f"({', '.join(str(count) for count in self.shell_atoms)})",
            f"  gap requested {self.requested_gap:.2f} A, achieved "
            f"{self.achieved_gap_mean:.2f} A on average, "
            f"{self.achieved_gap_p5:.2f} A at the 5th percentile, "
            f"{self.achieved_gap_min:.2f} A at the single closest atom",
        ]
        if self.n_shells > 1:
            spacing = ", ".join(f"{value:.2f}"
                                for value in self.achieved_interlayer)
            lines.append(
                f"  interlayer requested {self.interlayer:.2f} A, achieved "
                f"{spacing} A")
        for index, (counts, budget, verdict) in enumerate(
                zip(self.ring_counts, self.ring_budget, self.verdict,
                    strict=True)):
            census = " ".join(f"{size}:{count}"
                              for size, count in sorted(counts.items()))
            strain = ""
            if self.bond_p95 and self.bond_max:
                strain = (f", bonds p95 {self.bond_p95[index]:.3f} / max "
                          f"{self.bond_max[index]:.3f} A")
            lines.append(
                f"  shell {index + 1}: rings {census}, sum(6-n)="
                f"{budget[0]} (Euler asks {budget[1]}){strain}, {verdict}")
        if self.dopant:
            lines.append(
                f"  doped with {self.dopant} at "
                f"{100 * self.dopant_fraction:.1f} % of the shell carbons")
        if self.graft:
            lines.append(f"  {self.graft_count} x {self.graft} grafted on "
                         "the outer face")
        lines += [f"  ! {text}" for text in self.warnings]
        return "\n".join(lines)


def vdw_envelope(core: Atoms, gap: float,
                 element_gaps: Mapping[str, float] | None = None,
                 ) -> tuple[Field, float]:
    """The surface at distance ``gap`` from the nearest core atom.

    Parameters
    ----------
    core
        The particle to wrap. Only its positions and symbols are used.
    gap
        Centre-to-centre distance from the nearest core atom, Å.
    element_gaps
        Per-element overrides, so a core of two very differently sized
        elements can have its envelope follow both. An element not named
        uses ``gap``.

    Returns
    -------
    (field, extent)
        The signed-distance field and the half-width of a cube that
        contains its zero level set, which is what sizes the sampling
        grid.

    Notes
    -----
    ``min_i (|r - r_i| - g_i)`` is the exact signed distance to the union
    of balls whenever the balls do not overlap, and an underestimate in
    the crevices where they do -- which is the safe direction: an
    underestimate puts the surface slightly further out, never inside the
    core. Evaluated with a k-d tree over the core atoms, which is what
    makes it cheap enough to call from inside the remesher's projection
    loop.
    """
    positions = np.asarray(core.get_positions(), dtype=float)
    if len(positions) == 0:
        raise ValueError("the core has no atoms to wrap.")
    symbols = core.get_chemical_symbols()
    if element_gaps:
        radii = np.asarray([float(element_gaps.get(symbol, gap))
                            for symbol in symbols])
    else:
        radii = np.full(len(positions), float(gap))
    if np.any(radii <= 0):
        raise ValueError("every gap must be positive.")
    tree = cKDTree(positions)
    uniform = bool(np.ptp(radii) < 1e-12)
    # With one gap for every atom the nearest atom is the nearest
    # surface, so one query answers it. With per-element gaps it is not
    # -- a nearer small atom can be further from its own surface than a
    # more distant large one -- so enough neighbours are queried to
    # cover the spread.
    neighbours = 1 if uniform else min(len(positions), 12)
    spread = float(radii.max() - radii.min())

    def field(points: np.ndarray) -> np.ndarray:
        query = np.asarray(points, dtype=float)
        flat = query.reshape(-1, 3)
        distance, index = tree.query(flat, k=neighbours, workers=-1)
        if neighbours == 1:
            value = distance - radii[index]
        else:
            value = np.min(np.asarray(distance) - radii[np.asarray(index)],
                           axis=1)
        return value.reshape(query.shape[:-1])

    reach = float(np.linalg.norm(positions, axis=1).max())
    return field, reach + float(radii.max()) + spread + 1.0


def graphitic_coat(
    core: Atoms,
    gap: float = GRAPHITIC_GAP,
    shells: int = 1,
    interlayer: float = GRAPHITIC_GAP,
    bond: float = CC_BOND,
    element_gaps: Mapping[str, float] | None = None,
    grid_resolution: int = 80,
    remesh_iterations: int = 25,
    place_curvature: bool = True,
    anneal_sweeps: int = 0,
    relax_iterations: int = 3000,
    wall_anchor: float = 0.7,
    roughness: float = 0.0,
    vacuum: float = DEFAULT_VACUUM_1D,
    seed: int | None = 0,
) -> tuple[list[Atoms], CoatReport]:
    """Build the graphitic shell(s) around ``core``, without the core.

    Returns the shells themselves, so a caller can dope or functionalise
    them before putting the structure together; :func:`encapsulate` is
    the one-call version that does all of it.

    Parameters
    ----------
    core
        The particle to wrap.
    gap
        Distance from the nearest core atom to the first shell's surface,
        Å. See the module docstring on why 3.35 is a van der Waals
        contact and 2.1 is a carbide interface.
    shells
        How many concentric shells -- a **nano-onion** around the core,
        which is what arc-discharge encapsulation produces.
    interlayer
        Spacing between consecutive shells, Å.
    bond
        Target C-C bond length.
    element_gaps
        Per-element gap overrides; see :func:`vdw_envelope`.
    grid_resolution
        Minimum marching-cubes grid points per axis. The actual
        resolution tracks the box so the voxel stays near
        :data:`VOXEL_FRACTION` of the gap -- the corner rounding has the
        gap as its radius, and a voxel approaching it facets the corners.
    remesh_iterations, relax_iterations, roughness, vacuum, seed
        As for the other implicit builders.
    place_curvature
        Move the disclinations to where the surface's own Gaussian
        curvature asks for them, by Stone-Wales flips after the remesh.
        **On by default here, and off for the junction** -- the same
        switch, opposite defaults, for a reason that is about the surface
        and not about taste.

        A junction saddles: its Gaussian curvature is negative
        everywhere, pentagons do not belong on it at all, and a
        spread-out population of 5-7 pairs is how the net covers it. A
        coating's envelope is the offset of a solid, so it is **convex
        everywhere**: positive curvature, pentagons wanted, heptagons
        pure defects. Measured on a 9 Å gold sphere, placement alone
        takes the census from 39 pentagons / 25 heptagons / 1 octagon to
        **17 / 5 / 0** -- a net deficit of 12 either way, as Euler
        demands, but 22 disclinations instead of 65 -- and the wall that
        results is both smoother and better bonded:

            placement  anneal  anchor  bonds          angle_max  closest C-core
              off        0       1.0   1.316-1.542      132.9      2.08 A
              ON         0       1.0   1.327-1.546      129.1      2.20 A
              off       20       1.0   1.288-1.620      133.8      1.77 A
              ON        20       1.0   1.310-1.619      131.7      1.75 A
              off       60       1.0   1.318-1.563      132.7      1.86 A

        It costs about twice the remesh time (4.7 s to 8.9 s there).
    anneal_sweeps
        Metropolis flip-annealing. **0, and the table above is why**: it
        does cut the stray-pair count, and it makes the wall worse at
        every anchor tried -- 1.620 Å bonds against 1.546. That is the
        same lesson the junction builder records, reached from the other
        side: annealing optimises the *census*, and the census is not
        what a wall is judged by. Curvature placement optimises the
        placement, which is.
    wall_anchor
        Stiffness of a harmonic anchor holding the wall on its envelope,
        along the normal only. **On here and off for the junction**, for
        a physical reason rather than a tuning one: a junction's wall has
        nothing behind it, while a coated particle's wall has the core in
        the way, so a free relaxation that lets the wall move inward is
        relaxing it through solid matter. Measured, a free relaxation of
        this shell gives excellent carbon (1.335-1.494 Å) with an atom
        sitting **0.87 Å** from a gold one.

        It is a trade and the default sits at the useful end of it. On
        the 9 Å gold sphere, with curvature placement on:

            anchor  closest  5th pct  mean   bonds          verdict
             0.70    1.90 A   2.40 A  3.08   1.335-1.550    strained
             1.00    2.01 A   2.45 A  3.08   1.329-1.564    broken
             1.50    2.13 A   2.52 A  3.09   1.322-1.582    broken

        0.7 is the stiffest setting whose bonds are still inside the sp2
        window. Raising it buys a tenth of an Ångström of clearance and
        spends more than that in bond strain. ``relax_iterations`` past
        3000 changes none of these numbers to three decimals, so they are
        converged rather than truncated.

        **There is no setting that makes a coated particle's wall as good
        as a free cage's**, and there should not be: a shell conforming
        to a faceted core is strained, and "strained" is the honest
        verdict rather than a failure. What would be a failure is carbon
        inside the metal, and that is what the anchor prevents.

    Returns
    -------
    (shell_list, report)
        One :class:`ase.Atoms` of pure carbon per shell, innermost
        first.

    Raises
    ------
    ValueError
        For a non-positive shell count, or a gap small enough that the
        shell would be inside the core.
    RuntimeError
        From the shared finisher, if the meshed surface is not closed or
        its ring census does not match its own Euler characteristic.
    """
    if shells < 1:
        raise ValueError("shells must be at least 1.")
    if gap < CORE_SHELL_FLOOR:
        raise ValueError(
            f"a {gap:.2f} A gap is shorter than {CORE_SHELL_FLOOR} A, "
            "which is a bond length rather than a gap: the shell would be "
            "inside the core's own atoms. Graphite's interlayer spacing is "
            f"{GRAPHITIC_GAP} A and a carbide interface is about 2.1 A."
        )
    if interlayer < 2.0:
        raise ValueError(
            f"an interlayer spacing of {interlayer:.2f} A merges consecutive "
            "shells into one solid; graphite's is "
            f"{GRAPHITIC_GAP} A."
        )

    warnings: list[str] = []
    metals = sorted(set(core.get_chemical_symbols()) & CARBIDE_FORMERS)
    if metals and gap > 2.6:
        warnings.append(
            f"the core contains {', '.join(metals)}, which form stable "
            f"carbides: carbon on them chemisorbs at about 2.1 A rather than "
            f"sitting at the {gap:.2f} A van der Waals contact this used. "
            "Both structures exist -- the contact is what a graphitic shell "
            "has over most of its area -- but they are different interfaces "
            "and nothing here computes which one your system prefers."
        )
    if metals and gap < 2.6:
        warnings.append(
            f"a {gap:.2f} A gap is a chemisorbed interface. It has not been "
            "relaxed against any interface energy, because this framework "
            "computes none: the carbon network is relaxed, its distance to "
            "the core is held where you put it."
        )

    rng = make_rng(seed)
    built: list[Atoms] = []
    for index in range(shells):
        distance = gap + index * interlayer
        field, extent = vdw_envelope(core, distance, element_gaps)
        voxel_target = max(VOXEL_FRACTION * distance, 0.5)
        needed = int(math.ceil(2.0 * extent / voxel_target))
        resolution = int(np.clip(needed, grid_resolution,
                                 MAX_GRID_RESOLUTION))
        mesh = rm.marching_cubes_mesh(field, extent, resolution=resolution)
        mesh = rm.isotropic_remesh(
            mesh, field, target_edge=math.sqrt(3.0) * bond,
            iterations=remesh_iterations, anneal_sweeps=anneal_sweeps,
            place_curvature=place_curvature, rng=rng,
        )
        shell = _finish(
            mesh,
            field=field,
            wall_anchor=wall_anchor,
            bond=bond,
            relax_iterations=relax_iterations,
            vacuum=vacuum,
            roughness=roughness,
            rng=rng,
            info={
                "structure_type": "graphitic shell",
                "quality_family": "sp2",
                "shell_index": index,
                "requested_gap": distance,
                "bond": bond,
                "wall_anchor": wall_anchor,
                "place_curvature": place_curvature,
                "anneal_sweeps": anneal_sweeps,
                "roughness": roughness,
            },
        )
        # `_finish` centres each shell in its own cell, which would lose
        # the registry with the core. Put it back where the field was.
        shell.set_positions(shell.get_positions()
                            - shell.get_positions().mean(axis=0)
                            + _field_centre(core))
        _recentre(shell, vacuum)
        built.append(shell)

    report = _measure_coating(core, built, gap, interlayer, warnings)
    return built, report


def _recentre(shell: Atoms, vacuum: float) -> None:
    """Resize the shell's own cell after the lift moved its atoms."""
    positions = shell.get_positions()
    span = positions.max(axis=0) - positions.min(axis=0)
    shell.set_cell(np.diag(span + float(vacuum)))
    shell.pbc = False


def _field_centre(core: Atoms) -> np.ndarray:
    """The point the envelope was built around.

    The envelope is defined by the core's own coordinates, so the shell
    belongs at the core's centroid -- and this has to be undone
    explicitly because the shared finisher re-centres what it returns in
    its own cell, which for a standalone junction is right and here
    would silently slide the shell off the core.
    """
    return np.asarray(core.get_positions(), dtype=float).mean(axis=0)


def _measure_coating(core: Atoms, shells: Sequence[Atoms], gap: float,
                     interlayer: float,
                     warnings: list[str]) -> CoatReport:
    """Measure what was built, rather than restate what was requested."""
    core_tree = cKDTree(np.asarray(core.get_positions(), dtype=float))
    gaps_min: list[float] = []
    gaps_p5: list[float] = []
    gaps_mean: list[float] = []
    for shell in shells:
        distance, _ = core_tree.query(shell.get_positions(), workers=-1)
        gaps_min.append(float(distance.min()))
        gaps_p5.append(float(np.percentile(distance, 5)))
        gaps_mean.append(float(distance.mean()))

    achieved_interlayer: list[float] = []
    for first, second in zip(shells, shells[1:], strict=False):
        tree = cKDTree(first.get_positions())
        distance, _ = tree.query(second.get_positions(), workers=-1)
        achieved_interlayer.append(float(np.median(distance)))

    counts: list[Mapping[int, int]] = []
    budgets: list[tuple[int, int]] = []
    verdicts: list[str] = []
    bond_max: list[float] = []
    bond_p95: list[float] = []
    for shell in shells:
        positions = shell.get_positions()
        lengths = np.asarray([
            float(np.linalg.norm(positions[int(b)] - positions[int(a)]))
            for a, b in shell.info["bonds"]])
        bond_max.append(float(lengths.max()))
        bond_p95.append(float(np.percentile(lengths, 95)))
        census = {int(size): int(number)
                  for size, number in shell.info["ring_counts"].items()}
        counts.append(census)
        deficit = sum((6 - size) * number for size, number in census.items())
        budgets.append((deficit, 6 * int(shell.info["euler"])))
        verdict, explanation = sp2_quality(shell.info["geometry"], "sp2")
        verdicts.append(f"{verdict} ({explanation})" if verdict != "clean"
                        else "clean")

    if gaps_min and gaps_min[0] < CORE_SHELL_FLOOR:
        warnings.append(
            f"the closest shell atom ended up {gaps_min[0]:.2f} A from the "
            f"core, under the {CORE_SHELL_FLOOR} A floor (the fifth "
            f"percentile is {gaps_p5[0]:.2f} A, so it is a few atoms rather "
            "than the wall). Raise `wall_anchor` a little, or the gap, or "
            "round the core's sharp features off."
        )
    if gaps_mean and abs(gaps_mean[0] - gap) > 0.5:
        warnings.append(
            f"the mean gap came out {gaps_mean[0]:.2f} A against the "
            f"{gap:.2f} A requested. The shell is a closed network with its "
            "own preferred size, so it does not sit wherever it is put; the "
            "achieved number is the one to quote."
        )
    for index, (deficit, expected) in enumerate(budgets):
        if deficit != expected:            # pragma: no cover - _finish raises
            warnings.append(
                f"shell {index + 1}: ring budget {deficit} against Euler's "
                f"{expected}."
            )
    return CoatReport(
        n_shells=len(shells),
        shell_atoms=[len(shell) for shell in shells],
        requested_gap=gap,
        achieved_gap_min=min(gaps_min) if gaps_min else math.nan,
        achieved_gap_p5=gaps_p5[0] if gaps_p5 else math.nan,
        achieved_gap_mean=gaps_mean[0] if gaps_mean else math.nan,
        interlayer=interlayer,
        achieved_interlayer=achieved_interlayer,
        ring_counts=counts,
        ring_budget=budgets,
        verdict=verdicts,
        bond_max=bond_max,
        bond_p95=bond_p95,
        warnings=warnings,
    )


def encapsulate(
    core: Atoms,
    gap: float = GRAPHITIC_GAP,
    shells: int = 1,
    interlayer: float = GRAPHITIC_GAP,
    dopant: str | None = None,
    dopant_conc: float = 0.0,
    dopant_site: str = "random",
    codope_spec: Sequence[tuple[str, float]] = (),
    codope_affinity: str = "random",
    graft: str | None = None,
    graft_coverage: float = 0.0,
    graft_where: str = "all",
    seed: int | None = 0,
    vacuum: float = DEFAULT_VACUUM_1D,
    **coat_options: Any,
) -> tuple[Atoms, CoatReport]:
    """Core plus graphitic shell(s), doped and functionalised, in one call.

    Parameters
    ----------
    core
        The particle to encapsulate, from
        :func:`~nanocarbon_lab.nanoparticle.carve.carve_nanoparticle` or
        from anywhere else.
    gap, shells, interlayer
        As for :func:`graphitic_coat`.
    dopant, dopant_conc, dopant_site
        Substitutional heteroatoms **in the shell only**. The core is
        never touched: substituting a nitrogen for an iron is not doping,
        it is a different compound, and the dopant chemistry in
        :mod:`nanocarbon_lab.dopants` is about what an sp2 carbon
        lattice tolerates. ``dopant_site`` accepts ``"random"`` or a ring
        size -- ``"pentagon"`` is the interesting one on a shell, because
        the twelve pentagons Euler demands are where a closed cage
        carries its curvature and its reactivity.
    codope_spec, codope_affinity
        Several heteroatoms in the shell at once, as
        ``[("N", 0.05), ("B", 0.05)]``, each fraction against the
        **same** denominator. Mutually exclusive with ``dopant`` for the
        reason :class:`~nanocarbon_lab.jobs.Job` gives: running one
        substitution after another makes the second fraction a fraction
        of what the first left.
    graft, graft_coverage, graft_where
        A functional group on the shell's **outer** face. Inner-face
        grafting is refused: the inside of the shell is where the core
        is, and a hydroxyl pointing into a gold particle is not a
        structure.
    seed
        Threaded into the meshing, the doping and the grafting.
    vacuum
        Padding in the finished cell, Å.
    **coat_options
        Passed to :func:`graphitic_coat` -- ``bond``, ``element_gaps``,
        ``grid_resolution``, ``remesh_iterations``, ``relax_iterations``,
        ``wall_anchor``, ``roughness``.

    Returns
    -------
    (atoms, report)
        The whole structure: core atoms first, then the shells in order,
        then whatever was grafted. ``atoms.info`` carries the core's own
        ``nanoparticle`` report, the coating report and a ``core_indices``
        / ``shell_indices`` split so a later step can tell them apart --
        which it has to, because almost every analysis in this framework
        is about carbon and would otherwise run over the metal too.
    """
    shell_list, report = graphitic_coat(
        core, gap=gap, shells=shells, interlayer=interlayer,
        vacuum=vacuum, seed=seed, **coat_options)

    if dopant and codope_spec:
        raise ValueError(
            "pass either `dopant` or `codope_spec`, not both: they both "
            "substitute carbons, so running one after the other would make "
            "the second one's fraction a fraction of what the first left."
        )
    if codope_spec:
        shell_list = [codope(shell, list(codope_spec),
                             affinity=codope_affinity, seed=seed)
                      for shell in shell_list]
        report.dopant = "+".join(element for element, _ in codope_spec)
        report.dopant_fraction = float(sum(fraction
                                           for _, fraction in codope_spec))
    elif dopant and dopant_conc > 0:
        shell_list = [_dope_shell(shell, dopant, dopant_conc, dopant_site,
                                  seed) for shell in shell_list]
        report.dopant = dopant
        report.dopant_fraction = dopant_conc

    grafted_total = 0
    if graft and graft_coverage > 0:
        # Only the outermost shell is exposed; grafting an inner one
        # would put the group in the gap between shells, which is 3.35 A
        # and holds nothing.
        outer = shell_list[-1]
        before = len(outer)
        outer = functionalize(outer, graft, coverage=graft_coverage,
                              where=graft_where, face="outer", seed=seed)
        grafted_total = len(outer) - before
        shell_list = [*shell_list[:-1], outer]
        report.graft = graft
        report.graft_count = grafted_total
        if len(shell_list) > 1:
            report.warnings.append(
                "the groups went on the outermost shell only: the 3.35 A "
                "between two shells does not hold a functional group."
            )

    combined = core.copy()
    combined.set_cell(None)
    combined.pbc = False
    core_count = len(combined)
    for shell in shell_list:
        piece = shell.copy()
        piece.set_cell(None)
        piece.pbc = False
        combined += piece

    info: dict[str, Any] = dict(core.info)
    info.update({
        "structure_type": "encapsulated nanoparticle",
        "quality_family": None,
        "core_indices": [0, core_count],
        "shell_indices": [core_count, len(combined)],
        "coating": _coat_as_dict(report),
        "shell_ring_counts": [dict(counts) for counts in report.ring_counts],
    })
    combined.info = info
    positions = combined.get_positions()
    span = positions.max(axis=0) - positions.min(axis=0)
    combined.set_cell(np.diag(span + float(vacuum)))
    combined.center()
    return combined, report


#: Ring names accepted for ``dopant_site``, and the size each means.
RING_SITES = {"pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8}


def _dope_shell(shell: Atoms, dopant: str, concentration: float,
                site: str, seed: int | None) -> Atoms:
    """Substitute heteroatoms into one shell, by site or at random.

    Note what a ring-selected concentration is a fraction **of**: the
    sites of that ring size, not the whole shell. On a coating that is a
    large factor -- the twelve pentagons' worth of deficit is a few dozen
    carbons out of several hundred -- and
    :func:`~nanocarbon_lab.dopants.rings.dope_rings` records the achieved
    fraction against the whole structure for exactly that reason.
    """
    if site in RING_SITES:
        return dope_rings(shell, dopant, ring_size=RING_SITES[site],
                          concentration=concentration, seed=seed)
    if site == "random":
        return dope_random(shell, dopant, concentration, seed=seed)
    raise ValueError(
        f"dopant_site={site!r} is not supported on a shell; use 'random' or "
        f"one of {sorted(RING_SITES)}."
    )


def _coat_as_dict(report: CoatReport) -> dict[str, Any]:
    """The coating report as plain types, so it survives into a file."""
    return {
        "n_shells": report.n_shells,
        "shell_atoms": list(report.shell_atoms),
        "requested_gap": report.requested_gap,
        "achieved_gap_min": report.achieved_gap_min,
        "achieved_gap_p5": report.achieved_gap_p5,
        "achieved_gap_mean": report.achieved_gap_mean,
        "interlayer": report.interlayer,
        "achieved_interlayer": list(report.achieved_interlayer),
        "ring_counts": [{int(k): int(v) for k, v in counts.items()}
                        for counts in report.ring_counts],
        "ring_budget": [list(pair) for pair in report.ring_budget],
        "verdict": list(report.verdict),
        "bond_max": list(report.bond_max),
        "bond_p95": list(report.bond_p95),
        "dopant": report.dopant,
        "dopant_fraction": report.dopant_fraction,
        "graft": report.graft,
        "graft_count": report.graft_count,
        "warnings": list(report.warnings),
    }


__all__ = [
    "CARBIDE_FORMERS",
    "GRAPHITIC_GAP",
    "CORE_SHELL_FLOOR",
    "VOXEL_FRACTION",
    "RING_SITES",
    "CoatReport",
    "encapsulate",
    "graphitic_coat",
    "vdw_envelope",
]
