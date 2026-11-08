"""Geometric porosity: does anything fit in there, and how much of it.

This runs **before** a grand-canonical simulation, not after, and on a
structure that has never been near one. Four reasons, in the order they
matter:

1. **It answers the first question about any new framework.** A
   superstructure with no channel wider than a hydrogen molecule will
   adsorb nothing, and finding that out costs seconds here against
   hours of Monte Carlo. :attr:`PorosityReport.largest_free_sphere` is
   the number: a probe larger than it cannot cross the cell at all.
2. **A weight percentage needs a denominator.** Quoting mmol/g or wt%
   means committing to what counts as the adsorbent, and the pore
   volume is how that is normally stated.
3. **Absolute and excess adsorption differ by ``ρ_bulk × V_void``**, so
   comparing a simulation with an experiment needs the void volume.
   The thermodynamically correct one is the **helium void volume**,
   which is not the geometric one and is the thing the experimental
   literature means. Both are computed here and they are not
   interchangeable.
4. **The geometric surface area is what a BET number is compared
   with**, and the comparison is routinely made without saying that
   they are different quantities: BET is a model fitted to a measured
   nitrogen isotherm, this is a probe rolled over hard spheres.
   Agreement within tens of per cent is normal; agreement to three
   figures would be suspicious.

**Everything here is stochastic and everything takes a seed.** The
errors reported are the Monte Carlo errors and they are the honest
precision of each number — a void fraction quoted to four decimals from
ten thousand samples is three decimals of noise.

**What a "radius" means here.** The framework atoms are hard spheres of
their Lennard-Jones σ/2 by default, not their covalent or van der Waals
radius. That choice is what makes the result comparable with an
adsorption simulation that uses the same σ, which is the whole point of
computing it in this module rather than in a structure-analysis one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree

from .potentials import H2_C

#: Avogadro's number.
AVOGADRO: float = 6.02214076e23

#: Kinetic diameter of H2 in Å, the usual probe for "will hydrogen fit".
#: Note this is NOT the Lennard-Jones σ of 2.958 Å: the kinetic
#: diameter is the one quoted for sieving, and it is the smaller.
H2_KINETIC_DIAMETER: float = 2.89

#: Helium's Lennard-Jones parameters (ε/k in K, σ in Å), for the void
#: volume. Helium is the experimental standard because it is small and
#: barely adsorbs, so the volume it reports is close to the geometric
#: one without anyone having to define a hard sphere.
HELIUM_EPSILON: float = 10.9
HELIUM_SIGMA: float = 2.64


@dataclass
class PorosityReport:
    """Everything the geometry says, with the error on each number.

    Attributes
    ----------
    void_fraction, void_fraction_error
        Fraction of the cell a hard-sphere probe of ``probe_radius``
        can reach, and its Monte Carlo error.
    pore_volume_cm3_g
        That fraction as a specific volume.
    helium_void_fraction
        The thermodynamic void fraction, ``<exp(-βU_He)>``, which is
        what converts absolute adsorption to excess. Larger than the
        geometric one, because a helium atom samples the attractive
        region too.
    accessible_area_m2_g
        Probe-accessible surface area. Compare with BET only knowing
        they are different quantities.
    largest_included_sphere
        Diameter (Å) of the largest sphere that fits anywhere in the
        cell. The size of the biggest cavity.
    largest_free_sphere
        Diameter (Å) of the largest sphere that can cross the cell
        along the best periodic direction, estimated from the void
        network. A probe bigger than this cannot percolate: the
        material is closed to it however large its cavities are.
    pore_diameters, pore_histogram
        Pore size distribution: bin centres in Å and the fraction of
        void volume in each.
    """

    probe_radius: float
    void_fraction: float
    void_fraction_error: float
    pore_volume_cm3_g: float
    helium_void_fraction: float
    accessible_area_m2_g: float
    largest_included_sphere: float
    largest_free_sphere: float
    cell_volume_a3: float
    framework_mass_g: float
    pore_diameters: np.ndarray = field(default_factory=lambda: np.empty(0))
    pore_histogram: np.ndarray = field(default_factory=lambda: np.empty(0))
    samples: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """A few lines for a report, with the errors kept attached."""
        lines = [
            f"probe radius {self.probe_radius:.3f} A",
            f"  void fraction      {self.void_fraction:.4f} "
            f"+- {self.void_fraction_error:.4f}   "
            f"({self.pore_volume_cm3_g:.4f} cm3/g)",
            f"  helium void        {self.helium_void_fraction:.4f}"
            "   <- the one that converts absolute to excess",
            f"  accessible area    {self.accessible_area_m2_g:.0f} m2/g",
            f"  largest cavity     {self.largest_included_sphere:.2f} A",
            f"  largest channel    {self.largest_free_sphere:.2f} A"
            "   <- a probe above this cannot cross the cell",
        ]
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def _edges(atoms: Atoms) -> np.ndarray:
    """Orthorhombic cell edges, or a clear refusal."""
    cell = np.asarray(atoms.cell, dtype=float)
    edges = np.diag(cell)
    if not np.allclose(cell - np.diag(edges), 0.0, atol=1e-8):
        raise ValueError(
            "this cell is triclinic, and every neighbour search here is a "
            "cKDTree with boxsize, which is orthorhombic-only. A triclinic "
            "cell would be sampled against the wrong minimum images and the "
            "void fraction would come back plausible and wrong."
        )
    if np.any(edges <= 0.0):
        raise ValueError("the structure has no cell to sample inside")
    return edges


def _distance_to_framework(
    points: np.ndarray, tree: cKDTree, edges: np.ndarray
) -> np.ndarray:
    """Distance from each point to the nearest framework atom."""
    wrapped = np.mod(points, edges)
    np.clip(wrapped, 0.0, np.nextafter(edges, 0.0), out=wrapped)
    distances, _ = tree.query(wrapped, k=1)
    return distances


def analyse_porosity(
    atoms: Atoms,
    probe_radius: float = 0.5 * H2_KINETIC_DIAMETER,
    atom_radius: float = 0.5 * H2_C.sigma,
    samples: int = 200_000,
    area_points: int = 500,
    temperature: float = 298.0,
    bins: int = 40,
    seed: int = 0,
) -> PorosityReport:
    """Measure a framework's porosity by Monte Carlo insertion.

    Parameters
    ----------
    atoms
        The framework, in an orthorhombic periodic cell.
    probe_radius
        Hard-sphere radius of the probe in Å. The default is half the
        kinetic diameter of H2.
    atom_radius
        Hard-sphere radius of a framework atom in Å. The default is
        half the H2-C Lennard-Jones σ, so that "fits" here means the
        same thing it means to the simulation in
        :mod:`carbonforge.adsorption.gcmc`. A van der Waals radius
        (1.70 Å for carbon) gives a smaller void fraction and is the
        convention much of the Zeo++ literature uses — say which when
        quoting a number.
    samples
        Random points for the void fraction and the pore size
        distribution. The error scales as ``1/sqrt(samples)``.
    area_points
        Points per atom for the accessible area. 500 gives about 2 %.
    temperature
        For the helium void volume only, in kelvin. 298 K is the
        experimental convention.
    bins
        Bins in the pore size distribution.
    seed
        Required: this is a stochastic measurement.

    Returns
    -------
    PorosityReport
        Every quantity with its error and its caveats.
    """
    edges = _edges(atoms)
    positions = np.mod(np.asarray(atoms.get_positions(), dtype=float), edges)
    np.clip(positions, 0.0, np.nextafter(edges, 0.0), out=positions)
    tree = cKDTree(positions, boxsize=edges)
    volume = float(np.prod(edges))
    mass_g = sum(atoms.get_masses()) / AVOGADRO
    rng = np.random.default_rng(seed)
    warnings: list[str] = []

    # --- void fraction, and the free radius at every sampled point.
    # One tree query does both: a point is inside the void when the
    # nearest framework atom is further than atom_radius + probe_radius,
    # and how much further is what the pore size distribution needs.
    points = rng.random((samples, 3)) * edges
    nearest = _distance_to_framework(points, tree, edges)
    free_radius = nearest - atom_radius
    inside = free_radius >= probe_radius
    fraction = float(inside.mean())
    error = float(np.sqrt(max(fraction * (1.0 - fraction), 0.0) / samples))

    # --- helium void volume: Widom insertion, not a hard sphere.
    # This is the quantity the adsorption literature means by "void
    # volume", and it is what converts an absolute isotherm to an
    # excess one. It is larger than the geometric fraction because a
    # helium atom is also attracted.
    helium = _helium_void_fraction(points, tree, edges, temperature)

    # --- accessible surface area: roll the probe over every atom.
    area = _accessible_area(positions, tree, edges, atom_radius,
                            probe_radius, area_points, rng)

    # --- the two sphere diameters.
    largest_included = 2.0 * float(free_radius.max()) if samples else 0.0
    largest_free = 2.0 * _percolating_radius(
        points, free_radius, edges, probe_radius)

    if fraction <= 0.0:
        warnings.append(
            f"no sampled point admits a probe of radius {probe_radius:.2f} A: "
            "this framework is closed to it, and a GCMC run would return "
            "zero after however long it was given.")
    elif largest_free < probe_radius:
        warnings.append(
            f"the probe fits in cavities up to {largest_included:.2f} A "
            f"across but cannot cross the cell (largest channel "
            f"{largest_free:.2f} A). Anything adsorbed would have to be "
            "built in, not loaded in.")
    if error > 0.1 * max(fraction, 1e-9):
        warnings.append(
            f"the void fraction's own error is {100 * error / max(fraction, 1e-9):.0f} % "
            "of it; raise `samples`.")

    centres, histogram = _pore_distribution(
        free_radius[inside], bins=bins)

    return PorosityReport(
        probe_radius=probe_radius,
        void_fraction=fraction,
        void_fraction_error=error,
        pore_volume_cm3_g=(fraction * volume * 1e-24 / mass_g
                           if mass_g > 0 else 0.0),
        helium_void_fraction=helium,
        # Angstrom squared to metres squared: 1e-20, and forgetting it
        # gives an area of 1e23 m2/g, which at least fails loudly.
        accessible_area_m2_g=(area * 1e-20 / mass_g) if mass_g > 0 else 0.0,
        largest_included_sphere=largest_included,
        largest_free_sphere=largest_free,
        cell_volume_a3=volume,
        framework_mass_g=mass_g,
        pore_diameters=centres,
        pore_histogram=histogram,
        samples=samples,
        warnings=warnings,
    )


def _helium_void_fraction(
    points: np.ndarray, tree: cKDTree, edges: np.ndarray, temperature: float,
    cutoff: float = 12.0,
) -> float:
    """``<exp(-βU)>`` for a helium atom, the thermodynamic void volume.

    Lorentz-Berthelot between helium and the H2-C carbon parameters, so
    the carbon is described the same way the adsorption run describes
    it. Classical: helium at 298 K needs no quantum correction, which
    is part of why the convention settled on it.
    """
    from .potentials import PairParameters

    pair = PairParameters(
        epsilon=float(np.sqrt(HELIUM_EPSILON * H2_C.epsilon)),
        sigma=0.5 * (HELIUM_SIGMA + H2_C.sigma),
        reduced_mass=4.0026,
        source="Lorentz-Berthelot of helium with the H2-C carbon",
    )
    wrapped = np.mod(points, edges)
    np.clip(wrapped, 0.0, np.nextafter(edges, 0.0), out=wrapped)
    neighbours = tree.query_ball_point(wrapped, cutoff)
    total = 0.0
    beta = 1.0 / temperature
    for point, near in zip(wrapped, neighbours):
        if not near:
            total += 1.0
            continue
        delta = tree.data[near] - point
        delta -= edges * np.round(delta / edges)
        r = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
        # Classical, deliberately: helium at 298 K is heavy enough and
        # warm enough that the Feynman-Hibbs term is a fraction of a
        # per cent, and the experimental convention this number exists
        # to match is the classical one.
        sr6 = (pair.sigma / r) ** 6
        energy = float(np.sum(4.0 * pair.epsilon * (sr6 * sr6 - sr6)))
        total += float(np.exp(-beta * np.clip(energy, -1e4, 1e4)))
    return total / len(points)


def _accessible_area(
    positions: np.ndarray, tree: cKDTree, edges: np.ndarray,
    atom_radius: float, probe_radius: float, per_atom: int,
    rng: np.random.Generator,
) -> float:
    """Probe-accessible surface area in Å², by rolling a probe.

    Points are spread over each atom's probe sphere (radius
    ``atom_radius + probe_radius``) and kept when no other atom's probe
    sphere contains them. The kept fraction times the sphere's area,
    summed, is the accessible surface — Shrake-Rupley, with periodic
    neighbours.
    """
    shell = atom_radius + probe_radius
    # One set of directions for every atom: the same quadrature
    # everywhere keeps the comparison between atoms exact, and the
    # randomness that matters is already in `rng`'s one draw.
    directions = rng.normal(size=(per_atom, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    sphere_area = 4.0 * np.pi * shell**2
    total = 0.0
    for centre in positions:
        probes = centre + shell * directions
        wrapped = np.mod(probes, edges)
        np.clip(wrapped, 0.0, np.nextafter(edges, 0.0), out=wrapped)
        # `shell` is the radius every other atom blocks within, and the
        # atom itself sits at exactly `shell`, so the query has to
        # exclude it: count neighbours strictly closer.
        counts = tree.query_ball_point(wrapped, shell - 1e-9,
                                       return_length=True)
        total += sphere_area * float(np.mean(counts == 0))
    return total


def _percolating_radius(
    points: np.ndarray, free_radius: np.ndarray, edges: np.ndarray,
    probe_radius: float,
) -> float:
    """Largest probe radius whose void still crosses the cell.

    A bisection on the radius. At each trial the sampled points that
    admit a probe of that size are linked into a graph, and the
    question asked of the graph is whether any component **wraps**:
    whether walking its edges can return to the starting point in a
    different periodic image. That is the definition of percolation
    and it is not the same as "the component is wide" — a spherical
    cavity large enough to span the cell still does not percolate, and
    a proxy on point spacing or on extent says it does.

    Union-find with image offsets does it in one pass: each point
    carries the displacement to its component's root, and an edge
    whose two ends are already in the same component closes a loop
    whose offset is the lattice vector it wraps by. Nonzero offset,
    percolating.

    The estimate is from a point cloud, so it can **under**-report a
    channel too narrow to be sampled; it cannot over-report one,
    because every link is a real pair of void points. The report says
    so rather than presenting it as exact.
    """
    if not len(points):
        return 0.0
    high = float(free_radius.max())
    if high <= 0.0:
        return 0.0
    low = 0.0
    for _ in range(16):
        trial = 0.5 * (low + high)
        if _void_percolates(points[free_radius >= trial], edges, trial):
            low = trial
        else:
            high = trial
    return low


def _void_percolates(
    points: np.ndarray, edges: np.ndarray, radius: float
) -> bool:
    """Does a probe of ``radius`` have a path across the periodic cell?

    Two void points are linked when they are within ``2 * radius``: the
    probe can roll from one to the other without its centre leaving the
    void. Whether the resulting graph wraps is decided by union-find
    with offsets, as described in :func:`_percolating_radius`.
    """
    if len(points) < 2 or radius <= 0.0:
        return False
    wrapped = np.mod(points, edges)
    np.clip(wrapped, 0.0, np.nextafter(edges, 0.0), out=wrapped)
    tree = cKDTree(wrapped, boxsize=edges)
    pairs = tree.query_pairs(2.0 * radius, output_type="ndarray")
    if not len(pairs):
        return False

    parent = np.arange(len(wrapped))
    offset = np.zeros((len(wrapped), 3))

    def find(node: int) -> tuple[int, np.ndarray]:
        shift = np.zeros(3)
        while parent[node] != node:
            shift += offset[node]
            node = parent[node]
        return node, shift

    for a, b in pairs:
        # The image of b that is actually within reach of a.
        delta = wrapped[int(b)] - wrapped[int(a)]
        image = -edges * np.round(delta / edges)
        root_a, shift_a = find(int(a))
        root_b, shift_b = find(int(b))
        # Displacement from a's root to b's root through this edge.
        through = shift_a + (delta + image) - shift_b
        if root_a == root_b:
            # A closed loop's discrepancy is either zero or a lattice
            # vector, so the test is per axis against that axis's own
            # edge -- not against the smallest, which would judge a
            # wrap along a long cell by a short cell's standard.
            if np.any(np.abs(through) > 0.5 * edges):
                return True
            continue
        parent[root_b] = root_a
        offset[root_b] = through
    return False


def _pore_distribution(
    free_radius: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pore size distribution over the void points.

    Each void point contributes the diameter of the largest sphere
    centred on it that does not overlap the framework. That is the
    simple definition, not the Gelb-Gubbins one, which asks for the
    largest sphere *containing* the point wherever its centre sits and
    gives a distribution shifted to larger pores. The simple one is
    reported because it is the one these samples support; the
    difference matters for a narrow channel and is worth saying.
    """
    if not len(free_radius):
        return np.empty(0), np.empty(0)
    diameters = 2.0 * free_radius
    histogram, borders = np.histogram(diameters, bins=bins, density=False)
    centres = 0.5 * (borders[:-1] + borders[1:])
    total = histogram.sum()
    return centres, histogram / total if total else histogram.astype(float)
