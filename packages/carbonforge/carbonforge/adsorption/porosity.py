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
    mean_wall_distance
        Mean distance (Å) from a point of the accessible volume to the
        nearest framework atom's SURFACE. It says what
        ``largest_free_sphere`` cannot: that number is the width of the
        tightest place a probe must squeeze through, so two pores with
        the same one can hold very different amounts of volume far from
        any wall — and far from a wall is where nothing is adsorbed.
        For a slit it has a closed form, ``probe_radius / 2 +
        largest_free_sphere / 4``, which is what the test pins; a
        channel of the same free width reads higher the less slit-like
        it is.
    wall_distances, wall_distance_histogram
        That distribution: bin centres in Å and the fraction of
        accessible volume in each.
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
    mean_wall_distance: float = 0.0
    wall_distances: np.ndarray = field(default_factory=lambda: np.empty(0))
    wall_distance_histogram: np.ndarray = field(
        default_factory=lambda: np.empty(0))
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
            f"  mean wall distance {self.mean_wall_distance:.2f} A"
            f"   (a slit of this channel would read "
            f"{0.5 * self.probe_radius + 0.25 * self.largest_free_sphere:.2f})",
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
    grid_spacing: float = 0.4,
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
    grid_spacing
        Resolution in Å of the grid the two sphere diameters are
        measured on. It is **not** the sampling of the void fraction,
        which stays Monte Carlo: a volume is an integral and a channel
        is a yes/no, and they want different tools. Both diameters are
        lower bounds limited by this number — a channel narrower than
        the spacing can be missed — so halve it if a framework reports
        no channel and you expect one. The cost is the cube of it.
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
                            probe_radius, area_points)

    # --- the two sphere diameters, from a grid rather than the cloud.
    # Both are geometric extrema, and a maximum over random samples is
    # biased low by however unlucky the draw was, with no error bar that
    # shrinks usefully. The grid's error is its spacing.
    free_radius_max, included_radius = _percolating_radius(
        tree, edges, atom_radius, probe_radius, grid_spacing)
    largest_free = 2.0 * free_radius_max
    largest_included = 2.0 * included_radius

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
    wall_centres, wall_histogram = _wall_distance_distribution(
        free_radius[inside], bins=bins)
    mean_wall = float(free_radius[inside].mean()) if inside.any() else 0.0

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
        mean_wall_distance=mean_wall,
        wall_distances=wall_centres,
        wall_distance_histogram=wall_histogram,
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


def _sphere_directions(count: int) -> np.ndarray:
    """``count`` unit vectors spread evenly over the sphere.

    The golden-spiral (Fibonacci) lattice: the ``k``-th point takes
    ``cos θ = 1 − 2(k + ½)/n``, which spaces the points equally in
    ``cos θ`` so every band of latitude gets its due area, and winds the
    azimuth by the golden angle so no two points line up. Deterministic,
    and its error falls as ``1/n``.

    **It replaced a random draw, and that was a real bug rather than a
    refinement.** The draw was taken once and reused for every atom, on
    the reasoning that one quadrature everywhere keeps the atoms
    comparable. It does — and it also makes every atom's error *the same
    error*, so nothing averages out over the structure however large it
    is. Measured on the validation bundle: three seeds at 200 points
    gave 2966, 2330 and 2693 m²/g, a 27 % spread, and at 2000 points it
    was still 2593 to 2698. The number also moved 34 % between 200 and
    500 points, which is what a convergence knob must never do. Tiling
    the cell changed it by nothing at all — the giveaway, since a
    quarter-cell and the full cell are the same material and the errors
    were identical rather than merely close.

    With this lattice the same bundle reads 2616, 2581, 2600 and 2588
    m²/g at 200, 500, 2000 and 8000 points, and independent per-atom
    draws agree at 2571-2600, so ~2590 m²/g is the answer both routes
    converge to.
    """
    index = np.arange(count, dtype=float) + 0.5
    cos_theta = 1.0 - 2.0 * index / count
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    # The golden angle, as a turn per point.
    azimuth = np.pi * (1.0 + 5.0**0.5) * index
    return np.column_stack((np.cos(azimuth) * sin_theta,
                            np.sin(azimuth) * sin_theta,
                            cos_theta))


def _accessible_area(
    positions: np.ndarray, tree: cKDTree, edges: np.ndarray,
    atom_radius: float, probe_radius: float, per_atom: int,
) -> float:
    """Probe-accessible surface area in Å², by rolling a probe.

    Points are spread over each atom's probe sphere (radius
    ``atom_radius + probe_radius``) and kept when no other atom's probe
    sphere contains them. The kept fraction times the sphere's area,
    summed, is the accessible surface — Shrake-Rupley, with periodic
    neighbours.

    **This takes no seed.** The quadrature is deterministic (see
    :func:`_sphere_directions` for the bug that made it so), which means
    ``area_points`` is an honest convergence knob: raise it and the
    number settles instead of moving to a different draw.
    """
    shell = atom_radius + probe_radius
    directions = _sphere_directions(per_atom)
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


def _free_radius_grid(
    tree: cKDTree, edges: np.ndarray, atom_radius: float, spacing: float
) -> np.ndarray:
    """The free-sphere radius on a regular grid over the cell.

    A grid rather than the random samples the rest of this module uses,
    and the reason is not speed alone. A channel is a *geometric*
    feature: whether one exists is a yes/no about the structure, and a
    random cloud answers it by luck — it under-reports a narrow channel
    it happened not to sample, and the error does not shrink predictably
    because it is a maximum over samples, not a mean. A grid's error is
    its spacing, which is a number the caller sets.
    """
    counts = np.maximum(2, np.ceil(edges / spacing).astype(int))
    axes = [np.linspace(0.0, edge, n, endpoint=False)
            for edge, n in zip(edges, counts)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    distances, _ = tree.query(mesh.reshape(-1, 3), k=1)
    return distances.reshape(counts) - atom_radius


def _mask_percolates(mask: np.ndarray) -> bool:
    """Does a connected region of ``mask`` wrap the periodic cell?

    Label the mask **non-periodically** first, which ``scipy.ndimage``
    does in C, and only then join the labels that touch across opposite
    faces, carrying the lattice offset each join crosses. A component
    that reaches itself with a nonzero accumulated offset has returned
    to its own starting cell in a different image, which is the
    definition of percolation — and is not the same as "the component
    is wide", which a spherical cavity spanning the cell also satisfies.

    The work is the labelling, which is linear and compiled; the
    union-find is over the few hundred labels and the few thousand face
    cells, not over every pair of void points. The version this replaced
    linked every pair of sampled points within twice the probe radius,
    which on an open framework is most of the cell: measured on a Schwarz
    P cell, 0.5 s at 2000 samples, 2.6 at 5000 and 19.7 at 10 000 — cubic,
    and hours at the sample count the void fraction wants.
    """
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return False

    parent = list(range(count + 1))
    offset = [np.zeros(3, dtype=int) for _ in range(count + 1)]

    def find(node: int) -> tuple[int, np.ndarray]:
        shift = np.zeros(3, dtype=int)
        while parent[node] != node:
            shift = shift + offset[node]
            node = parent[node]
        return node, shift

    for axis in range(3):
        last = np.take(labels, -1, axis=axis).ravel()
        first = np.take(labels, 0, axis=axis).ravel()
        touching = (last > 0) & (first > 0)
        if not touching.any():
            continue
        step = np.zeros(3, dtype=int)
        step[axis] = 1
        for a, b in zip(last[touching], first[touching]):
            root_a, shift_a = find(int(a))
            root_b, shift_b = find(int(b))
            through = shift_a + step - shift_b
            if root_a == root_b:
                if np.any(through != 0):
                    return True
                continue
            parent[root_b] = root_a
            offset[root_b] = through
    return False


def _percolating_radius(
    tree: cKDTree, edges: np.ndarray, atom_radius: float,
    probe_radius: float, spacing: float,
) -> tuple[float, float]:
    """``(largest free sphere, largest included sphere)`` radii, in Å.

    The free one is the largest probe whose void still **crosses** the
    cell, by bisection on :func:`_mask_percolates`. The included one is
    the largest that fits anywhere, which is the same grid's maximum and
    comes for free. They are different questions and a sealed cage
    separates them: large included, zero free.

    Both are limited by ``spacing`` rather than by luck, and both are
    **lower** bounds — a channel narrower than the grid can be missed,
    and no feature is ever invented, because every grid point's free
    radius is a real distance to a real atom.
    """
    free = _free_radius_grid(tree, edges, atom_radius, spacing)
    included = float(free.max())
    if included <= 0.0:
        return 0.0, max(included, 0.0)
    low, high = 0.0, included
    for _ in range(20):
        trial = 0.5 * (low + high)
        if _mask_percolates(free >= trial):
            low = trial
        else:
            high = trial
    return (low if low >= probe_radius * 1e-6 else 0.0), included


def _wall_distance_distribution(
    free_radius: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """How far the accessible volume sits from the nearest wall.

    The same per-point number the pore size distribution doubles, kept
    undoubled, which is what it already is: the distance from the point
    to the surface of the nearest framework atom. Doubling it names a
    pore, and naming a pore from one point is only right at the deepest
    point of it; leaving it alone names a distance, and that is right
    everywhere.

    It answers the question `largest_free_sphere` cannot. That diameter
    is a bottleneck -- the narrowest place on the widest path -- so it
    is a bound on the pore and not a description of its shape. A slit
    and a hexagonal channel with the same bottleneck differ in how much
    of their volume sits far from any wall, and far from a wall is
    where the framework does nothing and the gas is simply compressed.
    """
    if not len(free_radius):
        return np.empty(0), np.empty(0)
    histogram, borders = np.histogram(free_radius, bins=bins, density=False)
    centres = 0.5 * (borders[:-1] + borders[1:])
    total = histogram.sum()
    return centres, histogram / total if total else histogram.astype(float)


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
