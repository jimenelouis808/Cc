"""Carbon nanocoil builder.

A carbon nanocoil is a helical single-wall carbon nanotube. Two
complementary construction routes exist in the literature:

1. **Geometric coil**. A straight CNT is mapped onto a helical path. The
   local tube cross-section is preserved, so C-C bond lengths remain close
   to their equilibrium value as long as the coil radius is large compared
   to the tube radius. This is the standard starting point for DFT or MD
   relaxation.
2. **Topological coil** (Dunlap / Ihara). Intrinsic curvature is sustained
   by a regular pattern of pentagon-heptagon (5-7) defect pairs that inject
   positive / negative Gaussian curvature along the spine.

Here we implement the geometric construction, with an **optional** post-hoc
insertion of Stone-Wales 5-7-7-5 defects at a controllable density. The
resulting structure is physically meaningful (bond lengths within ~2% of
1.42 Å for typical coils) and is meant to be relaxed by the user's
external force field / DFT.

Parameters follow the experimental literature:
  * coil radius ``R`` in Å (typical: 20-200 Å experimentally)
  * pitch ``P`` in Å (vertical advance per turn)
  * number of turns ``n_turns``
  * underlying CNT chirality ``(n, m)``
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.geometry import center_in_cell, guess_bonds, minimum_image_distances
from ..utils.rng import make_rng
from ..validation.quality import sp2_quality
from .capped_cnt import geometry_report
from .centerline import ARTISTIC_STRAIN_LIMIT, helix_curvature
from .cnt import build_cnt
from .lattice_edits import ring_census

#: Gap a graphite interlayer needs, added to the tube's own diameter to
#: get the smallest pitch at which successive turns do not interpenetrate.
#: The same rule the periodic-coil builder already applies.
TURN_CLEARANCE = 3.4


def _helical_frame(
    s: np.ndarray,
    coil_radius: float,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Frenet-like (T, N, B) frame along a helix parameterised
    by arc length ``s`` (in Å).

    The helix is ``C(s) = (R cos θ, R sin θ, h θ)`` with ``θ = s / a`` and
    ``a = sqrt(R² + h²)``, where ``h = P / (2π)``.

    The returned ``N`` points radially inward (toward the helix axis), so
    positive ``x_local`` in the straight CNT maps to the inner side of the
    coil, negative to the outer side.

    Parameters
    ----------
    s
        Arc-length parameter for each atom, shape ``(N,)``.
    coil_radius, pitch
        Helix parameters in Å.

    Returns
    -------
    (C, N, B)
        Each of shape ``(N, 3)``. ``C`` is the centreline position, ``N``
        the inward normal, ``B`` the binormal.
    """
    R = float(coil_radius)
    h = float(pitch) / (2.0 * math.pi)
    a = math.sqrt(R * R + h * h)
    theta = s / a
    c, si = np.cos(theta), np.sin(theta)
    zeros = np.zeros_like(theta)
    ones = np.ones_like(theta)

    C = np.stack([R * c, R * si, h * theta], axis=-1)
    T = np.stack([-R * si, R * c, h * ones], axis=-1) / a
    N = np.stack([-c, -si, zeros], axis=-1)  # inward normal
    B = np.cross(T, N)
    # Normalise B (should already be unit, but guard against FP drift).
    B /= np.linalg.norm(B, axis=-1, keepdims=True)
    return C, N, B


def _apply_stone_wales_to_outer_bonds(
    atoms: Atoms,
    density: float,
    coil_axis: int,
    seed: int | None,
) -> Atoms:
    """Insert Stone-Wales 5-7-7-5 defects preferentially on the outer wall.

    On the outer side of a coil, the curvature favours heptagon formation
    (tensile in-plane stress is released). We mimic this preference by
    sampling bonds with probability weighted by the atom's distance from
    the coil axis.

    Parameters
    ----------
    atoms
        Geometric coil (untouched topologically).
    density
        Fraction of bonds to rotate (0-1). Values > 0.02 are unphysical and
        rejected.
    coil_axis
        Cartesian axis of the coil helix (0, 1 or 2).
    seed
        RNG seed.
    """
    if density <= 0:
        return atoms
    if density > 0.02:
        raise ValueError(
            f"Stone-Wales density {density} is unphysically large (>0.02)."
        )

    from ..defects.stone_wales import stone_wales_defect

    rng = make_rng(seed)
    positions = atoms.get_positions()
    # Radial distance from the helix axis projected on its perpendicular plane.
    perp = [i for i in (0, 1, 2) if i != coil_axis]
    center = positions[:, perp].mean(axis=0)
    radial = np.linalg.norm(positions[:, perp] - center, axis=1)
    weights = radial / radial.max()

    dmat = minimum_image_distances(atoms)
    np.fill_diagonal(dmat, np.inf)
    n = len(atoms)
    bonds: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if 0.9 < dmat[i, j] < 1.8:
                w = 0.5 * (weights[i] + weights[j])
                bonds.append((i, j, w))
    if not bonds:
        return atoms

    n_defects = max(1, int(round(density * len(bonds))))
    ps = np.array([b[2] for b in bonds])
    ps = ps / ps.sum()
    idx = rng.choice(len(bonds), size=min(n_defects, len(bonds)),
                     replace=False, p=ps)
    out = atoms
    for k in idx:
        i, j, _ = bonds[int(k)]
        try:
            out = stone_wales_defect(out, bond=(i, j))
        except Exception:
            # Skip bonds that fail (e.g. degenerate local frames).
            continue
    return out


def build_nanocoil(
    n: int = 6,
    m: int = 6,
    coil_radius: float = 25.0,
    pitch: float = 12.0,
    n_turns: float = 1.0,
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_1D,
    stone_wales_density: float = 0.0,
    seed: int | None = None,
) -> Atoms:
    """Build a carbon nanocoil by winding a straight ``(n, m)`` CNT on a helix.

    Parameters
    ----------
    n, m
        Chirality indices of the underlying CNT.
    coil_radius
        Radius of the helical spine in Å. Should be at least 3 × the CNT
        radius to keep bond-length distortion below a few percent.
    pitch
        Vertical advance per turn (Å).
    n_turns
        Number of helical turns. Fractional values are allowed.
    bond
        C-C bond length (Å).
    vacuum
        Vacuum padding (Å) applied on all three axes (the coil is treated
        as a finite object).
    stone_wales_density
        If > 0, randomly rotate this fraction of bonds with a Stone-Wales
        operation, biased toward the outer wall of the coil. Physically
        meaningful values are in ``[0, 0.02]``.
    seed
        RNG seed for the (optional) defect placement.

    Returns
    -------
    ase.Atoms
        Finite (non-periodic) nanocoil centred in the cell. Metadata under
        ``atoms.info`` records ``coil_radius``, ``pitch``, ``n_turns``,
        ``arc_length`` and the underlying ``(n, m)``.

    Raises
    ------
    ValueError
        If parameters violate the geometric assumptions (e.g.
        ``coil_radius`` comparable to the CNT radius).

    Notes
    -----
    **Bond length preservation.** A straight CNT segment of arc length
    ``L = n_turns * sqrt((2πR)² + P²)`` is bent along the helix. The
    resulting bond lengths deviate from ``bond`` by a factor of
    ``1 ± r_tube/R`` at leading order, where ``r_tube`` is the CNT radius.
    """
    if n_turns <= 0:
        raise ValueError("n_turns must be positive.")
    if coil_radius <= 0 or pitch <= 0:
        raise ValueError("coil_radius and pitch must be positive.")

    arc_length = n_turns * math.sqrt((2.0 * math.pi * coil_radius) ** 2 + pitch ** 2)
    straight = build_cnt(n=n, m=m, length=arc_length, bond=bond,
                         vacuum=0.0, axis=2)
    r_tube = float(straight.info["radius"])
    if coil_radius < 2.0 * r_tube:
        raise ValueError(
            f"coil_radius={coil_radius} Å is too small for a CNT of radius "
            f"{r_tube:.2f} Å (need at least 2×)."
        )

    positions = straight.get_positions()
    # Only keep atoms within one arc length (CNT cell may be slightly longer).
    mask = (positions[:, 2] >= 0.0) & (positions[:, 2] <= arc_length)
    positions = positions[mask]

    # Local cross-section: subtract the tube's transverse centre.
    transverse_centre = positions[:, :2].mean(axis=0)
    x_local = positions[:, 0] - transverse_centre[0]
    y_local = positions[:, 1] - transverse_centre[1]
    s = positions[:, 2]

    C, N, B = _helical_frame(s, coil_radius, pitch)
    coil_pos = C + x_local[:, None] * N + y_local[:, None] * B

    # Bounding box + vacuum → cubic-ish cell.
    extents = coil_pos.max(axis=0) - coil_pos.min(axis=0)
    cell = np.diag(extents + vacuum)
    atoms = Atoms(symbols=["C"] * len(coil_pos), positions=coil_pos,
                  cell=cell, pbc=False)
    center_in_cell(atoms, axes=(0, 1, 2))

    atoms.info.update(
        {
            "structure_type": "nanocoil",
            "n": n,
            "m": m,
            "coil_radius": coil_radius,
            "pitch": pitch,
            "n_turns": n_turns,
            "arc_length": arc_length,
            "tube_radius": r_tube,
            "bond": bond,
            "stone_wales_density": stone_wales_density,
            "seed": seed,
        }
    )

    if stone_wales_density > 0:
        atoms = _apply_stone_wales_to_outer_bonds(
            atoms, density=stone_wales_density, coil_axis=2, seed=seed
        )

    # --- what the sweep actually produced, measured rather than assumed.
    #
    # This builder bends a finished all-hexagon lattice, so it can only
    # stretch: nothing here relieves curvature the way a meshed wall does
    # with 5-7 pairs. Two separate things can therefore ruin it, and
    # before this they both did so silently.
    #
    # Wall strain, r_tube * kappa, is the obvious one -- a (6,6) tube on
    # an 18 Å coil is strained 22% and comes back with 1.17 Å bonds.
    #
    # Turns colliding is the one that is easy to miss, because it has
    # nothing to do with the coil radius: a (10,10) tube is 13.6 Å across,
    # so on a 13 Å pitch successive turns interpenetrate at *every* radius.
    # Measured at R = 120 Å, where the strain is a comfortable 5.6%: 5211
    # overlapping pairs and 0.69 Å "bonds". Widening the coil cannot fix
    # it; only widening the pitch can.
    bonds = [(i, j) for i, j, _ in guess_bonds(atoms)]
    quality = geometry_report(atoms.get_positions(), bonds)
    strain = r_tube * helix_curvature(coil_radius, pitch)
    clearance = 2.0 * r_tube + TURN_CLEARANCE
    counts = ring_census(atoms.get_positions(), bonds)
    atoms.info.update({
        "bonds": [[int(i), int(j)] for i, j in bonds],
        "ring_counts": counts,
        "geometry": quality,
        "wall_strain": float(strain),
        "min_pitch": float(clearance),
        # A rolled lattice, defects aside: its rings are hexagons, so the
        # pristine sp2 window is the right one to judge it against.
        "quality_family": "sp2" if set(counts) <= {6} else "haeckelite",
    })

    verdict, why = sp2_quality(quality, atoms.info["quality_family"])
    if verdict == "broken":
        cause = (
            f"the pitch is {pitch:.1f} Å but this tube is {2 * r_tube:.1f} Å "
            f"across, so successive turns pass through each other; raise the "
            f"pitch above {clearance:.1f} Å"
            if pitch < clearance else
            f"the wall is strained {strain:.0%} by a coil radius of "
            f"{coil_radius:.0f} Å; this route bends a finished hexagonal "
            f"lattice, so widen the coil (clean from about "
            f"{_clean_coil_radius(r_tube, pitch):.0f} Å here) or use a "
            f"thinner tube"
        )
        raise ValueError(
            f"This sweep tears the wall rather than winding it: {why} "
            f"The cause is that {cause}."
        )
    if pitch < clearance:
        warnings.warn(
            f"Pitch {pitch:.1f} Å leaves less than a graphite interlayer gap "
            f"between turns of a {2 * r_tube:.1f} Å tube (needs "
            f"{clearance:.1f} Å). The geometry survived, but the turns are "
            "closer than any real coil's.",
            UserWarning, stacklevel=2,
        )
    elif strain > ARTISTIC_STRAIN_LIMIT:
        warnings.warn(
            f"A coil radius of {coil_radius:.0f} Å strains this tube's outer "
            f"wall by {strain:.0%}, past the {ARTISTIC_STRAIN_LIMIT:.0%} sp2 "
            "limit. Real carbon nanocoils have coil radii of hundreds of Å "
            "for exactly this reason.",
            UserWarning, stacklevel=2,
        )
    return atoms


def _clean_coil_radius(tube_radius: float, pitch: float,
                       max_strain: float = 0.08) -> float:
    """Smallest coil radius whose wall strain stays inside ``max_strain``.

    Inverting ``strain = r * R / (R^2 + c^2)`` for R gives a quadratic
    with two roots; the larger is the one that means "wide enough", the
    smaller a coil tighter than the tube itself. Returns the larger.

    This is the number that makes "I want a small, clean coil" an
    impossible request rather than a matter of tuning: at a 13 Å pitch a
    (6,6) tube needs about 50 Å of coil radius, which is 6000 atoms of
    tube, and a fatter tube needs more.
    """
    c = pitch / (2.0 * math.pi)
    ratio = tube_radius / max_strain
    discriminant = ratio * ratio - 4.0 * c * c
    if discriminant <= 0.0:
        return float("inf")
    return 0.5 * (ratio + math.sqrt(discriminant))
