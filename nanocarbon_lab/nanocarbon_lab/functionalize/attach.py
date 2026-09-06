"""Grafting groups onto a surface: where it points, and what fits.

The group library says what a hydroxyl *is*; this says where it goes.
Three questions, and each was a wrong answer first.

**Which way is out.** A group must point away from the material, and the
outward direction is not a free choice -- put it inward and the group is
inside the tube, overlapping the far wall. For an atom with three
neighbours the bond plane fixes the normal only up to sign, and the sign
cannot be taken per atom: deciding each one against its own local
centroid gives neighbouring normals 178 degrees apart on a saddle, which
is the failure `tmd/curved.py` already records. So the field is made
**continuous first** by propagating signs across the bond graph, and
flipped **once per connected component** afterwards. The global flip uses
the divergence theorem -- for an outward field on a closed surface,
`sum(n_i . (r_i - centre))` is three times the enclosed volume, so its
sign is unambiguous and its magnitude says whether the structure encloses
anything at all. A flat sheet gives nearly zero, and that is exactly the
case where "outer" is meaningless and the caller must pick a face.

**Which atoms may be functionalised.** On carbon, any wall atom; on an
MX2 sandwich, **only the chalcogen**. The metal is buried between two
chalcogen planes and nothing can reach it -- a group grafted onto it
would be threaded through the surface. This is enforced, not suggested.

**Whether the group fits.** Coverage is requested and *achieved*, and the
two differ. Each placement is tested against every atom already present,
including groups grafted earlier in the same call; a site whose group
would overlap is skipped and counted. That is why -F reaches near-total
coverage on graphene and -COOH does not: the carboxyl is simply too big,
and the number of sites it fits into is a measurement rather than an
input.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from ase import Atoms

from ..utils.constants import COVALENT_RADII
from ..utils.geometry import guess_bonds
from .groups import (
    FunctionalGroup,
    build_bridging_positions,
    build_positions,
    get_group,
)

#: Chalcogens, so an MX2 surface can be told from its buried metal
#: without enumerating every metal -- the same role-based test
#: ``tmd/quality.py`` uses.
CHALCOGENS = frozenset({"S", "Se", "Te"})

#: Closest a grafted atom may come to an atom it is not bonded to.
#: 2.0 Å is below any van der Waals contact and above every real bond,
#: so it rejects a genuine overlap without rejecting the group's own
#: 1-3 neighbours on the surface.
MIN_CONTACT: float = 2.0

#: Fraction of the covalent-radii sum below which two atoms are treated
#: as overlapping regardless of ``MIN_CONTACT``. Needed for the heavy
#: chalcogens, where 2.0 Å is *shorter* than a legitimate Te-Te contact.
CONTACT_FRACTION: float = 0.75

#: Rotations about the surface bond tried per site, keeping the one with
#: the most clearance. A group on one single bond rotates freely, so this
#: is a real degree of freedom; 12 steps is 30 degrees apart, finer than
#: the difference between a rotamer that fits and one that does not.
ROTAMERS: int = 12

#: How far the unit bond vectors at a site must fail to cancel before the
#: site counts as pyramidal and its outward direction is taken as the
#: direction they lean away from.
#:
#: Graphene gives 0 (three coplanar bonds at 120 degrees cancel exactly);
#: an MX2 chalcogen above its three metals gives about 1.6; the tightest
#: nanotube, where the two circumferential bonds tilt inward, gives 0.26.
#: The threshold sits above the tube and below the sandwich, and nothing
#: delicate rides on where exactly: in the overlap region both branches
#: return the same direction, verified against a (6,6) tube where the
#: plane normal is 0.996 radial.
PYRAMIDAL_SUM: float = 0.30


def _mic_vectors(atoms: Atoms, pairs: np.ndarray) -> np.ndarray:
    """Minimum-image displacement ``r_j - r_i`` for each pair.

    Done in fractional coordinates so a bond across a periodic seam is a
    bond and not a cell-length stretch -- the same requirement every
    geometric step in the schwarzite route carries.
    """
    positions = atoms.get_positions()
    delta = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    if not atoms.cell.rank or not any(atoms.pbc):
        return delta
    cell = np.asarray(atoms.cell)
    inverse = np.linalg.pinv(cell)
    fractional = delta @ inverse
    for axis in range(3):
        if atoms.pbc[axis]:
            fractional[:, axis] -= np.round(fractional[:, axis])
    return fractional @ cell


def bond_pairs(atoms: Atoms, tolerance: float = 0.30) -> np.ndarray:
    """The bond list as an ``(n_bonds, 2)`` integer array.

    Prefers ``atoms.info["bonds"]`` when the builder recorded one, for
    the reason :func:`~nanocarbon_lab.topology.graph.coordination_numbers`
    does: on a curved shell a distance cutoff sweeps up neighbours that
    are close but not bonded, and a mesh-derived graph is exact.
    """
    recorded = atoms.info.get("bonds")
    if recorded is not None and len(recorded):
        pairs = np.asarray(recorded, dtype=int)
        if pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.max() < len(atoms):
            return pairs
    guessed = guess_bonds(atoms, tolerance=tolerance)
    if not guessed:
        return np.zeros((0, 2), dtype=int)
    return np.asarray([(i, j) for i, j, _ in guessed], dtype=int)


def surface_normals(atoms: Atoms,
                    face: str = "outer",
                    tolerance: float = 0.30) -> np.ndarray:
    """Outward unit normal at every atom.

    Parameters
    ----------
    atoms
        Structure to analyse.
    face
        ``"outer"`` or ``"inner"``. For a tube, cage or network these
        mean what they say. For a **sheet** the two faces are physically
        equivalent and the labels are only a deterministic way to name
        them; :func:`is_enclosing` reports which case a structure is.
    tolerance
        Bond-detection slack, used only when the structure records no
        bond graph of its own.

    Returns
    -------
    numpy.ndarray, shape ``(n_atoms, 3)``
        Unit vectors. An atom with no bonds gets a zero row rather than
        an invented direction; :func:`candidate_sites` excludes those.

    Notes
    -----
    The per-atom normal comes from its bonded neighbours: the plane
    normal for three or more, the outward bisector for two (an edge atom,
    whose group points along the sheet, not across it), and the reversed
    bond for one. Only the three-neighbour case is sign-ambiguous, and
    that sign is resolved globally rather than per atom.
    """
    if face not in ("outer", "inner"):
        raise ValueError(
            f"face must be 'outer' or 'inner', not {face!r}. They are the two "
            "sides of the surface; for a flat sheet the choice is arbitrary "
            "but reproducible, and functionalize(face='both') uses each once."
        )

    n = len(atoms)
    normals = np.zeros((n, 3), dtype=float)
    if n == 0:
        return normals

    pairs = bond_pairs(atoms, tolerance=tolerance)
    neighbours: list[list[int]] = [[] for _ in range(n)]
    offsets: list[list[np.ndarray]] = [[] for _ in range(n)]
    if len(pairs):
        vectors = _mic_vectors(atoms, pairs)
        for (i, j), vector in zip(pairs, vectors, strict=True):
            neighbours[i].append(int(j))
            offsets[i].append(vector)
            neighbours[j].append(int(i))
            offsets[j].append(-vector)

    ambiguous = np.zeros(n, dtype=bool)
    for index in range(n):
        local = offsets[index]
        if not local:
            continue
        stack = np.asarray(local, dtype=float)
        units = stack / np.linalg.norm(stack, axis=1, keepdims=True)
        away = -units.sum(axis=0)
        norm = float(np.linalg.norm(away))

        if norm >= PYRAMIDAL_SUM:
            # The bonds lean to one side, so the direction they lean away
            # from *is* the outward one, with no sign to resolve. This is
            # the only correct rule for a pyramidal site: an MX2 sulphur
            # sits above three metals whose bond vectors are **not**
            # coplanar, so the plane-normal branch has no plane to find
            # and returns whichever direction the three happen to vary
            # least along. That refused every group on every MoS2 surface.
            normals[index] = away / norm
        elif len(stack) >= 3:
            # A planar site: the bonds cancel, so there is no leaning to
            # measure and the normal is the direction they span least.
            # Sign is not decided here -- see the propagation below.
            _, _, right = np.linalg.svd(stack, full_matrices=True)
            normals[index] = right[-1]
            ambiguous[index] = True
        else:
            # One or two bonds that cancel: collinear. Any perpendicular
            # will do, and the propagation makes the choice consistent.
            helper = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(helper, units[0])) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
            perpendicular = np.cross(units[0], helper)
            normals[index] = perpendicular / np.linalg.norm(perpendicular)
            ambiguous[index] = True

    _propagate_signs(normals, neighbours, ambiguous)
    _orient_outward(atoms, normals, neighbours, face)
    return normals


def _propagate_signs(normals: np.ndarray,
                     neighbours: list[list[int]],
                     ambiguous: np.ndarray) -> None:
    """Make the sign-ambiguous normals agree with their neighbours.

    A breadth-first walk of the bond graph, flipping each normal to have
    a positive dot product with the one it was reached from. This is what
    turns a set of independent plane normals into an *orientation*; doing
    it per atom against a local centroid is the mistake that gave normals
    turning 178 degrees on a saddle.
    """
    n = len(normals)
    seen = np.zeros(n, dtype=bool)
    for start in range(n):
        if seen[start] or not np.any(normals[start]):
            continue
        seen[start] = True
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for other in neighbours[current]:
                if seen[other] or not np.any(normals[other]):
                    continue
                seen[other] = True
                if ambiguous[other] and np.dot(normals[current],
                                               normals[other]) < 0.0:
                    normals[other] *= -1.0
                queue.append(other)


def _components(neighbours: list[list[int]], n: int) -> list[list[int]]:
    """Connected components of the bond graph, as index lists."""
    seen = np.zeros(n, dtype=bool)
    groups: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        members = [start]
        while queue:
            current = queue.popleft()
            for other in neighbours[current]:
                if not seen[other]:
                    seen[other] = True
                    members.append(other)
                    queue.append(other)
        groups.append(members)
    return groups


def enclosure(atoms: Atoms, normals: np.ndarray,
              members: list[int]) -> float:
    """Mean outward projection of a component's normal field, in Å.

    ``sum(n_i . (r_i - centre)) / count``. By the divergence theorem this
    is positive for an outward field on a closed surface and scales with
    its radius; a flat sheet gives about zero, because half its atoms
    project one way and half the other only if the field is *not*
    oriented -- once oriented, every atom projects the same tiny amount
    and the sum stays near zero because the centre lies in the plane.

    That is what separates "this structure has an inside" from "this
    structure has two equivalent faces", and the caller needs to know
    which, because only the second one makes ``face`` a free choice.
    """
    positions = atoms.get_positions()[members]
    centre = positions.mean(axis=0)
    projections = np.einsum("ij,ij->i", normals[members], positions - centre)
    return float(projections.mean())


#: Mean outward projection above which a component is treated as
#: enclosing a volume. A tube of radius 3.4 Å -- the smallest that exists
#: -- gives 3.4; a sheet gives under 0.1. One Ångström sits between them
#: with a wide margin at both ends.
ENCLOSURE_THRESHOLD: float = 1.0


def is_enclosing(atoms: Atoms, tolerance: float = 0.30) -> bool:
    """Whether the structure has a well-defined inside.

    ``True`` for tubes, cages, networks and schwarzites; ``False`` for
    sheets and ribbons, where ``face="outer"`` names a side rather than
    describing one.
    """
    normals = surface_normals(atoms, tolerance=tolerance)
    pairs = bond_pairs(atoms, tolerance=tolerance)
    neighbours: list[list[int]] = [[] for _ in range(len(atoms))]
    for i, j in pairs:
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))
    for members in _components(neighbours, len(atoms)):
        if len(members) < 4:
            continue
        if abs(enclosure(atoms, normals, members)) >= ENCLOSURE_THRESHOLD:
            return True
    return False


def _orient_outward(atoms: Atoms, normals: np.ndarray,
                    neighbours: list[list[int]], face: str) -> None:
    """Flip each component once, so its continuous field points outward.

    Per component, not globally: a bundle of tubes and a multi-wall tube
    are several closed surfaces, and each has its own outside. Flipping
    them together would put half the groups inside their own tube.
    """
    sign = 1.0 if face == "outer" else -1.0
    for members in _components(neighbours, len(atoms)):
        if not members:
            continue
        projection = enclosure(atoms, normals, members)
        if abs(projection) >= ENCLOSURE_THRESHOLD:
            if projection < 0.0:
                normals[members] *= -1.0
        else:
            # A sheet: no inside, so "outer" cannot be derived. Fix it
            # deterministically instead -- align the mean normal with
            # whichever Cartesian axis it lies closest to, which for a
            # sheet in the xy plane is +z, the answer anyone would
            # expect. Reproducible across runs is the property that
            # matters; which face is called "outer" is not.
            mean = normals[members].mean(axis=0)
            if np.linalg.norm(mean) > 1e-8:
                axis = int(np.argmax(np.abs(mean)))
                if mean[axis] < 0.0:
                    normals[members] *= -1.0
        normals[members] *= sign


def candidate_sites(atoms: Atoms,
                    where: str = "all",
                    tolerance: float = 0.30) -> list[int]:
    """Atom indices a group may be grafted onto.

    Parameters
    ----------
    atoms
        Structure to inspect.
    where
        ``"all"``      every bonded surface atom of the graftable species;
        ``"edge"``     only under-coordinated atoms -- an open tube end, a
                       ribbon edge, a vacancy rim. This is where oxidation
                       actually happens, so it is not a niche option;
        ``"defect"``   atoms in a non-hexagonal ring, read from
                       ``info["rings"]``. Raises if the structure did not
                       record its rings, rather than re-deriving them from
                       distances -- the failure
                       :mod:`~nanocarbon_lab.builders.fullerene_mesh` exists
                       to prevent;
        ``"ring:N"``   atoms in a ring of exactly ``N`` members.
    tolerance
        Bond-detection slack when no bond graph was recorded.

    Returns
    -------
    list of int
        Sorted atom indices.

    Notes
    -----
    On a dichalcogenide only chalcogens are returned. The metal sits
    between the two chalcogen planes with no line of sight to the
    outside, so a group on it would pass through the surface.
    """
    symbols = atoms.get_chemical_symbols()
    pairs = bond_pairs(atoms, tolerance=tolerance)
    degree = np.zeros(len(atoms), dtype=int)
    for i, j in pairs:
        degree[int(i)] += 1
        degree[int(j)] += 1

    has_chalcogen = any(s in CHALCOGENS for s in symbols)
    metals = {s for s in symbols if s not in CHALCOGENS and s != "C"}
    is_mx2 = has_chalcogen and bool(metals)

    def graftable(index: int) -> bool:
        if degree[index] == 0:
            return False
        if is_mx2:
            return symbols[index] in CHALCOGENS
        return True

    if where == "all":
        chosen = [i for i in range(len(atoms)) if graftable(i)]
    elif where == "edge":
        # "Under-coordinated" against the bulk value for the species:
        # 3 for sp2 carbon, 3 for a chalcogen in an MX2 sandwich.
        bulk = 3
        chosen = [i for i in range(len(atoms))
                  if graftable(i) and degree[i] < bulk]
    elif where == "defect" or where.startswith("ring:"):
        rings = atoms.info.get("rings")
        if not rings:
            raise ValueError(
                f"where={where!r} needs the recorded ring metadata, and this "
                "structure has none. Every mesh-based builder records "
                "info['rings']; re-deriving rings from distances on a curved "
                "or periodic shell produces silently wrong counts, so there "
                "is deliberately no fallback. Use where='all' or 'edge'."
            )
        if where == "defect":
            wanted = [ring for ring in rings if len(ring) != 6]
        else:
            size = int(where.split(":", 1)[1])
            wanted = [ring for ring in rings if len(ring) == size]
        members = {int(i) for ring in wanted for i in ring}
        chosen = [i for i in sorted(members) if graftable(i)]
    else:
        raise ValueError(
            f"Unknown site selection {where!r}. Use 'all', 'edge', 'defect' "
            "or 'ring:N'."
        )
    return chosen


def _contact_floor(first: str, second: str) -> float:
    """Closest two non-bonded atoms of these elements may come."""
    radii = COVALENT_RADII.get(first, 0.8) + COVALENT_RADII.get(second, 0.8)
    return max(MIN_CONTACT, CONTACT_FRACTION * radii)


#: How far inward a bonded neighbour must sit before a site's inner face
#: counts as blocked. An MX2 chalcogen has its three metals 1.56 Å below
#: it and genuinely has only one exposed side; a fullerene's carbons lean
#: inward by 0.29 Å and its cavity is real; a nanotube's by 0.18 Å.
INNER_BLOCKED_DEPTH: float = 0.5


def inner_face_blocked(atoms: Atoms, normals: np.ndarray,
                       tolerance: float = 0.30) -> np.ndarray:
    """Per atom, whether its inward side is occupied by its own material.

    A single atomic layer -- graphene -- has two free faces, and that is
    what makes the chair conformation of fluorographene possible. A
    three-plane MX2 sandwich does not: under every surface chalcogen sit
    the metals it is bonded to and then the far chalcogen plane, so a
    group aimed inward is threaded through the slab.

    This is not a detail. ``face="both"`` alternates by sublattice, and
    an MX2 bond graph is bipartite **between the metal and the
    chalcogen** -- so every chalcogen takes the same colour, and
    alternating sent every single group into the sandwich. Coverage came
    out at exactly zero with the refusals blamed on sterics, which was
    true and useless.

    Measured by the bonded neighbours, which are already minimum-image
    corrected: a neighbour more than :data:`INNER_BLOCKED_DEPTH` below
    the site along its own normal blocks that side.
    """
    pairs = bond_pairs(atoms, tolerance=tolerance)
    blocked = np.zeros(len(atoms), dtype=bool)
    if not len(pairs):
        return blocked
    vectors = _mic_vectors(atoms, pairs)
    for (i, j), vector in zip(pairs, vectors, strict=True):
        if np.dot(vector, normals[i]) < -INNER_BLOCKED_DEPTH:
            blocked[int(i)] = True
        if np.dot(-vector, normals[j]) < -INNER_BLOCKED_DEPTH:
            blocked[int(j)] = True
    return blocked


def preserve_vacuum(original: Atoms, grown: Atoms) -> None:
    """Re-pad ``grown``'s non-periodic axes to the vacuum ``original`` had.

    Grafting makes a structure wider, and a finite builder's cell is a
    bounding box with the requested vacuum already inside it. Leave it
    alone and the groups stick out past the box: a (6,6) tube built with
    12 Å of vacuum came back with 8.55 Å once hydroxylated, and
    ``run_basic_checks`` refused it -- correctly, since a group closer to
    the cell edge than the vacuum requirement really is interacting with
    its own image.

    Only non-periodic axes are touched. A periodic axis is the physics,
    and stretching it to fit a substituent would change the crystal
    rather than the box -- the rule ``cell.to_unit_cell`` already keeps.
    """
    cell = np.array(grown.cell, dtype=float)
    before = original.get_positions()
    after = grown.get_positions()
    for axis in range(3):
        if grown.pbc[axis] or np.linalg.norm(cell[axis]) < 1e-8:
            continue
        length = float(np.linalg.norm(cell[axis]))
        span = float(before[:, axis].max() - before[:, axis].min())
        vacuum = max(0.0, length - span)
        new_span = float(after[:, axis].max() - after[:, axis].min())
        cell[axis] = cell[axis] / length * (new_span + vacuum)
    # The cell grows; the atoms do not move. Re-centring here would
    # translate every pre-existing atom, so a caller holding coordinates
    # from before the graft -- or comparing against them -- would find
    # them silently shifted. `check_vacuum` measures the cell length
    # against the atomic span and does not care where in the box that
    # span sits, so there is nothing to gain by moving anything.
    grown.set_cell(cell)


def _pick_side(face: str, parity: int, blocked: bool) -> float:
    """Which side of the surface a group at this site goes on.

    ``+1`` is along the outward normal. A site whose inner face is
    blocked always takes the outer one whatever was asked: there is no
    inner face to put anything on, and pretending otherwise buries the
    group in the material.
    """
    if blocked:
        return 1.0
    if face == "both":
        return 1.0 if parity == 0 else -1.0
    return 1.0 if face == "outer" else -1.0


def sublattice_parity(atoms: Atoms, tolerance: float = 0.30) -> np.ndarray:
    """A two-colouring of the bond graph, as 0/1 per atom.

    On a honeycomb this is exactly the A/B sublattice, and it is what
    makes ``face="both"`` reach full coverage instead of a third of it.
    Fluorographene is the **chair** conformation: every carbon carries an
    F, alternating strictly from one carbon to the next. Alternate by
    anything else -- a shuffled index, say -- and adjacent carbons take
    the same face perhaps half the time, putting their two fluorines
    1.42 Å apart, which is a bond length rather than a contact. Measured:
    alternating by shuffle order reached 42% coverage, by sublattice 100%.

    A honeycomb is bipartite, so the colouring is exact. Any structure
    with odd rings -- a fullerene, a schwarzite's pentagons -- is not, and
    the walk then leaves frustrated bonds whose two ends share a colour.
    That is a property of the surface, not an error: those sites simply
    cannot all alternate, and the steric test refuses the ones that
    collide.
    """
    n = len(atoms)
    parity = np.zeros(n, dtype=int)
    seen = np.zeros(n, dtype=bool)
    neighbours: list[list[int]] = [[] for _ in range(n)]
    for first, second in bond_pairs(atoms, tolerance=tolerance):
        neighbours[int(first)].append(int(second))
        neighbours[int(second)].append(int(first))
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for other in neighbours[current]:
                if not seen[other]:
                    seen[other] = True
                    parity[other] = 1 - parity[current]
                    queue.append(other)
    return parity


def _terminal_atoms(group: FunctionalGroup, root: int) -> list[int]:
    """Group atoms that are complete with a single neighbour.

    A carboxyl's carbonyl oxygen has one neighbour and a bond order of
    two, so it is finished. Coordination counting cannot tell that from
    an atom that lost a partner -- the count is 1 either way -- so
    without recording it, validation warned "dangling" about every
    carboxyl, carbonyl, nitro and aldehyde this package can graft.
    Monovalent atoms need no entry: their element's own ceiling of 1
    already says they are complete.
    """
    has_child = {atom.parent for atom in group.atoms if atom.parent >= 0}
    # The root counts too. A carbonyl is one doubly bonded oxygen with no
    # parent at all, so requiring a parent here left the single group
    # whose *whole point* is a terminal double bond still reported as
    # dangling.
    return [root + index for index, atom in enumerate(group.atoms)
            if index not in has_child and atom.order > 1.0]


def _bonded_exemptions(group: FunctionalGroup, anchor: int,
                       neighbours_of: list[list[int]]) -> list[set[int]]:
    """Per group atom, the surface atoms it is within two bonds of.

    Once the group's root bonds to the anchor, a group atom at depth
    ``dg`` from the root and a surface atom at depth ``ds`` from the
    anchor are ``dg + 1 + ds`` bonds apart. Two bonds or fewer means the
    separation is set by a bond length and a bond angle, so it is not a
    steric contact and must not be scored as one -- that is ``dg + ds <=
    1``, which is the root against the anchor and its neighbours, plus
    the root's own children against the anchor.

    Three bonds out is a torsion. Those are left in: they are real, and
    the rotamer search is what resolves them.
    """
    depth = [0] * group.n_atoms
    for index, atom in enumerate(group.atoms):
        if atom.parent >= 0:
            depth[index] = depth[atom.parent] + 1
    near_anchor = {anchor}
    first_shell = near_anchor | {int(i) for i in neighbours_of[anchor]}
    return [first_shell if d == 0 else near_anchor if d == 1 else set()
            for d in depth]


class _ClashChecker:
    """Asks whether a candidate atom overlaps anything already placed.

    Two reasons this is a class rather than a loop over ``atoms``.

    **It must be periodic.** A group grafted near a schwarzite's cell
    seam has its own image on the other side, and a plain distance to the
    stored coordinates never sees it -- the cell would come back looking
    fine and be broken only once a neighbouring image was drawn. Each
    query point is therefore tested at every relevant lattice
    translation, along the periodic axes only.

    **It must not be linear in the structure.** The wall does not change
    during a run, so it goes into a KD-tree once; only the handful of
    atoms grafted so far are scanned directly. Without that, grafting
    onto a 10 000-atom network is quadratic in exactly the way
    ``guess_bonds`` already had to be rescued from.
    """

    def __init__(self, atoms: Atoms, radius: float) -> None:
        from scipy.spatial import cKDTree

        self._symbols = atoms.get_chemical_symbols()
        self._tree = cKDTree(atoms.get_positions())
        self._radius = radius
        self._grafted = np.zeros((0, 3), dtype=float)
        self._grafted_symbols: list[str] = []
        self._translations = self._lattice_translations(atoms, radius)

    @staticmethod
    def _lattice_translations(atoms: Atoms, radius: float) -> np.ndarray:
        """Lattice vectors an overlapping image could sit at.

        Only the periodic axes contribute, and only the neighbouring
        image on each: a cell narrower than the contact radius would be
        unphysical for any structure this deals with.
        """
        cell = np.asarray(atoms.cell)
        steps = []
        for axis in range(3):
            if atoms.pbc[axis] and np.linalg.norm(cell[axis]) > 1e-8:
                steps.append((-1, 0, 1))
            else:
                steps.append((0,))
        vectors = [
            i * cell[0] + j * cell[1] + k * cell[2]
            for i in steps[0] for j in steps[1] for k in steps[2]
        ]
        return np.asarray(vectors, dtype=float)

    def add(self, positions: np.ndarray, symbols: list[str]) -> None:
        """Record a group that has been accepted, so later ones see it."""
        self._grafted = np.vstack([self._grafted, positions])
        self._grafted_symbols.extend(symbols)

    def margin(self, positions: np.ndarray, symbols: list[str],
               exempt: list[set[int]]) -> float:
        """Smallest ``distance - floor`` over every pair, in Å.

        Negative means an overlap. Returning the margin rather than a
        yes/no is what lets the caller compare rotamers: a group joined
        by one single bond spins freely about it, and which way it
        happens to face should not decide whether it fits.

        ``exempt[k]`` holds the surface atoms group atom ``k`` is within
        two bonds of once the new bond exists. Those separations are
        fixed by bond lengths and angles, so measuring them as clearances
        is a category error -- and it refused **every** placement on every
        structure until it was handled. A hydroxyl's oxygen is 2.09 Å
        from the three carbons around its anchor and its hydrogen 1.96 Å
        from the anchor itself; both are simply what a 108.5 degree
        C-O-H angle puts there. Anything three bonds out or further is a
        real contact and is measured.
        """
        smallest = np.inf
        for offset, point in enumerate(positions):
            spared = exempt[offset]
            for image in self._translations:
                shifted = point + image
                for index in self._tree.query_ball_point(shifted, self._radius):
                    # Exempt in every image, not only the home one. A site
                    # at a cell face has its bonded neighbour stored on
                    # the far side, so the copy that is actually 1.42 Å
                    # away is an *image* -- restricting the exemption to
                    # the home cell refused every group along every seam,
                    # which on a 6x6 graphene cell was a third of them.
                    # Safe because the exempt atoms are within two bonds
                    # and every cell here is far wider than that.
                    if index in spared:
                        continue
                    gap = float(np.linalg.norm(self._tree.data[index] - shifted))
                    gap -= _contact_floor(symbols[offset], self._symbols[index])
                    smallest = min(smallest, gap)
                if len(self._grafted):
                    distances = np.linalg.norm(self._grafted - shifted, axis=1)
                    for index in np.flatnonzero(distances < self._radius):
                        gap = float(distances[index]) - _contact_floor(
                            symbols[offset], self._grafted_symbols[index])
                        smallest = min(smallest, gap)
        return smallest


def functionalize(atoms: Atoms,
                  group: FunctionalGroup | str,
                  coverage: float | None = None,
                  count: int | None = None,
                  where: str = "all",
                  face: str = "outer",
                  seed: int | None = None,
                  min_separation: float = 0.0,
                  tolerance: float = 0.30) -> Atoms:
    """Graft a functional group onto a structure's surface.

    Parameters
    ----------
    atoms
        Structure to functionalise. Not modified; a copy is returned.
    group
        A :class:`~nanocarbon_lab.functionalize.groups.FunctionalGroup`,
        or the name of a shipped one. Build a variant with
        :func:`~nanocarbon_lab.functionalize.groups.substitute` and pass
        the result here -- the geometry follows the substitution.
    coverage
        Fraction of the candidate sites to graft, in ``(0, 1]``. Mutually
        exclusive with ``count``.
    count
        Exact number of groups to graft instead of a fraction.
    where
        Site selection, see :func:`candidate_sites`.
    face
        ``"outer"``, ``"inner"`` or ``"both"``. ``"both"`` alternates by
        **sublattice** (see :func:`sublattice_parity`), which is the
        chair conformation real fluorographene and graphene oxide adopt,
        and the only way full coverage is sterically reachable at all.
    seed
        Seeds the site shuffle. Required for reproducibility; ``None``
        gives a fresh pattern each call.
    min_separation
        Minimum distance (Å) between two grafted **sites**. ``0`` leaves
        the steric test to decide, which is usually right: a fluorine
        reaches near-total coverage and a carboxyl does not, and that
        difference is the chemistry rather than a rule.
    tolerance
        Bond-detection slack when no bond graph was recorded.

    Returns
    -------
    ase.Atoms
        A copy with the groups appended after the original atoms, so
        every existing index still refers to the same atom.
        ``info["bonds"]`` is extended with the new bonds when the
        structure recorded one, and ``info["functionalization"]`` records
        what was grafted and what was refused.

    Raises
    ------
    ValueError
        If neither or both of ``coverage`` and ``count`` are given, if the
        selection yields no sites, or if a bridging group is requested --
        those span two atoms and are not placed from a single anchor.

    Notes
    -----
    **Achieved coverage is measured, not assumed.** A site whose group
    would come within :data:`MIN_CONTACT` of any atom -- including a
    group placed earlier in the same call -- is skipped, and the count of
    those is reported. Asking for full coverage with a carboxyl therefore
    returns the density that fits, not an interpenetrating structure that
    only a relaxation would reveal as broken.
    """
    if isinstance(group, str):
        group = get_group(group)
    if (coverage is None) == (count is None):
        raise ValueError(
            "Give exactly one of coverage (a fraction of the candidate sites) "
            "or count (a number of groups)."
        )
    if coverage is not None and not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], not {coverage}.")
    if face not in ("outer", "inner", "both"):
        raise ValueError(
            f"face must be 'outer', 'inner' or 'both', not {face!r}."
        )

    if group.bridging:
        return _graft_bridging(atoms, group, coverage=coverage, count=count,
                               where=where, face=face, seed=seed,
                               tolerance=tolerance)

    sites = candidate_sites(atoms, where=where, tolerance=tolerance)
    if not sites:
        raise ValueError(
            f"No graftable sites for where={where!r}. On a dichalcogenide only "
            "the chalcogens are graftable, since the metal is buried between "
            "them; 'edge' needs under-coordinated atoms and a closed structure "
            "has none."
        )

    requested = count if count is not None else int(round(coverage * len(sites)))
    requested = max(0, min(requested, len(sites)))

    outer = surface_normals(atoms, face="outer", tolerance=tolerance)
    neighbours_of: list[list[int]] = [[] for _ in range(len(atoms))]
    for first, second in bond_pairs(atoms, tolerance=tolerance):
        neighbours_of[int(first)].append(int(second))
        neighbours_of[int(second)].append(int(first))
    parity = (sublattice_parity(atoms, tolerance=tolerance)
              if face == "both" else np.zeros(len(atoms), dtype=int))
    blocked = inner_face_blocked(atoms, outer, tolerance=tolerance)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(sites)))

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    existing = len(atoms)
    new_symbols: list[str] = []
    new_positions: list[np.ndarray] = []
    new_bonds: list[tuple[int, int]] = []
    grafted: list[int] = []
    faces: list[str] = []
    terminal: list[int] = []
    refused_steric = 0
    refused_separation = 0

    # The largest floor any pair in this group could have, so the tree
    # query radius is generous enough to never miss a clash.
    reach = max(_contact_floor(a, b)
                for a in set(group.elements()) | {"C"}
                for b in set(symbols) | set(group.elements()))
    checker = _ClashChecker(atoms, reach)

    for position in order:
        if len(grafted) >= requested:
            break
        site = sites[position]
        normal = outer[site]
        if not np.any(normal):
            continue

        if min_separation > 0.0 and grafted:
            # Minimum image, not raw coordinates. Two sites on opposite
            # sides of a cell seam are neighbours, and measuring them
            # straight through the cell called them the full cell length
            # apart -- so the separation was enforced everywhere except
            # exactly where crowding happens.
            spans = _mic_vectors(
                atoms, np.array([[site, other] for other in grafted]))
            if float(np.linalg.norm(spans, axis=1).min()) < min_separation:
                refused_separation += 1
                continue

        side = _pick_side(face, parity[site], blocked[site])

        # Try the rotamers and keep the roomiest. A single bond to the
        # surface is a free rotation, so refusing a site because the
        # frame's arbitrary tangent aimed a hydrogen at the wall would
        # measure the frame rather than the chemistry.
        exempt = _bonded_exemptions(group, int(site), neighbours_of)
        best_margin, best = -np.inf, None
        for twist in np.linspace(0.0, 360.0, ROTAMERS, endpoint=False):
            placed_symbols, placed = build_positions(
                group, positions[site], side * normal,
                surface_symbol=symbols[site], twist=float(twist))
            clearance = checker.margin(placed, placed_symbols, exempt)
            if clearance > best_margin:
                best_margin, best = clearance, (placed_symbols, placed)
            if group.n_atoms == 1:
                break  # a lone atom has no rotamers to try

        if best is None or best_margin < 0.0:
            refused_steric += 1
            continue
        placed_symbols, placed = best

        root = existing + len(new_symbols)
        terminal.extend(_terminal_atoms(group, root))
        new_bonds.append((int(site), root))
        for offset, atom in enumerate(group.atoms):
            if atom.parent >= 0:
                new_bonds.append((root + atom.parent, root + offset))
        new_symbols.extend(placed_symbols)
        new_positions.extend(placed)
        grafted.append(int(site))
        faces.append("outer" if side > 0 else "inner")
        checker.add(placed, placed_symbols)

    out = atoms.copy()
    if new_symbols:
        out += Atoms(symbols=new_symbols, positions=np.asarray(new_positions))
        preserve_vacuum(atoms, out)
    out.info = dict(atoms.info)

    if terminal or atoms.info.get("terminal_atoms"):
        out.info["terminal_atoms"] = [
            *(int(i) for i in atoms.info.get("terminal_atoms", [])), *terminal]

    recorded = atoms.info.get("bonds")
    if recorded is not None and len(recorded):
        out.info["bonds"] = [list(map(int, pair)) for pair in recorded] + \
            [[int(a), int(b)] for a, b in new_bonds]

    out.info["functionalization"] = {
        "group": group.name,
        "formula": group.formula,
        "elements": list(group.elements()),
        "where": where,
        "face": face,
        "sites": [int(i) for i in grafted],
        "faces": faces,
        "n_grafted": len(grafted),
        "n_requested": requested,
        "n_candidate_sites": len(sites),
        "coverage": len(grafted) / len(sites) if sites else 0.0,
        "requested_coverage": requested / len(sites) if sites else 0.0,
        "refused_steric": refused_steric,
        "refused_separation": refused_separation,
        "site_hybridisation": group.site_hybridisation,
        "seed": seed,
    }
    return out


def _graft_bridging(atoms: Atoms,
                    group: FunctionalGroup,
                    coverage: float | None,
                    count: int | None,
                    where: str,
                    face: str,
                    seed: int | None,
                    tolerance: float) -> Atoms:
    """Place groups that span a surface **bond** rather than an atom.

    The candidates are bonds with both ends graftable, and coverage is a
    fraction of those -- not of the atoms, which would silently mean
    something different for the same number. Each accepted bridge
    consumes both of its atoms, since an epoxide and a hydroxyl cannot
    share a carbon.
    """
    pairs = bond_pairs(atoms, tolerance=tolerance)
    allowed = set(candidate_sites(atoms, where=where, tolerance=tolerance))
    bonds = [(int(i), int(j)) for i, j in pairs
             if int(i) in allowed and int(j) in allowed]
    if not bonds:
        raise ValueError(
            f"No bond has both ends graftable for where={where!r}, so a "
            f"bridging group such as {group.name!r} has nothing to span."
        )

    if coverage is not None and not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], not {coverage}.")
    requested = count if count is not None else int(round(coverage * len(bonds)))
    requested = max(0, min(requested, len(bonds)))

    outer = surface_normals(atoms, face="outer", tolerance=tolerance)
    parity = (sublattice_parity(atoms, tolerance=tolerance)
              if face == "both" else np.zeros(len(atoms), dtype=int))
    blocked = inner_face_blocked(atoms, outer, tolerance=tolerance)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bonds))

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    neighbours_of: list[list[int]] = [[] for _ in range(len(atoms))]
    for first, second in pairs:
        neighbours_of[int(first)].append(int(second))
        neighbours_of[int(second)].append(int(first))

    reach = max(_contact_floor(a, b)
                for a in set(group.elements()) | {"C"}
                for b in set(symbols) | set(group.elements()))
    checker = _ClashChecker(atoms, reach)

    existing = len(atoms)
    new_symbols: list[str] = []
    new_positions: list[np.ndarray] = []
    new_bonds: list[tuple[int, int]] = []
    used: set[int] = set()
    grafted: list[list[int]] = []
    terminal: list[int] = []
    refused_steric = 0
    refused_geometry = 0
    refused_occupied = 0

    for slot in order:
        if len(grafted) >= requested:
            break
        first, second = bonds[int(slot)]
        if first in used or second in used:
            # An epoxide and its neighbour cannot share a carbon, so
            # accepting one bond rules out the four touching it. The
            # ceiling this imposes is a maximum matching on the surface,
            # about a third of the bonds on a honeycomb -- not a steric
            # refusal, and counted apart from one.
            refused_occupied += 1
            continue

        # The minimum-image copy of the partner, so a bond across a cell
        # seam is bridged where it actually is rather than across the
        # whole cell.
        offset = _mic_vectors(atoms, np.array([[first, second]]))[0]
        partner = positions[first] + offset

        side = _pick_side(face, parity[first],
                          blocked[first] or blocked[second])
        normal = side * (outer[first] + outer[second])
        if not np.any(normal):
            continue

        try:
            placed_symbols, placed = build_bridging_positions(
                group, positions[first], partner, normal,
                surface_symbol=symbols[first])
        except ValueError:
            # This particular bond is too long for this group to span.
            refused_geometry += 1
            continue

        # Both anchors and both their neighbourhoods are within two bonds
        # of the bridging atom, for the same reason a terminal group's
        # root is.
        spared = {first, second,
                  *neighbours_of[first], *neighbours_of[second]}
        exempt = [spared if index == 0 else {first, second}
                  for index in range(group.n_atoms)]
        if checker.margin(placed, placed_symbols, exempt) < 0.0:
            refused_steric += 1
            continue

        root = existing + len(new_symbols)
        terminal.extend(_terminal_atoms(group, root))
        new_bonds.extend([(first, root), (second, root)])
        for index, atom in enumerate(group.atoms):
            if atom.parent >= 0:
                new_bonds.append((root + atom.parent, root + index))
        new_symbols.extend(placed_symbols)
        new_positions.extend(placed)
        checker.add(placed, placed_symbols)
        used.update((first, second))
        grafted.append([first, second])

    out = atoms.copy()
    if new_symbols:
        out += Atoms(symbols=new_symbols, positions=np.asarray(new_positions))
        preserve_vacuum(atoms, out)
    out.info = dict(atoms.info)

    if terminal or atoms.info.get("terminal_atoms"):
        out.info["terminal_atoms"] = [
            *(int(i) for i in atoms.info.get("terminal_atoms", [])), *terminal]

    recorded = atoms.info.get("bonds")
    if recorded is not None and len(recorded):
        out.info["bonds"] = [list(map(int, pair)) for pair in recorded] + \
            [[int(a), int(b)] for a, b in new_bonds]

    out.info["functionalization"] = {
        "group": group.name,
        "formula": group.formula,
        "elements": list(group.elements()),
        "where": where,
        "face": face,
        "bridges": [[int(a), int(b)] for a, b in grafted],
        "sites": sorted({int(i) for pair in grafted for i in pair}),
        "n_grafted": len(grafted),
        "n_requested": requested,
        "n_candidate_sites": len(bonds),
        "coverage": len(grafted) / len(bonds) if bonds else 0.0,
        "requested_coverage": requested / len(bonds) if bonds else 0.0,
        "refused_steric": refused_steric,
        "refused_geometry": refused_geometry,
        "refused_occupied": refused_occupied,
        "refused_separation": 0,
        "site_hybridisation": group.site_hybridisation,
        "bridging": True,
        "seed": seed,
    }
    return out


def describe_functionalization(atoms: Atoms) -> str:
    """A human summary of what was grafted, or why nothing was.

    Says the achieved coverage next to the requested one whenever they
    differ, because a silent shortfall is the one outcome a user must not
    discover later from an atom count.
    """
    record: dict[str, Any] | None = atoms.info.get("functionalization")
    if not record:
        return "No functionalization recorded on this structure."

    unit = "surface bonds" if record.get("bridging") else "candidate sites"
    lines = [
        f"{record['group']} ({record['formula']}) on {record['n_grafted']} of "
        f"{record['n_candidate_sites']} {unit} "
        f"({100 * record['coverage']:.1f}% coverage, "
        f"selection {record['where']!r}, face {record['face']!r}).",
        f"The site becomes {record['site_hybridisation']}.",
    ]
    shortfall = record["n_requested"] - record["n_grafted"]
    if shortfall > 0:
        reasons = []
        if record["refused_steric"]:
            reasons.append(f"{record['refused_steric']} would have overlapped "
                           "a neighbouring group or the wall")
        if record.get("refused_occupied"):
            reasons.append(f"{record['refused_occupied']} shared an atom with a "
                           "bridge already placed, which no two can do")
        if record.get("refused_geometry"):
            reasons.append(f"{record['refused_geometry']} spanned atoms too far "
                           "apart for this group to reach across")
        if record["refused_separation"]:
            reasons.append(f"{record['refused_separation']} were closer than "
                           "the requested site separation")
        lines.append(
            f"{shortfall} of the {record['n_requested']} requested did not fit"
            + (": " + "; ".join(reasons) if reasons else "")
            + ". That is the density the group's own size and the surface "
              "allow, not a placement failure."
        )
    return " ".join(lines)


__all__ = [
    "CHALCOGENS",
    "CONTACT_FRACTION",
    "ENCLOSURE_THRESHOLD",
    "INNER_BLOCKED_DEPTH",
    "MIN_CONTACT",
    "PYRAMIDAL_SUM",
    "bond_pairs",
    "candidate_sites",
    "describe_functionalization",
    "enclosure",
    "functionalize",
    "inner_face_blocked",
    "is_enclosing",
    "preserve_vacuum",
    "sublattice_parity",
    "surface_normals",
]
