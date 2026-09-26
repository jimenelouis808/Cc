"""Attaching functional groups to a structure.

The public entry points are :func:`functionalize`, which decorates chosen
sites with one group, and :func:`functionalize_random`, which picks sites at
random from a reproducible seed.

Every attachment records what it did in ``atoms.info["functionalization"]``,
so a structure carries its own provenance into the exported input file and
the dataset metadata.
"""

from __future__ import annotations

from typing import Optional, Sequence  # noqa: F401

import numpy as np
from ase import Atoms

from ..utils.constants import HARD_MIN_DISTANCE
from ..utils.geometry import ensure_vacuum
from ..utils.rng import make_rng
from .groups import EDGE_ONLY_GROUPS, FunctionalGroup, get_group
from .sites import (
    AttachmentSite,
    SiteKind,
    find_bridge_sites,
    find_sites,
    local_frame,
)


def _place(
    group: FunctionalGroup,
    origin: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Map a group's local coordinates onto a site."""
    x_hat, y_hat, z_hat = local_frame(direction)
    basis = np.vstack([x_hat, y_hat, z_hat])
    return origin + group.positions @ basis


def _best_torsion(
    atoms: Atoms,
    anchor: int,
    positions: np.ndarray,
    step_deg: float = 15.0,
) -> np.ndarray:
    """Rotate a group about its anchor bond to keep clear of everything else.

    The group library places each group in a fixed local frame that knows
    nothing of the neighbours. On an edge that frame can fold the H of an
    -OH back onto the lattice (0.5 Å from the next carbon). The rotation is
    a free torsion about the anchor-to-first-atom bond, so it changes
    nothing chemical; the orientation farthest from every other atom wins,
    the first found among equals, so the result is reproducible. Distances
    are minimum-image, so a group near a periodic boundary sees its images.
    """
    from ase.geometry import get_distances

    if len(positions) < 2 or len(atoms) < 2:
        return positions
    others = np.array([i for i in range(len(atoms)) if i != anchor])
    pivot = positions[0]
    axis = pivot - atoms.positions[anchor]
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return positions
    axis = axis / norm
    moving = positions[1:] - pivot

    best, best_gap = positions, -1.0
    for angle in np.radians(np.arange(0.0, 360.0, step_deg)):
        c, s = np.cos(angle), np.sin(angle)
        rotated = (moving * c + np.cross(axis, moving) * s
                   + np.outer(moving @ axis, axis) * (1 - c))
        trial = np.vstack([pivot, pivot + rotated])
        _, distances = get_distances(trial[1:], atoms.positions[others],
                                     cell=atoms.cell, pbc=atoms.pbc)
        gap = float(distances.min())
        if gap > best_gap + 1e-6:
            best, best_gap = trial, gap
    return best


def attach_group(
    atoms: Atoms,
    site: AttachmentSite,
    group_key: str,
) -> Atoms:
    """Attach one group at one site.

    Parameters
    ----------
    atoms
        Structure to decorate (not mutated).
    site
        Where to attach, from :func:`~carbonforge.functionalization.sites.find_sites`.
    group_key
        Key into :data:`~carbonforge.functionalization.groups.GROUPS`.

    Returns
    -------
    ase.Atoms
        New structure with the group appended.

    Raises
    ------
    ValueError
        If the group cannot chemically go on that site — a carbonyl needs a
        free double-bond valence and so only fits an edge carbon, and a
        bridging epoxide needs :func:`attach_bridging_group` instead.
    """
    group = get_group(group_key)

    if group.bridging:
        raise ValueError(
            f"'{group_key}' es un grupo puente entre dos carbonos. "
            "Usa attach_bridging_group()."
        )
    if group_key in EDGE_ONLY_GROUPS and site.kind != "edge":
        raise ValueError(
            f"'{group_key}' ({group.formula}) consume dos valencias, así que "
            "solo cabe en un carbono de borde (coordinación 2). El sitio "
            f"{site.index} tiene coordinación {site.coordination}."
        )

    out = atoms.copy()
    out.info = {**atoms.info}
    positions = _best_torsion(atoms, site.index, _place(group, site.origin, site.direction))
    out += Atoms(symbols=list(group.symbols), positions=positions)

    record = out.info.setdefault("functionalization", [])
    record.append(
        {
            "group": group_key,
            "formula": group.formula,
            "anchor": int(site.index),
            "site_kind": site.kind,
            "n_atoms_added": len(group),
        }
    )
    return out


def attach_bridging_group(
    atoms: Atoms,
    pair: tuple[int, int, np.ndarray, np.ndarray],
    group_key: str = "epoxy",
) -> Atoms:
    """Attach a bridging group (an epoxide) across two adjacent carbons."""
    group = get_group(group_key)
    if not group.bridging:
        raise ValueError(f"'{group_key}' no es un grupo puente.")

    i, j, midpoint, normal = pair
    out = atoms.copy()
    out.info = {**atoms.info}
    positions = _place(group, midpoint, normal)
    out += Atoms(symbols=list(group.symbols), positions=positions)

    record = out.info.setdefault("functionalization", [])
    record.append(
        {
            "group": group_key,
            "formula": group.formula,
            "anchor": [int(i), int(j)],
            "site_kind": "bridge",
            "n_atoms_added": len(group),
        }
    )
    return out


def _check_no_overlap(atoms: Atoms, context: str) -> None:
    """Raise if attaching produced atoms closer than the hard minimum.

    Two groups on neighbouring anchors can collide even when each is
    individually fine, and a silent overlap makes any downstream calculation
    meaningless. Better to fail here with the offending pair named.
    """
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    smallest = float(distances.min())
    if smallest < HARD_MIN_DISTANCE:
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        symbols = atoms.get_chemical_symbols()
        raise ValueError(
            f"{context}: los átomos {int(i)} ({symbols[int(i)]}) y {int(j)} "
            f"({symbols[int(j)]}) quedan a {smallest:.3f} Å, por debajo del "
            f"mínimo físico de {HARD_MIN_DISTANCE} Å. Suele pasar al poner "
            "grupos en carbonos vecinos: sube min_separation o pon menos "
            "grupos."
        )


def repad_vacuum(atoms: Atoms, min_vacuum: float = 12.0) -> Atoms:
    """Restore the vacuum padding consumed by attached groups.

    A group protruding from an edge eats into the padding the structure was
    built with. Call this after functionalising anything that will go to DFT.
    """
    return ensure_vacuum(atoms, min_vacuum=min_vacuum)


def functionalize(
    atoms: Atoms,
    group_key: str,
    indices: Optional[Sequence[int]] = None,
    site_kind: Optional[SiteKind] = None,
) -> Atoms:
    """Attach a group at explicit anchor indices, or at every matching site.

    Parameters
    ----------
    atoms
        Structure to decorate.
    group_key
        Which group to attach.
    indices
        Anchor atom indices. When ``None``, every site matching ``site_kind``
        is decorated — which for a basal group on a large sheet means a very
        heavily functionalised system, so pass indices or use
        :func:`functionalize_random` for realistic coverages.
    site_kind
        Restrict to ``"edge"`` or ``"basal"`` sites.

    Returns
    -------
    ase.Atoms

    Notes
    -----
    Sites are resolved against the **original** structure and the geometry of
    each is computed before anything is added, so attaching several groups
    does not shift the anchors of the later ones.
    """
    group = get_group(group_key)
    if group.bridging:
        raise ValueError(
            f"'{group_key}' es un grupo puente. Usa functionalize_bridges()."
        )

    available = find_sites(atoms, kind=site_kind)
    by_index = {site.index: site for site in available}

    if indices is None:
        chosen = available
    else:
        chosen = []
        for index in indices:
            site = by_index.get(int(index))
            if site is None:
                raise ValueError(
                    f"El átomo {index} no es un sitio de anclaje válido "
                    f"(¿no es carbono, o tiene coordinación distinta de 2 o 3?)."
                )
            chosen.append(site)

    out = atoms
    for site in chosen:
        out = attach_group(out, site, group_key)
    _check_no_overlap(out, f"functionalize('{group_key}')")
    return repad_vacuum(out)


def functionalize_bridges(
    atoms: Atoms,
    n_groups: Optional[int] = None,
    seed: Optional[int] = None,
    min_separation: float = 4.0,
) -> Atoms:
    """Decorate basal C-C bonds with epoxide bridges.

    Parameters
    ----------
    atoms
        Structure to decorate.
    n_groups
        How many epoxides to place. ``None`` places one on every eligible
        bond, which is unrealistically dense — pass a number.
    seed
        RNG seed for the choice of bonds.
    min_separation
        Minimum distance (Å) between the midpoints of chosen bonds, so the
        epoxides do not pile onto neighbouring bonds.
    """
    pairs = find_bridge_sites(atoms)
    if not pairs:
        raise ValueError(
            "No hay enlaces C-C basales disponibles. Los epóxidos necesitan "
            "carbonos con coordinación 3."
        )

    if n_groups is None:
        chosen = pairs
    else:
        rng = make_rng(seed)
        order = rng.permutation(len(pairs))
        chosen = []
        placed_midpoints: list[np.ndarray] = []
        for position in order:
            _, _, midpoint, _ = pairs[position]
            if all(
                np.linalg.norm(midpoint - other) >= min_separation
                for other in placed_midpoints
            ):
                chosen.append(pairs[position])
                placed_midpoints.append(midpoint)
            if len(chosen) >= n_groups:
                break
        if len(chosen) < n_groups:
            raise ValueError(
                f"Solo caben {len(chosen)} epóxidos separados al menos "
                f"{min_separation} Å, y se pidieron {n_groups}. Baja "
                "min_separation o usa una lámina más grande."
            )

    out = atoms
    for pair in chosen:
        out = attach_bridging_group(out, pair)
    _check_no_overlap(out, "functionalize_bridges()")
    out.info["functionalization_seed"] = seed
    return repad_vacuum(out)


def functionalize_at_sites(
    atoms: Atoms,
    group_key: str,
    count: int = 1,
    region: str = "edge",
    indices: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    min_separation: float = 2.5,
    avoid_radius: float = 2.6,
    avoid_occupied: bool = True,
    face: str = "+",
) -> Atoms:
    """Attach ``count`` groups at sites of a chosen kind.

    Parameters
    ----------
    atoms
        Structure to decorate (not mutated).
    group_key
        Group to attach (not the bridging epoxide: use
        :func:`functionalize_bridges`).
    count
        How many groups. Ignored with ``indices``.
    region
        Kind of site, from :data:`carbonforge.placement.REGIONS`: ``edge``,
        ``edge_armchair``, ``edge_zigzag``, ``basal``, ``pentagon``,
        ``heptagon``, ``defect_57``, ``near_defect``, ``vacancy_rim``,
        ``any``. An H-terminated edge carbon is a valid edge site: the group
        **replaces** its H. A vacancy rim is never an ``edge`` site -- a
        group there points into the hole -- so it has to be asked for.
    indices
        Exact anchor carbons; overrides ``region`` and ``count``.
    seed
        RNG seed.
    min_separation
        Minimum distance between anchors, Å.
    avoid_radius
        Minimum distance from existing heteroatoms and groups, Å -- the
        guard that keeps a group off the carbon next to a dopant.
    avoid_occupied
        Apply ``avoid_radius``.
    face
        On a flat structure, the side basal groups go (``"+"``/``"-"``).

    Returns
    -------
    ase.Atoms
    """
    from ..placement import candidate_sites, choose_sites, classify_sites, occupied_atoms

    group = get_group(group_key)
    if group.bridging:
        raise ValueError(f"'{group_key}' es un grupo puente. Usa functionalize_bridges().")
    info = classify_sites(atoms)
    edge_like = {"bare_edge", "terminated_edge", "vacancy_rim"}

    if indices is not None:
        chosen = [int(i) for i in indices]
        wrong = [i for i in chosen if i not in info or info[i].element != "C"
                 or info[i].substituents]
        if wrong:
            raise ValueError(f"Los átomos {wrong} no son carbonos libres de la red.")
        region = "índices"
    else:
        candidates = candidate_sites(atoms, region, "C", info=info)
        if group_key in EDGE_ONLY_GROUPS:
            candidates = [i for i in candidates if info[i].tags & edge_like]
        if not candidates:
            hint = (" Una lámina periódica sin bordes no tiene sitios de borde: usa "
                    "'basal', o construye una cinta o un fragmento finito."
                    if region.startswith("edge") else "")
            raise ValueError(f"No hay sitios '{region}' libres en esta estructura.{hint}")
        avoid = occupied_atoms(atoms, info) if avoid_occupied else []
        chosen = choose_sites(atoms, candidates, count, seed=seed,
                              min_separation=min_separation, avoid=avoid,
                              avoid_radius=avoid_radius)
    if group_key in EDGE_ONLY_GROUPS:
        not_edge = [i for i in chosen if not info[i].tags & edge_like]
        if not_edge:
            raise ValueError(
                f"'{group_key}' ({group.formula}) consume dos valencias: solo cabe en un "
                f"carbono de borde, y {not_edge} no lo son."
            )

    # Terminated edges: the group takes the place of the H.
    hydrogens = sorted({h for i in chosen for h in info[i].hydrogens}, reverse=True)
    out = atoms.copy()
    out.info = {**atoms.info}
    for h in hydrogens:
        del out[h]
    anchors = [i - sum(1 for h in hydrogens if h < i) for i in chosen]

    sites = {site.index: site for site in find_sites(out, face=face)}
    for anchor, original in zip(anchors, chosen, strict=True):
        site = sites.get(anchor)
        if site is None:
            raise ValueError(
                f"El carbono {original} no tiene una valencia libre para '{group_key}'."
            )
        out = attach_group(out, site, group_key)
        out.info["functionalization"][-1]["region"] = region
    _check_no_overlap(out, f"functionalize_at_sites('{group_key}')")
    out.info["functionalization_seed"] = seed
    return repad_vacuum(out)


def functionalize_random(
    atoms: Atoms,
    group_key: str,
    n_groups: int = 1,
    site_kind: Optional[SiteKind] = "edge",
    seed: Optional[int] = None,
    min_separation: float = 2.5,
) -> Atoms:
    """Attach ``n_groups`` copies of a group at randomly chosen sites.

    Parameters
    ----------
    atoms
        Structure to decorate.
    group_key
        Which group to attach.
    n_groups
        How many to place.
    site_kind
        ``"edge"`` (default) or ``"basal"``. Edge functionalisation is the
        chemically ordinary case; basal attachment forces sp3 and is what
        graphene oxide looks like.
    seed
        RNG seed, so the same call always gives the same structure.
    min_separation
        Minimum distance (Å) between chosen anchors. The default of 2.5 Å
        skips immediate neighbours, whose groups would otherwise overlap for
        anything bulkier than -H. Lower it for dense coverage, but expect the
        overlap guard to complain.

    Returns
    -------
    ase.Atoms
    """
    group = get_group(group_key)
    if group.bridging:
        raise ValueError(
            f"'{group_key}' es un grupo puente. Usa functionalize_bridges()."
        )
    if n_groups <= 0:
        raise ValueError("n_groups debe ser >= 1.")
    # Since the placement module exists, this is a thin wrapper over it: the
    # same draw, but H-terminated edges count (the group replaces the H),
    # vacancy rims do not, and groups keep clear of dopants and other groups.
    if site_kind is not None and not (group_key in EDGE_ONLY_GROUPS and site_kind == "basal"):
        return functionalize_at_sites(atoms, group_key, count=n_groups, region=site_kind,
                                      seed=seed, min_separation=min_separation)

    if group_key in EDGE_ONLY_GROUPS and site_kind == "basal":
        raise ValueError(
            f"'{group_key}' ({group.formula}) consume dos valencias, así que "
            "solo cabe en un carbono de borde. Usa site_kind='edge'."
        )
    available = find_sites(atoms, kind=site_kind)
    if group_key in EDGE_ONLY_GROUPS:
        available = [site for site in available if site.kind == "edge"]
    if not available:
        raise ValueError(
            f"No hay sitios de tipo '{site_kind}' disponibles. "
            "Una lámina periódica sin bordes no tiene sitios 'edge'; usa "
            "site_kind='basal', o construye una cinta o un fragmento finito."
        )
    if n_groups > len(available):
        raise ValueError(
            f"Se pidieron {n_groups} grupos pero solo hay {len(available)} "
            f"sitios de tipo '{site_kind}'."
        )

    rng = make_rng(seed)
    order = rng.permutation(len(available))
    chosen: list[AttachmentSite] = []
    for position in order:
        candidate = available[position]
        if min_separation > 0 and any(
            np.linalg.norm(candidate.origin - site.origin) < min_separation
            for site in chosen
        ):
            continue
        chosen.append(candidate)
        if len(chosen) >= n_groups:
            break

    if len(chosen) < n_groups:
        raise ValueError(
            f"Solo caben {len(chosen)} grupos separados al menos "
            f"{min_separation} Å, y se pidieron {n_groups}."
        )

    out = atoms
    for site in chosen:
        out = attach_group(out, site, group_key)
    _check_no_overlap(out, f"functionalize_random('{group_key}')")
    out.info["functionalization_seed"] = seed
    return repad_vacuum(out)


def passivate_edges(atoms: Atoms, group_key: str = "H") -> Atoms:
    """Saturate every edge carbon with a group, hydrogen by default.

    Leaving edge carbons bare gives spurious mid-gap states in a DFT
    calculation, so passivation is usually what you want before computing a
    band structure of a ribbon or a finite flake.
    """
    return functionalize(atoms, group_key, site_kind="edge")


def coverage(atoms: Atoms) -> dict[str, object]:
    """Summarise what has been attached to a structure."""
    records = atoms.info.get("functionalization", [])
    counts: dict[str, int] = {}
    added = 0
    for record in records:
        counts[record["group"]] = counts.get(record["group"], 0) + 1
        added += int(record["n_atoms_added"])
    carbons = sum(1 for s in atoms.get_chemical_symbols() if s == "C")
    return {
        "n_groups": len(records),
        "groups": counts,
        "atoms_added": added,
        "n_carbon": carbons,
        "groups_per_carbon": len(records) / carbons if carbons else 0.0,
    }
