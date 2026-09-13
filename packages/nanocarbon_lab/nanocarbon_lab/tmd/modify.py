"""Post-build edits to a finished MX2 structure.

Everything here takes a built dichalcogenide and changes its chemistry
without touching its geometry, which is what makes them composable: build
a monolayer, a tube or a schwarzite, then make it Janus, alloy it, and
knock sulphur out of it, in any order.

Four edits, chosen because they are the ones the literature actually
uses:

* **Janus** (MoSSe) -- one chalcogen species on top, another underneath.
  Breaks the layer's mirror symmetry, which switches on an out-of-plane
  dipole and piezoelectricity that neither parent has. It is a
  substitution rather than a rebuild: the sandwich already has two
  distinguishable chalcogen planes.
* **Chalcogen vacancies** -- the dominant point defect in grown MoS2, and
  the one that dopes it n-type and pins its photoluminescence.
* **Antisites** -- a metal sitting on a chalcogen site or the reverse,
  common in sulphur-poor growth.
* **Alloying** -- Mo(1-x)W(x)S2 and friends, where the band gap tunes
  continuously with x.

Which plane a chalcogen belongs to is decided per atom, from the sign of
its offset along the local layer normal rather than from a global z.
That is what lets these work on a rolled tube or a curved cell, where
"above" is a different direction at every site and a z test would slice
the structure in half.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..utils.metadata import keep_indices, remap_after_removal

#: Chalcogen species a Janus layer may be given, in order of radius.
JANUS_CHALCOGENS = ("S", "Se", "Te")


def _metal_and_chalcogen(atoms: Atoms) -> tuple[str, str]:
    metal = atoms.info.get("metal")
    chalcogen = atoms.info.get("chalcogen")
    if metal is None or chalcogen is None:
        raise ValueError(
            "This needs atoms.info['metal'] and ['chalcogen']; pass a "
            "structure built by nanocarbon_lab.tmd."
        )
    return str(metal), str(chalcogen)


def _chalcogen_sides(atoms: Atoms) -> np.ndarray:
    """Per-atom face label for chalcogens: +1 outward, -1 inward, 0 metal.

    Each chalcogen gets an unambiguous *outward* direction -- the vector
    from the centroid of its bonded metals to itself -- but which of the
    two faces that points to is only meaningful relative to its
    neighbours. So the label is **propagated**: two chalcogens sharing a
    metal lie on the same face when their outward vectors agree, and a
    breadth-first walk spreads that into two components, one per face.

    Not from a cross product of two bond vectors, which was the first
    attempt: its sign depends on the order the neighbour list happened to
    return, so the label flipped at random from site to site and a
    "Janus" nanotube came out with both species mixed at the same radius.
    """
    metal, chalcogen = _metal_and_chalcogen(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    ideal = float(atoms.info.get("bond_length", 2.4))

    from ase.neighborlist import neighbor_list

    first, second, offset = neighbor_list("ijD", atoms, cutoff=ideal * 1.25)
    metal_bond = (symbols[first] == chalcogen) & (symbols[second] == metal)

    outward: dict[int, np.ndarray] = {}
    shares_metal: dict[int, set[int]] = {}
    by_metal: dict[int, list[int]] = {}
    for position in np.where(metal_bond)[0]:
        chalcogen_index = int(first[position])
        metal_index = int(second[position])
        by_metal.setdefault(metal_index, []).append(chalcogen_index)

    for index in np.where(symbols == chalcogen)[0]:
        mask = metal_bond & (first == index)
        neighbours = offset[mask]
        if len(neighbours) < 1:
            continue
        vector = -neighbours.mean(axis=0)  # offset runs chalcogen -> metal
        length = np.linalg.norm(vector)
        if length < 1e-9:
            continue
        outward[int(index)] = vector / length

    for partners in by_metal.values():
        for a_index in partners:
            shares_metal.setdefault(a_index, set()).update(
                p for p in partners if p != a_index)

    sides = np.zeros(len(atoms))
    label: dict[int, int] = {}
    for start in sorted(outward):
        if start in label:
            continue
        label[start] = 1
        queue = [start]
        while queue:
            current = queue.pop()
            for partner in shares_metal.get(current, ()):
                if partner in label or partner not in outward:
                    continue
                same = float(np.dot(outward[current], outward[partner])) > 0.0
                label[partner] = label[current] if same else -label[current]
                queue.append(partner)

    for index, value in label.items():
        sides[index] = value

    # Name the faces. "+1" is the one further from the structure's centre,
    # which is the outer wall of a tube; for a flat layer the two are
    # equidistant, so fall back to the one higher in z.
    positions = atoms.get_positions()
    centre = positions.mean(axis=0)
    radius = np.linalg.norm(positions - centre, axis=1)
    plus = sides == 1
    minus = sides == -1
    if plus.any() and minus.any():
        gap = radius[plus].mean() - radius[minus].mean()
        if abs(gap) < 0.1:
            gap = positions[plus, 2].mean() - positions[minus, 2].mean()
        if gap < 0:
            sides = -sides
    return sides


def make_janus(atoms: Atoms, chalcogen: str = "Se",
               side: int = 1) -> Atoms:
    """Replace the chalcogens on one face, giving a Janus MXY layer.

    Parameters
    ----------
    atoms
        A structure from :mod:`nanocarbon_lab.tmd`.
    chalcogen
        The species to put on that face (``"S"``, ``"Se"`` or ``"Te"``).
    side
        ``+1`` for the outward face, ``-1`` for the inward one. On a tube
        those are the outer and inner walls; on a flat layer, top and
        bottom.

    Returns
    -------
    ase.Atoms
        A copy, with ``janus`` recorded in ``info``.

    Raises
    ------
    ValueError
        For an unknown chalcogen, a side other than +-1, or a structure
        whose faces could not be told apart (which happens when nothing
        is bonded, i.e. the input is not an intact MX2 sandwich).
    """
    if chalcogen not in JANUS_CHALCOGENS:
        raise ValueError(
            f"chalcogen must be one of {JANUS_CHALCOGENS}; got {chalcogen!r}.")
    if side not in (1, -1):
        raise ValueError("side must be +1 (outward) or -1 (inward).")

    metal, original = _metal_and_chalcogen(atoms)
    if chalcogen == original:
        raise ValueError(
            f"The layer is already all {original}; a Janus layer needs two "
            "different chalcogens."
        )
    sides = _chalcogen_sides(atoms)
    chosen = sides == side
    if not chosen.any():
        raise ValueError(
            "Could not tell the two chalcogen planes apart. That means no "
            "chalcogen has two metal neighbours, so this is not an intact "
            "MX2 sandwich."
        )

    out = atoms.copy()
    out.info = dict(atoms.info)
    symbols = list(out.get_chemical_symbols())
    for index in np.where(chosen)[0]:
        symbols[index] = chalcogen
    out.set_chemical_symbols(symbols)

    replaced = int(chosen.sum())
    out.info.update({
        "janus": True,
        "chalcogen_top": chalcogen if side == 1 else original,
        "chalcogen_bottom": original if side == 1 else chalcogen,
        "janus_replaced": replaced,
        "formula": out.get_chemical_formula(),
    })
    return out


def chalcogen_vacancies(atoms: Atoms, n_defects: int = 1, seed: int = 0,
                        paired: bool = False) -> Atoms:
    """Remove chalcogen atoms -- the commonest point defect in grown MoS2.

    Parameters
    ----------
    atoms
        A structure from :mod:`nanocarbon_lab.tmd`.
    n_defects
        How many vacancies to create.
    seed
        RNG seed; the same seed always removes the same sites.
    paired
        Remove both chalcogens of a site (a double vacancy) rather than
        one. The single vacancy is the common one; the double is what
        forms under electron irradiation.

    Returns
    -------
    ase.Atoms
        A copy with the sites removed and a ``defect_log`` in ``info``.

    Raises
    ------
    ValueError
        If more vacancies are asked for than there are chalcogens.
    """
    metal, chalcogen = _metal_and_chalcogen(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    candidates = np.where(symbols == chalcogen)[0]
    if n_defects < 0:
        raise ValueError("n_defects must be >= 0.")
    if n_defects > len(candidates):
        raise ValueError(
            f"Asked for {n_defects} vacancies but the structure has only "
            f"{len(candidates)} {chalcogen} atoms."
        )
    if n_defects == 0:
        return atoms.copy()

    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=n_defects, replace=False)

    remove = set(int(index) for index in chosen)
    if paired:
        # The partner is the other chalcogen bonded to the same three
        # metals, i.e. the nearest one across the sandwich.
        positions = atoms.get_positions()
        for index in list(remove):
            distances = np.linalg.norm(positions[candidates] - positions[index],
                                       axis=1)
            distances[candidates == index] = np.inf
            remove.add(int(candidates[int(np.argmin(distances))]))

    keep = keep_indices(len(atoms), remove)
    out = atoms[keep]
    # See utils/metadata: a curved MX2 records its own bond graph, and
    # copying those indices past a deletion corrupts it silently.
    out.info = remap_after_removal(atoms.info, keep)
    out.info.setdefault("defect_log", [])
    out.info["defect_log"] = list(out.info["defect_log"]) + [
        {"type": "chalcogen_vacancy", "count": len(remove),
         "paired": bool(paired), "seed": seed}
    ]
    out.info["formula"] = out.get_chemical_formula()
    return out


def alloy(atoms: Atoms, replacement: str, fraction: float = 0.5,
          seed: int = 0, site: str = "metal") -> Atoms:
    """Randomly substitute a fraction of one sublattice.

    Mo(1-x)W(x)S2 and MoS(2-2x)Se(2x) are the standard alloys, and their
    band gap moves continuously with ``x`` -- which is the point of
    making them.

    Parameters
    ----------
    atoms
        A structure from :mod:`nanocarbon_lab.tmd`.
    replacement
        Element substituted in.
    fraction
        Fraction of that sublattice replaced, in ``[0, 1]``.
    seed
        RNG seed.
    site
        ``"metal"`` or ``"chalcogen"`` -- which sublattice to alloy.

    Returns
    -------
    ase.Atoms
        A copy, with the achieved fraction recorded. The achieved value
        differs from the requested one whenever the sublattice count does
        not divide evenly, and it is the achieved one that describes the
        structure.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1].")
    if site not in ("metal", "chalcogen"):
        raise ValueError("site must be 'metal' or 'chalcogen'.")

    metal, chalcogen = _metal_and_chalcogen(atoms)
    target = metal if site == "metal" else chalcogen
    symbols = np.array(atoms.get_chemical_symbols())
    candidates = np.where(symbols == target)[0]
    if not len(candidates):
        raise ValueError(f"No {target} atoms to alloy.")

    count = int(round(fraction * len(candidates)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=count, replace=False) if count else []

    out = atoms.copy()
    out.info = dict(atoms.info)
    new_symbols = list(out.get_chemical_symbols())
    for index in chosen:
        new_symbols[int(index)] = replacement
    out.set_chemical_symbols(new_symbols)
    out.info.update({
        "alloy": {"site": site, "replacement": replacement,
                  "requested_fraction": fraction,
                  "achieved_fraction": count / len(candidates),
                  "seed": seed},
        "formula": out.get_chemical_formula(),
    })
    return out


def antisites(atoms: Atoms, n_defects: int = 1, seed: int = 0) -> Atoms:
    """Swap a metal onto a chalcogen site (``M_X``), as in sulphur-poor growth."""
    metal, chalcogen = _metal_and_chalcogen(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    candidates = np.where(symbols == chalcogen)[0]
    if n_defects > len(candidates):
        raise ValueError(
            f"Asked for {n_defects} antisites but there are only "
            f"{len(candidates)} {chalcogen} atoms."
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=n_defects, replace=False)

    out = atoms.copy()
    out.info = dict(atoms.info)
    new_symbols = list(out.get_chemical_symbols())
    for index in chosen:
        new_symbols[int(index)] = metal
    out.set_chemical_symbols(new_symbols)
    out.info.setdefault("defect_log", [])
    out.info["defect_log"] = list(out.info["defect_log"]) + [
        {"type": "antisite", "count": int(n_defects), "seed": seed}
    ]
    out.info["formula"] = out.get_chemical_formula()
    return out


__all__ = [
    "JANUS_CHALCOGENS",
    "alloy",
    "antisites",
    "chalcogen_vacancies",
    "make_janus",
]
