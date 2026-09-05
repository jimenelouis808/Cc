"""Keeping index-based metadata honest when atoms are removed.

Several builders record their own connectivity -- ``info["bonds"]`` as
pairs of atom indices, ``info["rings"]`` as lists of them -- precisely so
that nothing downstream has to re-derive rings from coordinates on a
curved shell, which is the mistake :mod:`nanocarbon_lab.builders.fullerene_mesh`
exists to prevent.

That recorded graph is only correct for the atom *numbering* it was built
against. Removing an atom renumbers every atom after it, and the stored
indices then point at the wrong atoms. Copying ``info`` across a deletion
therefore produces something worse than missing metadata: metadata that
is present, plausible and wrong.

It was wrong, and quietly. Three vacancies in a 240-atom capped tube left
bond indices running to 239 against 237 atoms;
:func:`~nanocarbon_lab.topology.graph.coordination_numbers` prefers the
recorded graph when it exists, so validation read the corrupted one and
still passed. The render bundle writes those same indices to JSON, so a
defected tube drew bonds between atoms that were never bonded.

:func:`remap_after_removal` is the fix, and every function that deletes
atoms must call it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

#: ``info`` keys holding atom indices, and how deep the nesting goes.
#: ``bonds`` is a list of pairs, ``rings`` a list of rings; both are
#: handled the same way because both are lists of index lists.
INDEX_LIST_KEYS = ("bonds", "rings")


def remap_after_removal(info: dict[str, Any],
                        keep: Sequence[int]) -> dict[str, Any]:
    """Return a copy of ``info`` renumbered for a structure cut to ``keep``.

    Parameters
    ----------
    info
        The original ``atoms.info``, in the original numbering.
    keep
        Indices of the surviving atoms, in the order they will appear in
        the new structure -- exactly what was passed to ``atoms[keep]``.

    Returns
    -------
    dict
        A new mapping. Bonds and rings that touched a removed atom are
        **dropped**, not repaired: a bond with one end missing is not a
        bond, and a pentagon missing an atom is not a pentagon. Dropping
        them is what makes the survivors trustworthy.

        ``ring_counts`` is recomputed from the surviving rings rather
        than carried over, since carrying it would contradict them. Any
        recorded dopant indices are remapped too, and dopants that were
        themselves removed disappear from the list.

    Notes
    -----
    Keys this does not know about are copied unchanged. That is the right
    default -- most of ``info`` is scalars like the tube radius -- but it
    means a **new index-carrying key must be added to this module**, not
    only to the builder that writes it.
    """
    out = dict(info)
    new_index = {old: position for position, old in enumerate(keep)}

    for key in INDEX_LIST_KEYS:
        groups = info.get(key)
        if not groups:
            continue
        out[key] = [
            [new_index[int(i)] for i in group]
            for group in groups
            if all(int(i) in new_index for i in group)
        ]

    if "rings" in out and "ring_counts" in info:
        # Recomputed, never carried: a census that disagrees with the
        # rings beside it is the kind of inconsistency that is only
        # noticed after it has been plotted.
        counts: dict[int, int] = {}
        for ring in out["rings"]:
            counts[len(ring)] = counts.get(len(ring), 0) + 1
        out["ring_counts"] = dict(sorted(counts.items()))

    dopants = info.get("dopants")
    if dopants:
        remapped = []
        for entry in dopants:
            survivors = [new_index[int(i)] for i in entry.get("indices", [])
                         if int(i) in new_index]
            if survivors:
                remapped.append({**entry, "indices": survivors})
        if remapped:
            out["dopants"] = remapped
        else:
            out.pop("dopants", None)

    return out


def keep_indices(n_atoms: int, removed: Iterable[int]) -> list[int]:
    """The surviving indices, in order, after removing ``removed``.

    A one-liner, but writing it once means the list handed to
    ``atoms[keep]`` and the list handed to :func:`remap_after_removal`
    cannot disagree -- and if they disagree the metadata is silently
    wrong rather than obviously broken.
    """
    gone = {int(i) for i in removed}
    return [i for i in range(n_atoms) if i not in gone]


__all__ = ["INDEX_LIST_KEYS", "keep_indices", "remap_after_removal"]
