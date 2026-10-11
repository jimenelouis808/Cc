"""Nitrogen at vacancies: the pyridinic and pyrrolic motifs, closed shell.

Substitution (:mod:`.substitutional`) gives graphitic N only. The other two
motifs every N-doped carbon is discussed in terms of (and the two that XPS
tells apart from graphitic N) need a vacancy, and a careless build of them
leaves an odd electron count. The two builders here are the closed-shell
versions in common use:

* :func:`pyridinic_divacancy` — "N4V2": the two carbons of a bond are removed
  and the four carbons that bonded to them become pyridinic N (two-coordinated,
  lone pair in plane). Removing 2 C (−8 e) and adding 4 N (+4 e) keeps the count
  even. The tetrapyridinic pocket of Fe-N4 and porphyrin-like sites.
* :func:`pyrrolic_vacancy` — one carbon outside a five-membered ring is removed;
  the ring atom it bonded to becomes N–H (pyrrolic: in a pentagon, bonded to two C
  and an H) and its other two neighbours are closed with H. −4 + 1 + 3 = 0.

Both take explicit indices: which bond or ring atom is chosen is a question
for whoever ranks the sites (an energy screening), not for the builder. The result
needs relaxing. Periodic cells are fine (minimum image).

**The three hydrogens of the pyrrolic motif cannot lie in the plane.** The three
atoms around a monovacancy are 2.46 Å apart, so three caps aimed at the empty site
along their bonds end 0.7-0.9 Å from each other -- an H3 cluster, which a relaxation
in a tight-binding model kept rather than broke apart (measured on the 204-atom coil;
GPAW then found 15-19 eV/Å forces). Each cap is therefore tilted out of the local
plane by ``tilt`` (75°), on alternating sides, which puts them ≥ 1.8 Å apart before
any relaxation; a test pins that.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..utils.metadata import keep_indices, remap_after_removal

C_H, N_H = 1.09, 1.01


def _neighbours(atoms: Atoms, index: int, cutoff: float = 1.80) -> list[int]:
    d = atoms.get_distances(index, range(len(atoms)), mic=True)
    return [int(k) for k in np.flatnonzero((d > 0.1) & (d < cutoff))]


def _vector(atoms: Atoms, a: int, b: int) -> np.ndarray:
    return atoms.get_distance(a, b, mic=True, vector=True)


def _finish(atoms: Atoms, removed: list[int], to_n: list[int],
            caps: list[tuple[int, np.ndarray, float]], record: dict) -> Atoms:
    out = atoms.copy()
    for k in to_n:
        out[k].symbol = "N"
    for parent, direction, length in caps:
        out.append("H")
        out.positions[-1] = out.positions[parent] + length * direction / np.linalg.norm(direction)
    keep = keep_indices(len(out), removed)
    info = remap_after_removal(dict(out.info), keep)
    result = out[keep]
    result.info = info
    old_to_new = {old: new for new, old in enumerate(keep)}
    record["nitrogen"] = [old_to_new[k] for k in to_n]
    result.info["nitrogen_motif"] = record
    return result


def pyridinic_divacancy(atoms: Atoms, bond: tuple[int, int], cutoff: float = 1.80) -> Atoms:
    """N4V2: remove the two carbons of ``bond``; their four other neighbours become N."""
    a, b = (int(x) for x in bond)
    if atoms[a].symbol != "C" or atoms[b].symbol != "C":
        raise ValueError("The bond must join two carbons.")
    if b not in _neighbours(atoms, a, cutoff):
        raise ValueError(f"Atoms {a} and {b} are not bonded.")
    rim = sorted({n for n in _neighbours(atoms, a, cutoff) + _neighbours(atoms, b, cutoff)}
                 - {a, b})
    if len(rim) != 4 or any(atoms[k].symbol != "C" for k in rim):
        raise ValueError(f"A divacancy here leaves {len(rim)} rim atoms, not four carbons; "
                         "choose a bond between two three-coordinated carbons.")
    return _finish(atoms, [a, b], rim, [], {"motif": "pyridinic N4V2", "removed": [a, b]})


def pyrrolic_vacancy(atoms: Atoms, ring_atom: int, removed: int,
                     cutoff: float = 1.80, tilt: float = 75.0) -> Atoms:
    """Remove ``removed`` (bonded to ``ring_atom`` from outside its pentagon): ``ring_atom``
    becomes N–H, the other two neighbours of ``removed`` become C–H, each H tilted
    ``tilt`` degrees out of the local plane on alternating sides (see the module note)."""
    ring_atom, removed = int(ring_atom), int(removed)
    around = _neighbours(atoms, removed, cutoff)
    if ring_atom not in around:
        raise ValueError(f"Atoms {ring_atom} and {removed} are not bonded.")
    others = [k for k in around if k != ring_atom]
    if len(others) != 2 or any(atoms[k].symbol != "C" for k in [ring_atom, *others]):
        raise ValueError("The removed atom must be a three-coordinated carbon bonded to "
                         "three carbons.")
    rim = [ring_atom, *others]
    vectors = [_vector(atoms, k, removed) for k in rim]          # rim atom -> vacancy
    points = np.array([np.zeros(3)] + [-v for v in vectors])     # vacancy at the origin
    normal = np.linalg.svd(points - points.mean(axis=0))[2][2]
    angle = np.radians(tilt)
    caps = []
    for k, v, side, length in zip(rim, vectors, (1, -1, 1), (N_H, C_H, C_H), strict=True):
        u = v / np.linalg.norm(v)
        caps.append((k, np.cos(angle) * u + side * np.sin(angle) * normal, length))
    return _finish(atoms, [removed], [ring_atom], caps,
                   {"motif": "pyrrolic N-H at a monovacancy", "removed": [removed]})
