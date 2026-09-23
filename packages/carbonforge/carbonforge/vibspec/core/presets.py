"""Functionalisation presets for finite nanoribbons, as ``Atoms -> Atoms`` transforms.

Each preset is one chemically complete motif placed at one reproducible site
(see :mod:`carbonforge.vibspec.core.sites`), built with the existing
:mod:`carbonforge.functionalization` machinery rather than a second copy of
it. Edge presets **replace** a terminal hydrogen, so the carbon they act on
keeps a sane valence; interior presets act on a carbon with no hydrogen.

All geometries are idealised starting points. They must be relaxed before
any vibrational calculation, and two of them need more than that:

* ``pyrrolic_precursor`` still has six-membered rings; the pentagon only
  forms on relaxation, and has to be confirmed afterwards.
* Presets that add an odd number of electrons (``graphitic``,
  ``carbonyl``) leave an open-shell system that needs a spin-polarised
  calculation. :func:`carbonforge.vibspec.core.checks.suggest_spin` says
  which, and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import numpy as np
from ase import Atoms

from ...builders.nanoribbon import DEFAULT_VACUUM_PER_SIDE, rebox
from ...dopants.substitutional import substitute_atoms
from ...functionalization.attach import attach_bridging_group, attach_group
from ...functionalization.nitrogen import (
    make_graphitic_n,
    make_pyridinic_n,
    make_pyrrolic_like,
)
from ...functionalization.sites import find_bridge_sites, find_sites
from .sites import (
    EdgeKind,
    interior_carbons,
    pick_edge_site,
    pick_interior_carbon,
    strip_hydrogen,
)

Placement = Literal["none", "edge", "interior", "bridge"]
Position = Union[str, int, None]

# Presets that go through the rng of the underlying builder use this seed, so
# the same preset on the same flake always gives the same atoms.
_SEED = 0


@dataclass(frozen=True)
class Preset:
    """One functionalisation motif."""

    key: str
    name: str
    family: Literal["", "N", "O"]
    placement: Placement
    apply: Callable[[Atoms, Position, Optional[EdgeKind]], Atoms]
    note: str = ""


def _turn_away(atoms: Atoms, anchor: int, n_added: int, step_deg: float = 15.0) -> Atoms:
    """Rotate a just-attached group about its bond to keep clear of its neighbours.

    The group library places a group in a fixed local frame, which knows
    nothing of what is next to the site. On an armchair edge that frame can
    put the H of an -OH 0.6 Å from the neighbouring edge hydrogen. Rotating
    about the anchor-to-first-atom bond changes nothing chemical (it is a
    free torsion) and picks the orientation farthest from everything else;
    among equally good ones, the first found, so the result is reproducible.
    """
    positions = atoms.get_positions()
    first = len(atoms) - n_added
    moving = np.arange(first + 1, len(atoms))
    if len(moving) == 0:
        return atoms
    others = np.array([i for i in range(first) if i != anchor])
    axis = positions[first] - positions[anchor]
    axis /= np.linalg.norm(axis)
    pivot = positions[first]

    def rotated(angle: float) -> np.ndarray:
        # Rodrigues' formula about ``axis`` through ``pivot``.
        v = positions[moving] - pivot
        c, s = np.cos(angle), np.sin(angle)
        return pivot + v * c + np.cross(axis, v) * s + np.outer(v @ axis, axis) * (1 - c)

    best, best_gap = positions[moving], -1.0
    for angle in np.radians(np.arange(0.0, 360.0, step_deg)):
        trial = rotated(angle)
        gap = float(np.min(np.linalg.norm(trial[:, None] - positions[others][None], axis=2)))
        if gap > best_gap + 1e-6:
            best, best_gap = trial, gap
    out = atoms.copy()
    out.info = atoms.info
    new = out.get_positions()
    new[moving] = best
    out.set_positions(new)
    return out


def _edge_group(group_key: str):
    """Replace the H of an edge carbon with ``group_key``."""

    def apply(atoms: Atoms, position: Position, edge: Optional[EdgeKind]) -> Atoms:
        site = pick_edge_site(atoms, position or "middle", edge=edge)
        out, carbon = strip_hydrogen(atoms, site)
        anchor = next(s for s in find_sites(out, kind="edge") if s.index == carbon)
        before = len(out)
        out = attach_group(out, anchor, group_key)
        out = _turn_away(out, carbon, len(out) - before)
        out.info["functionalization"][-1]["edge"] = site.edge
        return out

    return apply


def _pristine(atoms: Atoms, position: Position, edge: Optional[EdgeKind]) -> Atoms:
    out = atoms.copy()
    out.info = {**atoms.info}
    return out


def _graphitic(atoms: Atoms, position: Position, edge: Optional[EdgeKind]) -> Atoms:
    index = pick_interior_carbon(atoms, position or "center", edge=edge)
    return make_graphitic_n(atoms, indices=[index])


def _pyridinic_edge(atoms: Atoms, position: Position, edge: Optional[EdgeKind]) -> Atoms:
    site = pick_edge_site(atoms, position or "middle", edge=edge)
    out, carbon = strip_hydrogen(atoms, site)
    out = substitute_atoms(out, [carbon], "N")
    out.info.setdefault("nitrogen_configurations", []).append(
        {"type": "pyridinic_edge", "indices": [carbon], "edge": site.edge}
    )
    return out


def _pyridinic_vacancy(atoms: Atoms, position: Position, edge: Optional[EdgeKind]) -> Atoms:
    index = pick_interior_carbon(atoms, position or "center", edge=edge)
    return make_pyridinic_n(atoms, n_per_vacancy=3, sites=[index], seed=_SEED)


def _pyrrolic(atoms: Atoms, position: Position, edge: Optional[EdgeKind]) -> Atoms:
    index = pick_interior_carbon(atoms, position or "center", edge=edge)
    return make_pyrrolic_like(atoms, sites=[index], seed=_SEED)


def _pyridinic_n_oxide(atoms: Atoms, position: Position, edge: Optional[EdgeKind]) -> Atoms:
    out = _pyridinic_edge(atoms, position, edge)
    nitrogen = out.info["nitrogen_configurations"][-1]["indices"][0]
    anchor = next(s for s in find_sites(out, kind="edge", element="N") if s.index == nitrogen)
    out = attach_group(out, anchor, "O")
    out.info["nitrogen_configurations"][-1]["type"] = "pyridinic_n_oxide"
    return out


def _epoxide(atoms: Atoms, position: Position, edge: Optional[EdgeKind]) -> Atoms:
    interior = set(interior_carbons(atoms))
    first = pick_interior_carbon(atoms, position or "center", edge=edge)
    centre = atoms.get_positions()[first]
    pairs = [
        pair for pair in find_bridge_sites(atoms)
        if pair[0] in interior and pair[1] in interior and first in pair[:2]
    ]
    if not pairs:
        raise ValueError(f"El carbono {first} no tiene un vecino interior para el epóxido.")
    pair = min(pairs, key=lambda p: (float(((p[2] - centre) ** 2).sum()), p[0], p[1]))
    return attach_bridging_group(atoms, pair, "epoxy")


#: Every preset, keyed by the name the CLI and the GUI use.
PRESETS: dict[str, Preset] = {
    preset.key: preset
    for preset in (
        Preset("pristine", "Sin funcionalizar", "", "none", _pristine),
        Preset(
            "graphitic", "N grafítico (cuaternario)", "N", "interior", _graphitic,
            "Sustituye un C interior. Añade un electrón: sistema de capa abierta.",
        ),
        Preset(
            "pyridinic_edge", "N piridínico de borde", "N", "edge", _pyridinic_edge,
            "Sustituye un C-H del borde por N con su par libre en el plano. Capa cerrada.",
        ),
        Preset(
            "pyridinic_vacancy", "N piridínico en vacante (N3V)", "N", "interior",
            _pyridinic_vacancy,
            "Monovacante con sus tres carbonos de borde cambiados por N.",
        ),
        Preset(
            "pyrrolic_precursor", "Precursor de N pirrólico", "N", "interior", _pyrrolic,
            "Divacante con un N-H. El pentágono solo aparece al relajar: hay que comprobarlo.",
        ),
        Preset("amine", "Amina (-NH2)", "N", "edge", _edge_group("NH2")),
        Preset("nitrile", "Nitrilo (-C≡N)", "N", "edge", _edge_group("CN")),
        Preset(
            "pyridinic_n_oxide", "N-óxido piridínico", "N", "edge", _pyridinic_n_oxide,
            "N piridínico de borde con un O sobre el N.",
        ),
        Preset("hydroxyl", "Hidroxilo (-OH)", "O", "edge", _edge_group("OH")),
        Preset("carboxyl", "Carboxilo (-COOH)", "O", "edge", _edge_group("COOH")),
        Preset(
            "carbonyl", "Carbonilo (C=O)", "O", "edge", _edge_group("O"),
            "Un C=O aislado en el borde deja un electrón desapareado: capa abierta.",
        ),
        Preset(
            "epoxide", "Epóxido (C-O-C)", "O", "bridge", _epoxide,
            "Puente sobre dos C interiores: los vuelve sp3 y arruga la lámina.",
        ),
    )
}


def get_preset(key: str) -> Preset:
    """Look up a preset, with an error listing the options."""
    try:
        return PRESETS[key]
    except KeyError:
        raise ValueError(
            f"Preset desconocido: '{key}'. Disponibles: {', '.join(PRESETS)}."
        ) from None


def apply_preset(
    atoms: Atoms,
    key: str,
    position: Position = None,
    edge: Optional[EdgeKind] = None,
) -> Atoms:
    """Apply one preset to a terminated flake and return a new structure.

    Parameters
    ----------
    atoms
        Finite, hydrogen-terminated flake, e.g. from
        :func:`carbonforge.builders.build_finite_nanoribbon`. Not mutated.
    key
        Preset name; see :data:`PRESETS`.
    position
        Where to put it. ``None`` uses the preset's default: the middle of a
        long edge for edge presets, the centre of the flake for interior and
        bridge presets. Edge presets also accept ``"middle"``; interior ones
        ``"center"`` or ``"near_edge"``. An integer picks an atom index.
    edge
        For edge presets, which edge type to use (defaults to the ribbon's
        own). For ``near_edge``, which edge to stand next to.

    Returns
    -------
    ase.Atoms
        Re-boxed with the flake's original vacuum per side, so a group that
        sticks out of the edge does not eat into the padding.
    """
    preset = get_preset(key)
    if any(atoms.get_pbc()):
        raise ValueError(
            "Los presets de vibspec son para cintas finitas (sin periodicidad). "
            "Usa build_finite_nanoribbon()."
        )
    out = preset.apply(atoms, position, edge)
    rebox(out, float(atoms.info.get("vacuum_per_side", DEFAULT_VACUUM_PER_SIDE)))
    out.info["vibspec_preset"] = {
        "key": key,
        "position": position,
        "edge": edge,
    }
    return out


def describe_presets() -> str:
    """A table of the presets, for the CLI."""
    lines = [f"{'clave':20s} {'sitio':9s} nombre", "-" * 64]
    for preset in PRESETS.values():
        lines.append(f"{preset.key:20s} {preset.placement:9s} {preset.name}")
        if preset.note:
            lines.append(f"{'':31s}{preset.note}")
    return "\n".join(lines)
