"""Nitrogen configurations in carbon lattices.

"N-doped graphene" is not one thing. Experimentally (and in XPS, where the
N 1s binding energy separates them) at least four distinct configurations
coexist, and they have different electronic effects:

===================  ==================  ==============================
Configuration        N coordination      Character
===================  ==================  ==============================
Grafítico (cuaternario)   3              Sustituye un C del plano; dona
                                         electrones, desplaza E_F hacia
                                         arriba (dopaje tipo n).
Piridínico                2              N en el borde de una vacante o
                                         del borde; aporta 1 electrón π y
                                         deja un par libre en el plano.
Pirrólico                 2 (+H)         N en un anillo de 5 miembros.
N-óxido piridínico        2 (+O)         Piridínico con O unido al N.
===================  ==================  ==============================

The distinction matters: a paper reporting "5 % N" without saying which
configurations is reporting almost nothing, because their catalytic and
electronic behaviour differs.

Graphitic N is a plain substitution, so it reuses
:mod:`carbonforge.dopants`. Pyridinic N requires a vacancy first. Pyrrolic N
requires a **five-membered ring**, which is a topological change no simple
substitution produces: see :func:`make_pyrrolic_like` for what this module
can and cannot do there.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from ase import Atoms

from ..defects.vacancies import introduce_vacancies
from ..dopants.substitutional import substitute_atoms
from ..topology.graph import build_bond_graph
from ..utils.rng import make_rng
from .attach import attach_group
from .sites import find_sites

#: Approximate N 1s binding energies (eV) used to identify each configuration
#: in XPS. Ranges vary by a few tenths between studies and with the
#: calibration, so these are for orientation, not for fitting your spectrum.
XPS_BINDING_ENERGY_EV: dict[str, tuple[float, float]] = {
    "pyridinic": (398.1, 398.8),
    "pyrrolic": (399.8, 400.5),
    "graphitic": (400.9, 401.7),
    "oxide": (402.0, 403.5),
}


def make_graphitic_n(
    atoms: Atoms,
    n_sites: int = 1,
    seed: Optional[int] = None,
) -> Atoms:
    """Substitute basal carbons with nitrogen (quaternary / graphitic N).

    The nitrogen keeps the carbon's three bonds and donates roughly one extra
    electron to the π system, which is the n-type doping people usually mean
    by "N-doped graphene".

    Parameters
    ----------
    atoms
        Host structure.
    n_sites
        How many carbons to replace.
    seed
        RNG seed.

    Returns
    -------
    ase.Atoms
    """
    if n_sites <= 0:
        raise ValueError("n_sites debe ser >= 1.")

    basal = [site.index for site in find_sites(atoms, kind="basal")]
    if len(basal) < n_sites:
        raise ValueError(
            f"Se pidieron {n_sites} nitrógenos grafíticos pero solo hay "
            f"{len(basal)} carbonos con coordinación 3."
        )
    rng = make_rng(seed)
    chosen = rng.choice(basal, size=n_sites, replace=False)
    out = substitute_atoms(atoms, chosen.tolist(), "N")
    out.info.setdefault("nitrogen_configurations", []).append(
        {"type": "graphitic", "n": n_sites, "indices": sorted(map(int, chosen)),
         "seed": seed}
    )
    return out


def make_pyridinic_n(
    atoms: Atoms,
    n_defects: int = 1,
    n_per_vacancy: int = 1,
    seed: Optional[int] = None,
    min_separation: float = 6.0,
) -> Atoms:
    """Create pyridinic nitrogen: a vacancy whose rim carbons become N.

    Removing one carbon leaves three two-coordinated neighbours around a
    hole. Replacing one, two or three of them with nitrogen gives the
    pyridinic configuration, in which each N sits in a six-membered ring with
    two C neighbours and a lone pair pointing into the vacancy.

    Parameters
    ----------
    atoms
        Host structure.
    n_defects
        How many vacancies to create.
    n_per_vacancy
        How many rim carbons per vacancy to turn into nitrogen (1-3).
        Three gives the well-studied "pyridinic N3" trivacancy-like motif.
    seed
        RNG seed.
    min_separation
        Minimum distance between vacancies, in Å.

    Returns
    -------
    ase.Atoms

    Notes
    -----
    The rim atoms keep the positions they had in the pristine lattice. Real
    pyridinic sites relax outward because the C-N bond is shorter than C-C
    and the hole reconstructs, so relax before computing anything
    quantitative.
    """
    if not 1 <= n_per_vacancy <= 3:
        raise ValueError("n_per_vacancy debe estar entre 1 y 3.")

    # Record which atoms neighbour each site before removing anything, since
    # indices shift once atoms are deleted.
    graph = build_bond_graph(atoms)
    rng = make_rng(seed)

    defective = introduce_vacancies(
        atoms, n_defects=n_defects, kind="mono", seed=seed,
        min_separation=min_separation,
    )
    removed = set(defective.info["defects"][-1]["removed_indices"])

    # Map old indices to new ones after the removal.
    remaining = [i for i in range(len(atoms)) if i not in removed]
    old_to_new = {old: new for new, old in enumerate(remaining)}

    rim_new: list[int] = []
    for old_index in removed:
        neighbours = [
            old_to_new[n]
            for n in graph.neighbors(old_index)
            if n in old_to_new
        ]
        if not neighbours:
            continue
        take = min(n_per_vacancy, len(neighbours))
        chosen = rng.choice(neighbours, size=take, replace=False)
        rim_new.extend(int(i) for i in chosen)

    if not rim_new:
        raise RuntimeError(
            "La vacante no dejó carbonos de borde utilizables. ¿La estructura "
            "es demasiado pequeña?"
        )

    out = substitute_atoms(defective, rim_new, "N")
    out.info.setdefault("nitrogen_configurations", []).append(
        {
            "type": "pyridinic",
            "n_vacancies": n_defects,
            "n_per_vacancy": n_per_vacancy,
            "indices": sorted(rim_new),
            "seed": seed,
        }
    )
    return out


def make_pyrrolic_like(
    atoms: Atoms,
    n_defects: int = 1,
    seed: Optional[int] = None,
    min_separation: float = 6.0,
) -> Atoms:
    """Build a **precursor** to pyrrolic nitrogen, not the finished motif.

    True pyrrolic N sits in a five-membered ring bearing an N-H. Getting
    there from a hexagonal lattice needs a genuine topological change: two
    carbons are removed and the remaining rim reconstructs a pentagon. That
    reconstruction is driven by energy minimisation, not by geometry, so this
    function does what can be done honestly without a calculator:

    1. creates a divacancy,
    2. turns one rim carbon into nitrogen,
    3. adds the N-H hydrogen.

    The result has the right composition and connectivity *neighbourhood* for
    a pyrrolic site but still six-membered rings. **It must be relaxed** with
    a real force field or DFT before it becomes pyrrolic; check the ring
    statistics afterwards with
    :func:`carbonforge.topology.ring_statistics` to confirm a pentagon
    actually formed.

    The name says "like" for that reason: calling it pyrrolic before the
    relaxation would be a lie about what the file contains.
    """
    rng = make_rng(seed)
    graph = build_bond_graph(atoms)

    defective = introduce_vacancies(
        atoms, n_defects=n_defects, kind="di", seed=seed,
        min_separation=min_separation,
    )
    removed = set(defective.info["defects"][-1]["removed_indices"])
    remaining = [i for i in range(len(atoms)) if i not in removed]
    old_to_new = {old: new for new, old in enumerate(remaining)}

    rim: list[int] = []
    for old_index in removed:
        rim.extend(
            old_to_new[n] for n in graph.neighbors(old_index) if n in old_to_new
        )
    rim = sorted(set(rim))
    if not rim:
        raise RuntimeError("La divacante no dejó borde utilizable.")

    chosen = [int(rng.choice(rim))]
    out = substitute_atoms(defective, chosen, "N")

    # Add the N-H that defines the pyrrolic motif.
    nitrogen_sites = [
        site for site in find_sites(out, kind="edge", element="N")
        if site.index in chosen
    ]
    for site in nitrogen_sites:
        out = attach_group(out, site, "H")

    out.info.setdefault("nitrogen_configurations", []).append(
        {
            "type": "pyrrolic_precursor",
            "n_defects": n_defects,
            "indices": chosen,
            "seed": seed,
            "warning": (
                "Precursor sin relajar: todavía tiene anillos de 6, no de 5. "
                "Relaja y comprueba ring_statistics."
            ),
        }
    )
    return out


def make_pyridinic_n_oxide(
    atoms: Atoms,
    n_defects: int = 1,
    seed: Optional[int] = None,
) -> Atoms:
    """Pyridinic nitrogen carrying an oxygen on the N (pyridinic N-oxide).

    Built as a pyridinic site plus a carbonyl-style oxygen on the nitrogen.
    This is the configuration around 402-403 eV in N 1s XPS.
    """
    out = make_pyridinic_n(atoms, n_defects=n_defects, n_per_vacancy=1, seed=seed)
    nitrogen_indices = set(out.info["nitrogen_configurations"][-1]["indices"])

    oxidised = 0
    for site in find_sites(out, kind="edge", element="N"):
        if site.index in nitrogen_indices:
            out = attach_group(out, site, "O")
            oxidised += 1
            break  # one N-oxide per call keeps the composition predictable

    if oxidised == 0:
        raise RuntimeError(
            "No se encontró un nitrógeno piridínico con valencia libre para "
            "oxidar."
        )
    out.info["nitrogen_configurations"][-1]["type"] = "pyridinic_n_oxide"
    return out


def nitrogen_report(atoms: Atoms) -> str:
    """Summarise the nitrogen content and configurations of a structure."""
    symbols = atoms.get_chemical_symbols()
    n_total = sum(1 for s in symbols if s == "N")
    n_carbon = sum(1 for s in symbols if s == "C")
    if n_total == 0:
        return "La estructura no contiene nitrógeno."

    graph = build_bond_graph(atoms)
    by_coordination: dict[int, int] = {}
    for index, symbol in enumerate(symbols):
        if symbol != "N":
            continue
        degree = graph.degree[index] if index in graph else 0
        by_coordination[degree] = by_coordination.get(degree, 0) + 1

    lines = [
        f"Nitrógeno: {n_total} átomos "
        f"({100.0 * n_total / (n_total + n_carbon):.2f} % at. respecto a C+N)",
        "",
        "Por coordinación:",
    ]
    interpretation = {
        1: "colgante (revisa la estructura)",
        2: "compatible con piridínico o pirrólico",
        3: "compatible con grafítico (sustitucional)",
        4: "coordinación 4: no es físico en estos materiales",
    }
    for degree in sorted(by_coordination):
        lines.append(
            f"  {degree}: {by_coordination[degree]} átomo(s) — "
            f"{interpretation.get(degree, 'inesperada')}"
        )

    recorded = atoms.info.get("nitrogen_configurations", [])
    if recorded:
        lines.append("\nConfiguraciones creadas por carbonforge:")
        for entry in recorded:
            lines.append(f"  • {entry['type']}: {len(entry.get('indices', []))} N")
            if "warning" in entry:
                lines.append(f"    ⚠️  {entry['warning']}")

    lines.append(
        "\nLa coordinación sugiere la configuración pero no la demuestra: "
        "para distinguir piridínico de pirrólico hay que mirar el tamaño del "
        "anillo (ring_statistics) y, en el experimento, el N 1s de XPS."
    )
    return "\n".join(lines)
