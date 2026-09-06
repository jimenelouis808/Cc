"""Analyse a structure this package did not build.

Everything the framework writes carries its own metadata: the rings it
laid down, the bond graph it derived them from, the geometry it measured
after relaxing. A file from somewhere else carries none of that, and the
whole point of this module is to say what can be recovered from
coordinates alone -- **and to keep the two apart**.

That separation is the design. Every field carries a provenance:

``recorded``
    read from ``atoms.info``, so it is what the builder actually did.
``measured``
    computed from the coordinates and true of them: a bond length, a
    span, an element count.
``inferred``
    a judgement with a rule behind it: the bonds (a distance cutoff), the
    rings (face tracing or shortest paths), the shape (thresholds on
    spans and radial spread).

A report that blurred them would be worse than no report, because
"pentagons: 12" reads identically whether the builder placed twelve
pentagons or a cutoff guessed them -- and only one of those is a fact
about the structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ..functionalize.attach import bond_pairs
from ..topology import connected_components, coordination_numbers
from ..utils.geometry import guess_bonds
from ..validation import run_basic_checks
from .rings import ring_report
from .shape import describe_shape

#: Elements whose presence means the structure should be judged as a
#: dichalcogenide rather than as sp2 carbon -- a chalcogen and something
#: that is not carbon to pair it with.
CHALCOGENS = frozenset({"S", "Se", "Te"})


def read_structure(path: str | Path, index: int = -1) -> Atoms:
    """Read any file ASE can read, as one structure.

    Parameters
    ----------
    path
        The file. Format is taken from the extension, as ASE does.
    index
        Which frame of a trajectory; ``-1`` is the last, which for a
        relaxation is the relaxed one and is what a reader almost always
        means.

    Raises
    ------
    ValueError
        With the path and the underlying reason, since "could not read"
        is nearly always a wrong extension or a truncated file and the
        traceback from deep inside a parser says neither.
    """
    from ase import io

    path = Path(path)
    if not path.exists():
        raise ValueError(f"No such file: {path}")
    try:
        atoms = io.read(str(path), index=index)
    except Exception as exc:
        raise ValueError(
            f"Could not read {path} as a structure ({exc}). The format is "
            "taken from the extension; ASE reads xyz, extxyz, cif, POSCAR, "
            "pdb, cube, and the outputs of most codes."
        ) from None
    if isinstance(atoms, list):
        atoms = atoms[index]
    return atoms


def composition(atoms: Atoms) -> dict[str, Any]:
    """Element counts and fractions, plus the reduced formula."""
    symbols = atoms.get_chemical_symbols()
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    total = max(1, len(symbols))
    return {
        "formula": atoms.get_chemical_formula(),
        "counts": dict(sorted(counts.items())),
        "fractions": {element: round(count / total, 4)
                      for element, count in sorted(counts.items())},
        "n_atoms": len(atoms),
        "n_elements": len(counts),
    }


def bond_statistics(atoms: Atoms, tolerance: float = 0.30) -> dict[str, Any]:
    """Bond lengths, split by element pair.

    Split because a mixed structure has no single "bond length" worth
    quoting: an MX2 with a grafted thiol has 2.4 Å M-X bonds and 1.3 Å
    S-H ones, and their mean is a number describing nothing.
    """
    found = guess_bonds(atoms, tolerance=tolerance)
    symbols = atoms.get_chemical_symbols()
    per_pair: dict[str, list[float]] = {}
    for first, second, distance in found:
        key = "-".join(sorted((symbols[first], symbols[second])))
        per_pair.setdefault(key, []).append(distance)

    summary = {}
    for key, lengths in sorted(per_pair.items()):
        values = np.asarray(lengths)
        summary[key] = {
            "count": int(values.size),
            "min": round(float(values.min()), 3),
            "mean": round(float(values.mean()), 3),
            "max": round(float(values.max()), 3),
            "std": round(float(values.std()), 4),
        }
    return {"n_bonds": len(found), "by_pair": summary}


def coordination_census(atoms: Atoms, tolerance: float = 0.30) -> dict:
    """How many neighbours each element has, as a histogram per element."""
    numbers = coordination_numbers(atoms, tolerance=tolerance)
    symbols = np.array(atoms.get_chemical_symbols())
    census: dict[str, dict[int, int]] = {}
    for element in sorted(set(symbols)):
        values = numbers[symbols == element]
        histogram: dict[int, int] = {}
        for value in values:
            histogram[int(value)] = histogram.get(int(value), 0) + 1
        census[element] = dict(sorted(histogram.items()))
    return {
        "per_element": census,
        "mean": round(float(numbers.mean()), 3) if len(numbers) else 0.0,
    }


def _recorded(atoms: Atoms) -> dict[str, Any]:
    """What the file itself claims, as opposed to what we measured."""
    interesting = (
        "builder", "mode", "ring_counts", "euler", "genus", "geometry",
        "material", "metal", "chalcogen", "phase", "bond_length",
        "tube_radius", "chirality", "n_shells", "shell_radii", "dopants",
        "doping_mode", "functionalization", "defect_log", "twist_angle",
    )
    present = {key: atoms.info[key] for key in interesting if key in atoms.info}
    present["has_bond_graph"] = "bonds" in atoms.info
    present["has_ring_list"] = "rings" in atoms.info
    return present


def _verdict(atoms: Atoms, shape: dict, rings: dict,
             bonds: dict) -> dict[str, Any]:
    """A judgement on whether the geometry is physically sensible.

    Uses whichever quality model fits the composition -- the MX2 one for
    a dichalcogenide, the sp2 one for carbon -- because their thresholds
    are not interchangeable: judging a six-coordinate metal by carbon's
    "five or more is unphysical" rejects every correct MoS2 cell.
    """
    symbols = set(atoms.get_chemical_symbols())
    is_mx2 = bool(symbols & CHALCOGENS) and bool(symbols - CHALCOGENS - {"C"})

    if is_mx2:
        from ..tmd.quality import geometry_report, tmd_quality

        if "bond_length" not in atoms.info:
            # geometry_report needs an ideal bond to compare against, and
            # a foreign file has none. Take the measured M-X mode rather
            # than a default, so the verdict is against this structure's
            # own chemistry instead of MoS2's.
            metal_bonds = [stats["mean"] for pair, stats
                           in bonds["by_pair"].items()
                           if any(part in CHALCOGENS for part in pair.split("-"))
                           and not all(part in CHALCOGENS
                                       for part in pair.split("-"))]
            atoms = atoms.copy()
            atoms.info.setdefault("metal", "M")
            atoms.info.setdefault("chalcogen", "X")
            atoms.info["bond_length"] = (round(float(np.mean(metal_bonds)), 3)
                                         if metal_bonds else 2.4)
        try:
            report = geometry_report(atoms)
        except ValueError as exc:
            return {"model": "dichalcogenide", "verdict": "unknown",
                    "reason": str(exc)}
        verdict, reason = tmd_quality(
            report, expect_stoichiometric=False,
        )
        return {"model": "dichalcogenide", "verdict": verdict,
                "reason": reason, "measured": report}

    from ..validation.quality import sp2_quality

    if "geometry" in atoms.info:
        verdict, reason = sp2_quality(atoms.info["geometry"])
        return {"model": "sp2 carbon", "verdict": verdict, "reason": reason,
                "source": "recorded"}

    carbon = bonds["by_pair"].get("C-C")
    if not carbon:
        return {"model": "none", "verdict": "unknown",
                "reason": "no carbon-carbon bonds and no dichalcogenide, so "
                          "neither quality model applies. The validation "
                          "report above still holds."}
    measured = {
        "bond_min": carbon["min"], "bond_mean": carbon["mean"],
        "bond_max": carbon["max"], "bond_std": carbon["std"],
        "angle_min": float("nan"), "angle_mean": float("nan"),
        "angle_max": float("nan"), "n_close_contacts": 0,
    }
    verdict, reason = sp2_quality(measured)
    return {
        "model": "sp2 carbon", "verdict": verdict, "reason": reason,
        "source": "measured",
        "caveat": "Judged on bond lengths alone. The builders' verdict also "
                  "weighs bond angles and close contacts, which are recorded "
                  "at build time and cannot be recovered from coordinates "
                  "without assuming the bond graph is right.",
    }


def analyse(atoms: Atoms, tolerance: float = 0.30,
            max_ring_size: int | None = None) -> dict[str, Any]:
    """Everything that can be said about a structure, sorted by provenance.

    Parameters
    ----------
    atoms
        The structure. Metadata is used where present and its absence is
        never filled in with a guess dressed as a fact.
    tolerance
        Slack on the covalent-radii sum when inferring bonds. This is the
        single assumption the rest rests on, which is why it is a
        parameter and is reported back.
    max_ring_size
        Largest ring to look for; ``None`` uses the module default.

    Returns
    -------
    dict
        ``recorded`` / ``measured`` / ``inferred``, plus ``validation``
        and ``verdict``. Nothing appears in more than one of the three.
    """
    from .rings import MAX_RING_SIZE

    pairs = bond_pairs(atoms, tolerance=tolerance)
    bonds = bond_statistics(atoms, tolerance=tolerance)
    shape = describe_shape(atoms)
    rings = ring_report(atoms, pairs,
                        max_ring_size if max_ring_size else MAX_RING_SIZE)
    components = connected_components(atoms, tolerance=tolerance)
    validation = run_basic_checks(atoms)

    cell = np.asarray(atoms.cell)
    volume = float(abs(np.linalg.det(cell))) if cell.any() else 0.0
    density = None
    if shape["dimensionality"] == 3 and volume > 0:
        # g/cm3. Only for a genuine bulk: a slab's cell is mostly vacuum,
        # so a density computed from it describes the padding.
        density = round(float(sum(atoms.get_masses()) / volume * 1.66053907), 4)

    rings_from_file = "rings" in atoms.info
    return {
        "recorded": _recorded(atoms),
        "measured": {
            "composition": composition(atoms),
            "bonds": bonds,
            "coordination": coordination_census(atoms, tolerance=tolerance),
            "cell_volume": round(volume, 3),
            "density_g_cm3": density,
            "n_components": len(components),
            "component_sizes": [len(part) for part in components[:10]],
        },
        "inferred": {
            "shape": shape,
            "rings": {key: value for key, value in rings.items()
                      if key != "rings"},
            "bond_tolerance": tolerance,
            "rings_are_recorded": rings_from_file,
        },
        "validation": {
            "ok": validation.ok,
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
            "info": dict(validation.info),
        },
        "verdict": _verdict(atoms, shape, rings, bonds),
    }


def format_report(result: dict[str, Any], name: str = "structure") -> str:
    """The analysis as text, with provenance kept visible throughout."""
    measured = result["measured"]
    inferred = result["inferred"]
    shape = inferred["shape"]
    rings = inferred["rings"]
    recorded = result["recorded"]
    lines: list[str] = []

    composition_ = measured["composition"]
    lines.append(f"{name}: {composition_['formula']}, "
                 f"{composition_['n_atoms']} atoms")
    lines.append("")

    lines.append("MEASURED  (true of the coordinates)")
    fractions = ", ".join(f"{element} {100 * value:.1f}%"
                          for element, value in composition_["fractions"].items())
    lines.append(f"  composition   {fractions}")
    for pair, stats in measured["bonds"]["by_pair"].items():
        lines.append(f"  {pair:<12s}  {stats['count']:5d} bonds  "
                     f"{stats['min']:.3f} / {stats['mean']:.3f} / "
                     f"{stats['max']:.3f} Å  (std {stats['std']:.4f})")
    for element, histogram in measured["coordination"]["per_element"].items():
        spread = ", ".join(f"{count}x{number}"
                           for number, count in histogram.items())
        lines.append(f"  {element} coordination  {spread}")
    if measured["n_components"] > 1:
        lines.append(f"  components    {measured['n_components']} disjoint "
                     f"pieces, sizes {measured['component_sizes']}")
    if measured["density_g_cm3"] is not None:
        lines.append(f"  density       {measured['density_g_cm3']} g/cm3")
    lines.append("")

    lines.append(f"INFERRED  (a rule, not a fact; bond cutoff = radii sum "
                 f"+ {inferred['bond_tolerance']} Å)")
    lines.append(f"  shape         {shape['dimensionality']}D {shape['shape']}"
                 f"  — {shape['reason']}")
    lines.append(f"  cell says     pbc {shape['declared_pbc']}, of which "
                 f"{shape['periodic_axes']} really repeat"
                 + (f"; {shape['vacuum_axes']} are vacuum"
                    if shape["vacuum_axes"] else ""))
    if inferred["rings_are_recorded"]:
        lines.append("  rings         not inferred — the file records them "
                     "(see RECORDED below)")
    elif rings["method"] == "none":
        lines.append(f"  rings         not attempted. {rings['caveat']}")
    else:
        method = ("faces of the embedded surface" if rings["method"] == "faces"
                  else "shortest-path rings")
        lines.append(f"  rings         {rings['counts']}  "
                     f"sum(6-n) = {rings['euler_deficit']:+d}   [{method}]")
        if rings["caveat"]:
            lines.append(f"                {rings['caveat']}")
    lines.append("")

    if len(recorded) > 2:
        lines.append("RECORDED  (what the file itself says)")
        for key, value in recorded.items():
            if key in ("has_bond_graph", "has_ring_list"):
                continue
            text = str(value)
            if len(text) > 96:
                text = text[:93] + "..."
            lines.append(f"  {key:<14s}{text}")
        lines.append("")
    else:
        lines.append("RECORDED  nothing: no builder metadata in this file, so "
                     "everything above is measured or inferred.")
        lines.append("")

    validation = result["validation"]
    lines.append(f"VALIDATION  {'passed' if validation['ok'] else 'FAILED'}")
    for error in validation["errors"]:
        lines.append(f"  error    {error}")
    for warning in validation["warnings"]:
        lines.append(f"  warning  {warning}")
    if validation["ok"] and not validation["warnings"]:
        lines.append("  no errors, no warnings")
    lines.append("")

    verdict = result["verdict"]
    lines.append(f"VERDICT  [{verdict['model']}]  "
                 f"{str(verdict['verdict']).upper()}: {verdict['reason']}")
    if verdict.get("caveat"):
        lines.append(f"  {verdict['caveat']}")
    return "\n".join(lines)


__all__ = [
    "CHALCOGENS",
    "analyse",
    "bond_statistics",
    "composition",
    "coordination_census",
    "format_report",
    "read_structure",
]
