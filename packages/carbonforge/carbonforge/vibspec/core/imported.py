"""Bringing your own geometry: a ribbon, its atoms and groups, from a file.

The presets cover the common motifs, but the model you want may already
exist -- drawn in Avogadro or GaussView, cut from a nanotube in
nanocarbon_lab, relaxed in an earlier calculation, or a preset you edited by
hand. :func:`load_structure` reads any format ASE knows (through
:func:`carbonforge.io.import_structure`, so the file is diagnosed on the way
in) and turns it into a model the rest of vibspec accepts:

* **No cell** (XYZ, MOL, PDB): it gets an orthorhombic box with the
  requested vacuum on every side.
* **A cell, declared periodic**: accepted only as a molecule in a box. If
  any bond crosses the cell boundary the structure is genuinely periodic,
  and it is refused with the reason -- an IR calculation by finite
  differences of the dipole needs a finite model, and quietly cutting the
  bonds would invent dangling carbons.
* **Edge type**: read from the geometry (armchair or zigzag C-H), so the
  presets' default site, the middle of a long edge, still works.

Whatever the file already carries -- dopants, groups, defects -- stays as it
is; a preset can be applied on top (:func:`carbonforge.vibspec.core.apply_preset`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ase import Atoms

from ...builders.nanoribbon import DEFAULT_VACUUM_PER_SIDE, MIN_VACUUM_PER_SIDE, rebox
from ...functionalization.nitrogen import nitrogen_report
from ...io import import_structure
from ...topology.graph import build_bond_graph
from .sites import edge_sites, zigzag_runs

#: One edge type must outnumber the other by this much to be taken as the
#: ribbon's edge; otherwise it is left for the user to say.
_EDGE_DOMINANCE = 1.5


def infer_edge(atoms: Atoms) -> tuple[Optional[str], dict[str, int]]:
    """The ribbon's edge type, if the geometry says so clearly.

    Corner carbons are left out: a two-site zigzag step is what every
    armchair corner looks like, and counting those would call a square
    armchair flake zigzag. Returns ``(edge or None, counts)``.
    """
    zigzag = sum(len(run) for run in zigzag_runs(atoms) if len(run) >= 3)
    armchair = sum(1 for site in edge_sites(atoms) if site.edge == "armchair")
    counts = {"armchair": armchair, "zigzag": zigzag}
    if armchair >= _EDGE_DOMINANCE * max(zigzag, 1) and armchair:
        return "armchair", counts
    if zigzag >= _EDGE_DOMINANCE * max(armchair, 1) and zigzag:
        return "zigzag", counts
    return None, counts

#: Extensions offered by the file dialogs; ASE reads more.
STRUCTURE_EXTENSIONS: tuple[str, ...] = (
    ".xyz", ".extxyz", ".cif", ".vasp", ".pdb", ".mol", ".sdf", ".traj", ".json", ".gen",
    ".in", ".out", ".pwo",
)


class ImportRefused(ValueError):
    """The file was read, but it cannot become a finite IR model as it is."""


def _crosses_boundary(atoms: Atoms) -> bool:
    """Whether any bond exists only through the periodic boundary."""
    periodic = atoms.copy()
    finite = atoms.copy()
    finite.pbc = False
    return build_bond_graph(periodic).number_of_edges() > build_bond_graph(finite).number_of_edges()


def load_structure(
    path: str | Path,
    vacuum_per_side: Optional[float] = None,
    index: int = -1,
) -> tuple[Atoms, str]:
    """Read a structure file and make it a finite, boxed vibspec model.

    Parameters
    ----------
    path
        Any file ASE can read. For multi-frame files (a relaxation
        trajectory, a QE output) ``index`` picks the frame; the default, the
        last, is the relaxed geometry.
    vacuum_per_side
        Vacuum around the atoms, Å. Defaults to the value stored in the file
        by vibspec, or :data:`DEFAULT_VACUUM_PER_SIDE`.
    index
        Frame to read.

    Returns
    -------
    (atoms, report)
        The model, non-periodic and re-boxed, with ``info["source_file"]``
        and an inferred ``info["edge"]``; and a report of what arrived and
        what was done, for the user to read before trusting it.

    Raises
    ------
    ImportRefused
        For a genuinely periodic structure, or too little vacuum requested.
    ValueError
        If the file cannot be read.
    """
    result = import_structure(path, index=index)
    atoms = result.atoms
    # A missing cell is expected (XYZ, MOL) and fixed below; listing it among
    # the problems that block an export would only alarm.
    no_cell = any(issue.code == "no_cell" for issue in result.issues)
    result.issues = [issue for issue in result.issues if issue.code != "no_cell"]
    lines = [result.summary()]
    if no_cell:
        lines.append("\nEl archivo no traía celda (normal en XYZ/MOL/PDB): se le pone una caja.")

    errors = [issue for issue in result.issues if issue.severity == "error"]
    if any(issue.code == "empty" for issue in errors):
        raise ImportRefused(f"{Path(path).name} no contiene ningún átomo.")
    overlapping = [issue for issue in errors if issue.code in ("overlap", "duplicates")]
    if overlapping:
        raise ImportRefused(
            "El archivo tiene átomos superpuestos o duplicados; corrígelo en el programa "
            "de origen (vibspec no mueve átomos para arreglarlo):\n"
            + "\n".join(issue.message for issue in overlapping)
        )

    if any(atoms.get_pbc()):
        if atoms.cell.rank == 3 and _crosses_boundary(atoms):
            raise ImportRefused(
                f"{Path(path).name} es periódica de verdad: hay enlaces que cruzan la celda. "
                "El IR por diferencias finitas del dipolo necesita un modelo finito. Corta un "
                "fragmento finito y termina sus bordes con H (por ejemplo con "
                "build_finite_nanoribbon o en tu editor molecular)."
            )
        lines.append("\nEl archivo declaraba periodicidad, pero ningún enlace cruza la celda: "
                     "se trata como molécula en una caja (pbc desactivado).")
        atoms.pbc = False

    if vacuum_per_side is None:
        vacuum_per_side = float(atoms.info.get("vacuum_per_side", DEFAULT_VACUUM_PER_SIDE))
    if vacuum_per_side < MIN_VACUUM_PER_SIDE:
        raise ImportRefused(
            f"vacuum_per_side={vacuum_per_side} Å: el mínimo es {MIN_VACUUM_PER_SIDE} Å."
        )
    rebox(atoms, vacuum_per_side)
    atoms.info["vacuum_per_side"] = vacuum_per_side
    atoms.info["source_file"] = str(Path(path).resolve())
    atoms.info.setdefault("structure_type", "imported")

    lines.append(
        f"\nModelo: {atoms.get_chemical_formula()}, {len(atoms)} átomos, "
        f"{vacuum_per_side} Å de vacío por lado."
    )
    if not edge_sites(atoms):
        lines.append("No hay carbonos de borde con un H: los presets de borde no se podrán usar.")
    elif "edge" in atoms.info:
        lines.append(f"Borde de la cinta (guardado en el archivo): {atoms.info['edge']}.")
    else:
        edge, counts = infer_edge(atoms)
        detail = f"{counts['armchair']} C-H armchair, {counts['zigzag']} C-H zigzag sin esquinas"
        if edge is None:
            lines.append(f"Bordes: {detail}. Ningún tipo domina: elige el tipo de borde del "
                         "sitio al aplicar un preset de borde.")
        else:
            atoms.info["edge"] = edge
            lines.append(f"Bordes: {detail}; se toma {edge} como borde de la cinta.")
    if "N" in atoms.get_chemical_symbols():
        lines.append("\n" + nitrogen_report(atoms))
    return atoms, "\n".join(lines)


def list_library(directory: str | Path) -> list[Path]:
    """Structure files in a folder of your own models, sorted by name.

    A plain folder is the library: drop an ``.xyz`` (or any listed format)
    in it and it appears in the window's list. Nothing is copied or indexed.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in STRUCTURE_EXTENSIONS)
