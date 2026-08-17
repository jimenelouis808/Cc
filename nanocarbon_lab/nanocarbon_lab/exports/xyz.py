"""Plain XYZ export, plus a companion "render bundle" for external tools.

Standard XYZ (element, x, y, z in Å) is the least common denominator most
visualization and 3D tools (Blender, OVITO, VMD, VESTA, ...) can read
directly. It carries no bond or ring information, though -- and for the
capped/defected structures built by
:mod:`nanocarbon_lab.builders.capped_cnt`, bonds and per-atom ring
membership are exactly known and worth keeping, e.g. to colour pentagons/
heptagons/octagons differently in a render. :func:`write_render_bundle`
writes the ``.xyz`` alongside a ``.json`` sidecar with that information,
consumed by the Blender import script in ``nanocarbon_lab/blender/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ase import Atoms


def write_xyz(atoms: Atoms, path: str | Path, comment: Optional[str] = None) -> Path:
    """Write a plain (non-extended) XYZ file.

    Parameters
    ----------
    atoms
        Structure to write.
    path
        Output file path (parent directories are created as needed).
    comment
        Text for the second header line. Defaults to the structure's
        ``structure_type`` (from ``atoms.info``) if set, else empty.

    Returns
    -------
    pathlib.Path
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    if comment is None:
        comment = str(atoms.info.get("structure_type", ""))
    lines = [str(len(atoms)), comment]
    for sym, (x, y, z) in zip(symbols, positions):
        lines.append(f"{sym:<2} {x:20.10f} {y:20.10f} {z:20.10f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_render_bundle(atoms: Atoms, path: str | Path) -> tuple[Path, Path]:
    """Write ``<path>.xyz`` plus a ``<path>.json`` sidecar for rendering.

    The JSON sidecar carries everything a renderer needs beyond raw
    coordinates: explicit bonds, ring membership (with each ring's size,
    so a consumer can colour pentagons/hexagons/heptagons/octagons
    differently), and a per-atom ``ring_sizes`` list (an atom on the
    honeycomb lattice sits on exactly 3 rings; an atom missing rings
    metadata -- e.g. one built by a plain :func:`nanocarbon_lab.builders.cnt.build_cnt`
    call rather than :func:`nanocarbon_lab.builders.capped_cnt.build_capped_cnt`
    -- is exported as an ``.xyz`` only, with an empty bundle.

    Parameters
    ----------
    atoms
        Structure to export. Ring/bond metadata is read from
        ``atoms.info["rings"]`` / ``atoms.info["bonds"]`` when present
        (as populated by :func:`nanocarbon_lab.builders.capped_cnt.build_capped_cnt`).
    path
        Output path **without** extension; ``.xyz`` and ``.json`` are
        appended.

    Returns
    -------
    (xyz_path, json_path)
    """
    path = Path(path)
    xyz_path = write_xyz(atoms, path.with_suffix(".xyz"))

    rings = atoms.info.get("rings", [])
    bonds = atoms.info.get("bonds", [])
    n = len(atoms)
    ring_sizes_per_atom: list[list[int]] = [[] for _ in range(n)]
    for ring in rings:
        for a in ring:
            ring_sizes_per_atom[a].append(len(ring))

    bundle = {
        "n_atoms": n,
        "bonds": bonds,
        "rings": rings,
        "ring_sizes_per_atom": ring_sizes_per_atom,
        "ring_counts": atoms.info.get("ring_counts", {}),
        "defect_log": atoms.info.get("defect_log", []),
        "structure_type": atoms.info.get("structure_type", ""),
    }
    json_path = path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(bundle, indent=2))
    return xyz_path, json_path
