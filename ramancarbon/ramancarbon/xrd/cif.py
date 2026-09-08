"""Reading (and writing) Crystallographic Information Files.

CIF is how the Crystallography Open Database distributes structures, so
being able to read one is what lets a user add a reference phase by
downloading a file instead of editing code. The parser handles what real
COD files contain: `data_` blocks, `loop_` tables, quoted and
semicolon-delimited text fields, and numeric values with a standard
uncertainty in parentheses (`3.7734(2)`).

Two decisions in here are worth stating.

The symmetry operations listed in the file are **closed under
composition** rather than used verbatim (see
:mod:`ramancarbon.xrd.symmetry`). For a complete COD file this changes
nothing and costs nothing; for a truncated or hand-written one it repairs
the list instead of silently generating half a structure.

A file with **no symmetry loop at all** is read as P1, and that is a real
risk rather than a convenience — a file listing only an asymmetric unit and
naming its space group in `_symmetry_space_group_name_H-M` without the
operations would be read as containing far too few atoms. So that case is
not silent: the reader records a warning on the structure saying the space
group was named but its operations were absent, and the intensity
calculation will be wrong.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

from .structure import Crystal, Lattice, Site, StructureError
from .symmetry import SymmetryError

#: Tags that hold a symmetry operation list, oldest spelling last.
SYMOP_TAGS = (
    "_space_group_symop_operation_xyz",
    "_symmetry_equiv_pos_as_xyz",
    "_space_group_symop.operation_xyz",
)

_NUMBER = re.compile(r"^([+-]?[\d.]+(?:[eE][+-]?\d+)?)(?:\((\d+)\))?$")


class CIFError(ValueError):
    """Raised when a CIF cannot be read as a structure."""


def _number(text: str, default: Optional[float] = None) -> float:
    """Parse ``3.7734(2)`` or ``0.25`` or ``.`` (unknown)."""
    cleaned = text.strip().strip("'\"")
    if cleaned in {".", "?", ""}:
        if default is None:
            raise CIFError(f"valor numérico ausente y sin valor por defecto")
        return default
    match = _NUMBER.match(cleaned)
    if not match:
        raise CIFError(f"no se puede leer {text!r} como número")
    return float(match.group(1))


def _tokenise(line: str) -> list[str]:
    """Split a CIF data line, respecting quotes."""
    tokens: list[str] = []
    current = ""
    quote: Optional[str] = None
    for character in line:
        if quote:
            if character == quote:
                tokens.append(current)
                current = ""
                quote = None
            else:
                current += character
        elif character in "'\"":
            if current:
                tokens.append(current)
                current = ""
            quote = character
        elif character.isspace():
            if current:
                tokens.append(current)
                current = ""
        else:
            current += character
    if current:
        tokens.append(current)
    return tokens


def _lines(text: str) -> Iterator[str]:
    """Yield logical lines, folding semicolon-delimited text fields.

    A multi-line ``;`` field becomes one token so a long note in the middle
    of a file cannot be mistaken for data.
    """
    inside = False
    buffer: list[str] = []
    for raw in text.splitlines():
        if raw.startswith(";"):
            if inside:
                yield "'" + " ".join(buffer).replace("'", "") + "'"
                buffer = []
                inside = False
            else:
                inside = True
                rest = raw[1:].strip()
                if rest:
                    buffer.append(rest)
            continue
        if inside:
            buffer.append(raw.strip())
            continue
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield stripped
    if inside:  # pragma: no cover - malformed file
        yield "'" + " ".join(buffer) + "'"


def parse_cif(text: str) -> list[dict]:
    """Split CIF text into one dictionary per ``data_`` block.

    Loop columns become lists under their tag; scalars become strings.
    """
    blocks: list[dict] = []
    current: Optional[dict] = None
    stream = list(_lines(text))
    index = 0
    while index < len(stream):
        line = stream[index]
        lowered = line.lower()
        if lowered.startswith("data_"):
            current = {"_block": line[5:]}
            blocks.append(current)
            index += 1
            continue
        if current is None:
            current = {"_block": ""}
            blocks.append(current)
        if lowered == "loop_":
            index += 1
            tags: list[str] = []
            while index < len(stream) and stream[index].startswith("_"):
                tags.append(stream[index].split()[0].lower())
                index += 1
            columns: list[list[str]] = [[] for _ in tags]
            row: list[str] = []
            while index < len(stream):
                candidate = stream[index]
                if candidate.startswith("_") or candidate.lower() in ("loop_",) or \
                        candidate.lower().startswith("data_"):
                    break
                row.extend(_tokenise(candidate))
                index += 1
                while len(row) >= len(tags):
                    for position in range(len(tags)):
                        columns[position].append(row[position])
                    row = row[len(tags):]
            for tag, column in zip(tags, columns):
                current[tag] = column
            continue
        if line.startswith("_"):
            tokens = _tokenise(line)
            tag = tokens[0].lower()
            if len(tokens) > 1:
                current[tag] = " ".join(tokens[1:])
                index += 1
            else:
                index += 1
                if index < len(stream) and not stream[index].startswith("_"):
                    current[tag] = _tokenise(stream[index])[0] \
                        if _tokenise(stream[index]) else ""
                    index += 1
                else:
                    current[tag] = ""
            continue
        index += 1
    return blocks


def _first(block: dict, *tags: str, default=None):
    for tag in tags:
        if tag in block:
            value = block[tag]
            return value[0] if isinstance(value, list) and value else value
    return default


def crystal_from_block(block: dict, name: str = "") -> Crystal:
    """Build a :class:`~ramancarbon.xrd.structure.Crystal` from one block."""
    try:
        lattice = Lattice(
            a=_number(str(_first(block, "_cell_length_a", default=""))),
            b=_number(str(_first(block, "_cell_length_b", default=""))),
            c=_number(str(_first(block, "_cell_length_c", default=""))),
            alpha=_number(str(_first(block, "_cell_angle_alpha", default="90")), 90.0),
            beta=_number(str(_first(block, "_cell_angle_beta", default="90")), 90.0),
            gamma=_number(str(_first(block, "_cell_angle_gamma", default="90")), 90.0),
        )
    except (CIFError, StructureError) as exc:
        raise CIFError(f"celda ilegible: {exc}") from exc

    operations: list[str] = []
    for tag in SYMOP_TAGS:
        if tag in block:
            value = block[tag]
            operations = list(value) if isinstance(value, list) else [value]
            break

    labels = block.get("_atom_site_label", [])
    types = block.get("_atom_site_type_symbol", labels)
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(types, str):
        types = [types]
    xs = block.get("_atom_site_fract_x", [])
    ys = block.get("_atom_site_fract_y", [])
    zs = block.get("_atom_site_fract_z", [])
    if isinstance(xs, str):
        xs, ys, zs = [xs], [ys], [zs]
    if not xs:
        raise CIFError("el archivo no contiene posiciones atómicas")
    occupancies = block.get("_atom_site_occupancy", ["1.0"] * len(xs))
    if isinstance(occupancies, str):
        occupancies = [occupancies]
    u_iso = block.get(
        "_atom_site_u_iso_or_equiv", block.get("_atom_site_uiso_or_equiv", [])
    )
    b_iso = block.get("_atom_site_b_iso_or_equiv", [])
    if isinstance(u_iso, str):
        u_iso = [u_iso]
    if isinstance(b_iso, str):
        b_iso = [b_iso]

    sites: list[Site] = []
    for index in range(len(xs)):
        element = str(types[index]) if index < len(types) else str(labels[index])
        if index < len(u_iso) and str(u_iso[index]) not in (".", "?", ""):
            displacement = _number(str(u_iso[index]), 0.005)
        elif index < len(b_iso) and str(b_iso[index]) not in (".", "?", ""):
            displacement = _number(str(b_iso[index]), 0.4) / (8.0 * 3.141592653589793**2)
        else:
            displacement = 0.005
        sites.append(
            Site(
                element=element,
                fract=(
                    _number(str(xs[index])),
                    _number(str(ys[index])),
                    _number(str(zs[index])),
                ),
                occupancy=_number(
                    str(occupancies[index]) if index < len(occupancies) else "1.0", 1.0
                ),
                u_iso=max(0.0, displacement),
                label=str(labels[index]) if index < len(labels) else element,
            )
        )

    group_name = str(
        _first(
            block,
            "_space_group_name_h-m_alt",
            "_symmetry_space_group_name_h-m",
            "_space_group_name_h-m",
            default="P1",
        )
    ).strip()

    notes = ""
    if not operations:
        if group_name and group_name.replace(" ", "").upper() not in {"P1", "P-1?"}:
            notes = (
                f"AVISO: el archivo declara el grupo espacial {group_name} pero NO "
                "trae la lista de operaciones de simetría. Se ha leído como P1, "
                "así que la celda contiene solo los átomos escritos y las "
                "intensidades calculadas serán incorrectas. Descarga el CIF "
                "completo (los de la COD siempre traen "
                "_symmetry_equiv_pos_as_xyz)"
            )
        operations = ["x,y,z"]

    try:
        crystal = Crystal(
            name=name or str(_first(block, "_chemical_name_mineral",
                                    "_chemical_name_systematic", default=""))
            or block.get("_block", "sin nombre"),
            lattice=lattice,
            sites=sites,
            operations=operations,
            space_group=group_name or "P1",
            formula=str(
                _first(block, "_chemical_formula_sum", "_chemical_formula_structural",
                       default="")
            ).replace("'", "").strip(),
            source=str(_first(block, "_journal_name_full", "_publ_section_title",
                              default="")).strip()[:200],
            confidence="unknown",
            notes=notes,
        )
    except SymmetryError as exc:
        raise CIFError(f"simetría ilegible: {exc}") from exc
    return crystal


def read_cif(path: str | Path, block: int = 0) -> Crystal:
    """Read one structure from a CIF file.

    Parameters
    ----------
    path:
        The ``.cif`` file, e.g. as downloaded from the Crystallography Open
        Database.
    block:
        Index of the ``data_`` block to read, for the multi-block files
        some depositions use.

    Returns
    -------
    Crystal
    """
    location = Path(path)
    if not location.is_file():
        raise CIFError(f"no existe el archivo {location}")
    blocks = parse_cif(location.read_text(encoding="utf-8", errors="replace"))
    if not blocks:
        raise CIFError(f"{location} no contiene ningún bloque data_")
    if block >= len(blocks):
        raise CIFError(
            f"{location} tiene {len(blocks)} bloque(s); se pidió el {block}"
        )
    crystal = crystal_from_block(blocks[block], name=location.stem)
    return crystal


def read_cif_directory(path: str | Path) -> list[Crystal]:
    """Read every ``.cif`` in a directory, skipping the unreadable ones.

    A directory of downloads is expected to contain the occasional broken
    or unrelated file. Those are skipped rather than aborting the load, and
    :func:`read_cif` can be called on one directly to see why.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise CIFError(f"{directory} no es un directorio")
    crystals: list[Crystal] = []
    for candidate in sorted(directory.glob("*.cif")):
        try:
            crystals.append(read_cif(candidate))
        except (CIFError, StructureError, SymmetryError):
            continue
    return crystals


def write_cif(crystal: Crystal, path: str | Path) -> Path:
    """Write a structure back out as a CIF.

    The symmetry is written as the *closed* operation list, so a file this
    produces is complete regardless of how the structure was built.
    """
    destination = Path(path)
    lines = [
        f"data_{crystal.name.replace(' ', '_')}",
        f"_chemical_formula_sum          '{crystal.formula or crystal.cell_formula()}'",
        f"_symmetry_space_group_name_H-M '{crystal.space_group}'",
        f"_cell_length_a                 {crystal.lattice.a:.6f}",
        f"_cell_length_b                 {crystal.lattice.b:.6f}",
        f"_cell_length_c                 {crystal.lattice.c:.6f}",
        f"_cell_angle_alpha              {crystal.lattice.alpha:.4f}",
        f"_cell_angle_beta               {crystal.lattice.beta:.4f}",
        f"_cell_angle_gamma              {crystal.lattice.gamma:.4f}",
        f"_cell_volume                   {crystal.lattice.volume:.4f}",
        "",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
    ]
    lines.extend(f"  '{op.to_xyz()}'" for op in crystal.operations)
    lines.extend([
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_U_iso_or_equiv",
    ])
    for site in crystal.sites:
        lines.append(
            f"  {site.label or site.element:<8s} {site.element:<4s} "
            f"{site.fract[0]:10.6f} {site.fract[1]:10.6f} {site.fract[2]:10.6f} "
            f"{site.occupancy:8.4f} {site.u_iso:8.5f}"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


__all__ = [
    "CIFError",
    "SYMOP_TAGS",
    "crystal_from_block",
    "parse_cif",
    "read_cif",
    "read_cif_directory",
    "write_cif",
]
