"""Chemical element table and object-name -> element resolution.

Atomic Blender (``io_mesh_atomic``) names the objects it creates after the
*full element name* ("Carbon", "Nitrogen", ...) and appends role suffixes
("_mesh", "_ball", "_sticks") plus Blender's own ``.001`` duplicates. Users
who build structures by hand tend to use symbols instead ("C", "N").

This module is deliberately free of ``bpy`` so it can be unit tested outside
Blender.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Element:
    """Static data for one chemical element.

    Attributes:
        symbol: IUPAC symbol, e.g. ``"C"``.
        name: English element name as used by Atomic Blender, e.g. ``"Carbon"``.
        number: Atomic number.
        cpk: CPK/Jmol colour as an sRGB hex string (``"#rrggbb"``).
        covalent: Covalent radius in Angstrom (used for bond guessing).
        vdw: Van der Waals radius in Angstrom (used for ball scaling).
    """

    symbol: str
    name: str
    number: int
    cpk: str
    covalent: float
    vdw: float


# Curated subset: everything that realistically shows up in nanocarbon /
# 2D-material / catalysis papers. Radii in Angstrom.
_ELEMENT_TABLE: tuple[Element, ...] = (
    Element("H", "Hydrogen", 1, "#ffffff", 0.31, 1.20),
    Element("He", "Helium", 2, "#d9ffff", 0.28, 1.40),
    Element("Li", "Lithium", 3, "#cc80ff", 1.28, 1.82),
    Element("Be", "Beryllium", 4, "#c2ff00", 0.96, 1.53),
    Element("B", "Boron", 5, "#ffb5b5", 0.84, 1.92),
    Element("C", "Carbon", 6, "#909090", 0.76, 1.70),
    Element("N", "Nitrogen", 7, "#3050f8", 0.71, 1.55),
    Element("O", "Oxygen", 8, "#ff0d0d", 0.66, 1.52),
    Element("F", "Fluorine", 9, "#90e050", 0.57, 1.47),
    Element("Ne", "Neon", 10, "#b3e3f5", 0.58, 1.54),
    Element("Na", "Sodium", 11, "#ab5cf2", 1.66, 2.27),
    Element("Mg", "Magnesium", 12, "#8aff00", 1.41, 1.73),
    Element("Al", "Aluminium", 13, "#bfa6a6", 1.21, 1.84),
    Element("Si", "Silicon", 14, "#f0c8a0", 1.11, 2.10),
    Element("P", "Phosphorus", 15, "#ff8000", 1.07, 1.80),
    Element("S", "Sulfur", 16, "#ffff30", 1.05, 1.80),
    Element("Cl", "Chlorine", 17, "#1ff01f", 1.02, 1.75),
    Element("Ar", "Argon", 18, "#80d1e3", 1.06, 1.88),
    Element("K", "Potassium", 19, "#8f40d4", 2.03, 2.75),
    Element("Ca", "Calcium", 20, "#3dff00", 1.76, 2.31),
    Element("Ti", "Titanium", 22, "#bfc2c7", 1.60, 2.11),
    Element("V", "Vanadium", 23, "#a6a6ab", 1.53, 2.07),
    Element("Cr", "Chromium", 24, "#8a99c7", 1.39, 2.06),
    Element("Mn", "Manganese", 25, "#9c7ac7", 1.39, 2.05),
    Element("Fe", "Iron", 26, "#e06633", 1.32, 2.04),
    Element("Co", "Cobalt", 27, "#f090a0", 1.26, 2.00),
    Element("Ni", "Nickel", 28, "#50d050", 1.24, 1.97),
    Element("Cu", "Copper", 29, "#c88033", 1.32, 1.96),
    Element("Zn", "Zinc", 30, "#7d80b0", 1.22, 2.01),
    Element("Ga", "Gallium", 31, "#c28f8f", 1.22, 1.87),
    Element("Ge", "Germanium", 32, "#668f8f", 1.20, 2.11),
    Element("As", "Arsenic", 33, "#bd80e3", 1.19, 1.85),
    Element("Se", "Selenium", 34, "#ffa100", 1.20, 1.90),
    Element("Br", "Bromine", 35, "#a62929", 1.20, 1.85),
    Element("Mo", "Molybdenum", 42, "#54b5b5", 1.54, 2.17),
    Element("Ru", "Ruthenium", 44, "#248f8f", 1.46, 2.13),
    Element("Rh", "Rhodium", 45, "#0a7d8c", 1.42, 2.10),
    Element("Pd", "Palladium", 46, "#006985", 1.39, 2.10),
    Element("Ag", "Silver", 47, "#c0c0c0", 1.45, 2.11),
    Element("Cd", "Cadmium", 48, "#ffd98f", 1.44, 2.18),
    Element("In", "Indium", 49, "#a67573", 1.42, 1.93),
    Element("Sn", "Tin", 50, "#668080", 1.39, 2.17),
    Element("Te", "Tellurium", 52, "#d47a00", 1.38, 2.06),
    Element("I", "Iodine", 53, "#940094", 1.39, 1.98),
    Element("W", "Tungsten", 74, "#2194d6", 1.62, 2.18),
    Element("Ir", "Iridium", 77, "#175487", 1.41, 2.13),
    Element("Pt", "Platinum", 78, "#d0d0e0", 1.36, 2.13),
    Element("Au", "Gold", 79, "#ffd123", 1.36, 2.14),
    Element("Hg", "Mercury", 80, "#b8b8d0", 1.32, 2.23),
    Element("Pb", "Lead", 82, "#575961", 1.46, 2.02),
)

#: Symbol -> :class:`Element`.
ELEMENTS: dict[str, Element] = {e.symbol: e for e in _ELEMENT_TABLE}

_BY_NAME: dict[str, str] = {e.name.lower(): e.symbol for e in _ELEMENT_TABLE}
_BY_SYMBOL_LOWER: dict[str, str] = {e.symbol.lower(): e.symbol for e in _ELEMENT_TABLE}

# Alternative spellings Atomic Blender / other tools may use.
_ALIASES: dict[str, str] = {
    "aluminum": "Al",
    "sulphur": "S",
    "vacancy": "",  # explicit "not an element"
    "dummy": "",
    "x": "",
}

#: Symbol used for atoms whose element could not be resolved.
UNKNOWN_SYMBOL = "X"

#: Fallback colour (grey) for :data:`UNKNOWN_SYMBOL`.
UNKNOWN_COLOR = "#b0b0b0"

# Suffixes Atomic Blender appends to the objects/materials it creates.
_ROLE_SUFFIXES = (
    "mesh",
    "ball",
    "balls",
    "atom",
    "atoms",
    "sphere",
    "spheres",
    "nurbs",
    "mat",
    "material",
    "shape",
    "rep",
)

# Tokens that identify bond/stick geometry rather than atoms.
_STICK_TOKENS = ("stick", "sticks", "bond", "bonds", "cylinder", "cylinders")


def _tokenize(name: str) -> list[str]:
    """Split a Blender object/material name into comparable tokens.

    ``"Carbon_ball.001"`` -> ``["carbon", "ball", "001"]``. Charge decorations
    used by Atomic Blender (``"Iron_F2+"``) survive as their own token and are
    simply ignored by the lookup.
    """
    cleaned = []
    for ch in name:
        cleaned.append(ch if ch.isalnum() else " ")
    return [tok.lower() for tok in "".join(cleaned).split() if tok]


def is_stick_name(name: str) -> bool:
    """Return ``True`` when *name* looks like Atomic Blender bond geometry."""
    return any(tok in _STICK_TOKENS for tok in _tokenize(name))


def guess_element(name: str) -> str | None:
    """Resolve a Blender object/material name to an element symbol.

    Full element names win over symbols, so ``"Carbon"`` never resolves to
    calcium and ``"Nitrogen_mesh"`` never resolves to nickel. Returns ``None``
    when nothing in *name* looks like an element, which is how stick objects,
    empties and user geometry are filtered out.

    Args:
        name: Object, mesh or material name as it appears in Blender.

    Returns:
        The element symbol, or ``None`` if unresolved.
    """
    tokens = _tokenize(name)
    if not tokens or is_stick_name(name):
        return None

    # Pass 1: full element names ("carbon") and aliases.
    for tok in tokens:
        if tok in _BY_NAME:
            return _BY_NAME[tok]
        if tok in _ALIASES:
            alias = _ALIASES[tok]
            return alias or None

    # Pass 2: bare symbols, ignoring role suffixes and pure numbers.
    for tok in tokens:
        if tok in _ROLE_SUFFIXES or tok.isdigit():
            continue
        if tok in _BY_SYMBOL_LOWER:
            return _BY_SYMBOL_LOWER[tok]
    return None


def element_color(symbol: str) -> str:
    """Return the CPK hex colour for *symbol*, grey when unknown."""
    element = ELEMENTS.get(symbol)
    return element.cpk if element else UNKNOWN_COLOR


def element_label(symbol: str) -> str:
    """Return a human readable label such as ``"C - Carbon"``."""
    element = ELEMENTS.get(symbol)
    return f"{symbol} - {element.name}" if element else f"{symbol} - unknown"


def sort_symbols(symbols: object) -> list[str]:
    """Sort element symbols by atomic number, unknowns last."""
    return sorted(
        symbols,  # type: ignore[arg-type]
        key=lambda s: (ELEMENTS[s].number if s in ELEMENTS else 999, s),
    )
