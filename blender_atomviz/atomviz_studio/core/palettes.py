"""Colour palettes for atomic structures.

A palette maps element symbols to sRGB hex colours. Three resolution modes are
supported:

``table``
    Fixed per-element colours (CPK and hand-tuned editorial variants).
``accent``
    Structural elements stay neutral, everything else (dopants, adsorbates,
    metals) is coloured from a single user accent — the classic
    "grey framework + one screaming colour" cover look.
``spectral``
    Colour is derived from the atomic number along a continuous ramp, which
    keeps large multi-element systems readable.

No ``bpy`` import: unit testable outside Blender.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .colors import darken, lighten, mix_hex
from .elements import ELEMENTS, UNKNOWN_COLOR, element_color, sort_symbols

#: Elements treated as "framework" by the accent palettes.
FRAMEWORK_SYMBOLS: frozenset[str] = frozenset({"C", "H"})


@dataclass(frozen=True)
class Palette:
    """A named element -> colour mapping.

    Attributes:
        key: Stable identifier used by the UI and the CLI.
        label: Human readable name.
        description: One-line summary shown as a tooltip.
        mode: ``"table"``, ``"accent"`` or ``"spectral"``.
        colors: Explicit per-symbol colours (``mode="table"``).
        default: Colour used for symbols missing from :attr:`colors`.
        framework: Neutral colour for framework atoms (``mode="accent"``).
        dark_background: Whether the palette was designed against a dark canvas.
    """

    key: str
    label: str
    description: str
    mode: str = "table"
    colors: dict[str, str] = field(default_factory=dict)
    default: str = UNKNOWN_COLOR
    framework: str = "#8a8f98"
    dark_background: bool = True


#: Ramp used by the ``spectral`` mode, from light elements to heavy ones.
_SPECTRAL_STOPS: tuple[str, ...] = (
    "#f8fafc",
    "#38bdf8",
    "#6366f1",
    "#a855f7",
    "#ec4899",
    "#f97316",
    "#facc15",
)

_PALETTE_LIST: tuple[Palette, ...] = (
    Palette(
        key="cpk",
        label="CPK / Jmol",
        description="Standard chemistry colours. Safe default for referees.",
        mode="table",
        colors={s: element_color(s) for s in ELEMENTS},
        default=UNKNOWN_COLOR,
        dark_background=False,
    ),
    Palette(
        key="pastel_lab",
        label="Pastel Lab",
        description="Desaturated CPK on a light canvas. Print friendly.",
        mode="table",
        colors={s: mix_hex(element_color(s), "#ffffff", 0.45) for s in ELEMENTS},
        default="#d8dade",
        dark_background=False,
    ),
    Palette(
        key="midnight_neon",
        label="Midnight Neon",
        description="Graphite framework with electric dopants. Dark covers.",
        mode="table",
        colors={
            "C": "#23262f",
            "H": "#e8eef7",
            "B": "#f472b6",
            "N": "#22d3ee",
            "O": "#fb7185",
            "F": "#a3e635",
            "P": "#fb923c",
            "S": "#facc15",
            "Si": "#94a3b8",
            "Fe": "#f97316",
            "Ni": "#4ade80",
            "Cu": "#f59e0b",
            "Pt": "#e2e8f0",
            "Au": "#fcd34d",
        },
        default="#38bdf8",
    ),
    Palette(
        key="ice_fire",
        label="Ice & Fire",
        description="Cold framework, hot heteroatoms. High contrast.",
        mode="table",
        colors={
            "C": "#7c93b0",
            "H": "#e2e8f0",
            "B": "#f97316",
            "N": "#38bdf8",
            "O": "#ef4444",
            "P": "#fb923c",
            "S": "#fbbf24",
            "Fe": "#dc2626",
            "Cu": "#f97316",
        },
        default="#60a5fa",
    ),
    Palette(
        key="gold_lab",
        label="Gold Lab",
        description="Warm metallic scheme for catalysis and plasmonics.",
        mode="table",
        colors={
            "C": "#7a6a4e",
            "H": "#f5efe3",
            "N": "#7dd3fc",
            "O": "#e11d48",
            "S": "#eab308",
            "P": "#fb923c",
            "Au": "#ffcf40",
            "Ag": "#dfe6ee",
            "Cu": "#e07a3c",
            "Pt": "#dbe2ea",
            "Pd": "#9fb3c8",
            "Ru": "#4fb0a5",
        },
        default="#c9a227",
    ),
    Palette(
        key="mono_accent",
        label="Mono + Accent",
        description="Neutral framework, every other element from your accent.",
        mode="accent",
        framework="#6b7280",
    ),
    Palette(
        key="spectral",
        label="Spectral by Z",
        description="Colour follows the atomic number. Good for many elements.",
        mode="spectral",
    ),
    Palette(
        key="blueprint",
        label="Blueprint",
        description="Cyan-on-navy technical look for schematic covers.",
        mode="table",
        colors={
            "C": "#7dd3fc",
            "H": "#e0f2fe",
            "B": "#f0abfc",
            "N": "#a5f3fc",
            "O": "#fda4af",
            "S": "#fde68a",
            "P": "#fdba74",
        },
        default="#93c5fd",
    ),
)

#: Palette key -> :class:`Palette`.
PALETTES: dict[str, Palette] = {p.key: p for p in _PALETTE_LIST}

#: Key of the palette used when the caller passes an unknown name.
DEFAULT_PALETTE = "midnight_neon"


def list_palettes() -> list[tuple[str, str, str]]:
    """Return ``(key, label, description)`` triples for UI enum construction."""
    return [(p.key, p.label, p.description) for p in _PALETTE_LIST]


def get_palette(key: str) -> Palette:
    """Return the palette for *key*, falling back to :data:`DEFAULT_PALETTE`."""
    return PALETTES.get(key, PALETTES[DEFAULT_PALETTE])


def _spectral_color(symbol: str) -> str:
    """Interpolate the spectral ramp using the atomic number of *symbol*."""
    element = ELEMENTS.get(symbol)
    if element is None:
        return UNKNOWN_COLOR
    # Z=1..~60 covers everything in the table without crushing the light end.
    t = min(1.0, max(0.0, (element.number - 1) / 59.0)) ** 0.6
    span = len(_SPECTRAL_STOPS) - 1
    pos = t * span
    idx = min(span - 1, int(pos))
    return mix_hex(_SPECTRAL_STOPS[idx], _SPECTRAL_STOPS[idx + 1], pos - idx)


def _accent_variants(accent: str, count: int) -> list[str]:
    """Derive *count* distinguishable colours from a single accent colour."""
    recipe = (
        lambda c: c,
        lambda c: lighten(c, 0.42),
        lambda c: darken(c, 0.32),
        lambda c: mix_hex(c, "#ffffff", 0.7),
        lambda c: mix_hex(c, "#000000", 0.55),
    )
    return [recipe[i % len(recipe)](accent) for i in range(max(0, count))]


def resolve_palette(
    key: str,
    symbols: object,
    accent: str = "#22d3ee",
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve a palette into concrete hex colours for the given symbols.

    Args:
        key: Palette key (see :data:`PALETTES`).
        symbols: Iterable of element symbols present in the structure.
        accent: Accent colour used by ``accent`` mode palettes.
        overrides: Optional per-symbol hex colours that win over everything.

    Returns:
        ``{symbol: "#rrggbb"}`` for every requested symbol.
    """
    palette = get_palette(key)
    ordered = sort_symbols(set(symbols))  # type: ignore[arg-type]
    resolved: dict[str, str] = {}

    if palette.mode == "accent":
        highlighted = [s for s in ordered if s not in FRAMEWORK_SYMBOLS]
        variants = _accent_variants(accent, len(highlighted))
        variant_by_symbol = dict(zip(highlighted, variants))
        for symbol in ordered:
            if symbol in FRAMEWORK_SYMBOLS:
                shade = palette.framework if symbol == "C" else lighten(palette.framework, 0.5)
                resolved[symbol] = shade
            else:
                resolved[symbol] = variant_by_symbol[symbol]
    elif palette.mode == "spectral":
        for symbol in ordered:
            resolved[symbol] = _spectral_color(symbol)
    else:
        for symbol in ordered:
            resolved[symbol] = palette.colors.get(symbol, palette.default)

    for symbol, value in (overrides or {}).items():
        if symbol in resolved:
            resolved[symbol] = value
    return resolved
