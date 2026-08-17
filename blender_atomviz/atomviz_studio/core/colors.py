"""Colour helpers.

Blender stores shader colours in **linear** space while palettes are far more
readable written as sRGB hex. Everything the add-on hands to Blender goes
through :func:`hex_to_linear` first.

No ``bpy`` import: unit testable outside Blender.
"""

from __future__ import annotations

RGBA = tuple[float, float, float, float]


def hex_to_srgb(value: str) -> tuple[float, float, float]:
    """Convert ``"#rrggbb"`` (or ``"rrggbb"``) to sRGB floats in ``[0, 1]``.

    Raises:
        ValueError: if *value* is not a 6-digit hexadecimal colour.
    """
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected a 6-digit hex colour, got {value!r}")
    try:
        r, g, b = (int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError as exc:  # pragma: no cover - message clarity only
        raise ValueError(f"invalid hex colour {value!r}") from exc
    return r, g, b


def srgb_to_linear(channel: float) -> float:
    """Convert one sRGB channel in ``[0, 1]`` to linear scene-referred space."""
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def linear_to_srgb(channel: float) -> float:
    """Inverse of :func:`srgb_to_linear`."""
    if channel <= 0.0031308:
        return channel * 12.92
    return 1.055 * (channel ** (1.0 / 2.4)) - 0.055


def hex_to_linear(value: str, alpha: float = 1.0) -> RGBA:
    """Convert an sRGB hex string to a linear RGBA tuple ready for Blender."""
    r, g, b = hex_to_srgb(value)
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), alpha)


def linear_to_hex(color: object) -> str:
    """Convert a linear RGB(A) sequence back to an sRGB hex string."""
    r, g, b = (float(c) for c in tuple(color)[:3])  # type: ignore[arg-type]
    channels = (linear_to_srgb(max(0.0, min(1.0, c))) for c in (r, g, b))
    return "#" + "".join(f"{round(c * 255):02x}" for c in channels)


def mix_hex(a: str, b: str, factor: float) -> str:
    """Blend two hex colours in sRGB space (``factor=0`` -> *a*, ``1`` -> *b*)."""
    factor = max(0.0, min(1.0, factor))
    ca, cb = hex_to_srgb(a), hex_to_srgb(b)
    mixed = tuple(x + (y - x) * factor for x, y in zip(ca, cb))
    return "#" + "".join(f"{round(c * 255):02x}" for c in mixed)


def lighten(value: str, amount: float) -> str:
    """Move a colour towards white by *amount* in ``[0, 1]``."""
    return mix_hex(value, "#ffffff", amount)


def darken(value: str, amount: float) -> str:
    """Move a colour towards black by *amount* in ``[0, 1]``."""
    return mix_hex(value, "#000000", amount)


def luminance(value: str) -> float:
    """Relative luminance (Rec. 709) of an sRGB hex colour, in ``[0, 1]``."""
    r, g, b = (srgb_to_linear(c) for c in hex_to_srgb(value))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
