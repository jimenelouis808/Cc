"""Assign styles and palettes to a detected structure."""

from __future__ import annotations

from dataclasses import dataclass

import bpy

from ..core.colors import hex_to_linear
from ..core.compat import find_input, set_principled
from ..core.detect import Structure
from ..core.elements import ELEMENTS, element_label
from ..core.palettes import FRAMEWORK_SYMBOLS, resolve_palette
from .styles import MATERIAL_PREFIX, StyleParams, build_material


@dataclass
class ApplyReport:
    """Summary of what a styling run touched."""

    structure: str
    materials: dict[str, str]
    objects: int
    sticks: int
    skipped: list[str]

    def summary(self) -> str:
        """One-line message suitable for ``operator.report``."""
        elements = ", ".join(sorted(self.materials)) or "none"
        return (
            f"{self.structure}: styled {self.objects} object(s) "
            f"[{elements}] + {self.sticks} bond object(s)"
        )


def material_name(style_key: str, tag: str) -> str:
    """Return the canonical material name for a style/element pair."""
    return f"{MATERIAL_PREFIX}_{style_key}_{tag}"


def assign(obj: bpy.types.Object, material: bpy.types.Material) -> bool:
    """Replace every material slot of *obj* with *material*.

    Returns:
        ``False`` when the object cannot hold materials (empties, lights).
    """
    data = getattr(obj, "data", None)
    if data is None or not hasattr(data, "materials"):
        return False
    data.materials.clear()
    data.materials.append(material)
    obj.active_material_index = 0
    return True


def boost_emission(material: bpy.types.Material, color: str, strength: float) -> None:
    """Add self-illumination to an already built material.

    Used for the "glowing dopants" trick: the framework stays physically
    plausible while heteroatoms carry the eye across the cover.
    """
    if strength <= 0.0 or not material.use_nodes:
        return
    linear = hex_to_linear(color)
    for node in material.node_tree.nodes:
        if node.bl_idname == "ShaderNodeBsdfPrincipled":
            set_principled(node, emission_color=linear, emission_strength=strength)
            return
    for node in material.node_tree.nodes:
        if node.bl_idname == "ShaderNodeEmission":
            socket = find_input(node, "Strength")
            if socket is not None:
                socket.default_value = max(socket.default_value, strength)
            return


def apply_style(
    structure: Structure,
    style_key: str,
    palette_key: str,
    params: StyleParams,
    accent: str = "#22d3ee",
    stick_style: str | None = None,
    stick_color: str = "#4b5563",
    overrides: dict[str, str] | None = None,
) -> ApplyReport:
    """Build and assign materials for every element of *structure*.

    Args:
        structure: Structure returned by :func:`~atomviz_studio.core.detect.detect_structures`.
        style_key: Atom style key (see :data:`~atomviz_studio.materials.styles.STYLES`).
        palette_key: Palette key (see :data:`~atomviz_studio.core.palettes.PALETTES`).
        params: Style tunables.
        accent: Accent colour for accent-mode palettes.
        stick_style: Style for bond geometry; ``None`` reuses *style_key*.
        stick_color: Bond colour.
        overrides: Per-element colour overrides (``{"N": "#ff0000"}``).

    Returns:
        An :class:`ApplyReport` describing the run.
    """
    colors = resolve_palette(palette_key, structure.symbols, accent, overrides)
    materials: dict[str, str] = {}
    touched = 0
    skipped: list[str] = []

    for symbol in structure.symbols:
        group = structure.groups[symbol]
        color = colors.get(symbol, accent)
        material = build_material(style_key, material_name(style_key, symbol), color, params)
        if params.emissive_dopants > 0.0 and symbol not in FRAMEWORK_SYMBOLS:
            boost_emission(material, color, params.emissive_dopants)
        materials[symbol] = material.name
        for obj in group.shaded:
            if assign(obj, material):
                touched += 1
            else:
                skipped.append(obj.name)

    stick_key = stick_style or style_key
    stick_count = 0
    if structure.sticks:
        stick_material = build_material(
            stick_key, material_name(stick_key, "sticks"), stick_color, params
        )
        materials["sticks"] = stick_material.name
        for obj in structure.sticks:
            if assign(obj, stick_material):
                stick_count += 1
            else:
                skipped.append(obj.name)

    return ApplyReport(
        structure=structure.name,
        materials=materials,
        objects=touched,
        sticks=stick_count,
        skipped=skipped,
    )


def apply_radii(structure: Structure, mode: str = "vdw", factor: float = 1.0) -> int:
    """Rescale the instanced atom spheres by element radius.

    Atomic Blender already scales its balls, but covers usually want the
    relationship exaggerated (small H, dominant metal centres) or flattened
    (uniform beads). Only the *shaded* objects are touched, so the underlying
    coordinates stay untouched and scientifically correct.

    Args:
        structure: Target structure.
        mode: ``"vdw"``, ``"covalent"`` or ``"uniform"``.
        factor: Global multiplier applied on top of the radius.

    Returns:
        Number of objects rescaled.
    """
    if mode not in {"vdw", "covalent", "uniform"}:
        raise ValueError(f"unknown radius mode {mode!r}")
    changed = 0
    for symbol in structure.symbols:
        element = ELEMENTS.get(symbol)
        if mode == "uniform" or element is None:
            radius = 1.0
        else:
            radius = element.vdw if mode == "vdw" else element.covalent
            # Normalise against carbon so "1.0" keeps the usual scene scale.
            reference = ELEMENTS["C"].vdw if mode == "vdw" else ELEMENTS["C"].covalent
            radius = radius / reference
        scale = max(1e-3, radius * factor)
        for obj in structure.groups[symbol].shaded:
            obj.scale = (scale, scale, scale)
            changed += 1
    return changed


def describe(structure: Structure) -> list[str]:
    """Return human readable ``"C - Carbon: 240 atoms"`` lines for the UI."""
    lines = []
    for symbol in structure.symbols:
        group = structure.groups[symbol]
        lines.append(f"{element_label(symbol)}: {group.atom_count} atoms")
    if structure.sticks:
        lines.append(f"bonds: {len(structure.sticks)} object(s)")
    return lines
