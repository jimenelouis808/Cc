"""Scene properties driving the AtomViz Studio panel.

Note: this module must **not** use ``from __future__ import annotations``.
Blender registers properties by reading the evaluated values of class
annotations; postponed evaluation would turn them into plain strings and
registration would fail.
"""

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from ..core.colors import hex_to_linear, linear_to_hex
from ..core.palettes import list_palettes
from ..core.presets import list_looks, list_render_presets
from ..effects.backgrounds import list_backgrounds
from ..effects.lighting import list_rigs
from ..materials.styles import list_styles


def _items(triples):
    """Convert ``(key, label, description)`` triples into Blender enum items."""
    return [(key, label, description) for key, label, description in triples]


def _color(name, default_hex, description):
    """Declare a linear RGB colour property initialised from a hex string."""
    return FloatVectorProperty(
        name=name,
        description=description,
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=hex_to_linear(default_hex)[:3],
    )


def hex_of(value):
    """Convert a Blender colour property value to an sRGB hex string."""
    return linear_to_hex(tuple(value))


class AtomVizProps(PropertyGroup):
    """All AtomViz Studio settings, stored on the scene."""

    # -- targeting ---------------------------------------------------------
    scope: EnumProperty(
        name="Scope",
        description="Which objects to treat as the structure",
        items=[
            ("SCENE", "Whole scene", "Every object in the scene"),
            ("SELECTED", "Selection", "Selected objects and their children"),
            ("COLLECTION", "Active collection", "Objects in the active collection"),
        ],
        default="SCENE",
    )

    # -- looks -------------------------------------------------------------
    look: EnumProperty(
        name="Cover look",
        description="Complete recipe: palette, shading, world, lights and effects",
        items=_items(list_looks()),
        default="neon_lab",
    )
    seed: IntProperty(
        name="Seed",
        description="Seed for arcs and beam jitter; same seed rebuilds the same result",
        default=0,
        min=0,
    )

    # -- materials ---------------------------------------------------------
    palette: EnumProperty(
        name="Palette",
        description="Element colour scheme",
        items=_items(list_palettes()),
        default="midnight_neon",
    )
    accent: _color("Accent", "#22d3ee", "Accent colour for accent-mode palettes")
    style: EnumProperty(
        name="Atom style",
        description="Shading style applied to the atoms",
        items=_items(list_styles()),
        default="glossy_ceramic",
    )
    stick_style: EnumProperty(
        name="Bond style",
        description="Shading style applied to bond geometry",
        items=_items(list_styles()),
        default="polished_metal",
    )
    stick_color: _color("Bond colour", "#4b5563", "Colour of the bond geometry")

    roughness: FloatProperty(name="Roughness", default=0.28, min=0.0, max=1.0)
    metallic: FloatProperty(name="Metallic", default=0.0, min=0.0, max=1.0)
    ior: FloatProperty(name="IOR", default=1.45, min=1.0, max=3.0)
    alpha: FloatProperty(name="Alpha", default=1.0, min=0.0, max=1.0)
    clearcoat: FloatProperty(name="Coat", default=0.4, min=0.0, max=1.0)
    subsurface: FloatProperty(name="Subsurface", default=0.35, min=0.0, max=1.0)
    emission_strength: FloatProperty(name="Emission", default=2.0, min=0.0, max=200.0)
    emissive_dopants: FloatProperty(
        name="Glowing dopants",
        description="Extra emission for every element that is not C or H",
        default=0.0,
        min=0.0,
        max=100.0,
    )
    toon_steps: FloatProperty(name="Toon steps", default=3.0, min=2.0, max=8.0)
    scanlines: FloatProperty(name="Scanlines", default=90.0, min=1.0, max=400.0)

    radius_mode: EnumProperty(
        name="Atom radii",
        description="How to scale the instanced atom spheres",
        items=[
            ("KEEP", "Keep", "Leave the importer scaling untouched"),
            ("VDW", "Van der Waals", "Scale by van der Waals radius"),
            ("COVALENT", "Covalent", "Scale by covalent radius"),
            ("UNIFORM", "Uniform", "Same size for every element"),
        ],
        default="KEEP",
    )
    radius_factor: FloatProperty(name="Radius factor", default=1.0, min=0.05, max=5.0)

    # -- world -------------------------------------------------------------
    background: EnumProperty(
        name="Background",
        description="Procedural world background",
        items=_items(list_backgrounds()),
        default="gradient",
    )
    bg_top: _color("Top", "#0d1b2a", "Colour towards the zenith")
    bg_bottom: _color("Bottom", "#05070f", "Colour towards the horizon")
    bg_accent: _color("Tint", "#1b3b6f", "Tint used by nebula, plasma and starfield")
    bg_strength: FloatProperty(name="World strength", default=1.0, min=0.0, max=20.0)

    # -- lighting ----------------------------------------------------------
    light_rig: EnumProperty(
        name="Light rig",
        description="Lighting setup built around the structure",
        items=_items(list_rigs()),
        default="three_point",
    )
    light_energy: FloatProperty(name="Energy", default=800.0, min=0.0, max=100000.0)
    light_color_a: _color("Light A", "#22d3ee", "Key / first rim colour")
    light_color_b: _color("Light B", "#f472b6", "Fill / second rim colour")
    haze_density: FloatProperty(
        name="Haze density",
        description="Participating medium around the structure; needed for beams and god rays",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=4,
    )

    # -- electricity -------------------------------------------------------
    arc_mode: EnumProperty(
        name="Arc mode",
        description="Where the discharges are anchored",
        items=[
            ("ATOMS", "Between atoms", "Arcs jump between randomly chosen atoms"),
            ("CAGE", "Cage", "Arcs travel across the bounding sphere"),
            ("DISCHARGE", "Discharge", "Arcs shoot outwards from surface atoms"),
        ],
        default="ATOMS",
    )
    arc_count: IntProperty(name="Arcs", default=6, min=1, max=200)
    arc_color: _color("Arc colour", "#7dd3fc", "Discharge colour")
    arc_thickness: FloatProperty(name="Thickness", default=0.03, min=0.001, max=2.0, precision=3)
    arc_strength: FloatProperty(name="Glow", default=40.0, min=0.0, max=500.0)
    arc_chaos: FloatProperty(name="Chaos", default=0.18, min=0.0, max=1.0)
    arc_branches: IntProperty(name="Branches", default=2, min=0, max=12)
    arc_flicker: BoolProperty(
        name="Flicker",
        description="Keyframe visibility so the discharge crackles over the frame range",
        default=False,
    )

    # -- lasers ------------------------------------------------------------
    laser_count: IntProperty(name="Beams", default=3, min=1, max=64)
    laser_color: _color("Beam colour", "#ff2d2d", "Laser colour")
    laser_radius: FloatProperty(name="Beam radius", default=0.05, min=0.001, max=2.0, precision=3)
    laser_strength: FloatProperty(name="Beam glow", default=90.0, min=0.0, max=1000.0)
    laser_distance: FloatProperty(name="Emitter distance", default=6.0, min=1.1, max=40.0)
    laser_impact: BoolProperty(name="Impact glow", default=True)
    laser_auto_haze: BoolProperty(
        name="Add haze",
        description="Beams are only visible inside a participating medium",
        default=True,
    )

    # -- camera ------------------------------------------------------------
    focal_mm: FloatProperty(name="Focal length", default=50.0, min=8.0, max=300.0)
    azimuth_deg: FloatProperty(name="Azimuth", default=35.0, min=-360.0, max=360.0)
    elevation_deg: FloatProperty(name="Elevation", default=15.0, min=-89.0, max=89.0)
    frame_margin: FloatProperty(name="Margin", default=1.25, min=1.0, max=4.0)
    use_dof: BoolProperty(name="Depth of field", default=True)
    fstop: FloatProperty(name="F-stop", default=2.8, min=0.1, max=32.0)

    # -- render ------------------------------------------------------------
    render_preset: EnumProperty(
        name="Format",
        description="Output resolution, engine and colour management",
        items=_items(list_render_presets()),
        default="preview_fast",
    )
    resolution_percentage: IntProperty(name="Scale %", default=100, min=5, max=200)
    output_path: StringProperty(
        name="Output",
        description="Where 'Render cover' writes the image",
        subtype="FILE_PATH",
        default="//atomviz_cover.png",
    )

    # -- post-processing ---------------------------------------------------
    glare: EnumProperty(
        name="Glow",
        description="Compositor glare applied to bright pixels",
        items=[
            ("NONE", "None", "No glow"),
            ("FOG_GLOW", "Fog glow", "Soft bloom around emissive areas"),
            ("STREAKS", "Streaks", "Anamorphic streaks"),
            ("GHOSTS", "Ghosts", "Lens ghosting"),
            ("SIMPLE_STAR", "Star", "Cross-shaped star filter"),
        ],
        default="FOG_GLOW",
    )
    glare_threshold: FloatProperty(name="Threshold", default=0.7, min=0.0, max=10.0)
    glare_streaks: BoolProperty(name="Add streaks", default=False)
    vignette: FloatProperty(name="Vignette", default=0.3, min=0.0, max=1.0)
    contrast: FloatProperty(name="Contrast", default=0.0, min=-50.0, max=50.0)
    saturation: FloatProperty(name="Saturation", default=1.0, min=0.0, max=3.0)


def register() -> None:
    """Register the property group and attach it to :class:`bpy.types.Scene`."""
    bpy.utils.register_class(AtomVizProps)
    bpy.types.Scene.atomviz = bpy.props.PointerProperty(type=AtomVizProps)


def unregister() -> None:
    """Detach and unregister the property group."""
    if hasattr(bpy.types.Scene, "atomviz"):
        del bpy.types.Scene.atomviz
    bpy.utils.unregister_class(AtomVizProps)
