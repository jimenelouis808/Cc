"""Procedural shading styles for atoms and bonds.

Each style is a small node-graph recipe that takes a base colour plus a few
tunable parameters and returns a ready material. Styles are registered in
:data:`STYLES` so the UI, the looks and the CLI all see the same catalogue.

Engine notes:

* ``toon_cel`` and ``hologram`` rely on **Shader to RGB**, which only works in
  EEVEE. They render, but flat, in Cycles.
* ``glass_crystal`` / ``frosted_glass`` look their best in Cycles; in EEVEE
  enable screen-space refraction (the add-on does that for you).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import bpy

from ..core import nodes as N
from ..core.colors import darken, hex_to_linear, lighten
from ..core.compat import set_material_blend, set_principled

#: Prefix for every material the add-on owns; used for cleanup and reuse.
MATERIAL_PREFIX = "AV"


@dataclass
class StyleParams:
    """Tunables shared by all styles (a style ignores what it does not use)."""

    roughness: float = 0.3
    metallic: float = 0.0
    ior: float = 1.45
    alpha: float = 1.0
    emission_strength: float = 2.0
    clearcoat: float = 0.4
    subsurface: float = 0.35
    steps: float = 3.0
    scanlines: float = 90.0
    bump: float = 0.0
    #: Extra emission applied to non-framework atoms (dopants, metals).
    emissive_dopants: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    def merged(self, overrides: dict[str, float] | None) -> "StyleParams":
        """Return a copy with *overrides* applied to the matching fields."""
        clone = StyleParams(**{**self.__dict__, "extras": dict(self.extras)})
        for key, value in (overrides or {}).items():
            if hasattr(clone, key) and key != "extras":
                setattr(clone, key, float(value))
            else:
                clone.extras[key] = float(value)
        return clone


# --------------------------------------------------------------------------
# Style builders
# --------------------------------------------------------------------------
def _finish(material: bpy.types.Material, color: str, transparent: bool = False) -> bpy.types.Material:
    """Set the viewport display colour and blend mode, then return *material*."""
    material.diffuse_color = hex_to_linear(color)
    set_material_blend(material, transparent)
    return material


def build_glossy_ceramic(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Glazed ceramic: opaque, saturated, clear-coated. The default cover ball."""
    material, tree = N.new_material(name)
    bsdf = N.new(tree, "ShaderNodeBsdfPrincipled", (0, 0))
    set_principled(
        bsdf,
        base_color=hex_to_linear(color),
        roughness=p.roughness,
        metallic=0.0,
        specular=0.6,
        coat=p.clearcoat,
        coat_roughness=0.05,
        ior=1.5,
    )
    N.link(tree, bsdf, N.material_output(tree), 0, "Surface")
    return _finish(material, color)


def build_polished_metal(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Anisotropic-ish polished metal with a slightly brighter grazing edge."""
    material, tree = N.new_material(name)
    bsdf = N.new(tree, "ShaderNodeBsdfPrincipled", (0, 0))
    set_principled(
        bsdf,
        base_color=hex_to_linear(color),
        metallic=1.0,
        roughness=max(0.03, p.roughness),
        anisotropic=0.25,
        coat=p.clearcoat * 0.5,
    )
    N.link(tree, bsdf, N.material_output(tree), 0, "Surface")
    return _finish(material, color)


def build_matte_clay(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Soft matte clay — reads well on white backgrounds and in print."""
    material, tree = N.new_material(name)
    bsdf = N.new(tree, "ShaderNodeBsdfPrincipled", (0, 0))
    set_principled(
        bsdf,
        base_color=hex_to_linear(color),
        roughness=max(0.55, p.roughness),
        specular=0.25,
        sheen=0.3,
    )
    N.link(tree, bsdf, N.material_output(tree), 0, "Surface")
    return _finish(material, color)


def build_glass_crystal(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Coloured transmissive glass with a tight specular highlight."""
    material, tree = N.new_material(name)
    bsdf = N.new(tree, "ShaderNodeBsdfPrincipled", (0, 0))
    set_principled(
        bsdf,
        base_color=hex_to_linear(lighten(color, 0.35)),
        transmission=1.0,
        roughness=min(0.2, p.roughness),
        ior=p.ior,
        metallic=0.0,
        alpha=1.0,
    )
    N.link(tree, bsdf, N.material_output(tree), 0, "Surface")
    if hasattr(material, "use_screen_refraction"):  # EEVEE legacy
        material.use_screen_refraction = True
    return _finish(material, color, transparent=True)


def build_frosted_glass(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Sand-blasted glass: transmissive but diffused, with a faint inner glow."""
    material, tree = N.new_material(name)
    bsdf = N.new(tree, "ShaderNodeBsdfPrincipled", (0, 0))
    set_principled(
        bsdf,
        base_color=hex_to_linear(lighten(color, 0.25)),
        transmission=1.0,
        roughness=max(0.2, p.roughness),
        ior=p.ior,
        emission_color=hex_to_linear(color),
        emission_strength=0.25,
    )
    N.link(tree, bsdf, N.material_output(tree), 0, "Surface")
    if hasattr(material, "use_screen_refraction"):
        material.use_screen_refraction = True
    return _finish(material, color, transparent=True)


def build_subsurface_jelly(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Translucent jelly — light bleeds through small atoms, very organic."""
    material, tree = N.new_material(name)
    bsdf = N.new(tree, "ShaderNodeBsdfPrincipled", (0, 0))
    set_principled(
        bsdf,
        base_color=hex_to_linear(color),
        roughness=max(0.15, p.roughness * 0.6),
        subsurface=p.subsurface,
        subsurface_radius=(0.6, 0.35, 0.25),
        subsurface_color=hex_to_linear(lighten(color, 0.3)),
        coat=p.clearcoat,
    )
    N.link(tree, bsdf, N.material_output(tree), 0, "Surface")
    return _finish(material, color)


def build_emissive_neon(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Self-illuminated atom: a glowing core wrapped in a thin glossy shell."""
    material, tree = N.new_material(name)
    emission = N.emission_shader(tree, color, p.emission_strength, (-260, -140))
    shell = N.new(tree, "ShaderNodeBsdfPrincipled", (-260, 160))
    set_principled(
        shell,
        base_color=hex_to_linear(darken(color, 0.55)),
        roughness=0.12,
        metallic=0.2,
        coat=0.8,
    )
    fresnel = N.new(tree, "ShaderNodeFresnel", (-260, 380), IOR=1.6)
    mix = N.new(tree, "ShaderNodeMixShader", (60, 0))
    N.link(tree, fresnel, mix, 0, "Fac")
    N.link(tree, emission, mix, 0, 1)
    N.link(tree, shell, mix, 0, 2)
    N.link(tree, mix, N.material_output(tree), 0, "Surface")
    return _finish(material, color)


def build_iridescent(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Pearlescent shell whose hue shifts with the viewing angle."""
    material, tree = N.new_material(name)
    layer = N.new(tree, "ShaderNodeLayerWeight", (-620, 120), Blend=0.35)
    color_ramp = N.new(tree, "ShaderNodeValToRGB", (-420, 120))
    N.ramp(
        color_ramp,
        [
            (0.0, darken(color, 0.35)),
            (0.35, color),
            (0.62, lighten(color, 0.55)),
            (1.0, "#ffffff"),
        ],
    )
    bsdf = N.new(tree, "ShaderNodeBsdfPrincipled", (-60, 0))
    set_principled(
        bsdf,
        metallic=0.85,
        roughness=max(0.08, p.roughness * 0.5),
        coat=1.0,
        coat_roughness=0.03,
    )
    N.link(tree, layer, color_ramp, "Facing", "Fac")
    N.link(tree, color_ramp, bsdf, "Color", "Base Color")
    N.link(tree, bsdf, N.material_output(tree), 0, "Surface")
    return _finish(material, color)


def build_toon_cel(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Flat cel shading with hard light steps (EEVEE only) plus a rim light."""
    material, tree = N.new_material(name)
    diffuse = N.new(tree, "ShaderNodeBsdfDiffuse", (-820, 0))
    diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    to_rgb = N.new(tree, "ShaderNodeShaderToRGB", (-620, 0))
    steps = max(2, int(round(p.steps)))
    stops = []
    for i in range(steps):
        position = i / max(1, steps - 1)
        shade = darken(color, 0.55 * (1.0 - position))
        stops.append((max(0.0, position - 0.5 / steps), shade))
    color_ramp = N.new(tree, "ShaderNodeValToRGB", (-420, 0))
    color_ramp.color_ramp.interpolation = "CONSTANT"
    N.ramp(color_ramp, stops)

    fresnel = N.new(tree, "ShaderNodeFresnel", (-420, -260), IOR=1.35)
    rim_ramp = N.new(tree, "ShaderNodeValToRGB", (-240, -260))
    rim_ramp.color_ramp.interpolation = "CONSTANT"
    N.ramp(rim_ramp, [(0.0, "#000000"), (0.72, lighten(color, 0.75))])

    mix_node, fac, socket_a, socket_b, result = N.mix_rgb(tree, (-40, 0), "ADD", 1.0)
    socket_a.default_value = hex_to_linear(color)
    emission = N.new(tree, "ShaderNodeEmission", (200, 0), Strength=1.0)

    N.link(tree, diffuse, to_rgb, 0, 0)
    N.link(tree, to_rgb, color_ramp, "Color", "Fac")
    N.link(tree, fresnel, rim_ramp, 0, "Fac")
    tree.links.new(color_ramp.outputs["Color"], socket_a)
    tree.links.new(rim_ramp.outputs["Color"], socket_b)
    tree.links.new(result, emission.inputs["Color"])
    N.link(tree, emission, N.material_output(tree), 0, "Surface")
    return _finish(material, color)


def build_xray_fresnel(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Transparent body with glowing silhouette — the classic X-ray/diagram look."""
    material, tree = N.new_material(name)
    layer = N.new(tree, "ShaderNodeLayerWeight", (-620, 0), Blend=0.28)
    falloff = N.new(tree, "ShaderNodeValToRGB", (-420, 0))
    N.ramp(falloff, [(0.15, "#000000"), (1.0, "#ffffff")])
    emission = N.emission_shader(tree, color, p.emission_strength, (-160, 140))
    transparent = N.new(tree, "ShaderNodeBsdfTransparent", (-160, -60))
    mix = N.new(tree, "ShaderNodeMixShader", (120, 0))
    N.link(tree, layer, falloff, "Facing", "Fac")
    N.link(tree, falloff, mix, "Color", "Fac")
    N.link(tree, transparent, mix, 0, 1)
    N.link(tree, emission, mix, 0, 2)
    N.link(tree, mix, N.material_output(tree), 0, "Surface")
    return _finish(material, color, transparent=True)


def build_hologram(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Emissive scan-lined hologram: bright rim, striped body, partly see-through."""
    material, tree = N.new_material(name)
    coord = N.new(tree, "ShaderNodeTexCoord", (-1000, -200))
    separate = N.new(tree, "ShaderNodeSeparateXYZ", (-820, -200))
    scan = N.new(
        tree,
        "ShaderNodeMath",
        (-640, -200),
        operation="MULTIPLY",
    )
    scan.inputs[1].default_value = max(1.0, p.scanlines)
    wave = N.new(tree, "ShaderNodeMath", (-460, -200), operation="SINE")
    stripes = N.new(tree, "ShaderNodeValToRGB", (-280, -200))
    N.ramp(stripes, [(0.35, darken(color, 0.6)), (0.55, lighten(color, 0.4))])

    layer = N.new(tree, "ShaderNodeLayerWeight", (-640, 220), Blend=0.3)
    rim = N.new(tree, "ShaderNodeValToRGB", (-460, 220))
    N.ramp(rim, [(0.2, "#000000"), (1.0, "#ffffff")])

    emission = N.new(tree, "ShaderNodeEmission", (40, -80), Strength=p.emission_strength)
    transparent = N.new(tree, "ShaderNodeBsdfTransparent", (40, 120))
    mix = N.new(tree, "ShaderNodeMixShader", (300, 0))
    mix.inputs[0].default_value = max(0.05, min(1.0, p.alpha))

    N.link(tree, coord, separate, "Object", 0)
    N.link(tree, separate, scan, "Z", 0)
    N.link(tree, scan, wave, 0, 0)
    N.link(tree, wave, stripes, 0, "Fac")
    N.link(tree, stripes, emission, "Color", "Color")
    N.link(tree, layer, rim, "Facing", "Fac")
    N.link(tree, rim, mix, "Color", "Fac")
    N.link(tree, transparent, mix, 0, 1)
    N.link(tree, emission, mix, 0, 2)
    N.link(tree, mix, N.material_output(tree), 0, "Surface")
    return _finish(material, color, transparent=True)


def build_graphite(name: str, color: str, p: StyleParams) -> bpy.types.Material:
    """Dark carbon with a fine procedural grain — good for the framework."""
    material, tree = N.new_material(name)
    noise = N.new(tree, "ShaderNodeTexNoise", (-620, -220), Scale=180.0, Detail=6.0)
    bump = N.new(tree, "ShaderNodeBump", (-380, -220), Strength=max(0.05, p.bump or 0.15))
    bsdf = N.new(tree, "ShaderNodeBsdfPrincipled", (-60, 0))
    set_principled(
        bsdf,
        base_color=hex_to_linear(darken(color, 0.25)),
        roughness=max(0.35, p.roughness),
        metallic=0.35,
        specular=0.45,
    )
    N.link(tree, noise, bump, "Fac", "Height")
    N.link(tree, bump, bsdf, "Normal", "Normal")
    N.link(tree, bsdf, N.material_output(tree), 0, "Surface")
    return _finish(material, color)


@dataclass(frozen=True)
class Style:
    """A registered shading style."""

    key: str
    label: str
    description: str
    builder: object
    transparent: bool = False


_STYLE_LIST: tuple[Style, ...] = (
    Style("glossy_ceramic", "Glossy Ceramic", "Glazed, clear-coated spheres. Safe default.", build_glossy_ceramic),
    Style("polished_metal", "Polished Metal", "Fully metallic, sharp reflections.", build_polished_metal),
    Style("matte_clay", "Matte Clay", "Soft diffuse look for print.", build_matte_clay),
    Style("glass_crystal", "Glass Crystal", "Transmissive coloured glass.", build_glass_crystal, True),
    Style("frosted_glass", "Frosted Glass", "Diffused glass with a faint glow.", build_frosted_glass, True),
    Style("subsurface_jelly", "Subsurface Jelly", "Translucent, light bleeds through.", build_subsurface_jelly),
    Style("emissive_neon", "Emissive Neon", "Glowing core inside a glossy shell.", build_emissive_neon),
    Style("iridescent", "Iridescent Pearl", "Hue shifts with the viewing angle.", build_iridescent),
    Style("toon_cel", "Toon / Cel", "Flat stepped shading (EEVEE).", build_toon_cel),
    Style("xray_fresnel", "X-Ray Fresnel", "Transparent body, glowing silhouette.", build_xray_fresnel, True),
    Style("hologram", "Hologram", "Scan-lined emissive projection (EEVEE).", build_hologram, True),
    Style("graphite", "Graphite Grain", "Dark carbon with procedural grain.", build_graphite),
)

#: Style key -> :class:`Style`.
STYLES: dict[str, Style] = {s.key: s for s in _STYLE_LIST}

#: Default style key.
DEFAULT_STYLE = "glossy_ceramic"


def list_styles() -> list[tuple[str, str, str]]:
    """Return ``(key, label, description)`` triples for UI enum construction."""
    return [(s.key, s.label, s.description) for s in _STYLE_LIST]


def build_material(style_key: str, name: str, color: str, params: StyleParams) -> bpy.types.Material:
    """Build (or rebuild) the material *name* using the given style and colour.

    Args:
        style_key: Key from :data:`STYLES`.
        name: Material name; existing materials with this name are rebuilt.
        color: Base colour as an sRGB hex string.
        params: Style tunables.

    Returns:
        The material, ready to be assigned to objects.
    """
    style = STYLES.get(style_key, STYLES[DEFAULT_STYLE])
    return style.builder(name, color, params)  # type: ignore[operator]
