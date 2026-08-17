"""World backgrounds: gradients, nebulae, starfields, plasma and studio white.

All of them are procedural — no HDRI files to ship or to lose — and every one
is driven by three colours (``bottom``, ``top``, ``accent``) so a background
can be re-tinted to match the structure palette in one click.
"""

from __future__ import annotations

from dataclasses import dataclass

import bpy

from ..core import nodes as N
from ..core.colors import darken, hex_to_linear, mix_hex

WORLD_NAME = "AV_World"


def _world(scene: bpy.types.Scene) -> tuple[bpy.types.World, bpy.types.NodeTree]:
    """Return the add-on's world, creating it and assigning it to *scene*."""
    world = bpy.data.worlds.get(WORLD_NAME) or bpy.data.worlds.new(WORLD_NAME)
    world.use_nodes = True
    scene.world = world
    N.clear(world.node_tree)
    return world, world.node_tree


def build_solid(scene: bpy.types.Scene, top: str, bottom: str, accent: str, strength: float) -> bpy.types.World:
    """Flat colour background (uses *bottom*)."""
    world, tree = _world(scene)
    background = N.new(tree, "ShaderNodeBackground", (0, 0), Strength=strength)
    background.inputs["Color"].default_value = hex_to_linear(bottom)
    N.link(tree, background, N.world_output(tree), 0, "Surface")
    return world


def build_gradient(scene: bpy.types.Scene, top: str, bottom: str, accent: str, strength: float) -> bpy.types.World:
    """Vertical two-colour gradient — the everyday cover background."""
    world, tree = _world(scene)
    background = N.gradient_world(tree, top=top, bottom=bottom, strength=strength)
    N.link(tree, background, N.world_output(tree), 0, "Surface")
    return world


def build_studio_white(scene: bpy.types.Scene, top: str, bottom: str, accent: str, strength: float) -> bpy.types.World:
    """Bright neutral environment for clean, print-friendly figures."""
    world, tree = _world(scene)
    background = N.gradient_world(tree, top="#ffffff", bottom="#dfe3e8", strength=max(1.0, strength))
    N.link(tree, background, N.world_output(tree), 0, "Surface")
    return world


def build_nebula(scene: bpy.types.Scene, top: str, bottom: str, accent: str, strength: float) -> bpy.types.World:
    """Soft cloudy field: fractal noise tinted between the three colours."""
    world, tree = _world(scene)
    coord = N.new(tree, "ShaderNodeTexCoord", (-1100, 0))
    mapping = N.new(tree, "ShaderNodeMapping", (-920, 0))
    mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    noise = N.new(tree, "ShaderNodeTexNoise", (-740, 0), Scale=1.8, Detail=8.0, Roughness=0.62)
    clouds = N.new(tree, "ShaderNodeValToRGB", (-540, 0))
    N.ramp(
        clouds,
        [
            (0.30, bottom),
            (0.52, mix_hex(bottom, accent, 0.65)),
            (0.68, accent),
            (0.85, top),
        ],
    )
    gradient = N.gradient_world(tree, top=top, bottom=darken(bottom, 0.25), strength=strength)
    mix_node, fac, socket_a, socket_b, result = N.mix_rgb(tree, (-260, -220), "SCREEN", 0.55)

    N.link(tree, coord, mapping, "Generated", "Vector")
    N.link(tree, mapping, noise, 0, "Vector")
    N.link(tree, noise, clouds, "Fac", "Fac")
    # gradient's Background node already holds the base colour; re-mix it in.
    base_link = gradient.inputs["Color"].links
    if base_link:
        tree.links.new(base_link[0].from_socket, socket_a)
    tree.links.new(clouds.outputs["Color"], socket_b)
    tree.links.new(result, gradient.inputs["Color"])
    N.link(tree, gradient, N.world_output(tree), 0, "Surface")
    return world


def build_starfield(scene: bpy.types.Scene, top: str, bottom: str, accent: str, strength: float) -> bpy.types.World:
    """Dark sky with sparse stars — pairs well with hologram/neon structures."""
    world, tree = _world(scene)
    coord = N.new(tree, "ShaderNodeTexCoord", (-1100, -200))
    voronoi = N.new(tree, "ShaderNodeTexVoronoi", (-900, -200), Scale=120.0)
    voronoi.feature = "F1"
    stars = N.new(tree, "ShaderNodeValToRGB", (-700, -200))
    N.ramp(stars, [(0.0, "#ffffff"), (0.035, "#000000")])
    twinkle = N.new(tree, "ShaderNodeTexNoise", (-900, -460), Scale=8.0, Detail=2.0)
    tint = N.new(tree, "ShaderNodeValToRGB", (-700, -460))
    N.ramp(tint, [(0.35, "#000000"), (0.75, accent)])

    gradient = N.gradient_world(tree, top=top, bottom=bottom, strength=strength)
    star_mix, _, star_a, star_b, star_out = N.mix_rgb(tree, (-460, -300), "ADD", 0.9)
    sky_mix, _, sky_a, sky_b, sky_out = N.mix_rgb(tree, (-260, -300), "ADD", 0.35)

    N.link(tree, coord, voronoi, "Generated", "Vector")
    N.link(tree, coord, twinkle, "Generated", "Vector")
    N.link(tree, voronoi, stars, "Distance", "Fac")
    N.link(tree, twinkle, tint, "Fac", "Fac")
    tree.links.new(stars.outputs["Color"], star_a)
    tree.links.new(tint.outputs["Color"], star_b)
    base_link = gradient.inputs["Color"].links
    if base_link:
        tree.links.new(base_link[0].from_socket, sky_a)
    tree.links.new(star_out, sky_b)
    tree.links.new(sky_out, gradient.inputs["Color"])
    N.link(tree, gradient, N.world_output(tree), 0, "Surface")
    return world


def build_plasma(scene: bpy.types.Scene, top: str, bottom: str, accent: str, strength: float) -> bpy.types.World:
    """Turbulent energy field: distorted noise with hot filaments."""
    world, tree = _world(scene)
    coord = N.new(tree, "ShaderNodeTexCoord", (-1200, 0))
    turbulence = N.new(
        tree,
        "ShaderNodeTexNoise",
        (-1000, -220),
        Scale=3.5,
        Detail=10.0,
        Roughness=0.75,
        Distortion=2.5,
    )
    filaments = N.new(tree, "ShaderNodeValToRGB", (-800, -220))
    N.ramp(
        filaments,
        [
            (0.38, bottom),
            (0.55, mix_hex(bottom, accent, 0.8)),
            (0.66, accent),
            (0.78, top),
            (0.92, "#ffffff"),
        ],
    )
    gradient = N.gradient_world(tree, top=darken(top, 0.35), bottom=darken(bottom, 0.5), strength=strength)
    mix_node, _, socket_a, socket_b, result = N.mix_rgb(tree, (-360, -260), "SCREEN", 0.75)

    N.link(tree, coord, turbulence, "Generated", "Vector")
    N.link(tree, turbulence, filaments, "Fac", "Fac")
    base_link = gradient.inputs["Color"].links
    if base_link:
        tree.links.new(base_link[0].from_socket, socket_a)
    tree.links.new(filaments.outputs["Color"], socket_b)
    tree.links.new(result, gradient.inputs["Color"])
    N.link(tree, gradient, N.world_output(tree), 0, "Surface")
    return world


def build_transparent(scene: bpy.types.Scene, top: str, bottom: str, accent: str, strength: float) -> bpy.types.World:
    """Dim neutral world plus a transparent film, for compositing elsewhere."""
    world = build_gradient(scene, top, bottom, accent, strength * 0.6)
    scene.render.film_transparent = True
    return world


@dataclass(frozen=True)
class Background:
    """A registered background recipe."""

    key: str
    label: str
    description: str
    builder: object


_BACKGROUND_LIST: tuple[Background, ...] = (
    Background("solid", "Solid", "Flat colour. Maximum focus on the structure.", build_solid),
    Background("gradient", "Gradient", "Vertical two-colour sweep.", build_gradient),
    Background("studio_white", "Studio White", "Bright neutral environment for print.", build_studio_white),
    Background("nebula", "Nebula", "Soft cosmic clouds.", build_nebula),
    Background("starfield", "Starfield", "Dark sky with sparse stars.", build_starfield),
    Background("plasma", "Plasma Field", "Turbulent energy with hot filaments.", build_plasma),
    Background("transparent", "Transparent film", "Renders with alpha for compositing.", build_transparent),
)

#: Background key -> :class:`Background`.
BACKGROUNDS: dict[str, Background] = {b.key: b for b in _BACKGROUND_LIST}

#: Default background key.
DEFAULT_BACKGROUND = "gradient"


def list_backgrounds() -> list[tuple[str, str, str]]:
    """Return ``(key, label, description)`` triples for UI enum construction."""
    return [(b.key, b.label, b.description) for b in _BACKGROUND_LIST]


def apply_background(
    scene: bpy.types.Scene,
    key: str,
    top: str = "#0d1b2a",
    bottom: str = "#05070f",
    accent: str = "#1b3b6f",
    strength: float = 1.0,
) -> bpy.types.World:
    """Build the requested world background.

    Args:
        scene: Target scene (its ``world`` is replaced).
        key: Background key.
        top: Colour towards the zenith.
        bottom: Colour towards the horizon/nadir.
        accent: Tint used by the procedural backgrounds.
        strength: World light contribution.

    Returns:
        The world data-block that was created or updated.
    """
    background = BACKGROUNDS.get(key, BACKGROUNDS[DEFAULT_BACKGROUND])
    if key != "transparent":
        scene.render.film_transparent = False
    return background.builder(scene, top, bottom, accent, strength)  # type: ignore[operator]


def add_backdrop(
    center: tuple[float, float, float],
    radius: float,
    color: str = "#101418",
    roughness: float = 0.6,
    name: str = "AV_Backdrop",
) -> bpy.types.Object:
    """Add a large curved-feel backdrop plane behind and below the structure.

    A physical backdrop gives contact shadows and reflections that a world
    background cannot; it is what makes "studio" renders read as photographed.

    Args:
        center: Structure centre in world space.
        radius: Structure bounding radius; the backdrop is scaled from it.
        color: Backdrop colour.
        roughness: Surface roughness.
        name: Object name (reused on repeated calls).

    Returns:
        The backdrop object.
    """
    from ..core.compat import ensure_group, link_object, set_principled

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    size = radius * 14.0
    verts = [(-size, -size, 0.0), (size, -size, 0.0), (size, size, 0.0), (-size, size, 0.0)]
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()

    existing = bpy.data.objects.get(name)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = (center[0], center[1], center[2] - radius * 1.6)
    link_object(obj, ensure_group("AV_Scene"))

    material, tree = N.new_material(f"{name}_mat")
    bsdf = N.new(tree, "ShaderNodeBsdfPrincipled", (0, 0))
    set_principled(bsdf, base_color=hex_to_linear(color), roughness=roughness, specular=0.35)
    N.link(tree, bsdf, N.material_output(tree), 0, "Surface")
    mesh.materials.append(material)
    return obj
