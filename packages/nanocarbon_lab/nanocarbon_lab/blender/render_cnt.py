#!/usr/bin/env blender --background --python
"""Render a nanocarbon_lab capped/defected CNT to a journal-cover-style PNG.

Run this **through Blender**, not through a plain Python interpreter --
``bpy``/``bmesh`` only exist inside Blender's embedded interpreter:

    blender -b -P nanocarbon_lab/blender/render_cnt.py -- \\
        --xyz out/cnt_cap.xyz --json out/cnt_cap.json \\
        --style nature_dark --mode ballstick \\
        --out out/cover.png --resolution 2000 2400 --samples 256

Everything after the lone ``--`` is this script's own argument list (the
part before it is Blender's own CLI, e.g. ``-b`` for background/headless
and ``-P`` to run a script). ``--xyz``/``--json`` are produced by
``nanocarbon cnt-cap ...`` (see the CLI) or directly by
:func:`nanocarbon_lab.exports.xyz.write_render_bundle`.

Pick ``--style`` from ``styles.STYLES`` in the sibling ``styles.py`` --
run with ``--list-styles`` to print every style with its description,
rather than keeping a copy of the list here that drifts from it.

Three axes are independent, and keeping them so is the point:

``--style``
    The *character* of the surfaces: metallic, roughness, clearcoat,
    which shader, what lights, what lens.
``--colour-by``
    What the hue *means*: ``ring`` (ring size, as this pipeline
    originally did and nothing else), ``element``, ``structure`` (which
    item of a composition an atom came from) or ``uniform``. Left out,
    each style's own ``default_colour_by`` decides, so a cover style
    keeps ring colouring and a figure style takes element.
``--background``
    The backdrop: ``gradient``, ``solid``, ``transparent`` or
    ``studio`` (a gradient plus a floor to sit on). Overrides the
    style's own.

``--save-blend out/scene.blend`` writes the assembled scene instead of
(or as well as) rendering it, which is how a scene gets opened in
Blender for adjusting by hand; ``--no-render`` skips the render
entirely. Run *without* ``-b`` and with ``--no-render`` to build the
scene inside an interactive Blender session.
"""

from __future__ import annotations

import argparse
import ctypes.util
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

import camera
import effects as fx
import mesh_builder
import styles as style_defs


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xyz", help="Path to the .xyz file.")
    p.add_argument("--json", default=None, help="Path to the .json sidecar (rings/bonds).")
    p.add_argument("--style", default="nature_dark", choices=style_defs.list_styles())
    p.add_argument("--mode", default="ballstick", choices=["ballstick", "surface", "both"])
    p.add_argument(
        "--colour-by", "--color-by", dest="colour_by", default=None,
        choices=list(style_defs.COLOUR_MODES),
        help="What an atom's colour means. Default: the style's own.",
    )
    p.add_argument(
        "--detail", default=None, choices=list(style_defs.DETAIL_LEVELS),
        help="Mesh resolution per atom. Chosen from the atom count when "
             "left out: a 60-atom cage gets every triangle, a "
             "2000-atom network gets the level where an atom is three "
             "pixels across and more would be invisible.",
    )
    p.add_argument(
        "--model", default="ballstick", choices=list(style_defs.MODELS),
        help="How the structure is drawn: ballstick, spacefill (van der "
             "Waals spheres, no sticks), auto (space-filling for metals "
             "and ball-and-stick for everything else, in one picture), "
             "licorice or wireframe. A style's own default is used when "
             "this is left out.",
    )
    p.add_argument(
        "--finish", default="style", choices=list(style_defs.FINISHES),
        help="Replace every material's character, keeping its colour: "
             "glass, glow, metal, brushed, wax... 'style' leaves the "
             "style's own alone. --list-finishes describes them.",
    )
    p.add_argument(
        "--engine", default=None, choices=["cycles", "eevee"],
        help="Renderer. Cycles (the default for every style) is a path "
             "tracer: real refraction, caustics and soft shadows, and "
             "slow. EEVEE is rasterised and takes seconds, for checking "
             "a composition before committing to the final render.",
    )
    p.add_argument(
        "--bond-finish", default=None, dest="bond_finish",
        choices=list(style_defs.FINISHES),
        help="A separate finish for the bonds. Default: the same as "
             "--finish. Glass atoms on metal bonds is two decisions.",
    )
    p.add_argument(
        "--bond-colour", "--bond-color", default="auto", dest="bond_colour",
        choices=list(style_defs.BOND_COLOURS),
        help="atoms: each half takes its own atom's colour. style: the "
             "style's single bond material. neutral: one grey, so the "
             "atoms carry all the colour. auto follows the colour mode.",
    )
    p.add_argument("--list-finishes", action="store_true",
                   dest="list_finishes",
                   help="Describe the finishes and exit.")
    p.add_argument(
        "--background", default=None, choices=list(style_defs.BACKGROUNDS),
        help="Override the style's backdrop.",
    )
    # Not BooleanOptionalAction: the default has to be None ("whatever the
    # style says") and not False, so --no-floor is a real third state
    # rather than the same as leaving it out.
    p.add_argument("--floor", dest="floor", action="store_true", default=None,
                   help="Add a ground plane whatever the background says.")
    p.add_argument("--no-floor", dest="floor", action="store_false",
                   help="Remove the ground plane a studio background adds.")
    p.add_argument(
        "--view", default="auto",
        choices=["auto", *sorted(camera.NAMED_VIEWS)],
        help="Which way to look from. 'auto' measures it: the direction "
             "whose silhouette shows the most structure, which is what "
             "stops a row of structures being photographed end-on.",
    )
    p.add_argument("--camera-fill", type=float, default=FRAME_FILL,
                   dest="camera_fill",
                   help="Fraction of the frame the subject fills along "
                        "whichever axis binds first.")
    p.add_argument("--lens", type=float, default=None,
                   help="Focal length in mm, overriding the style's.")
    p.add_argument("--atom-scale", type=float, default=1.0,
                   help="Multiplies every atom radius.")
    p.add_argument("--bond-scale", type=float, default=1.0,
                   help="Multiplies the bond cylinder radius.")
    p.add_argument("--out", default="render.png", help="Output image path.")
    p.add_argument(
        "--quality", default=None,
        choices=list(style_defs.QUALITY_PRESETS),
        help="Resolution and samples together: draft (640 px, 48), "
             "preview (1280, 128), print (2400, 512 -- 300 dpi across a "
             "journal column), cover (4096, 1024 -- 300 dpi across a "
             "page). Keeps the aspect ratio of --resolution. An explicit "
             "--samples still wins.",
    )
    p.add_argument("--resolution", type=int, nargs=2, default=[1920, 1920])
    p.add_argument("--samples", type=int, default=None, help="Overrides the style's default.")
    p.add_argument("--transparent-background", action="store_true")
    p.add_argument("--save-blend", default=None,
                   help="Write the assembled scene to this .blend file.")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="Assemble the scene but do not render it.")
    p.add_argument(
        "--effect", action="append", default=None, dest="effects",
        metavar="NAME:K=V,...",
        help="Add a scene effect. Repeatable. "
             f"One of: {', '.join(fx.EFFECTS)}. --list-effects prints "
             "every parameter.",
    )
    p.add_argument("--no-style-effects", dest="style_effects",
                   action="store_false",
                   help="Drop the effects the style brings of its own.")
    p.add_argument("--list-effects", action="store_true",
                   dest="list_effects",
                   help="Print the effects and their parameters, then exit.")
    p.add_argument("--list-styles", action="store_true", help="Print style descriptions and exit.")
    return p.parse_args(argv)


def material_from_spec(name: str, spec) -> bpy.types.Material:
    """Build a material from a :class:`styles.MaterialSpec`.

    ``spec.shader`` picks the surface node, because four of the five
    looks :data:`styles.SHADERS` offers cannot be expressed by
    Principled parameters at all: glass needs refraction through a
    closed surface, toon needs banded rather than smooth falloff,
    emission needs no surface response, and clay needs the specular
    lobe *gone* rather than merely rough. The remaining one, which is
    most styles, is Principled and takes every parameter.

    Input socket names on the Principled BSDF changed between Blender
    3.x and 4.x (plain ``"Emission"`` became ``"Emission Color"`` plus a
    separate ``"Emission Strength"``, ``"Specular"`` became
    ``"Specular IOR Level"``, ``"Subsurface"`` became
    ``"Subsurface Weight"``). This sets whichever of a small set of
    known aliases exists on the running Blender's node, so one
    :class:`styles.MaterialSpec` renders across versions instead of
    raising a ``KeyError``.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    if spec.shader != "principled":
        # Replace the default Principled node rather than leaving it
        # unlinked: an orphan node is harmless but it makes the material
        # confusing to adjust by hand after --save-blend, which is one of
        # the things this script is for.
        out = fx._fresh_tree(nodes)
        shader = _shader_node(nodes, spec)
        links.new(shader.outputs[0], out.inputs["Surface"])
        return mat

    bsdf = next((node for node in nodes
                 if node.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:                            # pragma: no cover - rare
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        links.new(bsdf.outputs[0], fx._output_node(nodes).inputs["Surface"])

    def set_input(aliases: list[str], value) -> None:
        for alias in aliases:
            if alias in bsdf.inputs:
                bsdf.inputs[alias].default_value = value
                return

    set_input(["Base Color"], spec.base_color)
    set_input(["Metallic"], spec.metallic)
    set_input(["Roughness"], spec.roughness)
    set_input(["Transmission Weight", "Transmission"], spec.transmission)
    set_input(["IOR"], spec.ior)
    set_input(["Coat Weight", "Clearcoat"], spec.clearcoat)
    set_input(["Emission Color", "Emission"], (*spec.emission_color, 1.0))
    set_input(["Emission Strength"], spec.emission_strength)
    set_input(["Subsurface Weight", "Subsurface"], spec.subsurface)
    set_input(["Subsurface Radius"], tuple(spec.subsurface_color))
    set_input(["Sheen Weight", "Sheen"], spec.sheen)
    set_input(["Anisotropic"], spec.anisotropy)
    set_input(["Anisotropic Rotation"], spec.anisotropy_rotation)
    set_input(["Coat Roughness"], spec.coat_roughness)
    set_input(["Coat Tint"], (*spec.coat_tint, 1.0))
    set_input(["Specular Tint"], (*spec.specular_tint, 1.0))
    # Thin-film interference arrived on the Principled BSDF in Blender
    # 4.3. On an older build the two sockets are simply absent and
    # set_input skips them, so a style that asks for iridescence renders
    # without it rather than failing.
    set_input(["Thin Film Thickness"], spec.thin_film)
    set_input(["Thin Film IOR"], spec.thin_film_ior)
    if spec.rim > 0:
        _add_rim(nodes, links, bsdf, spec)
    return mat


def _add_rim(nodes, links, bsdf, spec) -> None:
    """Add Fresnel-driven emission on top of a Principled surface.

    ``LayerWeight.Facing`` is 0 where the surface points at the camera
    and 1 where it turns away, so raising it to a power and driving an
    Emission with it lights exactly the silhouette -- every silhouette,
    including the far wall of a cage seen through the near one, which is
    what no amount of rim *lamps* can do.

    Wired as an Add Shader rather than a Mix, because a rim is light
    arriving on top of the material and not a replacement for it: mixing
    would dim the body to pay for the edge.
    """
    out = fx._output_node(nodes)
    weight = nodes.new("ShaderNodeLayerWeight")
    weight.inputs["Blend"].default_value = 0.5
    power = nodes.new("ShaderNodeMath")
    power.operation = "POWER"
    power.inputs[1].default_value = float(spec.rim_power)
    scale = nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = float(spec.rim)
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*spec.rim_color, 1.0)
    add = nodes.new("ShaderNodeAddShader")

    links.new(weight.outputs["Facing"], power.inputs[0])
    links.new(power.outputs[0], scale.inputs[0])
    links.new(scale.outputs[0], emission.inputs["Strength"])
    links.new(bsdf.outputs[0], add.inputs[0])
    links.new(emission.outputs[0], add.inputs[1])
    links.new(add.outputs[0], out.inputs["Surface"])


def _shader_node(nodes, spec):
    """The surface node for a non-Principled shader.

    Each is built from the same :class:`styles.MaterialSpec`, so a style
    can be re-coloured (``--colour-by element`` inside ``glass_cryo``)
    without knowing which shader it uses.
    """
    colour = spec.base_color
    if spec.shader == "glass":
        node = nodes.new("ShaderNodeBsdfGlass")
        node.inputs["Color"].default_value = colour
        node.inputs["Roughness"].default_value = spec.roughness
        node.inputs["IOR"].default_value = spec.ior
        return node
    if spec.shader == "toon":
        node = nodes.new("ShaderNodeBsdfToon")
        node.inputs["Color"].default_value = colour
        # Size is the width of the lit band and Smooth the softness of
        # its edge; a schematic wants a wide band and a hard edge.
        node.inputs["Size"].default_value = 0.6
        node.inputs["Smooth"].default_value = 0.05
        return node
    if spec.shader == "emission":
        node = nodes.new("ShaderNodeEmission")
        node.inputs["Color"].default_value = (
            *(spec.emission_color or colour[:3]), 1.0
        )
        node.inputs["Strength"].default_value = max(
            spec.emission_strength, 1.0
        )
        return node
    if spec.shader == "clay":
        node = nodes.new("ShaderNodeBsdfDiffuse")
        node.inputs["Color"].default_value = colour
        node.inputs["Roughness"].default_value = spec.roughness
        return node
    raise ValueError(f"no node setup for shader {spec.shader!r}.")


def _setup_world(background) -> None:
    """Build the world from a :class:`styles.Background`.

    All four kinds go through one node tree so they cannot diverge:
    ``solid`` is the gradient with both stops equal, and ``transparent``
    is a black world with the film alpha opened in
    :func:`_setup_render` -- the world still has to be *dark* rather than
    merely invisible, or it lights the subject from every direction and
    the alpha comes back over a washed-out render.
    """
    world = bpy.data.worlds.new("NanocarbonWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg = nodes.new("ShaderNodeBackground")
    out = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    if background.kind == "transparent":
        bg.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
        bg.inputs["Strength"].default_value = 0.2
        return

    bg.inputs["Strength"].default_value = background.strength
    if background.kind == "solid":
        bg.inputs["Color"].default_value = (*background.top, 1.0)
        return

    if background.has_stars:
        links.new(_starfield(nodes, links, background), bg.inputs["Color"])
        return

    coord = nodes.new("ShaderNodeTexCoord")
    gradient = nodes.new("ShaderNodeTexGradient")
    gradient.gradient_type = "LINEAR"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*background.bottom, 1.0)
    ramp.color_ramp.elements[1].color = (*background.top, 1.0)
    links.new(coord.outputs["Generated"], gradient.inputs["Vector"])
    links.new(gradient.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bg.inputs["Color"])


def _starfield(nodes, links, background):
    """A procedural night sky, and optionally a nebula behind it.

    Generated rather than loaded from an HDRI, which matters for three
    reasons: it needs no texture file, so nothing has to be shipped or
    found at render time; it is resolution independent, so a 6000 px
    cover has stars that are still points; and it is parametrised, so
    the density and brightness are dials rather than a fixed photograph.

    Stars are a **Voronoi** field, not noise. Noise gives soft blobs
    that read as dust; Voronoi has one sharp minimum per cell, so
    thresholding its distance just above zero leaves a scatter of points
    of genuinely different brightness -- which is what a sky looks like.

    Returns the socket carrying the finished colour.
    """
    coord = nodes.new("ShaderNodeTexCoord")
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.feature = "F1"
    voronoi.inputs["Scale"].default_value = float(background.star_density)
    links.new(coord.outputs["Generated"], voronoi.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    # A very tight ramp just above zero: only the cell centres survive,
    # and `star_amount` is how far up the distance field the cut is
    # made. Widen it and the sky fills with overlapping discs.
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    ramp.color_ramp.elements[1].position = max(1e-4,
                                               float(background.star_amount))
    ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    links.new(voronoi.outputs["Distance"], ramp.inputs["Fac"])

    bright = nodes.new("ShaderNodeMixRGB")
    bright.blend_type = "MULTIPLY"
    bright.inputs["Fac"].default_value = 1.0
    bright.inputs[2].default_value = (
        *(3 * [float(background.star_brightness)]), 1.0)
    links.new(ramp.outputs["Color"], bright.inputs[1])
    stars = bright.outputs["Color"]

    if background.kind != "nebula":
        return stars

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = float(background.nebula_scale)
    noise.inputs["Detail"].default_value = 8.0
    links.new(coord.outputs["Generated"], noise.inputs["Vector"])
    cloud = nodes.new("ShaderNodeValToRGB")
    cloud.color_ramp.elements[0].position = 0.42
    cloud.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cloud.color_ramp.elements[1].position = 0.70
    cloud.color_ramp.elements[1].color = (
        *background.nebula_colors[0], 1.0)
    links.new(noise.outputs["Fac"], cloud.inputs["Fac"])

    tint = nodes.new("ShaderNodeMixRGB")
    tint.blend_type = "ADD"
    tint.inputs["Fac"].default_value = float(background.nebula_strength)
    links.new(stars, tint.inputs[1])
    links.new(cloud.outputs["Color"], tint.inputs[2])
    return tint.outputs["Color"]


#: How far below the subject the floor sits, as a fraction of its radius.
#: Exactly 1.0 would put the plane through the lowest atom of a sphere;
#: a little under 1.05 reads as contact rather than as levitation.
FLOOR_DROP = 1.04

#: Floor extent, in subject radii. Large enough that its edge is outside
#: frame at every lens the styles use, since a visible edge reads as a
#: table rather than as a backdrop.
FLOOR_EXTENT = 24.0


def _add_floor(background, center: Vector, radius: float):
    """A ground plane for the subject to sit and cast onto.

    Returned so the caller can keep it *out* of the framing measurement:
    a plane 24 radii across would otherwise set the camera distance and
    the subject would be a speck in the middle of it.

    A shadow catcher is invisible itself but still receives shadow, so a
    transparent background keeps the contact shadow that makes a subject
    look placed rather than pasted.
    """
    mesh = bpy.data.meshes.new("Floor")
    plane = bpy.data.objects.new("Floor", mesh)
    bpy.context.collection.objects.link(plane)

    size = radius * FLOOR_EXTENT
    z = center.z - radius * FLOOR_DROP
    verts = [(center.x - size, center.y - size, z),
             (center.x + size, center.y - size, z),
             (center.x + size, center.y + size, z),
             (center.x - size, center.y + size, z)]
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()

    mat = material_from_spec(
        "mat_floor",
        style_defs.MaterialSpec(base_color=(*background.floor_color, 1.0),
                                roughness=background.floor_roughness,
                                metallic=background.floor_metallic),
    )
    mesh.materials.append(mat)
    if background.floor_is_shadow_catcher:
        # Cycles-only, and only from 3.0; a style asking for one on an
        # older build should still render rather than abort.
        try:
            plane.is_shadow_catcher = True
        except AttributeError:
            pass
    return plane


#: Subject radius the light rigs in ``styles.py`` are written against.
#: Positions and energies are scaled from this, so a style looks the same
#: on a 3 Å cage and a 45 Å network.
NOMINAL_RADIUS = 10.0


def _setup_lights(style, center: Vector, radius: float) -> None:
    """Place the style's lights, scaled to the subject.

    The rig used to be placed at the literal metres in
    :class:`styles.LightSpec` -- around 10 units from the origin -- while
    the camera distance scaled with the structure. Since these
    coordinates are Ångström, that meant the lights sat outside a
    fullerene, level with a nanotube, and *inside* a 45 Å network, whose
    far side then received nothing at all. Measured on ``nature_dark``,
    subject luminance fell from 0.50 on C60 to 0.32 on a 25 Å tube to
    **0.04** on a 90 Å one: a black subject on a black background.

    Scaling the positions with the subject and the energies with the
    square of that scale keeps the irradiance at the surface constant, so
    the style means the same thing at every size. That is the property a
    preset needs to have.
    """
    scale = max(radius, 1e-6) / NOMINAL_RADIUS
    for i, spec in enumerate(style.lights):
        light_data = bpy.data.lights.new(f"Light{i}", type=spec.kind)
        # A SUN is directional and infinitely distant, so its irradiance
        # does not fall off and must not be scaled; the others obey the
        # inverse square law and must.
        light_data.energy = (spec.energy if spec.kind == "SUN"
                             else spec.energy * scale**2)
        light_data.color = spec.color
        if spec.kind == "AREA":
            # Softness is relative: a 3 m softbox is a point source to a
            # 45 Å object and a wraparound to a 3 Å one.
            light_data.size = spec.size * scale
        obj = bpy.data.objects.new(f"Light{i}", light_data)
        bpy.context.collection.objects.link(obj)
        obj.location = center + Vector(spec.location) * scale
        direction = (center + Vector(spec.target) * scale) - obj.location
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _bounding_sphere(objects) -> tuple[Vector, float]:
    corners: list[Vector] = []
    for obj in objects:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))
    if not corners:
        return Vector((0, 0, 0)), 10.0
    center = sum(corners, Vector((0, 0, 0))) / len(corners)
    radius = max((c - center).length for c in corners)
    return center, max(radius, 1.0)


#: Fraction of the frame the subject should fill along whichever axis
#: binds first. Below 1.0 leaves a margin; cover art wants the subject to
#: breathe, and a structure touching the frame edge reads as cropped even
#: when it is not.
FRAME_FILL = 0.86


def _framing_distance(cam_data, radius: float, resolution) -> float:
    """Distance that fits a sphere of ``radius`` in frame.

    Kept because the *sphere* case is still the right question for the
    light rig and for anything round, and because it is the arithmetic
    that replaced a hardcoded ``3.2 * radius`` -- a constant that is only
    correct at 50 mm, while the styles use 50 to 90 mm, so most of them
    cropped the subject off the frame edge.

    :func:`camera.frame_camera` is what actually places the camera now;
    it frames the **projected bounding box** instead, which is the same
    answer for a sphere and a much better one for a row of structures.
    """
    sensor = cam_data.sensor_width
    width, height = resolution
    half_width = math.atan(0.5 * sensor / cam_data.lens)
    aspect = min(width, height) / max(width, height)
    half_fov = math.atan(math.tan(half_width) * aspect)
    return radius / math.sin(half_fov) / FRAME_FILL


def _setup_camera(style, objects, lens_mm: float, resolution,
                  view: str = "auto", fill: float = FRAME_FILL,
                  points=None) -> tuple:
    """Place the camera, choosing the angle when asked to.

    ``view`` is ``"auto"`` -- measured, see :func:`camera.best_view` --
    or any name in :data:`camera.NAMED_VIEWS`. The framing is always done
    on the projected bounding box rather than the bounding sphere.

    ``points`` are the atom positions when the caller has them, which is
    the honest input: an object's ``bound_box`` is an axis-aligned box
    around a whole merged mesh, so for a diagonal row it is mostly empty
    space and the silhouette measured from it is the box's, not the
    structure's. Falling back to the corners keeps this working for a
    skinned surface, which has no atom list of its own.

    Returns ``(centre, framed_radius)``: the subject's centre and the
    larger of its two **in-frame** half-extents. That is the scale
    anything decorative should be measured against -- see
    :func:`effects.build_effects`.
    """
    if points is None or len(points) < 3:
        points = _object_corners(objects)
    points = np.asarray(points, dtype=float)

    if view == "auto":
        direction = camera.best_view(points)
    else:
        direction = camera.named_direction(view)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens_mm
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    location, target, distance = camera.frame_camera(
        points, direction, lens_mm, cam_data.sensor_width, resolution,
        fill=fill,
    )
    half, _ = camera.projected_extent(points, direction)
    cam_obj.location = Vector(tuple(location))
    look = Vector(tuple(target)) - cam_obj.location
    cam_obj.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()

    if style.camera_dof_fstop is not None:
        cam_data.dof.use_dof = True
        cam_data.dof.focus_distance = distance
        # The f-stop is set after the focus distance because the depth of
        # field it produces has to be judged against the subject. A fixed
        # f-number blurs a 3 Å cage and a 45 Å network by wildly
        # different amounts; scaling the aperture with the subject keeps
        # the *relative* depth of field constant, which is what the style
        # is actually asking for.
        cam_data.dof.aperture_fstop = style.camera_dof_fstop * max(
            1.0, distance / (10.0 * FRAME_FILL)
        )
    return np.asarray(target, dtype=float), float(max(half[0], half[1]))


def _object_corners(objects):
    """Bounding-box corners of every object, in world coordinates."""
    corners = []
    for obj in objects:
        for corner in obj.bound_box:
            corners.append(tuple(obj.matrix_world @ Vector(corner)))
    return corners or [(0.0, 0.0, 0.0)]


#: What ``--engine`` names map to. Blender's own identifiers change
#: between versions (EEVEE was ``BLENDER_EEVEE_NEXT`` through 4.x), so
#: the aliases are tried in order and the first the running Blender
#: accepts wins.
ENGINE_IDS = {
    "cycles": ("CYCLES",),
    "eevee": ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"),
}


def _can_rasterise() -> bool:
    """Is there a GPU context for EEVEE to draw into?

    EEVEE is a **rasteriser**: it needs OpenGL, which headless Blender
    reaches through EGL. On a machine without it -- a container, a CI
    runner, a server with no GPU -- requesting EEVEE does not raise or
    return an error, it **aborts the process** with SIGABRT after
    printing one line about ``libEGL.so.1``. That cannot be caught from
    Python once it starts, so it has to be ruled out before.

    Cycles has no such requirement: it path-traces on the CPU when there
    is no GPU, which is why it is the default here and why this falls
    back to it rather than refusing.
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    # EGL specifically, not "any GL library". libGL is present on most
    # Linux boxes without a display and finding it proves nothing: this
    # container has libGL.so.1 and no libEGL, and the first version of
    # this check passed it straight through to the abort it exists to
    # prevent.
    return ctypes.util.find_library("EGL") is not None


def _set_engine(scene, wanted: str) -> str:
    """Set the render engine, and say which one actually took.

    Cycles is an **add-on**, so in a stock ``bpy`` wheel it is missing
    from ``render.engine``'s enum listing until it registers -- the
    listing is built before add-ons load. Assigning ``"CYCLES"`` works
    anyway and that is what these styles have always done; this falls
    back rather than assuming, so a build genuinely without Cycles
    renders in EEVEE and says so instead of dying.
    """
    if wanted == "eevee" and not _can_rasterise():
        print("note: EEVEE needs a GPU context (libEGL) and there is "
              "none here, so it would abort rather than fail; rendering "
              "with Cycles instead.", file=sys.stderr)
        wanted = "cycles"
    for identifier in ENGINE_IDS.get(wanted, (wanted,)):
        try:
            scene.render.engine = identifier
            return identifier
        except (TypeError, ValueError):
            continue
    print(f"note: {wanted} is not available in this Blender; "
          f"rendering with {scene.render.engine}.", file=sys.stderr)
    return scene.render.engine


def _setup_render(style, args) -> None:
    scene = bpy.context.scene
    wanted = args.engine or ("cycles" if style.engine == "CYCLES"
                             else "eevee")
    engine = _set_engine(scene, wanted)
    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    # Either the flag or the style's own backdrop opens the alpha. The
    # flag is kept because it predates ``Background`` and a caller may
    # want any style cut out, not only the one built for it.
    scene.render.film_transparent = (
        args.transparent_background or style.background.is_transparent
    )
    samples = args.samples or style.samples
    if engine == "CYCLES":
        scene.cycles.samples = samples
        try:
            scene.cycles.device = "GPU"
        except Exception:
            pass
    elif hasattr(scene, "eevee"):
        # EEVEE's samples are temporal anti-aliasing passes rather than
        # light paths, so the same number means something different and
        # far fewer are needed; capped so a Cycles-tuned 512 does not
        # make the "fast preview" slow.
        scene.eevee.taa_render_samples = max(16, min(128, samples))
    scene.render.filepath = str(Path(args.out).resolve())
    scene.render.image_settings.file_format = "PNG"
    # RGBA, or a transparent film is composited onto black on write and
    # the alpha this was all for never reaches the file.
    scene.render.image_settings.color_mode = (
        "RGBA" if scene.render.film_transparent else "RGB"
    )


def main() -> int:
    args = _parse_args()
    if args.list_effects:
        print(fx.describe_effects())
        return 0
    if args.list_finishes:
        print(style_defs.describe_finishes())
        return 0
    if args.list_styles:
        for name in style_defs.list_styles():
            style = style_defs.get_style(name)
            print(f"{name} [{style.background.kind}, "
                  f"colour by {style.default_colour_by}]: "
                  f"{style.description}")
        return 0
    if not args.xyz:
        print("error: --xyz is required (unless --list-styles).", file=sys.stderr)
        return 2

    if args.quality:
        preset = style_defs.QUALITY_PRESETS[args.quality]
        width, height = args.resolution
        # Scale the longer side to the preset and carry the aspect, so a
        # cover asked for at 2000x2400 stays 5:6 rather than becoming
        # square.
        factor = preset["long_side"] / max(width, height)
        args.resolution = [max(16, round(width * factor)),
                           max(16, round(height * factor))]
        if args.samples is None:
            args.samples = preset["samples"]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    style = style_defs.get_style(args.style)
    if args.background is not None:
        style = style_defs.with_background(style, args.background)
    colour_by = style_defs.resolve_colour_by(args.colour_by, style)
    # A style may prefer a representation -- a model-kit look is
    # ball-and-stick by definition, a metal product shot space-filling.
    # An explicit --model still wins.
    model = (args.model if args.model != "ballstick"
             else style.default_model)

    bundle = mesh_builder.load_bundle(args.xyz, args.json)

    objects: list[bpy.types.Object] = []
    if args.mode in ("ballstick", "both"):
        objects += mesh_builder.build_ball_and_stick(
            bundle, style, colour_by=colour_by,
            # The style's own scales multiply the command line's, so a
            # kit stays fat and a dense lattice stays fine without the
            # caller having to know which is which.
            atom_scale=args.atom_scale * style.atom_scale,
            bond_radius=0.12 * args.bond_scale * style.bond_scale,
            finish=args.finish, bond_finish=args.bond_finish,
            bond_colour=args.bond_colour, model=model,
            detail=args.detail,
        )
    if args.mode in ("surface", "both"):
        objects.append(mesh_builder.build_smooth_surface(
            bundle, style, finish=args.finish))

    # The light rig and the camera both need the subject's size, so it is
    # measured once here rather than twice with a risk of disagreeing --
    # and before the floor exists, since a plane two dozen radii across
    # would otherwise be what the camera frames.
    # The light rig and the floor are placed against the subject's
    # bounding sphere, which is what "how big is this thing" means for
    # them. Effects use the camera's framed extent instead; see below.
    center, radius = _bounding_sphere(objects)
    _setup_world(style.background)
    floor = style.background.has_floor if args.floor is None else args.floor
    if floor:
        _add_floor(style.background, center, radius)
    _setup_lights(style, center, radius)

    # The camera goes before the effects, and returns the scale they are
    # measured against. The bounding sphere is the wrong one for them: on
    # a row of three structures it is the row's half-diagonal, while the
    # camera frames the much tighter projected box, so arcs and
    # resonance rings written as "1.1 radii" came out larger than the
    # frame and cut across the picture. The framed half-extent makes
    # "just outside the subject" mean just outside what is on screen.
    centre, framed = _setup_camera(
        style, objects, args.lens or style.camera_lens_mm, args.resolution,
        view=args.view, fill=args.camera_fill, points=bundle["positions"])

    # Effect objects are deliberately not fed back into the framing: a
    # beam seven radii long would otherwise set the frame and leave the
    # structure a dot in the middle of it.
    specs = list(style.effects) if args.style_effects else []
    try:
        specs += [fx.parse_effect(text) for text in (args.effects or [])]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    fx.build_effects(specs, centre, framed, bundle["positions"])
    _setup_render(style, args)

    if args.save_blend:
        path = Path(args.save_blend).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        print(f"Wrote {path}")

    if not args.render:
        print(f"Assembled {len(objects)} objects; not rendering.")
        return 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    # Post passes run over the finished file. Only `glare` is one, and
    # it is why the effect appears in the catalogue but adds no objects.
    for spec in specs:
        if spec.kind == "glare":
            fx.apply_glare(args.out, spec)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
