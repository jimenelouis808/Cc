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
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    p.add_argument("--atom-scale", type=float, default=1.0,
                   help="Multiplies every atom radius.")
    p.add_argument("--bond-scale", type=float, default=1.0,
                   help="Multiplies the bond cylinder radius.")
    p.add_argument("--out", default="render.png", help="Output image path.")
    p.add_argument("--resolution", type=int, nargs=2, default=[1920, 1920])
    p.add_argument("--samples", type=int, default=None, help="Overrides the style's default.")
    p.add_argument("--transparent-background", action="store_true")
    p.add_argument("--save-blend", default=None,
                   help="Write the assembled scene to this .blend file.")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="Assemble the scene but do not render it.")
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
        out = nodes.get("Material Output")
        for node in list(nodes):
            if node is not out:
                nodes.remove(node)
        if out is None:
            out = nodes.new("ShaderNodeOutputMaterial")
        shader = _shader_node(nodes, spec)
        links.new(shader.outputs[0], out.inputs["Surface"])
        return mat

    bsdf = nodes.get("Principled BSDF")

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
    return mat


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

    coord = nodes.new("ShaderNodeTexCoord")
    gradient = nodes.new("ShaderNodeTexGradient")
    gradient.gradient_type = "LINEAR"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*background.bottom, 1.0)
    ramp.color_ramp.elements[1].color = (*background.top, 1.0)
    links.new(coord.outputs["Generated"], gradient.inputs["Vector"])
    links.new(gradient.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bg.inputs["Color"])


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
                                roughness=background.floor_roughness),
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


#: Fraction of the frame the subject should fill along its tightest axis.
#: Below 1.0 leaves a margin; cover art wants the subject to breathe.
FRAME_FILL = 0.82


def _framing_distance(cam_data, radius: float, resolution) -> float:
    """How far the camera must sit to fit a sphere of ``radius`` in frame.

    This used to be ``radius * 3.2``, a constant, and that is only
    correct for one lens. The half-angle a camera sees is
    ``atan(sensor / 2f)``, so fitting a sphere needs ``radius / sin(that)``
    -- 2.95 R at 50 mm but 4.83 R at 85 mm and 5.10 R at 90 mm. Four of
    the five styles use a longer lens than 50 mm, so four of five cropped
    the subject off the edges of the frame.

    The aspect ratio matters too. Blender's automatic sensor fit maps the
    sensor width onto the **longer** image axis, so the shorter axis sees
    a narrower angle and is the one that clips. A 2000x2400 cover crops
    vertically long before it crops horizontally.
    """
    import math

    sensor = cam_data.sensor_width
    width, height = resolution
    half_width = math.atan(0.5 * sensor / cam_data.lens)
    # The narrower of the two half-angles governs what fits.
    aspect = min(width, height) / max(width, height)
    half_fov = math.atan(math.tan(half_width) * aspect)
    return radius / math.sin(half_fov) / FRAME_FILL


def _setup_camera(style, objects, lens_mm: float, resolution) -> None:
    center, radius = _bounding_sphere(objects)
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens_mm
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    distance = _framing_distance(cam_data, radius, resolution)
    direction = Vector((1.0, -1.4, 0.6)).normalized()
    cam_obj.location = center + direction * distance
    look = center - cam_obj.location
    cam_obj.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()

    if style.camera_dof_fstop is not None:
        cam_data.dof.use_dof = True
        cam_data.dof.focus_distance = distance
        # The f-stop is set after the focus distance because the depth of
        # field it produces has to be judged against the subject. A fixed
        # f-number blurs a 3 Å cage and a 45 Å network by wildly
        # different amounts; scaling the aperture with the focal length
        # and the subject keeps the *relative* depth of field constant,
        # which is what the style is actually asking for.
        cam_data.dof.aperture_fstop = style.camera_dof_fstop * max(
            1.0, distance / (10.0 * FRAME_FILL)
        )


def _setup_render(style, args) -> None:
    scene = bpy.context.scene
    scene.render.engine = style.engine
    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    # Either the flag or the style's own backdrop opens the alpha. The
    # flag is kept because it predates ``Background`` and a caller may
    # want any style cut out, not only the one built for it.
    scene.render.film_transparent = (
        args.transparent_background or style.background.is_transparent
    )
    if style.engine == "CYCLES":
        scene.cycles.samples = args.samples or style.samples
        try:
            scene.cycles.device = "GPU"
        except Exception:
            pass
    scene.render.filepath = str(Path(args.out).resolve())
    scene.render.image_settings.file_format = "PNG"
    # RGBA, or a transparent film is composited onto black on write and
    # the alpha this was all for never reaches the file.
    scene.render.image_settings.color_mode = (
        "RGBA" if scene.render.film_transparent else "RGB"
    )


def main() -> int:
    args = _parse_args()
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

    bpy.ops.wm.read_factory_settings(use_empty=True)
    style = style_defs.get_style(args.style)
    if args.background is not None:
        style = style_defs.with_background(style, args.background)
    colour_by = style_defs.resolve_colour_by(args.colour_by, style)

    bundle = mesh_builder.load_bundle(args.xyz, args.json)

    objects: list[bpy.types.Object] = []
    if args.mode in ("ballstick", "both"):
        objects += mesh_builder.build_ball_and_stick(
            bundle, style, colour_by=colour_by,
            atom_scale=args.atom_scale,
            bond_radius=0.12 * args.bond_scale,
        )
    if args.mode in ("surface", "both"):
        objects.append(mesh_builder.build_smooth_surface(bundle, style))

    # The light rig and the camera both need the subject's size, so it is
    # measured once here rather than twice with a risk of disagreeing --
    # and before the floor exists, since a plane two dozen radii across
    # would otherwise be what the camera frames.
    center, radius = _bounding_sphere(objects)
    _setup_world(style.background)
    floor = style.background.has_floor if args.floor is None else args.floor
    if floor:
        _add_floor(style.background, center, radius)
    _setup_lights(style, center, radius)
    _setup_camera(style, objects, style.camera_lens_mm, args.resolution)
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
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
