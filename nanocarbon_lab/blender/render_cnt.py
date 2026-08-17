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

Pick ``--style`` from ``styles.STYLES`` in the sibling ``styles.py``
(``nature_dark``, ``acs_nano_vivid``, ``small_minimal``,
``blueprint_technical``, ``gold_nanotech``) -- run with ``--list-styles``
to print descriptions without rendering anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mesh_builder  # noqa: E402  (must follow sys.path fix-up)
import styles as style_defs  # noqa: E402


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xyz", help="Path to the .xyz file.")
    p.add_argument("--json", default=None, help="Path to the .json sidecar (rings/bonds).")
    p.add_argument("--style", default="nature_dark", choices=style_defs.list_styles())
    p.add_argument("--mode", default="ballstick", choices=["ballstick", "surface", "both"])
    p.add_argument("--out", default="render.png", help="Output image path.")
    p.add_argument("--resolution", type=int, nargs=2, default=[1920, 1920])
    p.add_argument("--samples", type=int, default=None, help="Overrides the style's default.")
    p.add_argument("--transparent-background", action="store_true")
    p.add_argument("--list-styles", action="store_true", help="Print style descriptions and exit.")
    return p.parse_args(argv)


def material_from_spec(name: str, spec) -> "bpy.types.Material":
    """Build a Principled-BSDF material from a :class:`styles.MaterialSpec`.

    Input socket names on the Principled BSDF node changed between
    Blender 3.x and 4.x (e.g. plain ``"Emission"`` became
    ``"Emission Color"`` + a separate ``"Emission Strength"``, and
    ``"Specular"`` became ``"Specular IOR Level"``). This sets whichever
    of a small set of known aliases exists on the running Blender's node,
    so the same :class:`styles.MaterialSpec` renders correctly across
    versions instead of raising a ``KeyError``.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")

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
    return mat


def _setup_world(style) -> None:
    world = bpy.data.worlds.new("NanocarbonWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    coord = nodes.new("ShaderNodeTexCoord")
    gradient = nodes.new("ShaderNodeTexGradient")
    gradient.gradient_type = "LINEAR"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*style.world_color_bottom, 1.0)
    ramp.color_ramp.elements[1].color = (*style.world_color_top, 1.0)
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = style.world_strength
    out = nodes.new("ShaderNodeOutputWorld")

    links.new(coord.outputs["Generated"], gradient.inputs["Vector"])
    links.new(gradient.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def _setup_lights(style) -> None:
    for i, spec in enumerate(style.lights):
        light_data = bpy.data.lights.new(f"Light{i}", type=spec.kind)
        light_data.energy = spec.energy
        light_data.color = spec.color
        if spec.kind == "AREA":
            light_data.size = spec.size
        obj = bpy.data.objects.new(f"Light{i}", light_data)
        bpy.context.collection.objects.link(obj)
        obj.location = spec.location
        direction = Vector(spec.target) - Vector(spec.location)
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


def _setup_camera(style, objects, lens_mm: float) -> None:
    center, radius = _bounding_sphere(objects)
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens_mm
    if style.camera_dof_fstop is not None:
        cam_data.dof.use_dof = True
        cam_data.dof.aperture_fstop = style.camera_dof_fstop
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    distance = radius * 3.2
    direction = Vector((1.0, -1.4, 0.6)).normalized()
    cam_obj.location = center + direction * distance
    look = center - cam_obj.location
    cam_obj.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    if style.camera_dof_fstop is not None:
        cam_data.dof.focus_distance = distance


def _setup_render(style, args) -> None:
    scene = bpy.context.scene
    scene.render.engine = style.engine
    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    scene.render.film_transparent = args.transparent_background
    if style.engine == "CYCLES":
        scene.cycles.samples = args.samples or style.samples
        try:
            scene.cycles.device = "GPU"
        except Exception:
            pass
    scene.render.filepath = str(Path(args.out).resolve())
    scene.render.image_settings.file_format = "PNG"


def main() -> int:
    args = _parse_args()
    if args.list_styles:
        for name in style_defs.list_styles():
            print(f"{name}: {style_defs.get_style(name).description}")
        return 0
    if not args.xyz:
        print("error: --xyz is required (unless --list-styles).", file=sys.stderr)
        return 2

    bpy.ops.wm.read_factory_settings(use_empty=True)
    style = style_defs.get_style(args.style)

    positions, bonds, ring_sizes_per_atom = mesh_builder.load_bundle(args.xyz, args.json)

    objects: list["bpy.types.Object"] = []
    if args.mode in ("ballstick", "both"):
        objects += mesh_builder.build_ball_and_stick(
            positions, bonds, ring_sizes_per_atom, style
        )
    if args.mode in ("surface", "both"):
        objects.append(mesh_builder.build_smooth_surface(positions, bonds, style))

    _setup_world(style)
    _setup_lights(style)
    _setup_camera(style, objects, style.camera_lens_mm)
    _setup_render(style, args)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
