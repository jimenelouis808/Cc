"""A cover project in Blender from one of the scenes' .extxyz, fast.

    <python con bpy> modelos_SEM/portada_blend.py -- \
        --extxyz modelos_SEM/xyz/escena_placas.extxyz \
        --blend modelos_SEM/blend/escena_placas.blend [--png vista.png]

    blender -b -P modelos_SEM/portada_blend.py -- --extxyz ... --blend ...

The studio is the one already chosen for these samples: the package's
`core_shell_cover` style, applied by `render_cnt.py`'s own functions --
its materials, world, floor, light rig, measured camera and render
settings. Nothing about the look is redefined here, so a change to the
style reaches this script by itself; `--style` takes any other.

What IS different is the geometry, and it is the whole reason for the
script. **Every atom is an INSTANCE.** `render_cnt.py` builds one
icosphere and one cylinder per atom and bond in Python: 88 minutes
measured for 10 000 atoms, and a file in the hundreds of megabytes.
Here each element is one mesh of bare vertices with a single sphere
instanced on them, and the C-C bonds are one mesh of tiny triangles --
each perpendicular to its bond at the midpoint -- with a single
cylinder instanced on the faces. A 78 000-atom scene builds in seconds
into a file of a few megabytes, and Cycles instances natively.

On top of the style, the camera gets depth of field focused on the
scene's centre (`--no-dof` to leave it out). The style's glare is a
pass over the finished PNG, so it is applied to `--png` and is not
something the .blend carries.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nanocarbon_lab" / "blender"))

import effects as fx  # noqa: E402
import render_cnt  # noqa: E402
import styles as style_defs  # noqa: E402

#: Sphere radius of the metals, in A: large enough that neighbours
#: fuse and a core reads as a solid. Carbon takes the style's own
#: ball-and-stick radius.
SOLID_RADIUS = {"Fe": 1.22, "Se": 1.30}

BOND_MAX = 1.75      # A, C-C

#: Van der Waals radii for the ``revista`` look, in A: Fe from Batsanov
#: (2001), Se and C from Bondi (1964). At these the metal is a closed
#: space-filling solid, which is how a cover draws a particle.
VDW_RADIUS = {"Fe": 2.00, "Se": 1.90}

#: The ``revista`` palette: a deep indigo field, carbon as pearl with a
#: cold rim, iron as polished copper and selenium as teal -- warm
#: against cold, so the metal and the shell never merge into one hue.
REVISTA_MATERIALS = {
    "C": style_defs.MaterialSpec(
        base_color=(0.62, 0.70, 0.80, 1.0), metallic=0.85, roughness=0.18,
        clearcoat=0.6, coat_roughness=0.05, specular_tint=(0.80, 0.90, 1.0),
        rim=2.2, rim_color=(0.35, 0.80, 1.00), rim_power=3.0),
    "Fe": style_defs.MaterialSpec(
        base_color=(0.96, 0.43, 0.15, 1.0), metallic=1.0, roughness=0.16,
        clearcoat=0.4, coat_roughness=0.04, specular_tint=(1.0, 0.70, 0.45),
        rim=0.8, rim_color=(1.00, 0.55, 0.25), rim_power=4.0),
    "Se": style_defs.MaterialSpec(
        base_color=(0.04, 0.58, 0.62, 1.0), metallic=0.9, roughness=0.20,
        clearcoat=0.4, coat_roughness=0.05, specular_tint=(0.55, 1.0, 0.95),
        rim=0.8, rim_color=(0.30, 1.00, 0.90), rim_power=4.0),
}
REVISTA_BOND = dataclasses.replace(REVISTA_MATERIALS["C"], rim=1.4)


def revista_style(base: style_defs.Style) -> style_defs.Style:
    """`base` re-lit for a high-impact cover.

    The field is a deep indigo graded to near-black, so the subject's
    cold rim reads against it; the key is warm and high, a strong cold
    back light draws every silhouette, and a magenta kicker low on the
    far side gives the shadows a colour instead of black.
    """
    return dataclasses.replace(
        base,
        name="revista",
        # A black mirror under the subject: dark enough to stay a
        # floor, glossy enough to give back a soft copy of the scene.
        background=style_defs.Background(
            kind="gradient", top=(0.020, 0.024, 0.075),
            bottom=(0.002, 0.003, 0.010), strength=1.0,
            floor=True, floor_color=(0.006, 0.007, 0.016),
            floor_roughness=0.07, floor_metallic=0.85),
        lights=[
            style_defs.LightSpec("AREA", 5800.0, (1.00, 0.88, 0.70),
                                 (7.0, -7.0, 11.0), size=7.0),
            style_defs.LightSpec("AREA", 7000.0, (0.40, 0.75, 1.00),
                                 (-4.0, 9.0, 10.0), size=5.0),
            style_defs.LightSpec("AREA", 2200.0, (0.85, 0.30, 0.95),
                                 (-9.0, -3.0, -4.0), size=10.0),
            style_defs.LightSpec("AREA", 1200.0, (0.55, 0.65, 1.00),
                                 (0.0, -10.0, 1.0), size=16.0),
        ],
        effects=(fx.EffectSpec("glare", {"strength": 0.42, "size": 0.55,
                                         "threshold": 0.70,
                                         "maximum": 3.0}),),
    )


def read_extxyz(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Symbols, positions and the `grupo` column (zeros if absent)."""
    lines = path.read_text().splitlines()
    count = int(lines[0])
    symbols, rows = [], []
    for line in lines[2:2 + count]:
        fields = line.split()
        symbols.append(fields[0])
        rows.append([float(v) for v in fields[1:5]] if len(fields) > 4
                    else [float(v) for v in fields[1:4]] + [0.0])
    table = np.asarray(rows)
    return symbols, table[:, :3], table[:, 3].astype(int)


def carbon_bonds(positions: np.ndarray, cutoff: float) -> np.ndarray:
    """Pairs closer than `cutoff`, by a sweep along x. numpy only."""
    order = np.argsort(positions[:, 0])
    sorted_positions = positions[order]
    last = np.searchsorted(sorted_positions[:, 0],
                           sorted_positions[:, 0] + cutoff)
    pairs = []
    for index in range(len(order)):
        window = sorted_positions[index + 1:last[index]]
        if not len(window):
            continue
        near = np.flatnonzero(
            ((window - sorted_positions[index]) ** 2).sum(axis=1)
            < cutoff * cutoff)
        pairs.extend((order[index], order[index + 1 + k]) for k in near)
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def link(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.scene.collection.objects.link(obj)
    return obj


def instanced(name: str, vertices: np.ndarray, faces: list,
              child: bpy.types.Object, kind: str) -> bpy.types.Object:
    """A carrier mesh with `child` instanced on its vertices or faces."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.update()
    carrier = link(bpy.data.objects.new(name, mesh))
    carrier.instance_type = kind
    carrier.show_instancer_for_render = False
    carrier.show_instancer_for_viewport = False
    child.parent = carrier
    return carrier


def primitive(name: str, make, mat: bpy.types.Material) -> bpy.types.Object:
    """One smooth-shaded primitive at the origin, carrying `mat`."""
    make()
    obj = bpy.context.active_object
    obj.name = name
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.data.materials.append(mat)
    return obj


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extxyz", type=Path, required=True)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--png", type=Path, default=None,
                        help="also render a still here")
    parser.add_argument("--style", default="core_shell_cover",
                        choices=[*style_defs.list_styles(), "revista"],
                        help="a package style, or `revista`: this "
                             "script's high-impact look, metals at vdW")
    parser.add_argument("--efecto", action="append", default=[],
                        help="an extra effect, as render_cnt takes it, "
                             "e.g. bokeh:color=cyan,count=40 (repeatable)")
    parser.add_argument("--view", default="auto")
    parser.add_argument("--elev", type=float, default=None,
                        help="camera elevation in degrees; with --azim, "
                             "overrides --view. On a floor this sets "
                             "where the horizon falls: higher puts it "
                             "above the subject")
    parser.add_argument("--azim", type=float, default=0.0,
                        help="camera azimuth in degrees from the front")
    parser.add_argument("--fill", type=float, default=0.95,
                        help="fraction of the frame the scene takes")
    parser.add_argument("--resolution", type=int, nargs=2,
                        default=[1600, 2000])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--atom-scale", type=float, default=0.55)
    parser.add_argument("--bond-scale", type=float, default=0.5)
    parser.add_argument("--no-dof", dest="dof", action="store_false")
    parser.add_argument("--floor", action="store_true",
                        help="keep the style's glossy floor. Off by "
                             "default for the package styles, on for "
                             "`revista`")
    parser.add_argument("--sin-piso", dest="no_floor", action="store_true",
                        help="no floor, even for `revista`")
    args = parser.parse_args(argv)

    symbols, positions, _groups = read_extxyz(args.extxyz)
    symbols = np.asarray(symbols)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    revista = args.style == "revista"
    style = (revista_style(style_defs.get_style("core_shell_cover"))
             if revista else style_defs.get_style(args.style))

    # --- atoms: one carrier of vertices per element, in the style's
    # own element materials.
    present = [key for key in ("C", "Fe", "Se") if (symbols == key).any()]
    specs = ([REVISTA_MATERIALS.get(key, REVISTA_MATERIALS["C"])
              for key in present] if revista
             else style_defs.materials_for(style, "element", present,
                                           "style"))
    radii = VDW_RADIUS if revista else SOLID_RADIUS
    carriers = []
    for key, spec in zip(present, specs, strict=True):
        size = radii.get(
            key, style_defs.element_radius(key) * args.atom_scale
            * style.atom_scale)
        ball = primitive(
            f"esfera_{key}",
            lambda size=size: bpy.ops.mesh.primitive_ico_sphere_add(
                subdivisions=2 if size < 1.0 else 3, radius=size),
            render_cnt.material_from_spec(f"mat_element_{key}", spec))
        carriers.append(instanced(f"atomos_{key}", positions[symbols == key],
                                  [], ball, "VERTS"))

    # --- bonds: one tiny triangle per C-C pair, normal along the bond.
    carbon = positions[symbols == "C"]
    pairs = carbon_bonds(carbon, BOND_MAX)
    start, end = carbon[pairs[:, 0]], carbon[pairs[:, 1]]
    axis = end - start
    length = np.linalg.norm(axis, axis=1)
    axis /= length[:, None]
    helper = np.where(np.abs(axis[:, :1]) < 0.9, [[1.0, 0.0, 0.0]],
                      [[0.0, 1.0, 0.0]])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(axis, u)
    middle = 0.5 * (start + end)
    corners = np.empty((len(pairs) * 3, 3))
    for k, angle in enumerate((0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)):
        corners[k::3] = middle + 0.05 * (np.cos(angle) * u
                                         + np.sin(angle) * v)
    faces = [(3 * k, 3 * k + 1, 3 * k + 2) for k in range(len(pairs))]
    bond_specs, _split = style_defs.bond_materials(
        style, "element", ["C"], "auto", "style")
    stick = primitive(
        "enlace",
        lambda: bpy.ops.mesh.primitive_cylinder_add(
            vertices=10, radius=0.12 * args.bond_scale * style.bond_scale,
            depth=float(np.median(length)), end_fill_type="NOTHING"),
        render_cnt.material_from_spec(
            "mat_bond", REVISTA_BOND if revista else bond_specs[0]))
    carriers.append(instanced("enlaces_CC", corners, faces, stick, "FACES"))

    # --- the studio: the style's own, placed by the pipeline's own
    # functions against this scene's size.
    bpy.context.view_layer.update()
    center, radius = render_cnt._bounding_sphere(carriers)
    render_cnt._setup_world(style.background)
    if args.elev is not None:
        elev, azim = np.radians(args.elev), np.radians(args.azim)
        render_cnt.camera.NAMED_VIEWS["portada"] = (
            float(np.cos(elev) * np.sin(azim)),
            float(-np.cos(elev) * np.cos(azim)), float(np.sin(elev)))
        args.view = "portada"
    use_floor = (args.floor or revista) and not args.no_floor
    if use_floor and args.view == "auto":
        # A floor wants the camera low. Seen from above, the auto view's
        # usual pick, it fills the frame and mirrors the light rig.
        args.view = "hero"
    if use_floor and style.background.has_floor:
        render_cnt._add_floor(style.background, center, radius,
                              bottom=float(positions[:, 2].min()) - 1.5)
    render_cnt._setup_lights(style, center, radius)
    # The pipeline's floor is FLOOR_EXTENT radii across, so its far edge
    # shows as a horizon line through the middle of the frame. Stretched
    # far past the camera's reach, the edge goes to infinity and the
    # floor fades into the field. (Hiding the lamps from reflections, to
    # keep them out of the mirror, was tried and undone: a metal's whole
    # brightness IS its reflection of the lamps, and the scene went dull.)
    floor_obj = bpy.data.objects.get("Floor")
    if floor_obj is not None:
        floor_obj.scale = (60.0, 60.0, 1.0)
        # Light linking: every lamp lights the subject and NOT the
        # floor, so the mirror gives back the scene and never a lamp.
        # Without it each lamp came back as a white rectangle under the
        # subject -- the back light even with the anemone's silhouette
        # in it -- and moving or hiding lamps from reflections either
        # only moved the rectangle or left the metal dull, since a
        # metal's highlights ARE its reflection of the lamps. The
        # collection holds the floor alone, marked EXCLUDE, so
        # everything else stays lit.
        excluded = bpy.data.collections.new("sin_luz_directa")
        excluded.objects.link(floor_obj)
        excluded.collection_objects[0].light_linking.link_state = "EXCLUDE"
        for obj in bpy.context.scene.objects:
            if obj.type == "LIGHT":
                obj.light_linking.receiver_collection = excluded
    centre, framed = render_cnt._setup_camera(
        style, carriers, style.camera_lens_mm, args.resolution,
        view=args.view, fill=args.fill, points=positions)
    specs = list(style.effects) + [fx.parse_effect(text)
                                   for text in args.efecto]
    fx.build_effects(specs, centre, framed, positions)
    render_cnt._setup_render(style, SimpleNamespace(
        engine="cycles", resolution=args.resolution,
        transparent_background=False, samples=args.samples,
        out=str(args.png or args.blend.with_suffix(".png"))))
    scene = bpy.context.scene
    scene.cycles.use_denoising = True
    try:
        scene.cycles.device = "CPU" if not render_cnt._can_rasterise() \
            else scene.cycles.device
    except Exception:
        pass

    camera = scene.camera
    if args.dof and camera is not None:
        # Blender reads 1 unit as 1 m, so a 500-unit model is a mountain
        # and a real f-number leaves it all sharp. Solve for the
        # f-number that blurs a point one scene radius behind the focus
        # by 1/1500 of the frame: the far arms soften, nothing smears.
        target = link(bpy.data.objects.new("foco", None))
        target.location = center
        distance = float((camera.location - center).length)
        focal = camera.data.lens / 1000.0
        blur = (camera.data.sensor_width / 1000.0) / 1500.0
        camera.data.dof.use_dof = True
        camera.data.dof.focus_object = target
        camera.data.dof.aperture_fstop = max(
            focal * focal / (blur * distance) * radius / (distance + radius),
            1e-4)

    args.blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()),
                                compress=True)
    print(f"{args.blend}: {len(positions)} átomos, {len(pairs)} enlaces "
          f"C-C, estilo {args.style}, radio {radius:.0f} Å")
    if args.png is not None:
        bpy.ops.render.render(write_still=True)
        for spec in specs:
            if spec.kind == "glare":
                fx.apply_glare(str(args.png.resolve()), spec)
        print(f"{args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
