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
                        choices=style_defs.list_styles())
    parser.add_argument("--view", default="auto")
    parser.add_argument("--fill", type=float, default=0.95,
                        help="fraction of the frame the scene takes")
    parser.add_argument("--resolution", type=int, nargs=2,
                        default=[1600, 2000])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--atom-scale", type=float, default=0.55)
    parser.add_argument("--bond-scale", type=float, default=0.5)
    parser.add_argument("--no-dof", dest="dof", action="store_false")
    parser.add_argument("--floor", action="store_true",
                        help="keep the style's glossy floor. Off here: "
                             "seen from above it mirrors the light rig "
                             "as bright rectangles behind the subject")
    args = parser.parse_args(argv)

    symbols, positions, _groups = read_extxyz(args.extxyz)
    symbols = np.asarray(symbols)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    style = style_defs.get_style(args.style)

    # --- atoms: one carrier of vertices per element, in the style's
    # own element materials.
    present = [key for key in ("C", "Fe", "Se") if (symbols == key).any()]
    specs = style_defs.materials_for(style, "element", present, "style")
    carriers = []
    for key, spec in zip(present, specs, strict=True):
        size = SOLID_RADIUS.get(
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
        render_cnt.material_from_spec("mat_bond", bond_specs[0]))
    carriers.append(instanced("enlaces_CC", corners, faces, stick, "FACES"))

    # --- the studio: the style's own, placed by the pipeline's own
    # functions against this scene's size.
    bpy.context.view_layer.update()
    center, radius = render_cnt._bounding_sphere(carriers)
    render_cnt._setup_world(style.background)
    if args.floor and style.background.has_floor:
        render_cnt._add_floor(style.background, center, radius,
                              bottom=float(positions[:, 2].min()) - 1.5)
    render_cnt._setup_lights(style, center, radius)
    render_cnt._setup_camera(
        style, carriers, style.camera_lens_mm, args.resolution,
        view=args.view, fill=args.fill, points=positions)
    specs = list(style.effects)
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
