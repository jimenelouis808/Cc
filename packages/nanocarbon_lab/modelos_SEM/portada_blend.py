"""A cover-grade Blender project from one of the scenes' .extxyz.

    <python con bpy> modelos_SEM/portada_blend.py -- \
        --extxyz modelos_SEM/xyz/escena_fese.extxyz \
        --blend modelos_SEM/blend/portada_fese.blend [--png vista.png]

    blender -b -P modelos_SEM/portada_blend.py -- --extxyz ... --blend ...

The .blend that comes out is a finished studio, not a pile of geometry:
materials, a graded background, softboxes, a framed portrait camera
with depth of field, and Cycles set up to render. Open it and press
F12, or move the camera first.

**Every atom is an INSTANCE, which is why the file is small and opens
at once.** `render_cnt.py` builds one icosphere and one cylinder per
atom and bond in Python; at 40 000 atoms that is minutes of meshing and
a file in the hundreds of megabytes. Here each element is one mesh of
bare vertices with a single sphere instanced on them, and the C-C bonds
are one mesh of tiny triangles -- each perpendicular to its bond at the
midpoint -- with a single cylinder instanced on the faces. Cycles
instances natively, so the render costs no more memory either.

Needs numpy and bpy only, so it runs under a Blender application as it
is. Nothing here imports the package: the scene arrives as a file.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np

#: Sphere radius per element, in A. Carbon is a ball-and-stick node;
#: the metals are large enough to fuse into a solid particle.
RADIUS = {"C": 0.46, "Fe": 1.22, "Se": 1.30}

#: (base colour, metallic, roughness) per material.
MATERIALS = {
    "carbono": ((0.340, 0.390, 0.470), 1.0, 0.26),
    "Fe": ((0.900, 0.300, 0.060), 1.0, 0.20),
    "Fe_carburo": ((0.800, 0.160, 0.060), 1.0, 0.26),
    "Se": ((1.000, 0.860, 0.350), 1.0, 0.22),
}

BOND_MAX = 1.75      # A, C-C
BOND_RADIUS = 0.20   # A

#: Background, bottom to top: near-black into a deep petrol blue.
BACKDROP = ((0.0025, 0.0035, 0.0070, 1.0), (0.018, 0.050, 0.085, 1.0))


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


def material(name: str) -> bpy.types.Material:
    colour, metallic, roughness = MATERIALS[name]
    out = bpy.data.materials.new(name)
    out.use_nodes = True
    shader = out.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*colour, 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return out


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


def softbox(name: str, where: np.ndarray, size: float, colour: tuple,
            strength: float) -> None:
    """An emissive plane facing the origin, seen in reflections only."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=where.tolist())
    plane = bpy.context.active_object
    plane.name = name
    direction = -where / np.linalg.norm(where)
    plane.rotation_euler = (
        math.acos(float(np.clip(direction[2], -1.0, 1.0))), 0.0,
        math.atan2(float(direction[0]), -float(direction[1])))
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*colour, 1.0)
    emission.inputs["Strength"].default_value = strength
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    plane.data.materials.append(mat)
    plane.visible_camera = False


def backdrop() -> None:
    """A vertical gradient in window space: it stays put under the camera."""
    world = bpy.data.worlds.new("fondo")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    coords = nodes.new("ShaderNodeTexCoord")
    split = nodes.new("ShaderNodeSeparateXYZ")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = BACKDROP[0]
    ramp.color_ramp.elements[1].color = BACKDROP[1]
    ramp.color_ramp.interpolation = "EASE"
    background = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(coords.outputs["Window"], split.inputs[0])
    links.new(split.outputs["Y"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs[0], output.inputs["Surface"])


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extxyz", type=Path, required=True)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--png", type=Path, default=None,
                        help="also render a still here")
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=35.0)
    parser.add_argument("--fill", type=float, default=0.86,
                        help="fraction of the frame the scene takes")
    parser.add_argument("--lens", type=float, default=85.0)
    parser.add_argument("--resolution", type=int, nargs=2,
                        default=[2400, 3000])
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--carbide-group", type=int, nargs="*", default=[1, 2],
                        help="`grupo` values whose iron is carbide")
    args = parser.parse_args(argv)

    symbols, positions, groups = read_extxyz(args.extxyz)
    positions = positions - 0.5 * (positions.max(axis=0)
                                   + positions.min(axis=0))
    symbols = np.asarray(symbols)
    radius = float(np.linalg.norm(positions, axis=1).max())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    mats = {name: material(name) for name in MATERIALS}

    # --- atoms: one carrier of vertices per material.
    iron = symbols == "Fe"
    carbide = iron & np.isin(groups, args.carbide_group)
    has_selenium = bool((symbols == "Se").any())
    if has_selenium:                     # FeSe: one iron, no carbide
        carbide[:] = False
    classes = (("carbono", symbols == "C", RADIUS["C"]),
               ("Fe", iron & ~carbide, RADIUS["Fe"]),
               ("Fe_carburo", carbide, RADIUS["Fe"]),
               ("Se", symbols == "Se", RADIUS["Se"]))
    for name, mask, size in classes:
        if not mask.any():
            continue
        ball = primitive(
            f"esfera_{name}",
            lambda size=size: bpy.ops.mesh.primitive_ico_sphere_add(
                subdivisions=2 if size < 1.0 else 3, radius=size),
            mats[name])
        instanced(f"atomos_{name}", positions[mask], [], ball, "VERTS")

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
    stick = primitive(
        "enlace",
        lambda: bpy.ops.mesh.primitive_cylinder_add(
            vertices=10, radius=BOND_RADIUS, depth=float(np.median(length)),
            end_fill_type="NOTHING"),
        mats["carbono"])
    instanced("enlaces_CC", corners, faces, stick, "FACES")

    # --- the studio. Softboxes scale with the scene, so the light on a
    # 100 A model and on a 600 A one is the same light.
    backdrop()
    softbox("luz_principal", radius * np.array([-2.2, -1.6, 2.6]),
            2.6 * radius, (1.00, 0.95, 0.88), 9.0)
    softbox("luz_contra", radius * np.array([2.4, 2.2, 1.0]),
            1.8 * radius, (0.35, 0.80, 1.00), 14.0)
    softbox("luz_relleno", radius * np.array([1.6, -2.8, -0.9]),
            3.0 * radius, (0.55, 0.65, 1.00), 1.6)
    sun = link(bpy.data.objects.new("sol", bpy.data.lights.new("sol", "SUN")))
    sun.data.energy = 2.2
    sun.data.angle = math.radians(12.0)
    sun.rotation_euler = (math.radians(48.0), 0.0, math.radians(-38.0))

    # --- the camera: a portrait frame the scene fills, focused on the
    # centre, with the far side falling softly out of focus.
    width, height = args.resolution
    camera = link(bpy.data.objects.new("camara",
                                       bpy.data.cameras.new("camara")))
    scene.camera = camera
    camera.data.lens = args.lens
    sensor = camera.data.sensor_width
    narrow = sensor * min(width, height) / max(width, height)
    half_angle = math.atan(0.5 * narrow / args.lens)
    distance = radius / (math.sin(half_angle) * args.fill)
    elev, azim = math.radians(args.elev), math.radians(args.azim)
    toward = np.array([math.cos(elev) * math.sin(azim),
                       -math.cos(elev) * math.cos(azim), math.sin(elev)])
    camera.location = (distance * toward).tolist()
    target = link(bpy.data.objects.new("foco", None))
    track = camera.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis, track.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    camera.data.clip_start = 0.01 * distance
    camera.data.clip_end = distance + 4.0 * radius
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = target
    # Blender reads 1 unit as 1 m, so a 500-unit model is a mountain
    # and a real f-number leaves it all sharp. Solve for the f-number
    # that blurs a point one scene radius behind the focus by 1/1500 of
    # the frame.
    focal = args.lens / 1000.0
    blur = (sensor / 1000.0) / 1500.0
    camera.data.dof.aperture_fstop = max(
        focal * focal / (blur * distance) * radius / (distance + radius),
        1e-4)

    # --- render settings.
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.film_transparent = False
    for transform in ("AgX", "Filmic"):
        try:
            scene.view_settings.view_transform = transform
            break
        except TypeError:
            continue
    for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue

    args.blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()),
                                compress=True)
    print(f"{args.blend}: {len(positions)} átomos, {len(pairs)} enlaces "
          f"C-C, radio {radius:.0f} Å, cámara a {distance:.0f} Å, "
          f"f/{camera.data.dof.aperture_fstop:.4f}")
    if args.png is not None:
        scene.render.filepath = str(args.png.resolve())
        bpy.ops.render.render(write_still=True)
        print(f"{args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
