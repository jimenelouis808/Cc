"""End-to-end check inside a real Blender.

Builds a small Atomic-Blender-shaped structure (or imports one when an XYZ
file is given), then applies **every** style, palette, background, light rig,
effect and cover look, and finally renders a tiny frame. Anything that only
Blender can validate — node names, socket names, property availability on this
exact version — fails loudly here.

Run either way::

    blender -b -P tools/integration_check.py
    python tools/integration_check.py            # when the `bpy` pip module is installed

Options after ``--``: ``--xyz FILE`` to use a real structure, ``--render``
to also render a 160 px frame per look (slow), ``--quick`` to test one look.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import bpy


def _bootstrap() -> None:
    """Make ``import atomviz_studio`` work when run as a loose script."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_args() -> argparse.Namespace:
    """Parse the arguments Blender leaves after ``--``."""
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="integration_check")
    parser.add_argument("--xyz", type=Path, default=None, help="Import this structure instead")
    parser.add_argument("--render", action="store_true", help="Render a small frame per look")
    parser.add_argument("--quick", action="store_true", help="Only exercise one look")
    parser.add_argument("--out", type=Path, default=Path("/tmp/atomviz_check"))
    return parser.parse_args(argv)


def build_fake_structure() -> None:
    """Recreate the object graph Atomic Blender produces, without the importer.

    ``Empty "test.xyz"`` -> ``Carbon_mesh`` (VERTS instancer) -> ``Carbon_ball``,
    same for nitrogen, plus a ``Sticks`` object.
    """
    import math

    root = bpy.data.objects.new("test.xyz", None)
    bpy.context.scene.collection.objects.link(root)

    def element(name: str, positions: list[tuple[float, float, float]], radius: float) -> None:
        mesh = bpy.data.meshes.new(f"{name}_mesh")
        mesh.from_pydata(positions, [], [])
        mesh.update()
        instancer = bpy.data.objects.new(f"{name}_mesh", mesh)
        instancer.instance_type = "VERTS"
        instancer.parent = root
        bpy.context.scene.collection.objects.link(instancer)

        ball = bpy.data.meshes.new(f"{name}_ball")
        # Cheap octahedron: enough to carry a material and be instanced.
        r = radius
        verts = [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]
        faces = [
            (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
            (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
        ]
        ball.from_pydata(verts, [], faces)
        ball.update()
        ball_obj = bpy.data.objects.new(f"{name}_ball", ball)
        ball_obj.parent = instancer
        bpy.context.scene.collection.objects.link(ball_obj)

    ring = [(math.cos(i * math.tau / 12) * 4.0, math.sin(i * math.tau / 12) * 4.0, 0.0) for i in range(12)]
    element("Carbon", ring + [(x, y, 2.5) for x, y, _ in ring], 0.35)
    element("Nitrogen", [(0.0, 0.0, 0.0), (0.0, 0.0, 2.5)], 0.4)

    stick_mesh = bpy.data.meshes.new("Sticks")
    stick_mesh.from_pydata([(0, 0, 0), (0, 0, 2.5), (0.1, 0, 0)], [], [(0, 1, 2)])
    stick_mesh.update()
    sticks = bpy.data.objects.new("Sticks", stick_mesh)
    sticks.parent = root
    bpy.context.scene.collection.objects.link(sticks)


def main() -> int:
    """Run every registry against a real Blender. Returns the exit code."""
    _bootstrap()
    args = parse_args()

    from atomviz_studio.core.detect import active_structure, detect_structures
    from atomviz_studio.core.palettes import PALETTES
    from atomviz_studio.core.presets import LOOKS
    from atomviz_studio.effects import backgrounds, electricity, lasers, lighting, postfx
    from atomviz_studio.looks.apply import apply_look, clear_all
    from atomviz_studio.materials.apply import apply_radii, apply_style
    from atomviz_studio.materials.styles import STYLES, StyleParams
    from atomviz_studio.scene.camera import frame_structure
    from atomviz_studio.scene.render import apply_render_preset, render_still

    failures: list[str] = []

    def step(label: str, function) -> object:
        """Run one step, recording the traceback instead of aborting the sweep."""
        try:
            result = function()
        except Exception as exc:  # noqa: BLE001 - this script exists to catch everything
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return None
        print(f"  ok  {label}")
        return result

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    print(f"Blender {bpy.app.version_string}")

    if args.xyz is not None:
        from atomviz_studio.ui.operators import enable_atomic_blender

        if not enable_atomic_blender():
            print("error: Atomic Blender unavailable in this build", file=sys.stderr)
            return 3
        bpy.ops.import_mesh.xyz(filepath=str(args.xyz.resolve()))
    else:
        build_fake_structure()

    structures = detect_structures(bpy.context, "SCENE")
    print(f"detected {len(structures)} structure(s)")
    structure = active_structure(bpy.context, "SCENE")
    if structure is None:
        print("error: detection found nothing", file=sys.stderr)
        return 4
    print(f"  {structure.name}: {structure.atom_count} atoms, {structure.symbols}, "
          f"{len(structure.sticks)} stick object(s)")
    center, radius = structure.bounds()
    print(f"  centre {tuple(round(c, 2) for c in center)}, radius {radius:.2f}")

    print("\n== styles ==")
    for key in STYLES:
        step(f"style {key}", lambda k=key: apply_style(
            structure, style_key=k, palette_key="midnight_neon",
            params=StyleParams(emissive_dopants=4.0), accent="#22d3ee",
        ))

    print("\n== palettes ==")
    for key in PALETTES:
        step(f"palette {key}", lambda k=key: apply_style(
            structure, style_key="glossy_ceramic", palette_key=k,
            params=StyleParams(), accent="#ff8800",
        ))

    print("\n== radii ==")
    for mode in ("vdw", "covalent", "uniform"):
        step(f"radii {mode}", lambda m=mode: apply_radii(structure, m, 1.0))

    print("\n== backgrounds ==")
    for key in backgrounds.BACKGROUNDS:
        step(f"background {key}", lambda k=key: backgrounds.apply_background(scene, k))
    step("backdrop object", lambda: backgrounds.add_backdrop(center, radius))

    print("\n== light rigs ==")
    for key in lighting.LIGHT_RIGS:
        step(f"rig {key}", lambda k=key: lighting.apply_rig(center, radius, k, {"energy": 500.0}))
    step("haze", lambda: lighting.add_haze(center, radius, 0.02, "#22d3ee"))
    step("sun", lambda: lighting.sun_from_camera(center, radius))

    print("\n== electricity ==")
    step("arcs between atoms", lambda: electricity.arcs_between_atoms(
        structure.atom_positions(), count=3, seed=1))
    step("cage arcs", lambda: electricity.cage_arcs(center, radius, count=3, seed=1))
    step("discharge", lambda: electricity.discharge_to_space(
        structure.atom_positions(), center, radius, count=2, seed=1))
    arcs = list(bpy.data.collections["AV_Electricity"].objects) if "AV_Electricity" in bpy.data.collections else []
    step("flicker", lambda: electricity.add_flicker(arcs, 1, 12, seed=1))

    print("\n== lasers ==")
    beams = step("laser rig", lambda: lasers.laser_rig(center, radius, count=2, seed=1)) or []
    step("emitter gizmo", lambda: lasers.add_emitter_gizmo(
        (center[0], center[1] - radius * 5, center[2]), center, radius * 0.4))
    step("sweep animation", lambda: lasers.sweep_animation(beams, center, 1, 12))

    print("\n== camera / render / post ==")
    step("render preset", lambda: apply_render_preset(scene, "preview_fast", percentage=10))
    step("frame structure", lambda: frame_structure(scene, structure, focal_mm=50.0))
    for glare in ("NONE", "FOG_GLOW", "STREAKS", "GHOSTS", "SIMPLE_STAR"):
        step(f"postfx {glare}", lambda g=glare: postfx.setup(
            scene, glare=g, streaks=True, vignette=0.4, contrast=2.0, saturation=1.2, tint="#22d3ee"))

    print("\n== looks ==")
    look_keys = ["neon_lab"] if args.quick else list(LOOKS)
    for key in look_keys:
        step(f"look {key}", lambda k=key: apply_look(scene, structure, k, seed=2))
        if args.render:
            scene.render.resolution_percentage = 5
            step(f"render {key}", lambda k=key: render_still(scene, args.out / f"{k}.png"))

    print("\n== cleanup ==")
    step("clear_all", clear_all)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} step(s)")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("ALL STEPS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
