"""Headless cover renderer.

Render a finished cover image straight from an ``.xyz`` file, without opening
the Blender GUI::

    blender -b -P atomviz_studio/cli/render_cover.py -- \\
        --xyz out/cnt/cnt.xyz --look neon_lab --out covers/cnt.png

Everything after the mandatory ``--`` is parsed by this script. Run with
``--list`` to print the available looks and formats.

The script is deliberately usable from a queue system: it is deterministic for
a given ``--seed``, writes exactly one file, and returns a non-zero exit code
when the structure could not be imported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    """Make ``import atomviz_studio`` work when run as a loose script."""
    package_root = Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the arguments that follow ``--`` on the Blender command line."""
    parser = argparse.ArgumentParser(
        prog="render_cover",
        description="Render a journal-cover image from an XYZ structure.",
    )
    parser.add_argument("--xyz", type=Path, help="Structure file to import (.xyz)")
    parser.add_argument("--look", default="neon_lab", help="Cover look key")
    parser.add_argument("--out", type=Path, help="Output image path (.png)")
    parser.add_argument("--format", dest="render_preset", default=None, help="Render preset key")
    parser.add_argument("--seed", type=int, default=0, help="Seed for arcs and beams")
    parser.add_argument("--samples", type=int, default=None, help="Override render samples")
    parser.add_argument("--percentage", type=int, default=100, help="Resolution percentage")
    parser.add_argument("--focal", type=float, default=None, help="Override focal length (mm)")
    parser.add_argument("--azimuth", type=float, default=None, help="Override camera azimuth (deg)")
    parser.add_argument("--elevation", type=float, default=None, help="Override camera elevation (deg)")
    parser.add_argument("--save-blend", type=Path, default=None, help="Also save the .blend file")
    parser.add_argument("--no-render", action="store_true", help="Set the scene up but do not render")
    parser.add_argument("--list", action="store_true", help="List looks and formats, then exit")
    return parser.parse_args(argv)


def _argv_after_dashes() -> list[str]:
    """Return the CLI arguments Blender left for the script."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    _bootstrap()
    args = parse_args(argv if argv is not None else _argv_after_dashes())

    from atomviz_studio.core.presets import LOOKS, RENDER_PRESETS

    if args.list:
        print("Looks:")
        for key, look in LOOKS.items():
            print(f"  {key:18s} {look.description}")
        print("Formats:")
        for key, preset in RENDER_PRESETS.items():
            print(f"  {key:18s} {preset.width}x{preset.height} ({preset.engine})")
        return 0

    import bpy

    from atomviz_studio.core.detect import active_structure
    from atomviz_studio.looks.apply import apply_look
    from atomviz_studio.scene import render as render_module
    from atomviz_studio.ui.operators import ATOMIC_BLENDER_HINT, enable_atomic_blender

    scene = bpy.context.scene

    if args.xyz is not None:
        path = args.xyz.expanduser().resolve()
        if not path.is_file():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
        # Start from an empty scene so the default cube never sneaks into a cover.
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        if not enable_atomic_blender():
            print(f"error: {ATOMIC_BLENDER_HINT}", file=sys.stderr)
            return 3
        bpy.ops.import_mesh.xyz(filepath=str(path))

    structure = active_structure(bpy.context, "SCENE")
    if structure is None:
        print("error: no atomic structure found in the scene", file=sys.stderr)
        return 4
    print(f"[AtomViz] structure {structure.name}: {structure.atom_count} atoms, {structure.symbols}")

    report = apply_look(scene, structure, args.look, seed=args.seed)
    print(f"[AtomViz] {report.summary()}")
    for warning in report.warnings:
        print(f"[AtomViz] warning: {warning}", file=sys.stderr)

    if args.render_preset:
        render_module.apply_render_preset(scene, args.render_preset, percentage=args.percentage)
    elif args.percentage != 100:
        scene.render.resolution_percentage = args.percentage

    if args.samples is not None:
        if scene.render.engine == "CYCLES" and hasattr(scene, "cycles"):
            scene.cycles.samples = args.samples
        else:
            scene.eevee.taa_render_samples = args.samples

    if any(v is not None for v in (args.focal, args.azimuth, args.elevation)):
        from atomviz_studio.core.presets import get_look
        from atomviz_studio.scene.camera import frame_structure

        look_camera = get_look(args.look).camera
        frame_structure(
            scene,
            structure,
            focal_mm=args.focal if args.focal is not None else float(look_camera.get("focal_mm", 50.0)),
            azimuth_deg=args.azimuth if args.azimuth is not None else float(look_camera.get("azimuth_deg", 35.0)),
            elevation_deg=(
                args.elevation if args.elevation is not None else float(look_camera.get("elevation_deg", 15.0))
            ),
            dof=bool(look_camera.get("dof", True)),
            fstop=float(look_camera.get("fstop", 2.8)),
        )

    for line in render_module.print_report(scene):
        print(f"[AtomViz] {line}")

    if args.save_blend is not None:
        blend_path = args.save_blend.expanduser().resolve()
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"[AtomViz] saved {blend_path}")

    if args.no_render:
        return 0
    if args.out is None:
        print("error: --out is required unless --no-render is given", file=sys.stderr)
        return 5
    written = render_module.render_still(scene, args.out)
    print(f"[AtomViz] rendered {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
