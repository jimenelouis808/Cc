"""Render one preview per cover look — a visual contact sheet.

Uses the same synthetic structure as ``integration_check.py`` unless an XYZ
file is given, so it doubles as a smoke test that actually produces pixels::

    blender -b -P tools/render_look_sheet.py -- --out /tmp/looks --width 480
    python tools/render_look_sheet.py -- --xyz mol.xyz --out /tmp/looks
"""

from __future__ import annotations

import argparse
import sys
import time
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
    parser = argparse.ArgumentParser(prog="render_look_sheet")
    parser.add_argument("--xyz", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("/tmp/atomviz_looks"))
    parser.add_argument("--width", type=int, default=480, help="Preview width in pixels")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--looks", default="", help="Comma-separated subset of look keys")
    return parser.parse_args(argv)


def main() -> int:
    """Render every look and print a timing table."""
    _bootstrap()
    args = parse_args()

    from atomviz_studio.core.detect import active_structure
    from atomviz_studio.core.presets import LOOKS
    from atomviz_studio.looks.apply import apply_look
    from atomviz_studio.scene.render import render_still

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from integration_check import build_fake_structure

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    if args.xyz is not None:
        from atomviz_studio.ui.operators import enable_atomic_blender

        if not enable_atomic_blender():
            print("error: Atomic Blender unavailable", file=sys.stderr)
            return 3
        bpy.ops.import_mesh.xyz(filepath=str(args.xyz.resolve()))
    else:
        build_fake_structure()

    structure = active_structure(bpy.context, "SCENE")
    if structure is None:
        print("error: no structure", file=sys.stderr)
        return 4

    keys = [k.strip() for k in args.looks.split(",") if k.strip()] or list(LOOKS)
    args.out.mkdir(parents=True, exist_ok=True)
    timings: list[tuple[str, float, int]] = []

    for key in keys:
        apply_look(scene, structure, key, seed=5)
        # Force every look to the same small format so the sheet is comparable.
        scene.render.resolution_x = args.width
        scene.render.resolution_y = int(args.width * 1.25)
        scene.render.resolution_percentage = 100
        if scene.render.engine == "CYCLES" and hasattr(scene, "cycles"):
            scene.cycles.samples = args.samples
        else:
            scene.eevee.taa_render_samples = args.samples
        start = time.perf_counter()
        path = render_still(scene, args.out / f"{key}.png")
        elapsed = time.perf_counter() - start
        size = path.stat().st_size if path and path.exists() else 0
        timings.append((key, elapsed, size))
        print(f"[{key}] {elapsed:5.1f}s  {size / 1024:6.0f} KiB  -> {path}")

    print("\nlook                 seconds   KiB")
    for key, elapsed, size in timings:
        print(f"{key:20s} {elapsed:7.1f} {size / 1024:6.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
