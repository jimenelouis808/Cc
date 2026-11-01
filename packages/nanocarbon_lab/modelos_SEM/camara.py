"""What the Blender camera would frame, without running Blender.

Usage::

    uv run python modelos_SEM/camara.py                 # every escena_*
    uv run python modelos_SEM/camara.py escena_barras

A render of a 25 000-atom scene costs minutes; deciding whether the
composition reads costs none of that, and the two questions are
separable. The camera is not re-invented here: the direction comes from
:func:`nanocarbon_lab.blender.camera.best_view` and the position from
:func:`~nanocarbon_lab.blender.camera.frame_camera`, which are the same
calls the render makes, with the same lens and the same frame fill. So
what this draws is where the camera WILL be, not an impression of it.

What it is not: a render. No lighting, no materials, no depth of field,
no shadow -- a thrown sphere per atom in painter's order. It answers
"is the subject in frame, at the right size, from an angle that shows
it", and nothing else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

from nanocarbon_lab.blender import camera as cam  # noqa: E402
from nanocarbon_lab.blender import styles as style_defs  # noqa: E402
from nanocarbon_lab.viz.plot import ELEMENT_COLOURS  # noqa: E402

HERE = Path(__file__).resolve().parent
XYZ = HERE / "xyz"
BUNDLE = HERE / "bundle"
IMG = HERE / "img"

#: The style the cover renders use, so the lens matches.
STYLE = "clusters_on_sheet"

#: Blender's default sensor width, which `render_cnt` reads off the
#: camera rather than setting.
SENSOR_MM = 36.0

#: A cover is portrait. Both numbers reach `frame_camera`, because
#: which axis crops first depends on the aspect.
RESOLUTION = (1600, 2000)

#: The same fraction of the frame the renderer fills.
FRAME_FILL = 0.86

#: Inches across the preview panel. The marker size is DERIVED from it
#: rather than tuned: a scatter's `s` is an area in points², and an
#: atom of radius r at depth d subtends r/d in the tangent units this
#: plots, which is (r/d)/(2*limit) of the panel width. Guessing a
#: constant instead gave markers of three thousandths of a point and a
#: preview that was an empty rectangle reporting "100 % in frame".
PANEL_INCHES = 5.0
POINTS_PER_INCH = 72.0


def solid_indices(label: str) -> set[int]:
    """Which atoms the model declared part of a solid."""
    sidecar = BUNDLE / f"{label}.json"
    if not sidecar.exists():
        return set()
    return {int(i) for i in json.loads(sidecar.read_text()).get("solid", [])}


def preview(label: str) -> dict:
    """Draw one scene as its camera will see it, and report the numbers."""
    atoms = read(XYZ / f"{label}.xyz")
    points = np.asarray(atoms.positions, dtype=float)
    symbols = atoms.get_chemical_symbols()
    solid = solid_indices(label)
    radii = np.array([
        style_defs.vdw_radius(s) * style_defs.SPACEFILL_SCALE
        if index in solid else style_defs.element_radius(s)
        for index, s in enumerate(symbols)])

    style = style_defs.get_style(STYLE)
    direction = cam.best_view(points)
    location, target, distance = cam.frame_camera(
        points, direction, style.camera_lens_mm, SENSOR_MM, RESOLUTION,
        fill=FRAME_FILL, radii=radii)
    right, up, back = cam.basis_for(direction)

    # True perspective, the same projection the camera performs: the
    # depth along the view axis divides. An orthographic preview would
    # frame correctly and lie about which object is in front, which is
    # most of what a depth composition is for.
    relative = points - location
    depth = -(relative @ back)
    depth = np.where(depth < 1e-6, 1e-6, depth)
    screen = np.stack([relative @ right, relative @ up], axis=1) / depth[:, None]

    width, height = RESOLUTION
    long_side, short_side = max(width, height), min(width, height)
    half_long = np.arctan(0.5 * SENSOR_MM / style.camera_lens_mm)
    half_short = np.arctan(np.tan(half_long) * short_side / long_side)
    half_h = half_long if width >= height else half_short
    half_v = half_short if width >= height else half_long
    limit_h, limit_v = np.tan(half_h), np.tan(half_v)

    aspect = height / width
    figure, axis = plt.subplots(
        figsize=(PANEL_INCHES, PANEL_INCHES * aspect))
    order = np.argsort(-depth)                 # far first
    # Perspective shrinks a distant atom, exactly as the render will.
    width_pts = PANEL_INCHES * POINTS_PER_INCH
    diameter = width_pts * (radii[order] / depth[order]) / limit_h
    sizes = np.maximum(diameter, 0.6) ** 2
    axis.scatter(screen[order, 0], screen[order, 1], s=sizes,
                 c=[ELEMENT_COLOURS.get(symbols[i], "#888888")
                    for i in order],
                 linewidths=0.0)
    axis.set_xlim(-limit_h, limit_h)
    axis.set_ylim(-limit_v, limit_v)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("#0d1b2a")
    for side in axis.spines.values():
        side.set_color("#888888")

    inside = (np.abs(screen[:, 0]) <= limit_h) & (np.abs(screen[:, 1]) <= limit_v)
    covered = (np.ptp(screen[inside, 0]) / (2 * limit_h)) if inside.any() else 0.0
    axis.set_title(f"{label}\n{len(atoms)} átomos · lente "
                   f"{style.camera_lens_mm:.0f} mm · cámara a "
                   f"{distance:.0f} Å · {100 * inside.mean():.1f} % en cuadro",
                   fontsize=8, color="#222222", pad=5)
    figure.tight_layout()
    out = IMG / f"_camara_{label}.png"
    figure.savefig(out, dpi=150, facecolor="white")
    plt.close(figure)
    print(f"  {label:26s} cámara a {distance:7.1f} Å  "
          f"{100 * inside.mean():5.1f} % en cuadro  ancho {100*covered:4.0f} %")
    return {"modelo": label, "distancia": distance,
            "en_cuadro": float(inside.mean()),
            "direccion": [float(v) for v in direction]}


def main(argv: list[str]) -> None:
    names = argv or sorted(p.stem for p in XYZ.glob("escena_*.xyz"))
    style = style_defs.get_style(STYLE)
    print(f"estilo {STYLE}, lente {style.camera_lens_mm:.0f} mm, "
          f"sensor {SENSOR_MM:.0f} mm, {RESOLUTION[0]}x{RESOLUTION[1]}, "
          f"relleno {FRAME_FILL:.0%}")
    rows = [preview(name) for name in names]
    (HERE / "camaras.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
