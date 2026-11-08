"""Exportación común: cada módulo escribe STEP y STL en out/."""
import os

import cadquery as cq

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def export(wp: cq.Workplane, name: str, out_dir: str = OUT) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    step = os.path.join(out_dir, f"{name}.step")
    stl = os.path.join(out_dir, f"{name}.stl")
    cq.exporters.export(wp, step)
    cq.exporters.export(wp, stl, tolerance=0.01, angularTolerance=0.1)
    return step, stl
