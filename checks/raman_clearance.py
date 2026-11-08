"""Holgura óptica del Raman sobre el ensamble base + anillo + tapa.

El objetivo se modela como cara frontal + nariz cónica + barril (params.RAMAN),
con su punta en z_tip = WD + window_t·(1 − 1/n_w) + gap·(1 − 1/n_e). Ninguna pieza
puede invadir su envolvente (más clearance_margin) ni el cono de luz de la NA
(más optical_margin), en todo el rango de gap que permiten las lainas.
"""
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._optics import light_cone, objective_envelope, z_tip  # noqa: E402

TOL = 1e-3  # mm³: residuo numérico de cortes coincidentes


def collision_volume(part: cq.Workplane, p=P, gap=None) -> float:
    s = part.val()
    return (s.intersect(objective_envelope(p, gap)).Volume()
            + s.intersect(light_cone(p, gap, margin=0.0)).Volume())


def run(p=P) -> list[tuple[str, float]]:
    from modules import lid_raman
    from modules.base_flat import build as base
    from modules.chamber_ring import build as ring

    out = []
    for shim in lid_raman.shim_range(p):
        g = lid_raman.gap_for_shim(p, shim)
        # la envolvente se calcula para el gap real: la tapa sube o baja con la laina
        for name, part in (("base", base(p)), ("anillo", ring(p)),
                           ("tapa", lid_raman.build(p, shim=shim))):
            out.append((f"gap {g:.2f}: {name}", collision_volume(part, p, g)))
    return out


if __name__ == "__main__":
    print(f"punta del objetivo a gap nominal: z = {z_tip(P):.3f} mm")
    bad = 0
    for name, v in run():
        ok = v < TOL
        bad += not ok
        print(f"[{'OK ' if ok else 'FALLA'}] {name:20s} invasión {v:.4f} mm³")
    sys.exit(1 if bad else 0)
