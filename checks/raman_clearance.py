"""Holgura del objetivo Raman sobre el anillo (y, después, sobre cada tapa).

La punta del objetivo (cilindro de Ø objective_tip_diameter) queda a
z_tip = gap + window_t + WD sobre el WE. Nada dentro de ese cilindro (más el
margen radial) puede estar por encima de z_tip − margen.
"""
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402


def z_tip(p=P) -> float:
    r = p.RAMAN
    return p.CELL["thin_gap_raman"] + r["window_t"] + r["working_distance"]


def objective_envelope(p=P, height: float = 100.0) -> cq.Solid:
    r = p.RAMAN
    rad = r["objective_tip_diameter"] / 2 + r["clearance_margin"]
    z0 = z_tip(p) - r["clearance_margin"]
    return cq.Solid.makeCylinder(rad, height, cq.Vector(0, 0, z0))


def collision_volume(part: cq.Workplane, p=P) -> float:
    return part.val().intersect(objective_envelope(p)).Volume()


if __name__ == "__main__":
    from modules.chamber_ring import build

    v = collision_volume(build())
    print(f"punta del objetivo en z = {z_tip():.2f} mm; "
          f"choque con el anillo: {v:.3f} mm³ {'OK' if v < 1e-6 else 'FALLA'}")
    sys.exit(0 if v < 1e-6 else 1)
