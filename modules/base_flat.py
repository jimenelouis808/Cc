"""Base para WE plano (WE B): sustrato cuadrado con su cara superior en z = 0.

- Alojamiento de sample_w + holgura, profundidad sample_t: la cara de la muestra
  queda al ras de la cara superior de la base (z = 0, plano de referencia).
- Lámina de contacto (Pt/Cu/Ni) en una ranura bajo la muestra que sale por la
  cara -x: el contacto queda fuera del área mojada (lo sella el O-ring del anillo).
  Muestras conductoras solo por arriba (película sobre vidrio): doblar la lámina o
  cinta conductora sobre el borde, fuera del O-ring.
- Agujero central para expulsar la muestra desde abajo.
- Tuercas M3 embebidas por abajo y pines guía a presión (Ø nominal sin offset).
Imprimir con la cara z = 0 hacia arriba.
"""
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._geom import body, bolt_stack, bolt_xy, compound, cyl  # noqa: E402


def features(p=P) -> dict:
    b, c, s = p.BASE_FLAT, p.CELL, p.WE_FLAT
    t = b["t"]
    pocket_w = s["sample_w"] + 2 * b["pocket_clearance"]
    z_pf = -s["sample_t"]                      # piso del alojamiento
    pocket = cq.Workplane("XY").rect(pocket_w, pocket_w).extrude(-s["sample_t"]).val()
    reliefs = [cyl(b["corner_relief_d"], z_pf, 0.01, sx * pocket_w / 2, sy * pocket_w / 2)
               for sx in (1, -1) for sy in (1, -1)]
    pocket = pocket.fuse(*reliefs).clean()

    x_out = c["body_w"] / 2 + 1
    contact = (cq.Workplane("XY")
               .box(x_out, b["contact_w"], b["contact_t"], centered=False)
               .translate((-x_out, -b["contact_w"] / 2, z_pf - b["contact_t"])).val())
    contact = contact.fuse(cq.Workplane("XY").box(b["contact_w"], b["contact_w"],
                                                  b["contact_t"], centered=(True, True, False))
                           .translate((0, 0, z_pf - b["contact_t"])).val())

    nut_d = p.hole(c["nut_af"]) / 0.8660254  # circunscrito del hexágono
    stack = bolt_stack(p, p.CHAMBER_RING["h"] - c["bolt_head_h"], -t)
    bolts, nuts = [], []
    for x, y in bolt_xy(p):
        bolts.append(cyl(p.hole(c["bolt_clearance_d"] - p.FDM["hole_offset"]), -t - 1, 1, x, y))
        nuts.append(cq.Workplane("XY").center(x, y).polygon(6, nut_d)
                    .extrude(stack["nut_top"] + t + 0.01).translate((0, 0, -t - 0.01)).val())
    pins = [cyl(c["pin_d"], -c["pin_depth"], 0.01, x, y) for x, y in c["pins"]]
    return {
        "pocket": pocket,
        "contact": contact,
        "eject": cyl(p.hole(b["eject_d"]), -t - 1, z_pf + 0.01),
        "bolts": compound(bolts),
        "nuts": compound(nuts),
        "pins": compound(pins),
    }


def build(p=P) -> cq.Workplane:
    solid = body(p, -p.BASE_FLAT["t"], 0.0)
    for f in features(p).values():
        solid = solid.cut(f)
    return cq.Workplane("XY").add(solid.clean())


if __name__ == "__main__":
    from export import export

    for f in export(build(), "base_flat"):
        print(f)
