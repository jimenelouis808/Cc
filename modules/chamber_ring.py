"""Anillo-cámara: se apoya en la base (z = 0) y sube hasta z = h.

- Cara inferior: escalón del O-ring que sella sobre la muestra. El anillo hace
  tope duro contra la base, así que la compresión del O-ring la fija la
  profundidad del escalón, no el apriete de los tornillos.
- Piso de espesor floor_t con un agujero de Ø exposed_d (área expuesta del WE).
  El WE queda rehundido floor_t: favorece una distribución primaria uniforme.
- Pozo de Ø well_d sobre el piso: zona de la tapa (émbolo/ventana) y reservorio
  donde viven RE y CE. well_d ≥ punta del objetivo + 2·margen (check Raman).
- Puertos laterales según params.RING_PORTS.
- Cara superior plana: el sello de cada tapa va en la tapa (tapas intercambiables),
  sujeta con insertos térmicos M3.
Imprimir boca abajo (escalón del O-ring hacia arriba).
"""
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._geom import (  # noqa: E402
    body, bolt_stack, bolt_xy, compound, cyl, lid_insert_xy, port_boss, port_solid,
    sample_seal)


def floor_t(p=P) -> float:
    return sample_seal(p)["step_depth"] + p.CHAMBER_RING["floor_lip"]


def features(p=P) -> dict:
    r, c = p.CHAMBER_RING, p.CELL
    h, ft, seal = r["h"], floor_t(p), sample_seal(p)
    bolts = [cyl(p.hole(c["bolt_clearance_d"] - p.FDM["hole_offset"]), -1, h + 1, x, y)
             .fuse(cyl(p.hole(c["bolt_head_d"]), h - c["bolt_head_h"], h + 1, x, y))
             for x, y in bolt_xy(p)]
    f = {
        "oring_step": cyl(seal["step_d"], -0.01, seal["step_depth"]),
        "bore": cyl(seal["bore_d"], -0.01, ft + 0.01),
        "well": cyl(r["well_d"], ft, h + 0.01),
        "bolts": compound(bolts),
        "pins": compound(cyl(p.hole(c["pin_d"]), -0.01, c["pin_depth"], x, y)
                         for x, y in c["pins"]),
        "lid_inserts": compound(cyl(c["insert_d"], h - c["insert_depth"], h + 0.01, x, y)
                                for x, y in lid_insert_xy(p)),
    }
    for spec, r_out, r_in in port_axes(p):
        f["port_" + spec["name"]] = port_solid(p, spec, r_out, r_in)
    return f


def port_axes(p=P):
    """(spec, cara exterior, interior de la cavidad) de cada puerto, sobre su eje."""
    r_in = p.CHAMBER_RING["well_d"] / 2 - 1.0
    return [(spec, p.CELL["body_w"] / 2, r_in) for spec in p.RING_PORTS]


def bosses(p=P) -> list:
    return [b for b in (port_boss(p, s, ro, ri) for s, ro, ri in port_axes(p)) if b is not None]


def build(p=P) -> cq.Workplane:
    solid = body(p, 0.0, p.CHAMBER_RING["h"])
    for b in bosses(p):
        solid = solid.fuse(b)
    for f in features(p).values():
        solid = solid.cut(f)
    return cq.Workplane("XY").add(solid.clean())


def summary(p=P) -> dict:
    seal = sample_seal(p)
    return {
        "area_expuesta_mm2": round(seal["area_mm2"], 2),
        "exposed_d": seal["bore_d"],
        "rehundido_WE": round(floor_t(p), 3),
        "oring_muestra": f"ID {seal['bore_d']} × cs {p.ORING['sample_cs']} (OD {seal['oring_od']})",
        "tornillo_anillo_base": "M3×%d" % bolt_stack(
            p, p.CHAMBER_RING["h"] - p.CELL["bolt_head_h"], -p.BASE_FLAT["t"])["length"],
    }


if __name__ == "__main__":
    from export import export

    for f in export(build(), "chamber_ring"):
        print(f)
    for k, v in summary().items():
        print(f"{k}: {v}")
