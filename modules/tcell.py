"""Celda T (tipo Swagelok) con electrodo de referencia, para supercapacitores y baterías.

Pila vertical dentro de un agujero de 1/2" (params.TCELL):

    tapa sup. ─ resorte ─ colector sup. ─ CE │ separador │ WE ─ colector inf. ─ tapa inf.

- z = 0 es la cara del electrodo WE. El colector inferior apoya en la tapa inferior (tope
  duro), así que z = 0 lo fija la geometría y no el apriete.
- El colector superior lo empuja un resorte alojado en la tapa superior: la presión sobre la
  pila es k·compresión / área del disco, la misma en cada armado (params.TCELL["pressure_range"]).
- Cada colector sella con un O-ring radial cerca de su cara; los terminales Ø4 salen por las
  tapas para las pinzas del potenciostato. La tapa inferior tiene patas: el terminal no toca
  la mesa.
- RE: la punta Ø4 de la Gamry entra por +y y queda re_recess fuera del agujero, en el plano
  medio del separador (que, mojado, llega hasta la boca del canal). El hombro Ø9 apoya en un
  resalte externo, como en las otras celdas.
- Llenado: el puerto −y (1/4-28) llega al mismo plano: con jeringa se llena el canal de la RE
  y se purgan burbujas; luego se tapa con un conector ciego.
- Dos electrodos (sin RE): misma celda, el puerto RE con un tapón.

Material: PEEK o PTFE maquinado (o PP impreso). Colectores: Ti grado 2 o carbón vítreo en
ácido o neutro; 316L sólo en electrolito neutro (se corroe en H2SO4).
"""
import math
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._geom import (  # noqa: E402
    block, compound, cyl, port_boss, port_solid, variant)


def bore_r(p=P) -> float:
    return p.hole(p.TCELL["rod_d"]) / 2


def layout(p=P) -> dict:
    t = p.TCELL
    z_we = -t["we_t"]                          # cara superior del colector inferior
    z_sep = 0.0
    z_ce = t["sep_t"]
    z_stack_top = t["sep_t"] + t["ce_t"]       # cara inferior del colector superior
    spring_l = t["spring_free_l"] - t["spring_compression"]
    rod_top = t["z1"] + t["spring_pocket_depth"] - spring_l
    return {
        "z_we": z_we, "z_sep": z_sep, "z_ce": z_ce, "z_stack_top": z_stack_top,
        "z_re": t["sep_t"] / 2,
        "rod_lo": (t["z0"], z_we), "rod_hi": (z_stack_top, rod_top),
        "spring": (rod_top, rod_top + spring_l), "spring_l": spring_l,
        "cap_lo": (t["z0"] - t["cap_t"], t["z0"]), "cap_hi": (t["z1"], t["z1"] + t["cap_t"]),
        "legs": (t["z0"] - t["cap_t"] - t["leg_h"], t["z0"] - t["cap_t"]),
        "pin_lo": (t["z0"] - t["cap_t"] - t["pin_out"], t["z0"]),
        "pin_hi": (rod_top, t["z1"] + t["cap_t"] + t["pin_out"]),
    }


def _p_re(p):
    """Copia de params en la que la punta de la RE queda re_recess FUERA del agujero."""
    return variant(p, ELECTRODES={"RE_protrusion": -p.TCELL["re_recess"]})


def port_specs(p=P) -> list[dict]:
    z = layout(p)["z_re"]
    return [dict(s, z=z) for s in p.TCELL_PORTS]


def port_axes(p=P):
    """(spec, cara exterior, r_in): r_in + 1 = pared del agujero (convención de re_boss_len)."""
    return [(s, p.TCELL["body_w"] / 2, bore_r(p) - 1.0) for s in port_specs(p)]


def screw_xy(p=P):
    r = p.TCELL["cap_screw_r"]
    return [(r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)]


def rod_glands(p=P) -> cq.Shape:
    t = p.TCELL
    g = p.oring_gland(p.ORING["re_cs"], "radial")
    d = p.hole(t["rod_d"] + 2 * g["depth"])
    a = p.ORING["re_groove_from_face"]
    return compound([cyl(d, t["z1"] - a - g["width"], t["z1"] - a),
                     cyl(d, t["z0"] + a, t["z0"] + a + g["width"])])


def features(p=P) -> dict:
    t, c = p.TCELL, p.CELL
    pr = _p_re(p)
    f = {
        "bore": cyl(2 * bore_r(p), t["z0"] - 1, t["z1"] + 1),
        "rod_glands": rod_glands(p),
        "lid_inserts": compound([cyl(c["insert_d"], z0, z1, x, y) for x, y in screw_xy(p)
                             for z0, z1 in ((t["z1"] - c["insert_depth"], t["z1"] + 0.01),
                                            (t["z0"] - 0.01, t["z0"] + c["insert_depth"]))]),
    }
    for spec, r_out, r_in in port_axes(p):
        f["port_" + spec["name"]] = port_solid(pr if spec["kind"] == "re_tip" else p,
                                               spec, r_out, r_in)
    return f


def bosses(p=P) -> list:
    pr = _p_re(p)
    return [b for b in (port_boss(pr, s, ro, ri) for s, ro, ri in port_axes(p)) if b is not None]


def build(p=P) -> cq.Workplane:
    t = p.TCELL
    solid = block(t["body_w"], t["body_w"], t["z0"], t["z1"], t["corner_chamfer"])
    for b in bosses(p):
        solid = solid.fuse(b)
    for f in features(p).values():
        solid = solid.cut(f)
    return cq.Workplane("XY").add(solid.clean())


def _cap(p, z0, z1, head_side: str) -> cq.Solid:
    t, c = p.TCELL, p.CELL
    s = block(t["body_w"], t["body_w"], z0, z1, t["corner_chamfer"])
    s = s.cut(cyl(p.hole(t["pin_d"] + 0.3), z0 - 1, z1 + 1))
    for x, y in screw_xy(p):
        s = s.cut(cyl(p.hole(c["bolt_clearance_d"] - p.FDM["hole_offset"]), z0 - 1, z1 + 1, x, y))
        zh = (z0 - 0.01, z0 + c["bolt_head_h"]) if head_side == "below" else (
            z1 - c["bolt_head_h"], z1 + 0.01)
        s = s.cut(cyl(p.hole(c["bolt_head_d"]), *zh, x, y))
    return s


def cap_bottom(p=P) -> cq.Workplane:
    t, lay = p.TCELL, layout(p)
    s = _cap(p, *lay["cap_lo"], "below")
    a = t["body_w"] / 2 - t["leg"] / 2 - 1.0
    for sx in (1, -1):
        for sy in (1, -1):
            s = s.fuse(block(t["leg"], t["leg"], lay["legs"][0], lay["legs"][1] + 0.01,
                             0.0, sx * a, sy * a))
    return cq.Workplane("XY").add(s.clean())


def cap_top(p=P) -> cq.Workplane:
    t, lay = p.TCELL, layout(p)
    z0, z1 = lay["cap_hi"]
    s = _cap(p, z0, z1, "above")
    s = s.cut(cyl(t["spring_od"] + 0.8, z0 - 0.01, z0 + t["spring_pocket_depth"]))
    return cq.Workplane("XY").add(s.clean())


# --- piezas compradas (para el ensamble y los checks) ---------------------------------

def collectors(p=P) -> list:
    """(nombre, sólido) de colectores con su terminal, discos de la pila y resorte."""
    t, lay = p.TCELL, layout(p)
    lo = cyl(t["rod_d"], *lay["rod_lo"]).fuse(cyl(t["pin_d"], *lay["pin_lo"]))
    hi = cyl(t["rod_d"], *lay["rod_hi"]).fuse(cyl(t["pin_d"], *lay["pin_hi"]))
    d = t["electrode_d"]
    return [("colector inferior (WE)", lo.clean()), ("colector superior (CE)", hi.clean()),
            ("electrodo WE", cyl(d, lay["z_we"], 0.0)),
            ("separador", cyl(d, 0.0, t["sep_t"])),
            ("electrodo CE", cyl(d, lay["z_ce"], lay["z_stack_top"]))]


def spring(p=P) -> cq.Solid:
    t, lay = p.TCELL, layout(p)
    z0, z1 = lay["spring"]
    r = (t["spring_od"] - t["spring_wire_d"]) / 2
    turns = 6
    helix = cq.Wire.makeHelix((z1 - z0) / turns, z1 - z0 - t["spring_wire_d"], r,
                              cq.Vector(0, 0, z0 + t["spring_wire_d"] / 2))
    prof = cq.Wire.makeCircle(t["spring_wire_d"] / 2, helix.startPoint(), helix.tangentAt(0))
    return cq.Solid.sweep(prof, [], helix, isFrenet=True)


def cap_screw_length(p=P) -> int:
    """M3 estándar más largo que cabe: tapa menos el avellanado más el inserto (−0.5)."""
    from modules._geom import STD_LENGTHS
    t, c = p.TCELL, p.CELL
    room = t["cap_t"] - c["bolt_head_h"] + c["insert_depth"] - 0.5
    return max(L for L in STD_LENGTHS if L <= room)


def summary(p=P) -> dict:
    t, lay = p.TCELL, layout(p)
    area = math.pi * (t["electrode_d"] / 2) ** 2
    force = t["spring_k"] * t["spring_compression"]
    lo, hi = lay["rod_lo"], lay["rod_hi"]
    return {
        "pila": f"WE {t['we_t']} + separador {t['sep_t']} + CE {t['ce_t']} mm; Ø{t['electrode_d']}",
        "área_del_electrodo_cm2": round(area / 100, 3),
        "fuerza_del_resorte_N": round(force, 1),
        "presión_sobre_la_pila_MPa": round(force / area, 3),
        "colector_inferior": f"Ø{t['rod_d']} × {lo[1] - lo[0]:.2f} + terminal "
                             f"Ø{t['pin_d']} × {lay['pin_lo'][1] - lay['pin_lo'][0]:.1f}",
        "colector_superior": f"Ø{t['rod_d']} × {hi[1] - hi[0]:.2f} + terminal "
                             f"Ø{t['pin_d']} × {lay['pin_hi'][1] - lay['pin_hi'][0]:.1f}",
        "RE": f"punta en y = {bore_r(p) + t['re_recess']:.2f}, z = {lay['z_re']:.2f} "
              "(plano medio del separador)",
        "tornillos_de_tapa": f"8 × M3×{cap_screw_length(p)} a insertos (4 por tapa)",
    }


if __name__ == "__main__":
    from export import export

    for name, part in (("tcell_body", build()), ("tcell_cap_bottom", cap_bottom()),
                       ("tcell_cap_top", cap_top())):
        for f in export(part, name):
            print(f)
    for k, v in summary().items():
        print(f"{k}: {v}")
