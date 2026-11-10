"""Celda de electrodeposición acústica (Chladni): placa-WE empotrada y vibrante.

Piezas (z = 0 es la cara superior de la placa, el WE):
- Placa: cuadrado libre plate_L entre líneas de empotramiento, más una franja de
  clamp_w a cada lado y una lengüeta que sale de los marcos por +x. La pinza de Pt
  toma la lengüeta afuera, en seco.
- Marco inferior (bottom_frame): sostiene la franja empotrada; su borde interior ES
  la línea de empotramiento. Deja libre la cara inferior de la placa para el piezo.
  Aluminio con una capa de Kapton en la cara de apoyo, o PEEK: la placa es el WE y
  no debe tocar metal a tierra.
- Pared (wall): marco superior y cámara en una pieza. Aprieta la franja empotrada;
  dentro de la línea de empotramiento no toca la placa (relief_gap), salvo el
  cordón de EPDM, que va justo dentro de esa línea, donde el desplazamiento es casi
  nulo. Un escalón sostiene el CE paralelo a ce_gap del WE. PEEK o PTFE maquinado.
- Tapa (lid): RE vertical con Luggin hasta una esquina, llenado y venteo.
- Piezo: disco pegado bajo la placa en piezo_center, con Kapton/epoxi entre ambos.
Sólo la cara superior de la placa, dentro del cordón, toca el electrolito.
"""
import math
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._geom import bolt_stack, compound, cyl, port_solid  # noqa: E402


def box(x0, x1, y0, y1, z0, z1) -> cq.Solid:
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))


def rsquare(half: float, r: float, z0: float, z1: float) -> cq.Solid:
    """Cuadrado de semilado `half` con esquinas de radio r, entre z0 y z1."""
    r = max(min(r, half - 1e-3), 1e-3)
    return (cq.Workplane("XY").rect(2 * half, 2 * half).extrude(z1 - z0)
            .edges("|Z").fillet(r).translate((0, 0, z0)).val())


def layout(p=P) -> dict:
    c = p.CHLADNI
    a = c["plate_L"] / 2                          # línea de empotramiento
    w = 1.5 * c["seal_cs"]                         # ancho de la ranura del cordón
    seal_out = a - c["seal_inset"]                 # borde exterior del cordón
    seal_in = seal_out - w                         # borde interior = área expuesta
    plate_half = a + c["clamp_w"]
    bolt_r = p.CELL["bolt_clearance_d"] / 2
    bolt_at = plate_half + c["bolt_margin"] + bolt_r
    outer = bolt_at + c["bolt_margin"] + bolt_r
    h = c["plate_h"]
    z_wall_top = c["electrolyte_h"] + c["headspace"]
    return {"a": a, "groove_w": w, "seal_out": seal_out, "seal_in": seal_in,
            "plate_half": plate_half, "bolt_at": bolt_at, "outer": outer,
            "z_plate_bottom": -h, "z_bottom": -h - c["bottom_t"],
            "z_wall_top": z_wall_top, "z_ce": c["ce_gap"],
            "exposed_mm2": (2 * seal_in) ** 2 - (4 - math.pi) * max(
                c["seal_corner_r"] - w, 0) ** 2}


def bolt_xy(p=P) -> list[tuple[float, float]]:
    """Tornillos alrededor de la placa, sin pasar por la lengüeta (+x)."""
    lay, c = layout(p), p.CHLADNI
    b = lay["bolt_at"]
    n = c["bolts_per_side"]
    pts = set()
    for i in range(n):
        t = -b + 2 * b * i / (n - 1)
        for x, y in ((t, b), (t, -b), (b, t), (-b, t)):
            pts.add((round(x, 6), round(y, 6)))
    tab_clear = c["tab_w"] / 2 + p.CELL["bolt_clearance_d"] / 2 + p.CELL["min_wall_dry"]
    return sorted((x, y) for x, y in pts if not (x > 0 and abs(y) < tab_clear))


def plate(p=P) -> cq.Solid:
    lay, c = layout(p), p.CHLADNI
    s = box(-lay["plate_half"], lay["plate_half"], -lay["plate_half"], lay["plate_half"],
            -c["plate_h"], 0.0)
    tab = box(lay["plate_half"] - 0.01, lay["outer"] + c["tab_out"], -c["tab_w"] / 2,
              c["tab_w"] / 2, -c["plate_h"], 0.0)
    return s.fuse(tab).clean()


def piezo(p=P) -> cq.Solid:
    c = p.CHLADNI
    x, y = c["piezo_center"]
    return cyl(c["piezo_d"], -c["plate_h"] - c["piezo_t"], -c["plate_h"], x, y)


def _bolt_holes(p, z0, z1) -> cq.Shape:
    return compound(cyl(p.hole(p.CELL["bolt_clearance_d"] - p.FDM["hole_offset"]), z0, z1, x, y)
                    for x, y in bolt_xy(p))


def bottom_features(p=P) -> dict:
    lay, c, cell = layout(p), p.CHLADNI, p.CELL
    zb = lay["z_bottom"]
    # La cara del marco inferior queda plate_preload bajo la cara de la placa: al apretar,
    # toda la fuerza va a la placa y los dos marcos nunca se tocan.
    z_face = -c["plate_preload"]
    stack = bolt_stack(p, lay["z_wall_top"] - cell["bolt_head_h"], zb)
    nut_d = p.hole(cell["nut_af"]) / 0.8660254
    nuts = compound(cq.Workplane("XY").center(x, y).polygon(6, nut_d)
                    .extrude(stack["nut_top"] - zb + 0.01).translate((0, 0, zb - 0.01)).val()
                    for x, y in bolt_xy(p))
    return {
        "opening": box(-lay["a"], lay["a"], -lay["a"], lay["a"], zb - 1, 1),
        "plate_pocket": box(-lay["plate_half"] - 0.1, lay["plate_half"] + 0.1,
                            -lay["plate_half"] - 0.1, lay["plate_half"] + 0.1,
                            -c["plate_h"], z_face + 1),
        "tab_slot": box(lay["plate_half"], lay["outer"] + 1, -c["tab_w"] / 2 - 0.2,
                        c["tab_w"] / 2 + 0.2, -c["plate_h"], z_face + 1),
        "bolts": _bolt_holes(p, zb - 1, 1),
        "nuts": nuts,
    }


def build_bottom(p=P) -> cq.Workplane:
    lay = layout(p)
    s = box(-lay["outer"], lay["outer"], -lay["outer"], lay["outer"], lay["z_bottom"],
            -p.CHLADNI["plate_preload"])
    s = (cq.Workplane("XY").add(s).edges("|Z").chamfer(p.CELL["corner_chamfer"]).val())
    for f in bottom_features(p).values():
        s = s.cut(f)
    return cq.Workplane("XY").add(s.clean())


def wall_features(p=P) -> dict:
    lay, c, cell = layout(p), p.CHLADNI, p.CELL
    zt = lay["z_wall_top"]
    depth = 0.75 * c["seal_cs"]                      # cordón comprimido al 25 %
    groove = rsquare(lay["seal_out"], c["seal_corner_r"], -0.01, depth).cut(
        rsquare(lay["seal_in"], c["seal_corner_r"] - lay["groove_w"], -0.02, depth + 0.01))
    ce_half = lay["seal_in"] + c["ce_ledge"]
    return {
        "cavity": rsquare(lay["seal_in"], c["seal_corner_r"] - lay["groove_w"], -0.01, zt + 0.01),
        # esquinas vivas, igual que la abertura del marco inferior: la línea de
        # empotramiento es la misma arriba y abajo
        "relief": box(-lay["a"], lay["a"], -lay["a"], lay["a"], -0.01, c["relief_gap"]),
        "groove": groove,
        "ce_seat": rsquare(ce_half, c["seal_corner_r"], lay["z_ce"] - 0.01, zt + 0.01),
        "bolts": _bolt_holes(p, -1, zt + 1).fuse(compound(
            cyl(p.hole(cell["bolt_head_d"]), zt - cell["bolt_head_h"], zt + 1, x, y)
            for x, y in bolt_xy(p))),
        "lid_inserts": compound(cyl(cell["insert_d"], zt - cell["insert_depth"], zt + 0.01, x, y)
                                for x, y in lid_insert_xy(p)),
        "port_CE": port_solid(p, ce_port(p), lay["outer"], ce_half - 1.0),
    }


def ce_port(p=P) -> dict:
    """El puerto del CE, a la altura del plano medio de la malla."""
    spec = dict(p.CHLADNI_CE_PORT)
    spec["z"] = layout(p)["z_ce"] + p.CHLADNI["ce_t"] / 2
    return spec


def lid_insert_xy(p=P) -> list[tuple[float, float]]:
    """Insertos de la tapa: en las diagonales, entre la cámara y los tornillos de esquina
    (en los ejes estorbarían a los puertos laterales)."""
    lay = layout(p)
    d = (lay["seal_in"] + p.CHLADNI["ce_ledge"] + lay["bolt_at"]) / 2
    return [(sx * d, sy * d) for sx in (1, -1) for sy in (1, -1)]


def build_wall(p=P) -> cq.Workplane:
    lay = layout(p)
    s = (cq.Workplane("XY").add(box(-lay["outer"], lay["outer"], -lay["outer"], lay["outer"],
                                    0.0, lay["z_wall_top"]))
         .edges("|Z").chamfer(p.CELL["corner_chamfer"]).val())
    for f in wall_features(p).values():
        s = s.cut(f)
    return cq.Workplane("XY").add(s.clean())


def re_xy(p=P) -> tuple[float, float]:
    lay, c = layout(p), p.CHLADNI
    sx, sy = c["re_corner"]
    d = lay["seal_in"] - c["re_inset"]
    return sx * d, sy * d


def ce_plate(p=P) -> cq.Solid:
    """CE paralelo apoyado en el escalón, con la esquina de la RE recortada."""
    lay, c = layout(p), p.CHLADNI
    half = lay["seal_in"] + c["ce_ledge"] - 0.2
    s = rsquare(half, c["seal_corner_r"], lay["z_ce"], lay["z_ce"] + c["ce_t"])
    x, y = re_xy(p)
    cut_r = p.hole(p.ELECTRODES["RE_tip_d"]) / 2 + 2.0      # sólo la punta baja hasta aquí
    return s.cut(cyl(2 * cut_r, lay["z_ce"] - 1, lay["z_ce"] + c["ce_t"] + 1, x, y))


def lid_features(p=P) -> dict:
    lay, c, cell = layout(p), p.CHLADNI, p.CELL
    zt, zl = lay["z_wall_top"], lay["z_wall_top"] + c["lid_t"]
    x, y = re_xy(p)
    f = {
        "re": re_port(p),
        "screws": compound(cyl(p.hole(cell["bolt_clearance_d"] - p.FDM["hole_offset"]),
                               zt - 1, zl + 1, sx, sy) for sx, sy in lid_insert_xy(p)),
        "bolt_heads": compound(cyl(p.hole(cell["bolt_head_d"]) + 1.0, zt - 1, zl + 1, bx, by)
                               for bx, by in bolt_xy(p)),
    }
    for spec in p.CHLADNI_LID_PORTS:
        px, py = spec["xy"]
        f["port_" + spec["name"]] = cyl(p.PORTS["tap_drill_d"], zl - p.PORTS["thread_depth"],
                                        zl + 1, px, py).fuse(
            cyl(p.hole(spec["through_d"]), zt - 1, zl - p.PORTS["thread_depth"] + 0.01, px, py))
    return f


def re_seat_z(p=P) -> float:
    """Altura del hombro de la RE: la punta (RE_tip_l) termina a luggin_h de la placa."""
    return p.CHLADNI["luggin_h"] + p.ELECTRODES["RE_tip_l"]


def re_port(p=P) -> cq.Shape:
    """Puerto vertical de la RE: sella sobre la punta Ø4 con O-ring radial; el hombro del
    cuerpo Ø9 apoya en el resalte, así la distancia punta–placa no depende de la mano."""
    lay, e = layout(p), p.ELECTRODES
    x, y = re_xy(p)
    zt, top = lay["z_wall_top"], re_seat_z(p)
    g = p.oring_gland(p.ORING["re_cs"], "radial")
    z_g = top - p.ORING["re_groove_from_face"]
    return (cyl(p.hole(e["RE_tip_d"]), zt - 1, top + 1, x, y)
            .fuse(cyl(p.hole(e["RE_tip_d"] + 2 * g["depth"]), z_g - g["width"], z_g, x, y)))


def build_lid(p=P) -> cq.Workplane:
    lay, c = layout(p), p.CHLADNI
    zt = lay["z_wall_top"]
    s = (cq.Workplane("XY").add(box(-lay["outer"], lay["outer"], -lay["outer"], lay["outer"],
                                    zt, zt + c["lid_t"]))
         .edges("|Z").chamfer(p.CELL["corner_chamfer"]).val())
    x, y = re_xy(p)
    if re_seat_z(p) > zt + c["lid_t"]:
        s = s.fuse(cyl(c["re_seat_d"], zt + c["lid_t"] - 0.01, re_seat_z(p), x, y))
    for f in lid_features(p).values():
        s = s.cut(f)
    return cq.Workplane("XY").add(s.clean())


def summary(p=P) -> dict:
    lay, c = layout(p), p.CHLADNI
    stack = bolt_stack(p, lay["z_wall_top"] - p.CELL["bolt_head_h"], lay["z_bottom"])
    return {
        "placa": f"{c['plate_material']}, {2 * lay['plate_half']:.0f} × {2 * lay['plate_half']:.0f}"
                 f" × {c['plate_h']} mm + lengüeta {c['tab_w']} × {c['tab_out']} mm afuera",
        "lado libre (empotrada)": f"{c['plate_L']} mm",
        "área expuesta": f"{2 * lay['seal_in']:.1f} × {2 * lay['seal_in']:.1f} mm, "
                         f"{lay['exposed_mm2']:.0f} mm²",
        "cordón": f"EPDM cs {c['seal_cs']}, a {c['seal_inset']} mm del empotramiento",
        "celda": f"{2 * lay['outer']:.1f} × {2 * lay['outer']:.1f} mm, de z = "
                 f"{lay['z_bottom']:.1f} a {lay['z_wall_top'] + c['lid_t']:.1f}",
        "tornillos": f"{len(bolt_xy(p))} × M3×{stack['length']} con tuerca embebida abajo",
        "CE": f"paralelo a {c['ce_gap']} mm del WE; RE en {tuple(round(v, 1) for v in re_xy(p))}",
    }


if __name__ == "__main__":
    from export import export

    for name, part in (("chladni_wall", build_wall()), ("chladni_bottom", build_bottom()),
                       ("chladni_lid", build_lid()), ("chladni_plate", cq.Workplane().add(plate())),
                       ("chladni_ce", cq.Workplane().add(ce_plate()))):
        for f in export(part, name):
            print(f)
    for k, v in summary().items():
        print(f"{k}: {v}")
