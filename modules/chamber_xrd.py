"""Cuerpo de la celda XRD (Bruker D8 θ-θ) con WE en pinza de Pt.

El haz va a lo largo de x. La cara superior del cuerpo es plana en z = 0, al ras
de la cara de la muestra: la película (Kapton / PP) apoya en ella y se sella con un
cordón de EPDM bajo el marco tensor (modules/lid_xrd.py). Durante la medición se
retira electrolito por un puerto hasta que la película queda sobre la muestra
(capa ≤ ~50 µm); para la electroquímica se infla.

- Muestra en un alojamiento, cara en z = 0. Bajo ella, una ranura para la
  lengüeta de contacto (tira de Pt/Ti si la muestra es rígida).
- Bolsillo de la pinza del lado +y, fuera del corredor del haz: la lengüeta baja
  ramp_l mm hasta la pinza, cuyo eje queda clip_depth mm bajo la cara. Una tapa al
  ras cubre el bolsillo para que la película no se hunda en él.
- Reservorio interno del lado −y con RE, CE, entrada y salida (params.XRD_PORTS),
  comunicado con la capa delgada por una ranura junto a la muestra.
- Nada en las caras ±x: es el eje del haz.
Ajuste de altura: lainas bajo la celda o la platina Z del D8 hasta que la cara de
la muestra quede en el plano del goniómetro.
"""
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._geom import (  # noqa: E402
    clamp_extent, compound, cyl, port_boss, port_bore_r, port_outer_r, port_solid)


def corridor_half(p=P) -> float:
    """Semiancho en y del corredor del haz (lo que la línea ilumina + margen)."""
    return p.XRD["beam_length_y"] / 2 + p.XRD["corridor_margin"]


def layout(p=P) -> dict:
    """Coordenadas de muestra, pinza, reservorio y caras del cuerpo."""
    s, c, k = p.WE_XRD, p.WE_CLAMP, p.CHAMBER_XRD
    sx = s["sample_x"] / 2 + s["pocket_clearance"]
    sy = s["sample_y"] / 2 + s["pocket_clearance"]
    z_clip = -s["clip_depth"]
    y_jaw0 = sy + s["ramp_l"]
    y_jaw1 = y_jaw0 + c["head_l"]                 # cabeza de la pinza, frente en y_jaw0
    pocket_hw = max(c["head_w"], c["rod_d"], s["tab_w"]) / 2 + k["pocket_margin"]
    pocket_y1 = y_jaw1 + 1.0
    up, down = clamp_extent(c, s.get("clamp_screw", "down"))
    pocket_floor = z_clip - max(down, c["rod_d"] / 2) - k["floor_clearance"]
    res_y0 = -(sy + k["slot_w"])
    res_y1 = res_y0 - k["reservoir_l"]
    z_rf, z_rr = k["reservoir_z"]
    # el reservorio cubre, en x, la boca de cada puerto −y
    edges = []
    for spec in p.XRD_PORTS:
        if spec["face"] == "-y":
            r = port_bore_r(p, spec)
            edges += [spec["offset"] - r, spec["offset"] + r]
    res_x0 = min(edges + [-sx]) - k["reservoir_margin"]
    res_x1 = max(edges + [sx]) + k["reservoir_margin"]
    lid = _lid_layout(p, sx, sy, pocket_y1 + k["cover_ledge"], (res_x0, res_x1, res_y1))
    y_pos = max(pocket_y1 + k["port_wall"], lid["y_screw_pos"] + p.LID_XRD["edge_margin"])
    y_neg = min(res_y1 - k["port_wall"], lid["y_screw_neg"] - p.LID_XRD["edge_margin"])
    z_bottom = min(z_rf, pocket_floor) - p.CELL["min_wall"]
    return {"sx": sx, "sy": sy, "z_clip": z_clip, "y_jaw0": y_jaw0, "y_jaw1": y_jaw1,
            "pocket_hw": pocket_hw, "pocket_y1": pocket_y1, "pocket_floor": pocket_floor,
            "res_y0": res_y0, "res_y1": res_y1, "z_rf": z_rf, "z_rr": z_rr,
            "res_x0": res_x0, "res_x1": res_x1,
            "x_half": lid["x_half"], "y_pos": y_pos, "y_neg": y_neg, "z_bottom": z_bottom,
            "lid": lid}


def _lid_layout(p, sx, sy, pocket_end, reservoir) -> dict:
    """Ventana, ranura del cordón y tornillos del marco (lo comparte lid_xrd)."""
    from checks.xrd_clearance import allowed_height

    L, k = p.LID_XRD, p.CHAMBER_XRD
    import math

    slope = math.tan(math.radians(L["bevel_deg"]))
    # borde interior del marco en los lados ±x: el más cercano que el haz permite
    fx_in = sx + L["window_margin"]
    while True:
        ok = True
        d = fx_in
        while d < fx_in + 60:
            th = min(L["t"], L["edge_t"] + (d - fx_in) * slope)
            if th > allowed_height(p, d) - L["clearance_margin"]:
                ok = False
                break
            d += 0.5
        if ok:
            break
        fx_in += 0.25
    full_at = fx_in + (L["t"] - L["edge_t"]) / slope
    wy0 = -(sy + k["slot_w"] + L["window_margin"])
    wy1 = sy + k["cover_open"] + L["window_margin"]
    w = 1.5 * L["film_cs"]
    # el cordón va donde el marco, dentro del corredor, ya tiene groove_min_t
    gx_in = fx_in + max(1.0, (L["groove_min_t"] - L["edge_t"]) / slope)
    # el cordón encierra la ventana y además todo el bolsillo de la pinza (+y)
    gy0_in, gy1_in = wy0 - 1.5, max(wy1, pocket_end) + 1.5
    head = L["screw_head_d"] / 2
    # fuera del corredor (|y| > cy) el marco tiene espesor completo: los tornillos no
    # necesitan esperar al final del bisel
    x_screw = gx_in + w + 1.2 + head
    cy = corridor_half(p)
    y_screw_pos = gy1_in + w + 1.2 + head
    # del lado −y los insertos (5 mm de hondo) no pueden caer sobre el reservorio
    insert_r = p.CELL["insert_d"] / 2
    y_screw_neg = min(gy0_in - w - 1.2 - head,
                      reservoir[2] - p.CELL["min_wall"] - insert_r)
    return {"fx_in": fx_in, "full_at": full_at, "slope": slope, "wy0": wy0, "wy1": wy1,
            "groove_w": w, "gx_in": gx_in, "gy0_in": gy0_in, "gy1_in": gy1_in,
            "x_screw": x_screw, "y_screw_pos": y_screw_pos, "y_screw_neg": y_screw_neg,
            "y_screw_corridor": cy + head + 0.5,
            "x_half": x_screw + L["edge_margin"]}


def screw_xy(p=P, lay=None) -> list[tuple[float, float]]:
    """Tornillos del marco, todos fuera del corredor del haz.

    En los lados ±x, uno a cada lado del corredor. En las filas ±y, a paso
    screw_pitch, saltando los que caerían sobre un puerto (los insertos bajan
    insert_depth mm) y lejos del chaflán de las esquinas.
    """
    import math

    lay = lay or layout(p)
    L, lz, c = p.LID_XRD, lay["lid"], p.CELL
    xs = lz["x_screw"]
    pts = [(sx * xs, sy * lz["y_screw_corridor"]) for sx in (1, -1) for sy in (1, -1)]
    # distancia al chaflán de la esquina ≥ pared seca + radio del inserto
    x_row = min(xs, lay["x_half"] - c["corner_chamfer"] / 2
                - (c["min_wall_dry"] + c["insert_d"] / 2) * math.sqrt(2))
    n = max(2, math.ceil(2 * x_row / L["screw_pitch"]) + 1)
    for y, face in ((lz["y_screw_pos"], "+y"), (lz["y_screw_neg"], "-y")):
        busy = []
        for spec in p.XRD_PORTS:
            if spec["face"] == face:
                r = port_outer_r(p, spec)
                x_port = spec["offset"] if face == "-y" else -spec["offset"]
                busy.append((x_port, r))
        for i in range(n):
            x = -x_row + 2 * x_row * i / (n - 1)
            if all(abs(x - xp) > r + c["insert_d"] / 2 + c["min_wall_dry"] for xp, r in busy):
                pts.append((x, y))
    return pts


def body_block(p, lay) -> cq.Solid:
    w = 2 * lay["x_half"]
    h = lay["y_pos"] - lay["y_neg"]
    wp = (cq.Workplane("XY").box(w, h, -lay["z_bottom"], centered=(True, False, False))
          .translate((0, lay["y_neg"], lay["z_bottom"])).edges("|Z")
          .chamfer(p.CELL["corner_chamfer"]))
    return wp.val()


def box(x0, x1, y0, y1, z0, z1) -> cq.Solid:
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))


def groove(p, lay) -> cq.Shape:
    """Ranura del cordón de EPDM bajo la película: rectángulo redondeado."""
    lz = lay["lid"]
    w, depth = lz["groove_w"], 0.75 * p.LID_XRD["film_cs"]

    def rrect(off):
        x = lz["gx_in"] + off
        y0, y1 = lz["gy0_in"] - off, lz["gy1_in"] + off
        r = 2.0 + off
        return (cq.Workplane("XY").center(0, (y0 + y1) / 2).rect(2 * x, y1 - y0)
                .extrude(-depth).edges("|Z").fillet(r).val())
    return rrect(w).cut(rrect(0.0))


def features(p=P) -> dict:
    s, k, lay = p.WE_XRD, p.CHAMBER_XRD, layout(p)
    sx, sy = lay["sx"], lay["sy"]
    f = {
        "sample_pocket": box(-sx, sx, -sy, sy, -s["sample_t"], 0.01),
        # la lengüeta toca la muestra sólo en su borde +y (tab_overlap), fuera del haz
        "tab_slot": box(-s["tab_w"] / 2 - 0.2, s["tab_w"] / 2 + 0.2, sy - s["tab_overlap"],
                        sy + 0.01,
                        -s["sample_t"] - s["tab_t"] - 0.05, -s["sample_t"] + 0.01),
        "clip_pocket": box(-lay["pocket_hw"], lay["pocket_hw"], sy - 0.01, lay["pocket_y1"],
                           lay["pocket_floor"], 0.01),
        "cover_recess": box(-lay["pocket_hw"] - k["cover_ledge"],
                            lay["pocket_hw"] + k["cover_ledge"],
                            sy + k["cover_open"], lay["pocket_y1"] + k["cover_ledge"],
                            -k["cover_t"], 0.01),
        "slot": box(-sx, sx, lay["res_y0"], -sy + 0.01, lay["z_rr"] - 0.01, 0.01),
        "reservoir": box(lay["res_x0"], lay["res_x1"], lay["res_y1"], lay["res_y0"],
                         lay["z_rf"], lay["z_rr"]),
        "groove": groove(p, lay),
        "screws": compound(cyl(p.CELL["insert_d"], -p.CELL["insert_depth"], 0.01, x, y)
                           for x, y in screw_xy(p, lay)),
    }
    for spec, r_out, r_in in port_axes(p, lay):
        f["port_" + spec["name"]] = port_solid(p, spec, r_out, r_in)
    return f


def port_axes(p=P, lay=None):
    """(spec, cara exterior, interior de la cavidad) de cada puerto, sobre su eje."""
    lay = lay or layout(p)
    out = []
    for spec in p.XRD_PORTS:
        if spec["face"] not in ("+y", "-y"):
            raise ValueError(f"puerto {spec['name']} en la cara {spec['face']}: en XRD sólo ±y "
                             "(las caras ±x están en el eje del haz)")
        if spec["face"] == "+y":
            out.append((spec, lay["y_pos"], lay["pocket_y1"] - 1.0))
        else:
            out.append((spec, -lay["y_neg"], -lay["res_y1"] - 1.0))
    return out


def bosses(p=P) -> list:
    return [b for b in (port_boss(p, s, ro, ri) for s, ro, ri in port_axes(p)) if b is not None]


def build(p=P) -> cq.Workplane:
    lay = layout(p)
    solid = body_block(p, lay)
    for b in bosses(p):
        solid = solid.fuse(b)
    for f in features(p).values():
        solid = solid.cut(f)
    return cq.Workplane("XY").add(solid.clean())


def sample(p=P) -> cq.Solid:
    s = p.WE_XRD
    return box(-s["sample_x"] / 2, s["sample_x"] / 2, -s["sample_y"] / 2, s["sample_y"] / 2,
               -s["sample_t"], 0.0)


def clamp(p=P) -> cq.Shape:
    """Envolvente del portaelectrodo: cabeza sobre la lengüeta + cuerpo hasta la cara +y."""
    c, lay = p.WE_CLAMP, layout(p)
    zc = lay["z_clip"]
    up, down = clamp_extent(c, p.WE_XRD.get("clamp_screw", "down"))
    jaw = box(-c["head_w"] / 2, c["head_w"] / 2, lay["y_jaw0"], lay["y_jaw1"], zc - down, zc + up)
    rod = cq.Solid.makeCylinder(c["rod_d"] / 2, lay["y_pos"] + 2 - lay["y_jaw1"],
                                cq.Vector(0, lay["y_jaw1"], zc), cq.Vector(0, 1, 0))
    return jaw.fuse(rod).clean()


def tab_path(p=P) -> list[tuple[float, float]]:
    """(y, z) del eje de la lengüeta: bajo la muestra, luego baja hasta la pinza."""
    s, lay = p.WE_XRD, layout(p)
    zt = -s["sample_t"] - s["tab_t"] / 2
    return [(lay["sy"] - s["tab_overlap"], zt), (lay["sy"], zt), (lay["y_jaw0"], lay["z_clip"]),
            (lay["y_jaw0"] + p.WE_CLAMP["grip_l"], lay["z_clip"])]


def summary(p=P) -> dict:
    from checks.xrd_clearance import footprint, min_two_theta_for_footprint

    lay, x = layout(p), p.XRD
    R, dv = x["goniometer_radius"], x["divergence_deg"]
    return {
        "cuerpo": f"{2 * lay['x_half']:.1f} (x, haz) × {lay['y_pos'] - lay['y_neg']:.1f} (y) × "
                  f"{-lay['z_bottom']:.1f} (z) mm",
        "huella a 2θ_min":
            f"{footprint(x['two_theta_min'], R, dv):.1f} mm a {x['two_theta_min']:g}°",
        "2θ mínimo con la huella dentro de la muestra":
            f"{min_two_theta_for_footprint(p.WE_XRD['sample_x'], R, dv):.1f}°",
        "corredor del haz": f"|y| ≤ {corridor_half(p):.1f} mm",
        "borde interior del marco": f"|x| = {lay['lid']['fx_in']:.2f} mm "
                                    f"(espesor completo desde {lay['lid']['full_at']:.1f})",
        "pinza": f"eje a z = {lay['z_clip']:.1f}, "
                 f"mandíbulas en y = {lay['y_jaw0']:.1f}–{lay['y_jaw1']:.1f}",
    }


if __name__ == "__main__":
    from export import export

    for f in export(build(), "chamber_xrd"):
        print(f)
    for k, v in summary().items():
        print(f"{k}: {v}")
