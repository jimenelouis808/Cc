"""Celda EPD / DEP selectiva: chip con electrodos interdigitados de gap graduado.

Apilado (z = 0 es la cara del chip con los electrodos):
  base (con ventana para luz transmitida)  →  chip (portaobjetos)  →  junta con el canal
  →  vidrio superior perforado  →  marco con ventana, conectores y tope duro.

- Electrodos: dos peines con dedos a lo largo de y que cruzan el canal. El gap es
  constante dentro de cada segmento y baja de la entrada a la salida (EPD["gaps"]):
  E ~ V/g y ∇|E|² ~ V²/g³ crecen río abajo, así que una partícula con más respuesta
  DEP se captura antes. Los buses van bajo la junta (secos) hasta dos pads en el
  extremo +x, fuera del marco, para las pinzas.
- Los electrodos se depositan por sputtering a través de una máscara de sombra:
  mask_dxf() exporta las aberturas y el contorno de la lámina.
- La junta define el canal; el marco apoya en la base con un tope duro que deja la
  junta comprimida gasket_compression: la altura del canal queda fija.
- Mojado: vidrio, electrodo (Pt/Au), junta (PTFE/FFKM), O-rings de cara (FFKM con
  acetona) y conectores PEEK. Marco y base no tocan el líquido.
"""
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._geom import compound, cyl  # noqa: E402


def box(x0, x1, y0, y1, z0, z1) -> cq.Solid:
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))


def slot(length: float, width: float, z0: float, z1: float) -> cq.Solid:
    """Canal: rectángulo de extremos redondeados centrado en el origen, a lo largo de x."""
    return (cq.Workplane("XY").slot2D(length, width).extrude(z1 - z0)
            .translate((0, 0, z0)).val())


# --- patrón de electrodos -------------------------------------------------------

def ide_rects(p=P) -> dict:
    """Rectángulos (x0, x1, y0, y1) de cada peine, segmento por segmento.

    Peine A (bus en +y) y peine B (bus en −y) alternan dedos. Cada dedo va de su bus
    hasta tip_gap antes del bus opuesto; tip_gap = mayor gap, para que la punta no
    domine el campo.
    """
    e = p.EPD
    W, ov, fw, bw = e["channel_w"], e["finger_overlap"], e["finger_w"], e["bus_w"]
    tip = max(e["gaps"])
    y_in = W / 2 + ov + tip                 # borde interior de cada bus
    segs = []
    total = len(e["gaps"]) * e["segment_l"] + (len(e["gaps"]) - 1) * e["segment_spacing"]
    x = -total / 2
    A, B = [], []
    for g in e["gaps"]:
        x_end = x + e["segment_l"]
        n = int((e["segment_l"] - fw) // (fw + g)) + 1
        span = n * fw + (n - 1) * g
        x0 = x + (e["segment_l"] - span) / 2
        fingers = []
        for i in range(n):
            fx = x0 + i * (fw + g)
            if i % 2 == 0:
                A.append((fx, fx + fw, -y_in + tip, y_in))
            else:
                B.append((fx, fx + fw, -y_in, y_in - tip))
            fingers.append(fx)
        segs.append({"gap": g, "x0": x0, "x1": x0 + span, "n": n, "fingers": fingers})
        x = x_end + e["segment_spacing"]
    # buses: de un extremo de la zona de dedos hasta los pads (+x)
    sx = e["slide"][0] / 2
    pad_x0 = sx - e["pad"][0]
    ide_x0, ide_x1 = segs[0]["x0"], segs[-1]["x1"]
    A.append((ide_x0, pad_x0 + 0.01, y_in, y_in + bw))
    B.append((ide_x0, pad_x0 + 0.01, -y_in - bw, -y_in))
    # pads: A arriba, B abajo, separados pad_gap; los buses bajan/suben hasta ellos
    py = e["pad"][1]
    A.append((pad_x0, sx, e["pad_gap"] / 2, e["pad_gap"] / 2 + py))
    B.append((pad_x0, sx, -e["pad_gap"] / 2 - py, -e["pad_gap"] / 2))
    return {"A": A, "B": B, "segments": segs, "y_bus_in": y_in, "ide_x": (ide_x0, ide_x1),
            "pad_x0": pad_x0}


def electrodes(p=P, t: float = 0.02) -> cq.Shape:
    """Electrodos en 3D (espesor simbólico t, sólo para ver y medir holguras)."""
    pat = ide_rects(p)
    return compound(box(x0, x1, y0, y1, 0.0, t) for x0, x1, y0, y1 in pat["A"] + pat["B"])


def mask_dxf(p=P, path=None) -> str:
    """DXF de la máscara de sombra: contorno de la lámina y aberturas (= electrodos).

    Coordenadas del portaobjetos (origen en su centro). Las aberturas se unen por
    peine: cada peine es una sola abertura conectada, y la lámina restante (los puentes
    entre dedos) es una sola pieza unida al borde.
    """
    from export import OUT

    e, pat = p.EPD, ide_rects(p)
    sx, sy, _ = e["slide"]
    b = e["mask_border"]
    sheet = cq.Workplane("XY").rect(sx + 2 * b, sy + 2 * b).extrude(1)
    for x0, x1, y0, y1 in pat["A"] + pat["B"]:
        sheet = sheet.cut(cq.Workplane("XY").add(box(x0, x1, y0, y1, -1, 2)))
    section = sheet.faces("<Z")
    path = path or os.path.join(OUT, "epd_mask.dxf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cq.exporters.exportDXF(section, path)
    return path


# --- geometría del apilado ------------------------------------------------------

def layout(p=P) -> dict:
    e = p.EPD
    sx, sy, st = e["slide"]
    cx, cy, ct = e["cover"]
    h_ch = e["gasket_t"] * (1 - e["gasket_compression"])
    z_cover_top = h_ch + ct
    hole_x = e["channel_l"] / 2 - e["channel_w"] / 2
    ide = ide_rects(p)
    win_x = max(abs(ide["ide_x"][0]), abs(ide["ide_x"][1])) + e["window_margin"]
    win_y = e["channel_w"] / 2 + e["window_margin"]
    fy = sy / 2 + e["frame_side"]
    fx = cx / 2 + 2.0
    return {"h_channel": h_ch, "z_cover_top": z_cover_top, "hole_x": hole_x,
            "win": (win_x, win_y), "frame": (fx, fy), "base_x": sx / 2 + 3.0,
            "z_base": -st - e["base_t"], "z_frame_top": z_cover_top + e["frame_t"],
            "screw_y": sy / 2 + e["frame_side"] / 2}


def gasket(p=P) -> cq.Solid:
    e, lay = p.EPD, layout(p)
    cx, cy, _ = e["cover"]
    g = box(-cx / 2, cx / 2, -cy / 2, cy / 2, 0, lay["h_channel"])
    return g.cut(slot(e["channel_l"], e["channel_w"], -1, 1))


def cover(p=P) -> cq.Solid:
    e, lay = p.EPD, layout(p)
    cx, cy, ct = e["cover"]
    c = box(-cx / 2, cx / 2, -cy / 2, cy / 2, lay["h_channel"], lay["z_cover_top"])
    for s in (-1, 1):
        c = c.cut(cyl(e["cover_hole_d"], 0, 5, s * lay["hole_x"], 0))
    return c


def chip(p=P) -> cq.Solid:
    sx, sy, st = p.EPD["slide"]
    return box(-sx / 2, sx / 2, -sy / 2, sy / 2, -st, 0)


def base_features(p=P) -> dict:
    e, lay = p.EPD, layout(p)
    sx, sy, st = e["slide"]
    wx, wy = lay["win"]
    return {
        "slide_pocket": box(-sx / 2 - 0.15, sx / 2 + 0.15, -sy / 2 - 0.15, sy / 2 + 0.15, -st, 1),
        "window": box(-wx, wx, -wy, wy, lay["z_base"] - 1, 1),
        "screws": compound(cyl(p.CELL["insert_d"], -p.CELL["insert_depth"], 0.01,
                               x, sgn * lay["screw_y"])
                           for x in e["screw_x"] for sgn in (1, -1)),
    }


def build_base(p=P) -> cq.Workplane:
    lay = layout(p)
    fy = lay["frame"][1]
    # en +x la base termina antes que el chip: el extremo con los pads queda en voladizo
    x_end = p.EPD["slide"][0] / 2 - p.EPD["clamp_overhang"]
    s = box(-lay["base_x"], x_end, -fy, fy, lay["z_base"], 0)
    for f in base_features(p).values():
        s = s.cut(f)
    return cq.Workplane("XY").add(s.clean())


def frame_features(p=P) -> dict:
    e, lay = p.EPD, layout(p)
    wx, wy = lay["win"]
    zc, zt = lay["z_cover_top"], lay["z_frame_top"]
    sx, sy, _ = e["slide"]
    o_id, o_cs = e["port_oring"]
    gland = p.oring_gland(o_cs, "face")
    f = {
        # todo lo que queda bajo la cara de apriete: chip, junta y vidrio superior (holgura)
        "stack": box(-e["cover"][0] / 2 - 0.3, e["cover"][0] / 2 + 0.3, -sy / 2 - 0.3, sy / 2 + 0.3,
                     -1, zc),
        # el marco no toca el portaobjetos fuera del vidrio superior: apoya en la base
        "chip_ends": box(-sx / 2 - 0.3, sx / 2 + 0.3, -sy / 2 - 0.3, sy / 2 + 0.3, -1,
                         p.EPD["frame_chip_clearance"]),
        "window": box(-wx, wx, -wy, wy, zc - 1, zt + 1),
        "screws": compound(
            cyl(p.hole(p.CELL["bolt_clearance_d"] - p.FDM["hole_offset"]), -1, zt + 1,
                x, sgn * lay["screw_y"])
            for x in e["screw_x"] for sgn in (1, -1)),
    }
    for spec in p.EPD_PORTS:
        x = spec["x_sign"] * lay["hole_x"]
        d_out = o_id + 2 * o_cs
        f["port_" + spec["name"]] = (
            cyl(p.PORTS["tap_drill_d"], zt - p.PORTS["thread_depth"], zt + 1, x, 0)
            .fuse(cyl(p.hole(spec["through_d"]), zc - 0.01,
                      zt - p.PORTS["thread_depth"] + 0.01, x, 0))
            .fuse(cyl(d_out, zc - 0.01, zc + gland["depth"], x, 0)
                  .cut(cyl(d_out - 2 * gland["width"], zc - 0.02,
                           zc + gland["depth"] + 0.01, x, 0))))
    return f


def build_frame(p=P) -> cq.Workplane:
    lay = layout(p)
    fx, fy = lay["frame"]
    s = box(-fx, fx, -fy, fy, 0.0, lay["z_frame_top"])     # sus patas tocan la base en z = 0
    for f in frame_features(p).values():
        s = s.cut(f)
    return cq.Workplane("XY").add(s.clean())


def summary(p=P) -> dict:
    e, lay, pat = p.EPD, layout(p), ide_rects(p)
    segs = ", ".join(f"{s['gap']*1000:.0f} µm ({s['n']} dedos)" for s in pat["segments"])
    return {
        "canal": f"{e['channel_l']} × {e['channel_w']} × {lay['h_channel']*1000:.0f} µm (junta "
                 f"{e['gasket_t']*1000:.0f} µm comprimida {e['gasket_compression']:.0%})",
        "segmentos": segs,
        "zona de electrodos": f"x = {pat['ide_x'][0]:.1f} … {pat['ide_x'][1]:.1f} mm",
        "pads": f"{e['pad'][0]} × {e['pad'][1]} mm en x > {pat['pad_x0']:.1f}; marco hasta "
                f"x = {lay['frame'][0]:.1f}",
        "ventana del marco": f"{2*lay['win'][0]:.1f} × {2*lay['win'][1]:.1f} mm",
        "volumen del canal": f"{(e['channel_l'] * e['channel_w']) * lay['h_channel']:.1f} µL",
    }


if __name__ == "__main__":
    from export import export

    for name, part in (("epd_base", build_base()), ("epd_frame", build_frame()),
                       ("epd_gasket", cq.Workplane().add(gasket())),
                       ("epd_cover", cq.Workplane().add(cover()))):
        for f in export(part, name):
            print(f)
    print(mask_dxf())
    for k, v in summary().items():
        print(f"{k}: {v}")
