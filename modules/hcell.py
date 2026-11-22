"""Celda H: dos compartimentos separados por una membrana de intercambio iónico (Nafion).

- Dos medias celdas IGUALES (`build_half`): se imprime o maquina la misma pieza dos veces.
  Cada una tiene una cámara cilíndrica, una ventana horizontal hacia su cara +x y, en esa
  cara, una ranura para un O-ring de cara alrededor de la ventana. La segunda media celda
  es la primera girada 180°: las ventanas y los O-rings quedan enfrentados, con la membrana
  entre los dos O-rings. Cuatro M3 pasantes por las bridas aprietan hasta que las caras
  topan con la membrana: la compresión del O-ring la fija la ranura, no el apriete.
- Tapa de trabajo (`build_lid("we")`): WE (GC Ø10 boca abajo, frente a la ventana), RE
  (cuerpo Ø9, punta a la altura del disco), entrada de gas (tubo de burbujeo hasta el fondo),
  salida de gas (al cromatógrafo o a una trampa) y muestreo de líquido (conector con septo).
- Tapa del contraelectrodo (`build_lid("ce")`): CE de Pt Ø6, entrada y salida de gas.
- z = 0 es el disco del WE, a la altura del centro de la ventana: el camino iónico WE → CE
  pasa recto por la membrana.
- Cada cámara admite una barra de agitación (stir_bar) bajo los electrodos.

Uso típico: electrólisis con análisis de productos (reducción de NO3⁻/NO2⁻ a NH3, evolución
de hidrógeno, CO2): los productos del CE no llegan al WE ni contaminan el análisis.
Material: PP impreso o PEEK/PTFE maquinado; O-rings de EPDM.
"""
import math
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._geom import STD_LENGTHS, axial_port, block, compound, cyl  # noqa: E402


def layout(p=P) -> dict:
    h = p.HCELL
    face = h["body_w"] / 2
    return {
        "face": face,                                   # cara de la membrana (x)
        "z_bottom": h["z_floor"] - h["floor_t"],
        "z_lid": (h["z_top"], h["z_top"] + h["lid_t"]),
        # centro de la otra media celda (girada 180°), con la membrana entre las caras
        "x_other": 2 * face + h["membrane_t"],
    }


def _xbar(d, x0, x1, y=0.0, z=0.0) -> cq.Solid:
    return cq.Solid.makeCylinder(d / 2, x1 - x0, cq.Vector(x0, y, z), cq.Vector(1, 0, 0))


def face_gland(p=P) -> dict:
    h = p.HCELL
    g = p.oring_gland(h["mem_oring_cs"], "face")
    r_in = h["mem_oring_id"] / 2
    return {"r_in": r_in, "r_out": r_in + g["width"], "depth": g["depth"]}


def bolt_yz(p=P):
    return [(sy * y, z) for y, z in p.HCELL["flange_bolts"] for sy in (1, -1)]


def lid_screw_xy(p=P):
    a = p.HCELL["lid_insert_xy"]
    return [(sx * a, sy * a) for sx in (1, -1) for sy in (1, -1)]


def half_features(p=P) -> dict:
    h, c, lay = p.HCELL, p.CELL, layout(p)
    face, gl = lay["face"], face_gland(p)
    z0 = lay["z_bottom"]
    ring = _xbar(2 * gl["r_out"], face - gl["depth"], face + 1).cut(
        _xbar(2 * gl["r_in"], face - gl["depth"] - 1, face + 2))
    return {
        "chamber": cyl(h["chamber_d"], h["z_floor"], h["z_top"] + 0.01),
        "window": _xbar(h["window_d"], 0.0, face + 1),
        "mem_gland": ring,
        "bolts": compound([_xbar(p.hole(c["bolt_clearance_d"] - p.FDM["hole_offset"]),
                                 face - h["flange_t"] - 1, face + 1, y, z)
                           for y, z in bolt_yz(p)]),
        "lid_inserts": compound([cyl(c["insert_d"], h["z_top"] - c["insert_depth"],
                                     h["z_top"] + 0.01, x, y) for x, y in lid_screw_xy(p)]),
        "_z0": z0,
    }


def half_outline(p=P) -> cq.Solid:
    h, lay = p.HCELL, layout(p)
    z0, z1 = lay["z_bottom"], h["z_top"]
    s = block(h["body_w"], h["body_w"], z0, z1, h["corner_chamfer"])
    fl = block(h["flange_t"], h["flange_w"], z0, z1, 0.0, lay["face"] - h["flange_t"] / 2, 0.0)
    return s.fuse(fl).clean()


def build_half(p=P) -> cq.Workplane:
    s = half_outline(p)
    for k, f in half_features(p).items():
        if not k.startswith("_"):
            s = s.cut(f)
    return cq.Workplane("XY").add(s.clean())


def lid_ports(p, which: str):
    return p.HCELL_WE_PORTS if which == "we" else p.HCELL_CE_PORTS


def lid_features(p=P, which: str = "we") -> dict:
    h, c, lay = p.HCELL, p.CELL, layout(p)
    z0, z1 = lay["z_lid"]
    g = p.oring_gland(h["lid_oring_cs"], "face")
    r_in = h["lid_oring_id"] / 2
    f = {
        "gland": cyl(2 * (r_in + g["width"]), z0 - 0.01, z0 + g["depth"]).cut(
            cyl(2 * r_in, z0 - 1, z0 + g["depth"] + 1)),
        "screws": compound([cyl(p.hole(c["bolt_clearance_d"] - p.FDM["hole_offset"]),
                                z0 - 1, z1 + 1, x, y) for x, y in lid_screw_xy(p)]),
    }
    for spec in lid_ports(p, which):
        f["port_" + spec["name"]] = axial_port(p, spec, z1, z0 - 0.5)
    return f


def build_lid(p=P, which: str = "we") -> cq.Workplane:
    h, lay = p.HCELL, layout(p)
    s = block(h["body_w"], h["body_w"], *lay["z_lid"], h["corner_chamfer"])
    for f in lid_features(p, which).values():
        s = s.cut(f)
    return cq.Workplane("XY").add(s.clean())


def other_side(shape: cq.Shape, p=P) -> cq.Shape:
    """Lleva una pieza de la media celda de trabajo a la del contraelectrodo."""
    return shape.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 180).translate(
        cq.Vector(layout(p)["x_other"], 0, 0))


def membrane(p=P) -> cq.Solid:
    h, face = p.HCELL, layout(p)["face"]
    return _xbar(h["membrane_d"], face, face + h["membrane_t"])


def mem_orings(p=P) -> list:
    """Los dos O-rings de la membrana, comprimidos (para el ensamble)."""
    h, face, gl = p.HCELL, layout(p)["face"], face_gland(p)
    cs = h["mem_oring_cs"]
    r = gl["r_in"] + cs / 2
    a = cq.Solid.makeTorus(r, cs / 2, cq.Vector(face - gl["depth"] / 2 - 0.1, 0, 0),
                           cq.Vector(1, 0, 0))
    a = a.intersect(_xbar(2 * gl["r_out"] + 2, face - gl["depth"], face))
    b = a.mirror("YZ", cq.Vector(face + h["membrane_t"] / 2, 0, 0))
    return [a, b]


def flange_bolt_length(p=P) -> int:
    h, c = p.HCELL, p.CELL
    need = 2 * h["flange_t"] + h["membrane_t"] + c["nut_h"] + 0.5   # + arandela
    return min(L for L in STD_LENGTHS if L >= need)


def bolts(p=P) -> list:
    h, c, face = p.HCELL, p.CELL, layout(p)["face"]
    x0 = face - h["flange_t"]
    L = flange_bolt_length(p)
    out = []
    for y, z in bolt_yz(p):
        shank = _xbar(3.0, x0, x0 + L, y, z)
        head = _xbar(c["bolt_head_d"], x0 - c["bolt_head_h"], x0, y, z)
        x_nut = face + h["membrane_t"] + h["flange_t"]
        nut = (cq.Workplane("YZ").polygon(6, c["nut_af"] / math.cos(math.pi / 6))
               .extrude(c["nut_h"]).translate((x_nut, y, z)).val())
        out.append(shank.fuse(head).fuse(nut))
    return out


def stir_bar(p=P, x: float = 0.0) -> cq.Solid:
    h = p.HCELL
    L, d = h["stir_bar"]
    s = cq.Solid.makeCylinder(d / 2, L - d, cq.Vector(x - (L - d) / 2, 0, h["z_floor"] + d / 2),
                              cq.Vector(1, 0, 0))
    for sx in (1, -1):
        s = s.fuse(cq.Solid.makeSphere(d / 2, cq.Vector(x + sx * (L - d) / 2, 0,
                                                        h["z_floor"] + d / 2)))
    return s


def summary(p=P) -> dict:
    h = p.HCELL
    r = h["chamber_d"] / 2
    vol = math.pi * r ** 2 * (h["z_fill"] - h["z_floor"]) / 1000
    return {
        "volumen_por_compartimento_mL": round(vol, 1),
        "membrana": f"Ø{h['membrane_d']} × {h['membrane_t']} mm; área expuesta "
                    f"{math.pi * (h['window_d'] / 2) ** 2 / 100:.2f} cm²",
        "O-rings": f"2 × {h['mem_oring_id']:g}×{h['mem_oring_cs']} (membrana), "
                   f"2 × {h['lid_oring_id']:g}×{h['lid_oring_cs']} (tapas), "
                   f"radiales sobre WE, RE y CE",
        "tornillería": f"4 × M3×{flange_bolt_length(p)} con tuerca (bridas); "
                       f"8 × M3×{lid_screw_length(p)} a insertos (tapas)",
        "WE": f"disco en z = 0, a {h['chamber_d'] / 2 - p.HCELL_WE_PORTS[0]['xy'][0] - 5:.1f} mm "
              "de la ventana",
        "nivel": f"electrolito hasta z = {h['z_fill']}; ventana hasta z = {h['window_d'] / 2}",
    }


def lid_screw_length(p=P) -> int:
    h, c = p.HCELL, p.CELL
    room = h["lid_t"] + c["insert_depth"] - 0.5
    return max(L for L in STD_LENGTHS if L <= room)


if __name__ == "__main__":
    from export import export

    for name, part in (("hcell_half", build_half()), ("hcell_lid_we", build_lid(which="we")),
                       ("hcell_lid_ce", build_lid(which="ce"))):
        for f in export(part, name):
            print(f)
    for k, v in summary().items():
        print(f"{k}: {v}")
