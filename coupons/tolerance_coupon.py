"""Cupón de tolerancias FDM (paso 1 del orden de trabajo).

Filas: un Ø nominal por fila (params.COUPON["hole_nominals"]).
Columnas: sobremedida modelada creciente (params.COUPON["hole_offsets"]),
numeradas con 1..n hoyuelos sobre el borde de la columna; la esquina biselada
marca el inicio (columna 1, primera fila).
Última fila: ranuras de O-ring de cara con profundidad = factor · cs.

Imprimir en PP, con las ranuras hacia arriba, y medir:
  - qué columna deja pasar un pin calibrado del Ø nominal → FDM["hole_offset"]
  - qué profundidad de ranura sella sin extruir el O-ring.
Cara superior en z = 0.
"""
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from params import COUPON, ORING, oring_gland  # noqa: E402


def groove_rings(p_coupon=COUPON, p_oring=ORING):
    """(d_exterior, d_interior, profundidad) de cada ranura de prueba.

    Sello de cara con presión interna: el O-ring se apoya en la pared exterior,
    así que el Ø exterior de la ranura = Ø exterior del O-ring.
    """
    cs, oid = p_oring["face_cs"], p_oring["face_id"]
    width = oring_gland(cs, "face")["width"]
    d_out = oid + 2 * cs
    d_in = d_out - 2 * width
    return [(d_out, d_in, round(f * cs, 3)) for f in p_coupon["groove_depth_factors"]]


def layout(p_coupon=COUPON, p_oring=ORING):
    """Dimensiones de la placa y centros de los elementos."""
    c = p_coupon
    n_col, n_row = len(c["hole_offsets"]), len(c["hole_nominals"])
    rings = groove_rings(c, p_oring)
    ring_pitch = max(c["pitch"], rings[0][0] + c["margin"])
    w_holes = (n_col - 1) * c["pitch"]
    w_rings = (len(rings) - 1) * ring_pitch
    width = max(w_holes, w_rings) + 2 * c["margin"] + max(c["hole_nominals"])
    rows_y = [i * c["pitch"] for i in range(n_row)]
    ring_y = rows_y[-1] + c["pitch"] / 2 + ring_pitch / 2 + max(c["hole_nominals"]) / 2
    length = ring_y + ring_pitch / 2 + 2 * c["margin"] + max(c["hole_nominals"]) / 2
    y0 = -length / 2 + c["margin"] + max(c["hole_nominals"]) / 2 + c["pitch"] / 2
    return {
        "width": width,
        "length": length,
        "holes": [((-w_holes / 2 + j * c["pitch"], y0 + rows_y[i]), c["hole_nominals"][i] + off)
                  for i in range(n_row) for j, off in enumerate(c["hole_offsets"])],
        "rings": [((-w_rings / 2 + k * ring_pitch, y0 + ring_y), r) for k, r in enumerate(rings)],
        "col_x": [-w_holes / 2 + j * c["pitch"] for j in range(n_col)],
    }


def build(p_coupon=COUPON, p_oring=ORING) -> cq.Workplane:
    c = p_coupon
    lay = layout(c, p_oring)
    t = c["thickness"]
    body = (cq.Workplane("XY").box(lay["width"], lay["length"], t)
            .translate((0, 0, -t / 2))
            .edges("|Z and <X and <Y").chamfer(c["corner_chamfer"]))
    for (x, y), d in lay["holes"]:
        body = body.cut(cq.Workplane("XY").center(x, y).circle(d / 2).extrude(-t))
    for (x, y), (d_out, d_in, depth) in lay["rings"]:
        body = body.cut(cq.Workplane("XY").center(x, y).circle(d_out / 2)
                        .circle(d_in / 2).extrude(-depth))
    # numeración de columnas: j+1 hoyuelos en el borde -Y
    y_mark = -lay["length"] / 2 + c["margin"] / 2
    for j, x in enumerate(lay["col_x"]):
        for k in range(j + 1):
            xm = x + (k - j / 2) * 1.8 * c["mark_d"]
            body = body.cut(cq.Workplane("XY").center(xm, y_mark)
                            .circle(c["mark_d"] / 2).extrude(-c["mark_depth"]))
    return body


if __name__ == "__main__":
    from export import export

    for f in export(build(), "tolerance_coupon"):
        print(f)
    lay = layout()
    print(f"placa {lay['width']:.1f} × {lay['length']:.1f} mm")
    for (_, _), (d_out, d_in, depth) in lay["rings"]:
        print(f"ranura Ø{d_in:.2f}–{d_out:.2f} mm, profundidad {depth} mm")
