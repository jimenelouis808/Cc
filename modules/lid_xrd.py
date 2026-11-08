"""Marco tensor de la película para la celda XRD, y la tapa del bolsillo de la pinza.

El marco aprieta la película contra el cordón de EPDM de la cara superior del
cuerpo (chamber_xrd). Dentro del corredor del haz (|y| ≤ semiancho) sus lados ±x
están biselados: el borde interior mide edge_t y sube con bevel_deg, por debajo de
la altura que el haz permite a cada distancia (checks/xrd_clearance.py). Fuera del
corredor el marco tiene su espesor completo y lleva los tornillos, que pueden
sobresalir: el haz no pasa por ahí.

Material: aluminio o acero (no toca el electrolito; la película queda debajo).
La película se perfora en los tornillos, fuera del cordón.

La tapa del bolsillo es una placa al ras (z = 0) que se apoya en un escalón del
bolsillo de la pinza para que la película no se hunda en él al aspirar.
"""
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules.chamber_xrd import box, corridor_half, layout, screw_xy  # noqa: E402
from modules._geom import compound, cyl  # noqa: E402


def frame_profile(p, d: float) -> float:
    """Altura del marco a |x| = d dentro del corredor (0 dentro de la ventana)."""
    lz, L = layout(p)["lid"], p.LID_XRD
    if d < lz["fx_in"]:
        return 0.0
    return min(L["t"], L["edge_t"] + (d - lz["fx_in"]) * lz["slope"])


def build(p=P) -> cq.Workplane:
    lay = layout(p)
    lz, L = lay["lid"], p.LID_XRD
    w = 2 * lay["x_half"]
    h = lay["y_pos"] - lay["y_neg"]
    outer = (cq.Workplane("XY").box(w, h, L["t"], centered=(True, False, False))
             .translate((0, lay["y_neg"], 0)).edges("|Z").chamfer(p.CELL["corner_chamfer"]).val())
    frame = outer.cut(box(-lz["fx_in"], lz["fx_in"], lz["wy0"], lz["wy1"], -1, L["t"] + 1))
    # bisel de los lados ±x dentro del corredor
    cy = corridor_half(p) + 0.5
    x_end = lay["x_half"] + 5
    for sgn in (1, -1):
        z_end = L["edge_t"] + (x_end - lz["fx_in"]) * lz["slope"]
        pts = [(lz["fx_in"], L["edge_t"]), (x_end, z_end),
               (x_end, L["t"] + 5), (lz["fx_in"], L["t"] + 5)]
        pts = [(sgn * x, z) for x, z in pts]
        wedge = (cq.Workplane("XZ").polyline(pts).close().extrude(cy)
                 .union(cq.Workplane("XZ").polyline(pts).close().extrude(-cy)).val())
        frame = frame.cut(wedge)
    holes = compound(cyl(p.hole(L["screw_d"]), -1, L["t"] + 1, x, y) for x, y in screw_xy(p, lay))
    return cq.Workplane("XY").add(frame.cut(holes).clean())


def build_cover(p=P) -> cq.Workplane:
    lay, k = layout(p), p.CHAMBER_XRD
    c = 0.1
    hw = lay["pocket_hw"] + k["cover_ledge"] - c
    y0 = lay["sy"] + k["cover_open"] + c
    y1 = lay["pocket_y1"] + k["cover_ledge"] - c
    return cq.Workplane("XY").add(box(-hw, hw, y0, y1, -k["cover_t"], 0.0))


def summary(p=P) -> dict:
    lay = layout(p)
    lz, L = lay["lid"], p.LID_XRD
    return {
        "marco": f"{2 * lay['x_half']:.1f} × {lay['y_pos'] - lay['y_neg']:.1f} × {L['t']} mm",
        "ventana": f"|x| ≤ {lz['fx_in']:.2f}, y = {lz['wy0']:.1f} … {lz['wy1']:.1f} mm",
        "bisel": f"{L['bevel_deg']}° desde {L['edge_t']} mm en el borde (sólo en el corredor)",
        "cordón": f"EPDM cs {L['film_cs']}, rectángulo de "
                  f"{2 * (lz['gx_in'] + lz['groove_w'] / 2):.1f} × "
                  f"{lz['gy1_in'] - lz['gy0_in'] + lz['groove_w']:.1f} mm (eje)",
        "tornillos": f"{len(screw_xy(p, lay))} × M3 en insertos; ninguno dentro del corredor",
    }


if __name__ == "__main__":
    from export import export

    for f in export(build(), "lid_xrd"):
        print(f)
    for f in export(build_cover(), "xrd_pocket_cover"):
        print(f)
    for k, v in summary().items():
        print(f"{k}: {v}")
