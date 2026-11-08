"""Cámara para WE en pinza de Pt (variante de la celda R1 de docs/referencias.md).

Una sola pieza que reemplaza a base + anillo cuando el WE es una muestra sujeta
con el portaelectrodo de pinza. La cara superior es idéntica a la del anillo
(misma altura h, mismos insertos, mismo pozo), así que lid_raman sirve sin cambios.

- La muestra (tira sample_w × sample_l × sample_t) queda horizontal, cara arriba,
  con su cara en z = 0 (plano de referencia): la sostiene la pinza por un extremo
  y apoya en un pedestal bajo el punto de medición.
- La pinza entra por el puerto WE (cara −x, conector 1/4-28 sobre el cuerpo del
  portaelectrodo). Su borde queda a nose_d/2 + clip_clearance del eje óptico, fuera
  de la nariz de la tapa.
- Montaje: pasar el portaelectrodo desde afuera (la pinza debe caber por el puerto),
  poner la muestra en la pinza desde arriba con el pozo abierto, cerrar con la tapa.
- RE, CE, entrada y salida en el pozo (params.CLAMP_PORTS).

Área mojada: toda la tira fuera del agarre, ambas caras y bordes, más la pinza de Pt.
Para un área definida, enmascarar reverso y bordes (cinta de PTFE / epoxi resistente).
Imprimir boca abajo o maquinar en PTFE/PEEK.
"""
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._geom import body, compound, cyl, lid_insert_xy, port_solid  # noqa: E402


def _nose_r(p) -> float:
    from modules.lid_raman import nose_d

    return nose_d(p) / 2


def layout(p=P) -> dict:
    """Posiciones (x) de pinza y muestra, y alturas del piso y del fondo."""
    c = p.WE_CLAMP
    x_jaw = -(_nose_r(p) + c["clip_clearance"])          # borde de la pinza hacia el eje
    x0 = x_jaw - c["jaw_l"]                              # extremo de la muestra
    z_low = min(-c["sample_t"] - c["jaw_t"], p.CLAMP_PORTS[0]["z"] - c["rod_d"] / 2)
    z_floor = z_low - c["floor_clearance"]
    return {"x_jaw": x_jaw, "x0": x0, "x1": x0 + c["sample_l"],
            "z_floor": z_floor, "z_bottom": z_floor - p.CELL["min_wall"]}


def sample(p=P) -> cq.Solid:
    c, lay = p.WE_CLAMP, layout(p)
    return cq.Solid.makeBox(c["sample_l"], c["sample_w"], c["sample_t"],
                            cq.Vector(lay["x0"], -c["sample_w"] / 2, -c["sample_t"]))


def clamp(p=P) -> cq.Shape:
    """Envolvente del portaelectrodo dentro de la celda: pinza + cuerpo hasta la cara −x."""
    c, lay = p.WE_CLAMP, layout(p)
    zc = p.CLAMP_PORTS[0]["z"]
    zj0, zj1 = -c["sample_t"] - c["jaw_t"], c["jaw_t"]
    jaw = cq.Solid.makeBox(c["jaw_l"], c["clip_w"], zj1 - zj0,
                           cq.Vector(lay["x0"], -c["clip_w"] / 2, zj0))
    x_face = -p.CELL["body_w"] / 2 - 2
    rod = cq.Solid.makeCylinder(c["rod_d"] / 2, lay["x0"] - x_face, cq.Vector(x_face, 0, zc),
                                cq.Vector(1, 0, 0))
    return jaw.fuse(rod).clean()


def pedestal(p=P) -> cq.Solid | None:
    d = p.WE_CLAMP["pedestal_d"]
    if d <= 0:
        return None
    lay = layout(p)
    return cyl(d, lay["z_floor"] - 0.01, -p.WE_CLAMP["sample_t"])


def features(p=P) -> dict:
    h, lay = p.CHAMBER_RING["h"], layout(p)
    c = p.CELL
    f = {
        "well": cyl(p.CHAMBER_RING["well_d"], lay["z_floor"], h + 0.01),
        "lid_inserts": compound(cyl(c["insert_d"], h - c["insert_depth"], h + 0.01, x, y)
                                for x, y in lid_insert_xy(p)),
    }
    r_in = p.CHAMBER_RING["well_d"] / 2 - 1.0
    for spec in p.CLAMP_PORTS:
        f["port_" + spec["name"]] = port_solid(p, spec, c["body_w"] / 2, r_in)
    return f


def build(p=P) -> cq.Workplane:
    lay = layout(p)
    solid = body(p, lay["z_bottom"], p.CHAMBER_RING["h"])
    for f in features(p).values():
        solid = solid.cut(f)
    ped = pedestal(p)
    if ped is not None:
        solid = solid.fuse(ped)
    return cq.Workplane("XY").add(solid.clean())


def summary(p=P) -> dict:
    c, lay = p.WE_CLAMP, layout(p)
    free = c["sample_l"] - c["jaw_l"]
    one_face = c["sample_w"] * free
    wet = 2 * one_face + c["sample_t"] * (2 * free + c["sample_w"])
    return {
        "muestra": f"{c['sample_w']} × {c['sample_l']} × {c['sample_t']} mm, cara en z = 0",
        "pinza": f"borde en x = {lay['x_jaw']:.2f} (eje óptico en x = 0)",
        "punto_de_medición": f"a {lay['x1']:.2f} mm del borde libre de la tira",
        "área_cara_superior_mm2": round(one_face, 1),
        "área_mojada_total_mm2": round(wet, 1),
        "fondo_de_la_cámara_z": round(lay["z_bottom"], 2),
    }


if __name__ == "__main__":
    from export import export

    for f in export(build(), "chamber_clamp"):
        print(f)
    for k, v in summary().items():
        print(f"{k}: {v}")
