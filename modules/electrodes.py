"""Modelos 3D de los electrodos del usuario (medidos el 2026-11-10), para ensambles.

Sólo para ver la configuración y comprobar que nada choca: no se fabrican. Cada
función devuelve una lista de (nombre, sólido, color) construida a lo largo de +x
con la punta (lo que entra a la celda) en x = 0; `place` la lleva a su puerto.

- RE Ag/AgCl Gamry: punta Ø4 × 20 mm y cuerpo de vidrio Ø9 hasta 75 mm.
- CE de Pt: cuerpo Ø6 con 35 mm de alambre, enrollado en espiral (como la celda R1 de
  docs/referencias.md): recto, 35 mm no caben en ninguna de estas celdas.
- WE: pinza CS945 (varilla Ø6 con ranura y tornillo hacia abajo) y double knob.
- GC: cuerpo con el disco al frente (para la base desde abajo, pendiente).
"""
import math
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402

COLORS = {                      # (r, g, b, alfa)
    "glass": (0.75, 0.88, 0.95, 0.55),
    "agcl": (0.55, 0.55, 0.60, 1.0),
    "pt": (0.85, 0.85, 0.88, 1.0),
    "peek": (0.80, 0.72, 0.55, 1.0),
    "ptfe": (0.96, 0.96, 0.94, 1.0),
    "gold": (0.90, 0.72, 0.20, 1.0),
    "black": (0.12, 0.12, 0.12, 1.0),
    "gc": (0.15, 0.15, 0.18, 1.0),
    "cable": (0.10, 0.10, 0.10, 1.0),
}


def _bar(d, x0, x1, y=0.0, z=0.0) -> cq.Solid:
    return cq.Solid.makeCylinder(d / 2, x1 - x0, cq.Vector(x0, y, z), cq.Vector(1, 0, 0))


def _box(x0, x1, y0, y1, z0, z1) -> cq.Solid:
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))


def reference(p=P) -> list:
    e = p.ELECTRODES
    tip, length = e["RE_tip_l"], e["RE_l"]
    return [("RE punta (frita)", _bar(e["RE_tip_d"], 0, tip), COLORS["glass"]),
            ("RE cuerpo Ø9", _bar(e["RE_body_d"], tip, length), COLORS["glass"]),
            ("RE Ag/AgCl", _bar(1.0, 3, length - 5), COLORS["agcl"]),
            ("RE cable", _bar(3.0, length, length + 15), COLORS["cable"])]


def ce_lengths(p=P) -> tuple[float, float]:
    """(largo axial de la espiral, tramo recto hasta el cuerpo) del CE de Pt."""
    e = p.ELECTRODES
    turn = math.hypot(math.pi * e["CE_coil_d"], e["CE_coil_pitch"])
    straight = 2.0
    coil_h = (e["CE_rod_wire_l"] - straight) / turn * e["CE_coil_pitch"]
    return max(e["CE_coil_pitch"], coil_h), straight


def counter_pt(p=P) -> list:
    """Electrodo de Pt Ø6: 35 mm de alambre enrollados en espiral frente al cuerpo."""
    e = p.ELECTRODES
    d_w, coil_d, pitch = 0.5, e["CE_coil_d"], e["CE_coil_pitch"]
    straight = 2.0
    turn = math.hypot(math.pi * coil_d, pitch)
    height = max(pitch, (e["CE_rod_wire_l"] - straight) / turn * pitch)
    helix = cq.Wire.makeHelix(pitch, height, coil_d / 2)
    prof = cq.Workplane("XY").add(cq.Wire.makeCircle(d_w / 2, helix.startPoint(),
                                                     helix.tangentAt(0)))
    coil = cq.Solid.sweep(prof.val(), [], helix, isFrenet=True)
    # la espiral se construye sobre z: se gira para que su eje sea x, de x = 0 a height
    coil = coil.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 1, 0), 90)
    x_body = height + straight
    return [("CE espiral de Pt (35 mm)", coil, COLORS["pt"]),
            ("CE alambre", _bar(d_w, height - 0.2, x_body + 1), COLORS["pt"]),
            ("CE cuerpo Ø6", _bar(e["CE_rod_d"], x_body, x_body + e["CE_rod_l"]), COLORS["ptfe"]),
            ("CE conector", _bar(1.0, x_body + e["CE_rod_l"], x_body + e["CE_rod_l"] + 10),
             COLORS["gold"])]


def clamp_cs945(c=None, p=P) -> list:
    """CS945: x = 0 es la cara frontal (por donde entra la muestra); tornillo hacia −z."""
    c = c or p.CLAMPS["CS945"]
    r = c["rod_d"] / 2
    rod = _bar(c["rod_d"], 0, c["rod_l"])
    rod = rod.cut(_box(-1, c["grip_l"], -r - 1, r + 1,
                       -c["max_sample_t"] / 2, c["max_sample_t"] / 2))
    xs = c["grip_l"] / 2
    # el tornillo sube hasta la muestra y la aprieta: termina en el fondo de la ranura
    screw = cq.Solid.makeCylinder(1.5, r - c["max_sample_t"] / 2, cq.Vector(xs, 0, -r),
                                  cq.Vector(0, 0, 1))
    head = cq.Solid.makeCylinder(2.5, c["head_down"] - r, cq.Vector(xs, 0, -c["head_down"]),
                                 cq.Vector(0, 0, 1))
    return [("CS945 varilla PEEK Ø6", rod, COLORS["peek"]),
            ("CS945 contacto de Pt", _box(0.2, c["grip_l"], -r + 0.6, r - 0.6,
                                          -c["max_sample_t"] / 2 - 0.1,
                                          -c["max_sample_t"] / 2), COLORS["pt"]),
            ("CS945 tornillo", screw.fuse(head), COLORS["peek"]),
            ("CS945 conector", _bar(1.0, c["rod_l"], c["rod_l"] + 10), COLORS["gold"])]


def clamp_double_knob(c=None, p=P) -> list:
    c = c or p.CLAMPS["double_knob"]
    knob_h = 4.0
    head = _box(0, c["head_l"], -c["head_w"] / 2, c["head_w"] / 2,
                -c["head_down"], c["head_up"] - knob_h)
    head = head.cut(_box(-1, c["grip_l"], -c["head_w"] / 2 + 2, c["head_w"] / 2 - 2,
                         -c["max_sample_t"] / 2, c["max_sample_t"] / 2))
    knobs = [cq.Solid.makeCylinder(3.5, knob_h, cq.Vector(x, 0, c["head_up"] - knob_h),
                                   cq.Vector(0, 0, 1)) for x in (6.0, 15.0)]
    rod_l = c["rod_l"] or 100.0
    return [("double knob cabeza PTFE", head, COLORS["ptfe"]),
            ("double knob perillas", knobs[0].fuse(knobs[1]), COLORS["ptfe"]),
            ("double knob cuerpo Ø6", _bar(c["rod_d"], c["head_l"], c["head_l"] + rod_l),
             COLORS["ptfe"]),
            ("double knob conector", _bar(1.0, c["head_l"] + rod_l, c["head_l"] + rod_l + 10),
             COLORS["gold"])]


def glassy_carbon(option: int = 0, p=P) -> list:
    g = p.ELECTRODES["GC_options"][option]
    length = g["l"] or 60.0
    return [(f"GC Ø{g['disk_d']:g} cuerpo", _bar(g["body_d"], 0.05, length), COLORS["peek"]),
            (f"GC disco Ø{g['disk_d']:g}", _bar(g["disk_d"], 0, 0.05), COLORS["gc"])]


def place(parts: list, tip, outward_deg: float, vertical: str | None = None) -> list:
    """Coloca un electrodo construido sobre +x.

    tip: punto (x, y, z) del mundo donde queda x = 0 del modelo.
    outward_deg: ángulo en el plano XY de la dirección hacia afuera de la celda.
    vertical: "up" si el electrodo entra desde arriba (su +x apunta a +z).
    """
    out = []
    for name, s, col in parts:
        if vertical == "up":
            s = s.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 1, 0), -90)
        else:
            s = s.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), outward_deg)
        out.append((name, s.translate(cq.Vector(*tip)), col))
    return out


def port_tip(spec: dict, r_tip: float) -> tuple[float, float, float]:
    """Punto sobre el eje de un puerto lateral a distancia r_tip del centro."""
    from modules._geom import FACE_ANGLE

    a = math.radians(FACE_ANGLE[spec["face"]])
    off = spec["offset"]
    return (r_tip * math.cos(a) - off * math.sin(a), r_tip * math.sin(a) + off * math.cos(a),
            spec["z"])
