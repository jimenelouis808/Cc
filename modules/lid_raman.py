"""Tapa Raman (inVia): placa + émbolo cónico hueco + nariz con ventana al ras.

- La ventana (Ø window_d × window_t) va pegada en un rebaje en la punta de la
  nariz, al ras: el espesor de electrolito sobre el WE es exactamente `gap`.
- La nariz entra en el agujero del piso del anillo (Ø exposed_d) con holgura
  radial nose_clearance; el hombro queda shoulder_clearance sobre el piso.
- El interior es el cono de luz (NA, con refracción) + optical_margin, y por encima
  de la punta del objetivo, la envolvente del objetivo: el objetivo baja dentro de la
  tapa hasta su WD sin tocarla. Paredes del émbolo: CELL["min_wall"] (mojadas por fuera).
- La placa apoya sobre lainas (arandelas DIN 988 3×6) en los 4 tornillos:
      gap = thin_gap_raman + (laina − shim_nominal)
  El O-ring de cara de la placa sella contra la cara superior del anillo; su ranura
  es menos profunda que la estándar para compensar la laina nominal.
- Sin electrolito por encima de la placa: el émbolo y la placa son una pieza.

Material: PEEK o PTFE maquinado (la nariz es fina); PP impreso solo para ajuste.
Adhesivo de la ventana: verificar contra 3 M KOH / 0.5 M H2SO4 antes de usar.
Llenado: inclinar la celda con la salida (+x) arriba para purgar el aire del pozo.

z = 0 en el WE; la geometría se construye a gap nominal (laina = shim_nominal).
"""
import math
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._geom import body, compound, cyl, lid_insert_xy, sample_seal  # noqa: E402
from modules._optics import _revolve, light_cone, light_radius, objective_envelope  # noqa: E402
from modules.chamber_ring import floor_t  # noqa: E402


def z_plate(p=P) -> float:
    """Cara inferior de la placa a laina nominal."""
    return p.CHAMBER_RING["h"] + p.LID_RAMAN["shim_nominal"]


def nose_d(p=P) -> float:
    return sample_seal(p)["bore_d"] - 2 * p.LID_RAMAN["nose_clearance"]


def z_shoulder(p=P) -> float:
    return floor_t(p) + p.LID_RAMAN["shoulder_clearance"]


def _gaps(p):
    """Gaps extremos y nominal. En el marco de la tapa, el WE y el objetivo se
    desplazan −(gap − nominal) cuando cambia la laina."""
    lo, hi = p.CELL["gap_range_raman"]
    g0 = p.CELL["thin_gap_raman"]
    return [(g, g - g0) for g in (lo, g0, hi)]


def interior(p=P) -> cq.Shape:
    """Hueco óptico: cono de luz + envolvente del objetivo, en todo el recorrido de lainas."""
    parts = [light_cone(p, g).fuse(objective_envelope(p, g)).translate(cq.Vector(0, 0, -d))
             for g, d in _gaps(p)]
    return parts[0].fuse(*parts[1:]).clean()


def _interior_radius(p, z: float) -> float:
    """Radio del hueco óptico a la altura z (marco de la tapa, laina nominal)."""
    from modules._optics import z_tip

    r = p.RAMAN
    out = 0.0
    for g, d in _gaps(p):
        zz = z + d                                  # altura medida desde el WE con ese gap
        rad = light_radius(p, max(zz, 0.0), g) + r["optical_margin"]
        z0 = z_tip(p, g) - r["clearance_margin"]
        if zz >= z0:
            r0 = r["objective_tip_diameter"] / 2 + r["clearance_margin"]
            rb = r["objective_barrel_d"] / 2 + r["clearance_margin"]
            ang = math.radians(r["objective_nose_angle_deg"])
            rad = max(rad, min(rb, r0 + (zz - z0) * math.tan(ang)))
        out = max(out, rad)
    return out


def plunger_blank(p=P) -> cq.Solid:
    """Exterior macizo del émbolo (nariz + cuerpo), de la punta a la placa.

    El exterior sigue al hueco con un desplazamiento radial wall/cos(φ), donde φ es
    la mayor pendiente del hueco: la pared normal resulta ≥ min_wall. Donde el hueco
    se ensancha de golpe (cara frontal del objetivo) el exterior toma el máximo del
    hueco en los `wall` mm de arriba: queda un escalón macizo de espesor ≥ min_wall.
    """
    g = p.CELL["thin_gap_raman"]
    wall = p.CELL["min_wall"]
    phi = max(math.asin(p.RAMAN["numerical_aperture"]),
              math.radians(p.RAMAN["objective_nose_angle_deg"]))
    off = wall / math.cos(phi)
    rn = nose_d(p) / 2
    zs, zp = z_shoulder(p), z_plate(p)
    n, k = 120, 8

    look = wall + (zp - zs) / n      # un paso de muestreo extra: la esquina queda dentro

    def r_look_ahead(z):
        return max(_interior_radius(p, min(z + look * j / k, zp)) for j in range(k + 1))

    outer = []
    for i in range(n + 1):
        z = zs + (zp + 0.01 - zs) * i / n
        outer.append((max(rn, r_look_ahead(z) + off), z))
    pts = [(0, g), (rn, g), (rn, zs)] + outer + [(0, zp + 0.01)]
    return _revolve(pts)


def lid_gland(p=P) -> dict:
    o, L = p.ORING, p.LID_RAMAN
    cs = o["lid_cs"]
    depth = p.oring_gland(cs, "face")["depth"] - L["shim_nominal"]
    width = L["gland_width_factor"] * cs
    d_out = o["lid_id"] + 2 * cs
    return {"cs": cs, "depth": depth, "width": width, "d_out": d_out, "d_in": d_out - 2 * width}


def features(p=P) -> dict:
    c, L = p.CELL, p.LID_RAMAN
    zp = z_plate(p)
    zt = zp + L["plate_t"]
    gl = lid_gland(p)
    bolts = [cyl(p.hole(c["bolt_clearance_d"] - p.FDM["hole_offset"]), zp - 1, zt + 1, x, y)
             .fuse(cyl(p.hole(c["bolt_head_d"]), zt - c["bolt_head_h"], zt + 1, x, y))
             for x, y in lid_insert_xy(p)]
    g = p.CELL["thin_gap_raman"]
    return {
        "interior": interior(p),
        "window_recess": cyl(p.hole(L["window_d"]), g - 0.01, g + p.RAMAN["window_t"]),
        "gland": cyl(gl["d_out"], zp - 0.01, zp + gl["depth"]).cut(
            cyl(gl["d_in"], zp - 0.02, zp + gl["depth"] + 0.01)),
        "bolts": compound(bolts),
    }


def build(p=P, shim=None) -> cq.Workplane:
    """Tapa en posición de montaje. `shim` (mm) mueve la tapa respecto de la nominal."""
    zp = z_plate(p)
    solid = body(p, zp, zp + p.LID_RAMAN["plate_t"]).fuse(plunger_blank(p))
    for f in features(p).values():
        solid = solid.cut(f)
    solid = solid.clean()
    if shim is not None:
        solid = solid.translate((0, 0, shim - p.LID_RAMAN["shim_nominal"]))
    return cq.Workplane("XY").add(solid)


def gap_for_shim(p, shim: float) -> float:
    return p.CELL["thin_gap_raman"] + shim - p.LID_RAMAN["shim_nominal"]


def shim_range(p=P) -> tuple[float, float]:
    lo, hi = p.CELL["gap_range_raman"]
    s0 = p.LID_RAMAN["shim_nominal"] - p.CELL["thin_gap_raman"]
    return lo + s0, hi + s0


def lid_bolt(p=P) -> dict:
    """Tornillo placa → inserto: el más largo que no toca fondo a laina mínima."""
    from modules._geom import STD_LENGTHS

    c, L = p.CELL, p.LID_RAMAN
    s_lo, s_hi = shim_range(p)
    above = L["plate_t"] - c["bolt_head_h"]          # rosca que no engrana (placa)
    room = above + s_lo + c["insert_depth"] - 0.5
    Ls = [x for x in STD_LENGTHS if x <= room]
    if not Ls:
        raise ValueError("no cabe ningún tornillo estándar en la tapa")
    length = Ls[-1]
    return {"length": length, "engagement_min": length - above - s_hi}


def summary(p=P) -> dict:
    from modules._optics import z_tip

    gl = lid_gland(p)
    s_lo, s_hi = shim_range(p)
    return {
        "gap_nominal": p.CELL["thin_gap_raman"],
        "lainas": f"{s_lo:.2f}–{s_hi:.2f} mm → gap {p.CELL['gap_range_raman']}",
        "punta_objetivo_z": round(z_tip(p), 3),
        "nariz": f"Ø{nose_d(p):.2f}, hombro en z = {z_shoulder(p):.2f}",
        "ventana": f"Ø{p.LID_RAMAN['window_d']} × {p.RAMAN['window_t']}",
        "oring_tapa": f"ID {p.ORING['lid_id']} × cs {gl['cs']}, ranura prof. {gl['depth']:.3f}",
        "tornillo_tapa": "M3×%d (engrane mín. %.1f mm)" % (lid_bolt(p)["length"],
                                                           lid_bolt(p)["engagement_min"]),
    }


if __name__ == "__main__":
    from export import export

    for f in export(build(), "lid_raman"):
        print(f)
    for k, v in summary().items():
        print(f"{k}: {v}")
