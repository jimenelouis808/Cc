"""Celda de flujo wall-jet para detección amperométrica (nitrito por inyección en flujo).

    collar ┐
           │ GC (cuerpo Ø10), boca abajo; el disco en z = 0
    ───────┼──────────── cara superior (O-ring radial sobre el GC)
      RE ─▶│◀─ CE        cavidad Ø20: el chorro se abre sobre el disco y sube a la salida
       ↑ chorro (nozzle_d), a jet_h del disco, desde una boquilla cónica
    ── asiento del tubo (conector 1/4-28 de fondo plano en la cara inferior)
       patas: el tubo sale por abajo y se curva hacia un lado

- Wall-jet: el chorro choca en el centro del disco y fluye radialmente hacia afuera. La
  corriente límite (Albery y Brett) es
      i = 1.38 n F C D^(2/3) ν^(−5/12) V_f^(3/4) a^(−1/2) R^(3/4)
  con V_f el caudal, a el diámetro de la boquilla y R el radio del disco: depende del caudal
  y no de dónde queden la RE o el CE, y la distancia boquilla–disco (jet_h) sólo debe ser del
  orden de unos pocos a. `walljet_current` la calcula.
- z = 0 lo fija un collar sobre el cuerpo del GC que apoya en la cara superior. El collar se
  ajusta con la galga (`build_gauge`): el GC entra hasta el fondo de la galga y el collar se
  baja hasta su borde y se aprieta (tornillo de nylon M3 en un inserto).
- La salida está en lo alto de la cavidad: las burbujas no se quedan bajo el disco.
- RE (punta Ø4) y CE de Pt (cuerpo Ø6) entran por los lados, a la altura del disco y río
  abajo de él.

Material: PEEK o PTFE maquinado (la boquilla de 0.5 mm no sale impresa). PP impreso sirve
para probar el ajuste y para caudales con boquilla de tubo.
"""
import math
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._geom import (  # noqa: E402
    axial_port, block, cyl, port_boss, port_solid, r_inner)


def we(p=P) -> dict:
    return p.ELECTRODES["GC_options"][p.FLOW["we_option"]]


def layout(p=P) -> dict:
    f = p.FLOW
    z_nozzle = -f["jet_h"]
    z_floor = z_nozzle - f["boss_h"]
    z_seat = z_nozzle - f["nozzle_l"]
    z_bottom = z_seat - p.PORTS["thread_depth"]
    return {"z_nozzle": z_nozzle, "z_floor": z_floor, "z_seat": z_seat, "z_bottom": z_bottom,
            "z_skirt": z_bottom - f["skirt_h"],
            "collar": (f["z_top"], f["z_top"] + f["collar_h"])}


def port_axes(p=P):
    """(spec, cara exterior, r_in). La RE (centrada) usa cavity_r − 1 como las otras celdas:
    de ahí sale cuánto entra su punta. Los puertos desplazados se taladran hasta que toda su
    boca cae en la cavidad (r_inner)."""
    R = p.FLOW["cavity_d"] / 2
    return [(s, p.FLOW["body_w"] / 2, R - 1.0 if s["kind"] == "re_tip" else r_inner(p, s, R))
            for s in p.FLOW_PORTS]


def we_port(p=P) -> dict:
    return {"name": "WE", "xy": (0.0, 0.0), "kind": "radial", "rod_d": we(p)["body_d"]}


def inlet_port(p=P) -> dict:
    return {"name": "inlet", "xy": (0.0, 0.0), "kind": "fitting",
            "through_d": p.FLOW["nozzle_d"] - p.FDM["hole_offset"]}   # la boquilla, sin sobremedida


def features(p=P) -> dict:
    f, lay = p.FLOW, layout(p)
    feats = {
        "cavity": cyl(f["cavity_d"], lay["z_floor"], f["cavity_top"]),
        "port_WE": axial_port(p, we_port(p), f["z_top"], f["cavity_top"] - 0.01),
        "inlet": axial_port(p, inlet_port(p), lay["z_bottom"], lay["z_nozzle"] + 0.01),
    }
    for spec, r_out, r_in in port_axes(p):
        feats["port_" + spec["name"]] = port_solid(p, spec, r_out, r_in)
    return feats


def nozzle_boss(p=P) -> cq.Solid:
    f, lay = p.FLOW, layout(p)
    return cq.Solid.makeCone(f["boss_base_d"] / 2, f["boss_top_d"] / 2, f["boss_h"],
                             cq.Vector(0, 0, lay["z_floor"] - 0.01))


def outline(p=P) -> cq.Solid:
    f, lay = p.FLOW, layout(p)
    s = block(f["body_w"], f["body_w"], lay["z_bottom"], f["z_top"], f["corner_chamfer"])
    for b in (port_boss(p, sp, ro, ri) for sp, ro, ri in port_axes(p)):
        if b is not None:
            s = s.fuse(b)
    w, t = f["body_w"], f["skirt_wall"]
    for sx in (1, -1):         # dos paredes a lo largo de y: el tubo sale entre ellas
        s = s.fuse(block(t, w - 2 * f["corner_chamfer"], lay["z_skirt"], lay["z_bottom"] + 0.01,
                         0.0, sx * (w / 2 - t / 2), 0.0))
    return s.clean()


def build(p=P) -> cq.Workplane:
    s = outline(p)
    for k, ft in features(p).items():
        s = s.cut(ft)
    s = s.fuse(nozzle_boss(p).cut(features(p)["inlet"]))
    return cq.Workplane("XY").add(s.clean())


def build_collar(p=P) -> cq.Workplane:
    """Collar del GC: apoya en la cara superior y fija el disco en z = 0."""
    f, c, lay = p.FLOW, p.CELL, layout(p)
    z0, z1 = lay["collar"]
    s = cyl(f["collar_od"], z0, z1).cut(cyl(p.hole(we(p)["body_d"]), z0 - 1, z1 + 1))
    zc = (z0 + z1) / 2
    s = s.cut(cq.Solid.makeCylinder(c["insert_d"] / 2, f["collar_od"] / 2 + 1,
                                    cq.Vector(0, 0, zc), cq.Vector(1, 0, 0)))
    return cq.Workplane("XY").add(s.clean())


def build_gauge(p=P) -> cq.Workplane:
    """Galga: agujero ciego de profundidad z_top. GC hasta el fondo, collar hasta el borde."""
    f = p.FLOW
    t = f["gauge_floor_t"]
    s = cyl(f["collar_od"], -t, f["z_top"]).cut(cyl(p.hole(we(p)["body_d"]), 0.0, f["z_top"] + 1))
    return cq.Workplane("XY").add(s.clean().translate(cq.Vector(0, 0, 0)))


def walljet_current(p=P, conc_uM: float = 1.0, flow_ml_min: float | None = None) -> float:
    """Corriente límite wall-jet (A), Albery y Brett: 1.38 n F C D^2/3 ν^−5/12 V^3/4 a^−1/2 R^3/4.

    Unidades cgs: C en mol/cm³, D y ν en cm²/s, V en cm³/s, a y R en cm.
    """
    f = p.FLOW
    F = 96485.0
    C = conc_uM * 1e-9                          # µM → mol/cm³
    V = (flow_ml_min or f["flow_ml_min"]) / 60.0
    a = f["nozzle_d"] / 10
    R = we(p)["disk_d"] / 20
    return (1.38 * f["n_electrons"] * F * C * f["D_nitrite_cm2_s"] ** (2 / 3)
            * f["nu_cm2_s"] ** (-5 / 12) * V ** 0.75 * a ** -0.5 * R ** 0.75)


def nozzle_reynolds(p=P, flow_ml_min: float | None = None) -> float:
    f = p.FLOW
    V = (flow_ml_min or f["flow_ml_min"]) / 60.0
    a = f["nozzle_d"] / 10
    u = V / (math.pi * a ** 2 / 4)
    return u * a / f["nu_cm2_s"]


def summary(p=P) -> dict:
    f, lay = p.FLOW, layout(p)
    g = we(p)
    vol = (math.pi * (f["cavity_d"] / 2) ** 2 * (f["cavity_top"] - lay["z_floor"])
           - math.pi * (g["body_d"] / 2) ** 2 * f["cavity_top"]) / 1000
    return {
        "WE": f"{g['ref']}: disco Ø{g['disk_d']:g} en z = 0, boca abajo",
        "chorro": f"Ø{f['nozzle_d']} a {f['jet_h']} mm del disco (h/a = "
                  f"{f['jet_h'] / f['nozzle_d']:.1f}; R/a = {g['disk_d'] / 2 / f['nozzle_d']:.1f})",
        "volumen_de_la_cavidad_mL": round(vol, 2),
        "Re_en_la_boquilla": round(nozzle_reynolds(p), 0),
        "corriente_por_µM_de_nitrito_nA": round(walljet_current(p) * 1e9, 2),
        "collar": f"a {f['z_top']} mm del disco (galga de {f['z_top']} mm)",
    }


if __name__ == "__main__":
    from export import export

    for name, part in (("flow_body", build()), ("flow_collar", build_collar()),
                       ("flow_gauge", build_gauge())):
        for fn in export(part, name):
            print(fn)
    for k, v in summary().items():
        print(f"{k}: {v}")
