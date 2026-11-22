"""Espesores mínimos de pared entre elementos restados de un mismo cuerpo.

Cada módulo expone `features(p) -> {nombre: sólido}`. Aquí se mide la distancia
mínima (BRepExtrema) entre pares de elementos y entre cada elemento y el exterior.
Mojado (cavidad de electrolito): CELL["min_wall"]. Seco: CELL["min_wall_dry"].
"""
import os
import sys
from dataclasses import dataclass

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules._geom import body, compound, cyl  # noqa: E402


@dataclass
class Result:
    name: str
    value: float
    required: float

    @property
    def ok(self) -> bool:
        return self.value >= self.required - 1e-6

    def __str__(self):
        flag = "OK " if self.ok else "FALLA"
        return f"[{flag}] {self.name:45s} {self.value:6.2f} mm (mín {self.required:.2f})"


def lateral_outside(p, z0: float, z1: float) -> cq.Shape:
    """Todo lo que queda fuera de la huella, entre z0 y z1 (caras laterales)."""
    w = p.CELL["body_w"] + 40
    big = cq.Solid.makeBox(w, w, z1 - z0, cq.Vector(-w / 2, -w / 2, z0))
    return big.cut(body(p, z0 - 1, z1 + 1))


def slab(p, z: float, above: bool) -> cq.Shape:
    """Semiespacio (en la práctica, una losa gruesa) sobre o bajo el plano z."""
    w = p.CELL["body_w"] + 40
    z0 = z if above else z - 20
    return cq.Solid.makeBox(w, w, 20, cq.Vector(-w / 2, -w / 2, z0))


def dist(a: cq.Shape, b: cq.Shape) -> float:
    return a.distance(b)


def ring_checks(p=P) -> list[Result]:
    from modules.chamber_ring import features

    f = features(p)
    h = p.CHAMBER_RING["h"]
    wet, dry = p.CELL["min_wall"], p.CELL["min_wall_dry"]
    cavity = f["well"].fuse(f["bore"])
    ports = {k: v for k, v in f.items() if k.startswith("port_")}
    out = lateral_outside(p, 0, h)
    res = [
        Result("cavidad ↔ exterior lateral", dist(cavity, out), wet),
        Result("cavidad ↔ tornillos", dist(cavity, f["bolts"]), wet),
        Result("cavidad ↔ pines", dist(cavity, f["pins"]), wet),
        Result("cavidad ↔ insertos de tapa", dist(cavity, f["lid_inserts"]), wet),
        Result("escalón O-ring ↔ pines", dist(f["oring_step"], f["pins"]), dry),
        Result("tornillos ↔ exterior lateral", dist(f["bolts"], out), dry),
        Result("pines ↔ exterior lateral", dist(f["pins"], out), dry),
        Result("insertos ↔ exterior lateral", dist(f["lid_inserts"], out), dry),
        Result("insertos ↔ tornillos", dist(f["lid_inserts"], f["bolts"]), dry),
    ]
    names = sorted(ports)
    for i, a in enumerate(names):
        pa = ports[a]
        res += [
            Result(f"{a} ↔ cara superior", dist(pa, slab(p, h, True)), dry),
            Result(f"{a} ↔ cara inferior", dist(pa, slab(p, 0, False)), dry),
            Result(f"{a} ↔ tornillos", dist(pa, f["bolts"]), dry),
            Result(f"{a} ↔ pines", dist(pa, f["pins"]), dry),
            Result(f"{a} ↔ insertos", dist(pa, f["lid_inserts"]), dry),
        ]
        for b in names[i + 1:]:
            res.append(Result(f"{a} ↔ {b}", dist(pa, ports[b]), dry))
    # el puerto debe desembocar sobre el piso, no en él
    from modules.chamber_ring import floor_t
    for a in names:
        spec = next(s for s in p.RING_PORTS if "port_" + s["name"] == a)
        bottom = spec["z"] - _port_r(p, spec)
        res.append(Result(f"{a}: borde inferior sobre el piso del pozo", bottom - floor_t(p), 0.0))
    return res


def _port_r(p, spec) -> float:
    from modules._geom import port_bore_r
    return port_bore_r(p, spec)


def base_checks(p=P) -> list[Result]:
    from modules.base_flat import features

    f = features(p)
    t = p.BASE_FLAT["t"]
    dry = p.CELL["min_wall_dry"]
    out = lateral_outside(p, -t, 0)
    return [
        Result("alojamiento ↔ exterior lateral", dist(f["pocket"], out), dry),
        Result("alojamiento ↔ pines", dist(f["pocket"], f["pins"]), dry),
        Result("alojamiento ↔ tornillos", dist(f["pocket"], f["bolts"]), dry),
        Result("tuercas ↔ cara superior", dist(f["nuts"], slab(p, 0, True)), dry),
        Result("tuercas ↔ exterior lateral", dist(f["nuts"], out), dry),
        Result("tuercas ↔ pines", dist(f["nuts"], f["pins"]), dry),
        Result("pines ↔ cara inferior", dist(f["pins"], slab(p, -t, False)), dry),
        Result("ranura de contacto ↔ cara inferior", dist(f["contact"], slab(p, -t, False)), dry),
        Result("ranura de contacto ↔ tornillos", dist(f["contact"], f["bolts"]), dry),
    ]


def seal_checks(p=P) -> list[Result]:
    """El O-ring de la muestra debe apoyar entero sobre la muestra, lejos del borde."""
    from modules._geom import sample_seal

    s = sample_seal(p)
    margin = (p.WE_FLAT["sample_w"] - s["step_d"]) / 2
    return [Result("escalón O-ring ↔ borde de la muestra", margin, p.CELL["min_wall_dry"])]


def assembly_checks(p=P) -> list[Result]:
    """Base y anillo no se interpenetran."""
    from modules.base_flat import build as base
    from modules.chamber_ring import build as ring

    v = base(p).val().intersect(ring(p).val()).Volume()
    return [Result("interferencia base/anillo (−volumen)", -v, -1e-6)]


def _chamber(p, name):
    """(sólido, features, obstáculos internos) de la cámara sobre la que va la tapa."""
    if name == "ring":
        from modules import chamber_ring as m
        return m.build(p).val(), m.features(p), {}
    from modules import chamber_clamp as m
    return m.build(p).val(), m.features(p), {"pinza": m.clamp(p)}


def lid_raman_checks(p=P, chamber: str = "ring") -> list[Result]:
    from modules import lid_raman as L

    c, lp = p.CELL, p.LID_RAMAN
    dry, wet = c["min_wall_dry"], c["min_wall"]
    f = L.features(p)
    zp = L.z_plate(p)
    g = c["thin_gap_raman"]
    res = []

    # pared del émbolo: hueco óptico ↔ electrolito (todo lo que rodea al émbolo bajo la placa)
    blank = L.plunger_blank(p)
    z_wt = g + p.RAMAN["window_t"]   # bajo z_wt está la ventana (rebaje: check aparte)
    around = cyl(p.CHAMBER_RING["well_d"] + 10, z_wt, zp).cut(blank)
    hollow = f["interior"].intersect(cyl(200, z_wt, zp))
    res.append(Result("émbolo: pared hueco ↔ electrolito", dist(hollow, around), wet))
    # rebaje de la ventana ↔ exterior de la nariz, y apoyo de la ventana
    res.append(Result("rebaje de ventana ↔ exterior de la nariz",
                      (L.nose_d(p) - p.hole(lp["window_d"])) / 2, lp["window_recess_wall_min"]))
    from modules._optics import light_radius
    aperture = light_radius(p, g + p.RAMAN["window_t"]) + p.RAMAN["optical_margin"]
    res.append(Result("apoyo de la ventana fuera de la apertura",
                      lp["window_d"] / 2 - aperture, lp["window_land_min"]))

    # holguras contra la cámara en todo el recorrido de lainas
    ring, rf, obstacles = _chamber(p, chamber)
    # bocas reales: la parte del sólido del puerto que cae dentro del pozo no existe
    ports = compound(v for k, v in rf.items() if k.startswith("port_")).cut(rf["well"])
    plate_cut = cq.Solid.makeBox(200, 200, 100, cq.Vector(-100, -100, zp - 0.05))
    for s in L.shim_range(p) + (lp["shim_nominal"],):
        lid = L.build(p, shim=s).val()
        plunger = lid.cut(plate_cut.translate(cq.Vector(0, 0, s - lp["shim_nominal"])))
        tag = f"laina {s:.2f} (gap {L.gap_for_shim(p, s):.2f})"
        res.append(Result(f"{tag}: émbolo ↔ cámara",
                          dist(plunger, ring), min(lp["nose_clearance"], lp["well_clearance"])))
        res.append(Result(f"{tag}: émbolo ↔ bocas de puertos", dist(plunger, ports),
                          lp["port_clearance"]))
        for oname, obst in obstacles.items():
            res.append(Result(f"{tag}: émbolo ↔ {oname}", dist(plunger, obst),
                              p.WE_CLAMP["clip_clearance"]))
        res.append(Result(f"{tag}: gap", L.gap_for_shim(p, s), 0.05))
        v = lid.intersect(ring).Volume()
        res.append(Result(f"{tag}: interferencia tapa/cámara (−volumen)", -v, -1e-6))

    # O-ring de la tapa: compresión en todo el recorrido y llenado de la ranura
    gl = L.lid_gland(p)
    lo, hi = lp["oring_compression"]
    for s in L.shim_range(p):
        comp = (gl["cs"] - gl["depth"] - s) / gl["cs"]
        res.append(Result(f"O-ring tapa, laina {s:.2f}: compresión ≥ {lo:.0%}", comp, lo))
        res.append(Result(f"O-ring tapa, laina {s:.2f}: compresión ≤ {hi:.0%} (−)", -comp, -hi))
    import math
    fill = math.pi * gl["cs"] ** 2 / 4 / (gl["width"] * gl["depth"])
    res.append(Result("O-ring tapa: llenado de ranura (−)", -fill, -lp["gland_fill_max"]))
    res.append(Result("O-ring tapa apoya sobre el anillo, fuera del pozo",
                      (p.ORING["lid_id"] - p.CHAMBER_RING["well_d"]) / 2, dry))

    # placa
    out = lateral_outside(p, zp, zp + lp["plate_t"])
    res += [
        Result("placa: ranura ↔ tornillos", dist(f["gland"], f["bolts"]), dry),
        Result("placa: ranura ↔ hueco óptico", dist(f["gland"], f["interior"]), dry),
        Result("placa: tornillos ↔ hueco óptico", dist(f["bolts"], f["interior"]), dry),
        Result("placa: tornillos ↔ exterior lateral", dist(f["bolts"], out), dry),
        Result("tornillo de tapa: engrane mínimo en el inserto",
               L.lid_bolt(p)["engagement_min"], lp["bolt_engagement_min"]),
    ]
    return res


def _ports_checks(p, f, specs, z_top, z_bottom, cavity_floor) -> list[Result]:
    """Puertos: entre sí, contra caras, insertos y tornillos; boca sobre el piso;
    paso entre conectores de una misma cara."""
    import math

    dry = p.CELL["min_wall_dry"]
    ports = {k: v for k, v in f.items() if k.startswith("port_")}
    others = {k: f[k] for k in ("bolts", "pins", "lid_inserts", "screws") if k in f}
    names = sorted(ports)
    res = []
    for i, a in enumerate(names):
        pa = ports[a]
        res += [Result(f"{a} ↔ cara superior", dist(pa, slab(p, z_top, True)), dry),
                Result(f"{a} ↔ cara inferior", dist(pa, slab(p, z_bottom, False)), dry)]
        res += [Result(f"{a} ↔ {k}", dist(pa, v), dry) for k, v in others.items()]
        res += [Result(f"{a} ↔ {b}", dist(pa, ports[b]), dry) for b in names[i + 1:]]
    for s in specs:
        bottom = s["z"] - _port_r(p, s)
        floor = cavity_floor.get(s["name"], cavity_floor["*"]) if isinstance(
            cavity_floor, dict) else cavity_floor
        res.append(Result(f"port_{s['name']}: boca sobre el piso", bottom - floor, 0.0))
    fittings = [s for s in specs if s["kind"] == "fitting"]
    for i, a in enumerate(fittings):
        for b in fittings[i + 1:]:
            if a["face"] == b["face"]:
                d = math.hypot(a["offset"] - b["offset"], a["z"] - b["z"])
                res.append(Result(f"conectores {a['name']}/{b['name']}: paso",
                                  d, p.PORTS["fitting_pitch_min"]))
    return res


def _clamp_port_checks(p, ports) -> list[Result]:
    """La pinza entra por su puerto (si se monta así) y su cuerpo es el que sella el puerto."""
    c = p.WE_CLAMP
    we = next(s for s in ports if s["name"] == "WE")
    res = [Result("cuerpo de la pinza = Ø del puerto WE (−diferencia)",
                  -abs(we.get("rod_d", 0.0) - c["rod_d"]), 0.0)]
    if c["mount"] == "through_port":
        # sin el tornillo, la cabeza es la propia varilla: debe pasar por el agujero
        res.append(Result("cabeza (sin tornillo) pasa por el puerto WE (holgura)",
                          p.hole(we["rod_d"]) - c["head_w"], 0.0))
    res.append(Result("muestra cabe en la pinza (holgura de espesor)",
                      c["max_sample_t"] - c["sample_t"], 0.0))
    return res


def clamp_checks(p=P) -> list[Result]:
    """Cámara para pinza de Pt: paredes, puertos, pinza, muestra y pedestal."""

    from modules import chamber_clamp as m

    f = m.features(p)
    lay = m.layout(p)
    h = p.CHAMBER_RING["h"]
    c = p.WE_CLAMP
    wet, dry = p.CELL["min_wall"], p.CELL["min_wall_dry"]
    out = lateral_outside(p, lay["z_bottom"], h)
    res = [
        Result("cámara: pozo ↔ exterior lateral", dist(f["well"], out), wet),
        Result("cámara: pozo ↔ cara inferior",
               dist(f["well"], slab(p, lay["z_bottom"], False)), wet),
        Result("cámara: pozo ↔ insertos de tapa", dist(f["well"], f["lid_inserts"]), wet),
        Result("cámara: insertos ↔ exterior lateral", dist(f["lid_inserts"], out), dry),
    ]
    res += _ports_checks(p, f, p.CLAMP_PORTS, h, lay["z_bottom"], lay["z_floor"])

    res += _clamp_port_checks(p, p.CLAMP_PORTS)
    # pinza y muestra dentro del pozo, sin tocar la cámara
    chamber = m.build(p).val()
    holder = m.clamp(p).cut(lateral_outside(p, lay["z_bottom"] - 1, h + 1))
    port_we = f["port_WE"]
    res.append(Result("pinza ↔ cámara (fuera del puerto)",
                      dist(holder.cut(port_we), chamber), 0.3))
    smp = m.sample(p)
    ped = m.pedestal(p)
    walls_only = chamber.cut(ped) if ped is not None else chamber
    res.append(Result("muestra ↔ paredes y piso de la cámara", dist(smp, walls_only), 0.5))
    if ped is not None:
        res.append(Result("pedestal ↔ pinza", dist(ped, m.clamp(p)), 0.5))
        res.append(Result("pedestal bajo la muestra (margen al borde)",
                          min(lay["x1"], c["sample_w"] / 2) - c["pedestal_d"] / 2, 0.5))
    return res


def xrd_checks(p=P) -> list[Result]:
    """Celda XRD: haz libre, paredes, puertos, pinza, lengüeta, tapa y marco."""
    from checks.xrd_clearance import invasion
    from modules import chamber_xrd as m
    from modules import lid_xrd

    f = m.features(p)
    lay = m.layout(p)
    k, s = p.CHAMBER_XRD, p.WE_XRD
    wet, dry = p.CELL["min_wall"], p.CELL["min_wall_dry"]
    body = m.build(p)
    wp = cq.Workplane
    res = []
    # 1. nada tapa el haz, a ningún 2θ ≥ two_theta_min
    for name, part in (("cuerpo", body), ("marco", lid_xrd.build(p)),
                       ("tapa del bolsillo", lid_xrd.build_cover(p)),
                       ("pinza", wp().add(m.clamp(p))), ("muestra", wp().add(m.sample(p)))):
        res.append(Result(f"haz: {name} fuera de la zona prohibida (−mm³)",
                          -invasion(part, p), -1e-3))
    # 2. paredes
    sides = _box_outside(p, lay)
    top, bottom = slab(p, 0.0, True), slab(p, lay["z_bottom"], False)
    res += [
        Result("reservorio ↔ exterior lateral", dist(f["reservoir"], sides), wet),
        Result("reservorio ↔ cara inferior", dist(f["reservoir"], bottom), wet),
        Result("reservorio ↔ cara superior (techo)", dist(f["reservoir"], top), wet),
        Result("reservorio ↔ ranura del cordón", dist(f["reservoir"], f["groove"]), wet),
        Result("reservorio ↔ insertos del marco", dist(f["reservoir"], f["screws"]), wet),
        Result("bolsillo de la pinza ↔ exterior lateral", dist(f["clip_pocket"], sides), wet),
        Result("bolsillo de la pinza ↔ cara inferior", dist(f["clip_pocket"], bottom), wet),
        Result("bolsillo de la pinza ↔ ranura del cordón",
               dist(f["clip_pocket"], f["groove"]), dry),
        Result("bolsillo de la pinza ↔ insertos", dist(f["clip_pocket"], f["screws"]), wet),
        Result("alojamiento de muestra ↔ ranura del cordón",
               dist(f["sample_pocket"], f["groove"]), dry),
        Result("insertos ↔ exterior lateral", dist(f["screws"], sides), dry),
        Result("insertos ↔ ranura del cordón", dist(f["screws"], f["groove"]), dry),
    ]
    # el WE desemboca en el bolsillo de la pinza; los demás, en el reservorio
    res += _ports_checks(p, f, p.XRD_PORTS, 0.0, lay["z_bottom"],
                         {"WE": lay["pocket_floor"], "*": lay["z_rf"]})
    # 3. pinza, lengüeta y tapa
    port_we = f["port_WE"]
    res.append(Result("pinza ↔ cuerpo (fuera del puerto)",
                      dist(m.clamp(p).cut(port_we).cut(sides), body.val()), 0.3))
    path = m.tab_path(p)
    (y1, z1), (y2, z2) = path[1], path[2]
    y_cov = lay["sy"] + k["cover_open"]
    z_at = z1 + (z2 - z1) * (y_cov - y1) / (y2 - y1)
    res.append(Result("lengüeta bajo la tapa del bolsillo (holgura)", -k["cover_t"] - z_at, 0.3))
    from modules._geom import clamp_extent
    head_top = lay["z_clip"] + clamp_extent(p.WE_CLAMP, p.WE_XRD.get("clamp_screw", "down"))[0]
    res.append(Result("cabeza de la pinza bajo la tapa del bolsillo (holgura)",
                      -k["cover_t"] - head_top, 0.3))
    res += _clamp_port_checks(p, p.XRD_PORTS)
    from checks.xrd_clearance import min_two_theta_for_footprint
    x = p.XRD
    tt = min_two_theta_for_footprint(s["sample_x"], x["goniometer_radius"], x["divergence_deg"])
    res.append(Result("huella dentro de la muestra a 2θ_min (2θ_min − 2θ que cabe, °)",
                      x["two_theta_min"] - tt, 0.0))
    res.append(Result("muestra cubre el corredor del haz en y (mm)",
                      s["sample_y"] / 2 - m.corridor_half(p) + p.XRD["corridor_margin"], 0.0))
    return res


def chladni_checks(p=P) -> list[Result]:
    """Celda Chladni: empotramiento, sello, piezo, lengüeta, CE, RE, tapa y puertos."""
    from modules import chladni as m

    lay, c = m.layout(p), p.CHLADNI
    wet, dry = p.CELL["min_wall"], p.CELL["min_wall_dry"]
    wall, bottom, lid = m.build_wall(p).val(), m.build_bottom(p).val(), m.build_lid(p).val()
    plate, piezo, ce = m.plate(p), m.piezo(p), m.ce_plate(p)
    wf = m.wall_features(p)
    big = cq.Solid.makeBox(400, 400, 100, cq.Vector(-200, -200, -50))
    outside = big.cut(cq.Workplane("XY").add(m.box(-lay["outer"], lay["outer"], -lay["outer"],
                                                   lay["outer"], -50, 50))
                      .edges("|Z").chamfer(p.CELL["corner_chamfer"]).val())
    bolts = m._bolt_holes(p, lay["z_bottom"] - 1, lay["z_wall_top"] + 1)
    # a más de relief_gap dentro de la línea, la pared debe quedar ≥ relief_gap sobre la
    # placa (sobre la línea misma la pared sí apoya: es el empotramiento)
    e = lay["a"] - c["relief_gap"]
    inner = plate.intersect(m.box(-e, e, -e, e, -5, 5))
    res = [
        # empotramiento y sello
        Result("pared ↔ placa dentro del empotramiento (alivio)", dist(wall, inner),
               c["relief_gap"] - 1e-3),
        Result("cordón dentro de la línea de empotramiento",
               lay["a"] - lay["seal_out"], c["seal_inset"] - 1e-6),
        Result("marcos sin contacto entre sí (toda la fuerza a la placa)", dist(wall, bottom),
               c["plate_preload"] - 1e-3),
        Result("interferencia placa/marco inferior (−mm³)",
               -plate.intersect(bottom).Volume(), -1e-3),
        Result("interferencia placa/pared (−mm³)", -plate.intersect(wall).Volume(), -1e-3),
        Result("placa ↔ tornillos", dist(plate, bolts), 1.0),
        # piezo
        Result("piezo ↔ marco inferior", dist(piezo, bottom), c["piezo_margin"]),
        Result("lengüeta afuera: largo libre para la pinza",
               c["tab_out"] - max(cl["grip_l"] for cl in p.CLAMPS.values()), 2.0),
        # cámara
        Result("cámara ↔ exterior lateral", dist(wf["cavity"], outside), wet),
        Result("cámara ↔ tornillos", dist(wf["cavity"].fuse(wf["ce_seat"]), bolts), wet),
        Result("asiento del CE ↔ insertos de la tapa", dist(wf["ce_seat"], wf["lid_inserts"]), wet),
        Result("ranura del cordón ↔ tornillos", dist(wf["groove"], bolts), dry),
        Result("insertos de la tapa ↔ tornillos", dist(wf["lid_inserts"], bolts), dry),
        Result("puerto CE ↔ tornillos", dist(wf["port_CE"], bolts), dry),
        Result("puerto CE ↔ insertos de la tapa", dist(wf["port_CE"], wf["lid_inserts"]), dry),
        Result("puerto CE ↔ cara superior",
               dist(wf["port_CE"], slab(p, lay["z_wall_top"], True)), dry),
        # CE y RE
        Result("CE bajo el nivel del electrolito",
               c["electrolyte_h"] - (lay["z_ce"] + c["ce_t"]), 0.5),
        Result("interferencia CE/pared (−mm³)", -ce.intersect(wall).Volume(), -1e-3),
    ]
    rx, ry = m.re_xy(p)
    from modules.chladni import re_seat_z
    re_body = cyl(p.ELECTRODES["RE_tip_d"], c["luggin_h"], re_seat_z(p), rx, ry).fuse(
        cyl(p.ELECTRODES["RE_body_d"], re_seat_z(p), re_seat_z(p) + 40, rx, ry))
    res += [
        Result("RE ↔ CE (pasa por el recorte)", dist(re_body, ce), 0.5),
        Result("RE ↔ pared de la cámara", dist(re_body, wall), 0.5),
        Result("punta de la RE sobre la placa", c["luggin_h"], 0.5),
        Result("hombro de la RE sobre la tapa (no entra a la celda)",
               re_seat_z(p) - lay["z_wall_top"] - c["lid_t"], 0.0),
    ]
    lf = m.lid_features(p)
    names = [k for k in lf if k.startswith("port_")] + ["re"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            res.append(Result(f"tapa: {a} ↔ {b}", dist(lf[a], lf[b]), dry))
        res.append(Result(f"tapa: {a} ↔ tornillos",
                          dist(lf[a], lf["screws"].fuse(lf["bolt_heads"])), dry))
        res.append(Result(f"tapa: {a} ↔ pared (cae en la cámara)",
                          dist(lf[a].intersect(slab(p, lay["z_wall_top"], False)), wall), 0.3))
    res.append(Result("interferencia tapa/pared (−mm³)", -lid.intersect(wall).Volume(), -1e-3))
    return res


def epd_checks(p=P) -> list[Result]:
    """Celda EPD/DEP: máscara, electrodos, canal, apilado, conectores, pads y materiales."""
    from modules import epd as m

    e, lay, pat = p.EPD, m.layout(p), m.ide_rects(p)
    dry = p.CELL["min_wall_dry"]
    res = []
    # 1. máscara de sombra
    opens = pat["A"] + pat["B"]
    min_open = min(min(x1 - x0, y1 - y0) for x0, x1, y0, y1 in opens)
    res.append(Result("máscara: abertura mínima", min_open, e["mask_min_feature"]))
    webs = list(e["gaps"]) + [max(e["gaps"])]          # entre dedos y en las puntas
    res.append(Result("máscara: puente mínimo", min(webs), e["mask_min_web"]))
    web_len = e["channel_w"] + 2 * e["finger_overlap"]
    res.append(Result("máscara: esbeltez del puente más fino (−)",
                      -web_len / min(e["gaps"]), -e["mask_max_aspect"]))
    sx, sy, _ = e["slide"]
    b = e["mask_border"]
    sheet = m.box(-sx / 2 - b, sx / 2 + b, -sy / 2 - b, sy / 2 + b, 0, 1)
    for x0, x1, y0, y1 in opens:
        sheet = sheet.cut(m.box(x0, x1, y0, y1, -1, 2))
    res.append(Result("máscara: la lámina es una sola pieza (−piezas+1)",
                      -(len(sheet.Solids()) - 1), 0))
    for name in ("A", "B"):
        merged = cq.Workplane("XY")
        for x0, x1, y0, y1 in pat[name]:
            merged = merged.union(cq.Workplane("XY").add(m.box(x0, x1, y0, y1, 0, 1)))
        res.append(Result(f"peine {name}: un solo electrodo conectado (−piezas+1)",
                          -(len(merged.val().Solids()) - 1), 0))
    # 2. electrodos y canal
    W = e["channel_w"]
    channel = m.slot(e["channel_l"], W, 0, lay["h_channel"])
    buses = cq.Compound.makeCompound([m.box(*r[:4], 0, 0.02) for r in (pat["A"][-2], pat["B"][-2])])
    res.append(Result("buses fuera del canal (bajo la junta)", dist(buses, channel), 0.3))
    fingers = pat["A"][:-2] + pat["B"][:-2]
    fingers_ok = all(y0 <= -W / 2 and y1 >= W / 2 for x0, x1, y0, y1 in fingers)
    res.append(Result("cada dedo cruza el canal entero", 1.0 if fingers_ok else 0.0, 1.0))
    cx, cy, _ = e["cover"]
    seal = min(cy / 2 - W / 2, cx / 2 - e["channel_l"] / 2)
    res.append(Result("junta entre canal y borde exterior", seal, e["gasket_seal_min"]))
    ide0, ide1 = pat["ide_x"]
    res.append(Result("electrodos dentro del canal en x (margen a los agujeros)",
                      lay["hole_x"] - e["cover_hole_d"] / 2 - max(abs(ide0), abs(ide1)), 1.0))
    # 3. apilado
    frame, base = m.build_frame(p).val(), m.build_base(p).val()
    chip, cov, gk = m.chip(p), m.cover(p), m.gasket(p)
    for a_name, a, b_name, b in (("marco", frame, "vidrio superior", cov),
                                 ("marco", frame, "chip", chip),
                                 ("base", base, "chip", chip), ("marco", frame, "base", base),
                                 ("junta", gk, "vidrio superior", cov)):
        res.append(Result(f"interferencia {a_name}/{b_name} (−mm³)",
                          -a.intersect(b).Volume(), -1e-3))
    res.append(Result("marco ↔ chip (no lo toca)", dist(frame, chip),
                      e["frame_chip_clearance"] - 1e-3))
    # 4. marco: ventana, conectores, tornillos
    ff = m.frame_features(p)
    ports = [k for k in ff if k.startswith("port_")]
    for k in ports:
        res += [Result(f"{k} ↔ ventana", dist(ff[k], ff["window"]), dry),
                Result(f"{k} ↔ tornillos", dist(ff[k], ff["screws"]), dry)]
    res.append(Result("rosca del conector cabe en el marco",
                      e["frame_t"] - p.PORTS["thread_depth"], 1.0))
    res.append(Result("ventana ↔ tornillos", dist(ff["window"], ff["screws"]), dry))
    o_id, o_cs = e["port_oring"]
    res.append(Result("O-ring del conector cubre el agujero del vidrio",
                      o_id / 2 - e["cover_hole_d"] / 2, 0.3))
    res.append(Result("pads afuera del marco (largo para la pinza)",
                      sx / 2 - max(lay["frame"][0], pat["pad_x0"]), e["clip_l"]))
    # 5. materiales contra los disolventes
    bad = [(s, mat) for s in e["solvents"] for mat in e["wetted_materials"]
           if mat in p.SOLVENT_ATTACKS.get(s, ())]
    res.append(Result("materiales mojados compatibles con los disolventes (−incompatibles)",
                      -len(bad), 0))
    return res


def _outside_of(outline: cq.Shape) -> cq.Shape:
    """Todo lo que rodea a un contorno (para medir paredes contra caras de cualquier forma)."""
    bb = outline.BoundingBox()
    big = cq.Solid.makeBox(bb.xlen + 60, bb.ylen + 60, bb.zlen + 60,
                           cq.Vector(bb.xmin - 30, bb.ymin - 30, bb.zmin - 30))
    return big.cut(outline)


def _vol(a: cq.Shape, b: cq.Shape) -> float:
    if a.distance(b) > 1e-6:
        return 0.0
    return a.intersect(b).Volume()


def tcell_checks(p=P) -> list[Result]:
    """Celda T: paredes, puertos, sellos de los colectores, resorte y presión sobre la pila."""
    import math

    from modules import tcell as m
    from modules._geom import block

    t, lay = p.TCELL, m.layout(p)
    wet, dry = p.CELL["min_wall"], p.CELL["min_wall_dry"]
    f = m.features(p)
    outline = block(t["body_w"], t["body_w"], t["z0"], t["z1"], t["corner_chamfer"])
    for b in m.bosses(p):
        outline = outline.fuse(b)
    out = _outside_of(outline)
    res = [
        Result("celda T: agujero ↔ exterior lateral", dist(f["bore"], out.cut(
            cyl(60, t["z0"] - 40, t["z0"])).cut(cyl(60, t["z1"], t["z1"] + 40))), wet),
        Result("celda T: agujero ↔ insertos", dist(f["bore"], f["lid_inserts"]), wet),
        Result("celda T: ranuras de los colectores ↔ insertos",
               dist(f["rod_glands"], f["lid_inserts"]), dry),
        Result("celda T: ranuras de los colectores ↔ exterior lateral",
               dist(f["rod_glands"], out), dry),
    ]
    specs = m.port_specs(p)
    res += _ports_checks(p, f, specs, t["z1"], t["z0"], t["z0"])
    for s in specs:
        res.append(Result(f"celda T: port_{s['name']} ↔ ranuras de los colectores",
                          dist(f["port_" + s["name"]], f["rod_glands"]), dry))
    # RE: fuera del agujero, en el plano medio del separador
    res.append(Result("celda T: punta de la RE fuera del agujero (no toca la pila)",
                      t["re_recess"], 0.2))
    res.append(Result("celda T: la boca de la RE cubre el separador (radio − sep/2)",
                      p.hole(p.ELECTRODES["RE_tip_d"]) / 2 - t["sep_t"] / 2, 0.5))
    # colectores: cubren sus O-rings; los discos caben bajo el colector
    g = p.oring_gland(p.ORING["re_cs"], "radial")
    a, mg = p.ORING["re_groove_from_face"], t["rod_seal_margin"]
    res += [
        Result("colector superior cubre su O-ring (arriba)", lay["rod_hi"][1] - (t["z1"] - a), mg),
        Result("colector superior cubre su O-ring (abajo)",
               (t["z1"] - a - g["width"]) - lay["rod_hi"][0], mg),
        Result("colector inferior cubre su O-ring",
               lay["rod_lo"][1] - (t["z0"] + a + g["width"]), mg),
        Result("disco de electrodo ≤ colector", t["rod_d"] - t["electrode_d"], 0.0),
    ]
    # resorte y presión
    spring_l = lay["spring_l"]
    area = math.pi * (t["electrode_d"] / 2) ** 2
    pres = t["spring_k"] * t["spring_compression"] / area
    lo, hi = t["pressure_range"]
    res += [
        Result("resorte: no llega a bloque (largo − largo sólido)",
               spring_l - t["spring_solid_l"], 1.0),
        Result("resorte: cabe en el agujero", p.hole(t["rod_d"]) - t["spring_od"], 0.5),
        Result("resorte: el terminal pasa por dentro",
               t["spring_od"] - 2 * t["spring_wire_d"] - t["pin_d"], 0.5),
        Result(f"presión sobre la pila ≥ {lo} MPa", pres, lo),
        Result(f"presión sobre la pila ≤ {hi} MPa (−)", -pres, -hi),
        Result("terminal inferior no toca la mesa (patas − terminal)",
               t["leg_h"] - t["pin_out"], 2.0),
    ]
    # piezas: sin interferencias; patas lejos de las cabezas de los tornillos
    body, cb, ct = m.build(p).val(), m.cap_bottom(p).val(), m.cap_top(p).val()
    for n, s in m.collectors(p):
        for on, o in (("cuerpo", body), ("tapa inferior", cb), ("tapa superior", ct)):
            res.append(Result(f"celda T: {n} ↔ {on} (−mm³)", -_vol(s, o), -1e-3))
    res.append(Result("celda T: tapas ↔ cuerpo (−mm³)", -(_vol(cb, body) + _vol(ct, body)), -1e-3))
    heads = compound(cyl(p.CELL["bolt_head_d"] + 1, lay["legs"][0], lay["cap_lo"][0], x, y)
                     for x, y in m.screw_xy(p))
    res.append(Result("celda T: patas ↔ cabezas de tornillo (acceso)", dist(cb, heads), 0.0))
    return res


def hcell_checks(p=P) -> list[Result]:
    """Celda H: paredes de la media celda, sello de la membrana, tapas, niveles y agitación."""
    import math

    from modules import hcell as m
    from modules._geom import block

    h, c, lay = p.HCELL, p.CELL, m.layout(p)
    wet, dry = c["min_wall"], c["min_wall_dry"]
    f = m.half_features(p)
    out = _outside_of(m.half_outline(p))
    res = [
        Result("celda H: cámara ↔ exterior lateral", dist(f["chamber"], out.cut(
            cyl(200, h["z_top"], h["z_top"] + 50))), wet),
        Result("celda H: cámara ↔ cara inferior",
               dist(f["chamber"], slab(p, lay["z_bottom"], False)), wet),
        Result("celda H: ranura de la membrana ↔ ventana", dist(f["mem_gland"], f["window"]), dry),
        Result("celda H: ranura de la membrana ↔ tornillos", dist(f["mem_gland"], f["bolts"]), dry),
        Result("celda H: tornillos de brida ↔ cámara", dist(f["bolts"], f["chamber"]), wet),
        # el agujero sale por las dos caras de la brida: se mide contra lo de arriba, abajo y
        # los lados de la brida
        Result("celda H: tornillos de brida ↔ exterior", dist(f["bolts"], out.cut(
            cq.Solid.makeBox(h["flange_t"] + 20, 200, 200,
                             cq.Vector(lay["face"] - h["flange_t"] - 10, -100, -100)).cut(
                block(h["flange_t"] + 20, h["flange_w"] + 40, lay["z_bottom"], h["z_top"],
                      0.0, lay["face"] - h["flange_t"] / 2, 0.0).cut(
                    block(h["flange_t"] + 20, h["flange_w"], lay["z_bottom"] - 1,
                          h["z_top"] + 1, 0.0, lay["face"] - h["flange_t"] / 2, 0.0))))), dry),
        Result("celda H: insertos de tapa ↔ cámara", dist(f["lid_inserts"], f["chamber"]), wet),
        Result("celda H: insertos de tapa ↔ exterior lateral",
               dist(f["lid_inserts"], out.cut(cyl(200, h["z_top"], h["z_top"] + 50))), dry),
    ]
    # cabeza y tuerca de los tornillos de brida fuera del cuerpo
    nut_r = c["nut_af"] / math.sqrt(3)
    y_min = min(abs(y) for y, _ in m.bolt_yz(p))
    res.append(Result("celda H: cabeza/tuerca de brida fuera del cuerpo",
                      y_min - max(c["bolt_head_d"] / 2, nut_r) - h["body_w"] / 2, 0.5))
    # membrana
    gl = m.face_gland(p)
    res.append(Result("membrana cubre los O-rings", h["membrane_d"] / 2 - gl["r_out"], 1.5))
    near = min(math.hypot(y, z) for y, z in m.bolt_yz(p))
    res.append(Result("membrana dentro de los tornillos",
                      near - p.hole(c["bolt_clearance_d"]) / 2 - h["membrane_d"] / 2, 1.0))
    comp = (h["mem_oring_cs"] - gl["depth"]) / h["mem_oring_cs"]
    res.append(Result("O-rings de la membrana: compresión ≥ 15 %", comp, 0.15))
    res.append(Result("O-rings de la membrana: compresión ≤ 35 % (−)", -comp, -0.35))
    res.append(Result("ventana bajo el nivel del electrolito",
                      h["z_fill"] - h["window_d"] / 2, 3.0))
    # tapas
    R = h["chamber_d"] / 2
    for which in ("we", "ce"):
        lf = m.lid_features(p, which)
        lid_out = _outside_of(block(h["body_w"], h["body_w"], *lay["z_lid"], h["corner_chamfer"]))
        ports = {k: v for k, v in lf.items() if k.startswith("port_")}
        names = sorted(ports)
        for i, a in enumerate(names):
            res += [Result(f"tapa {which}: {a} ↔ ranura del O-ring",
                           dist(ports[a], lf["gland"]), dry),
                    Result(f"tapa {which}: {a} ↔ tornillos", dist(ports[a], lf["screws"]), dry)]
            res += [Result(f"tapa {which}: {a} ↔ {b}", dist(ports[a], ports[b]), dry)
                    for b in names[i + 1:]]
        specs = m.lid_ports(p, which)
        from modules._geom import axial_port_r
        for s in specs:
            r_b, _ = axial_port_r(p, s)
            res.append(Result(f"tapa {which}: {s['name']} desemboca en la cámara",
                              R - math.hypot(*s["xy"]) - r_b, 0.5))
        fit = [s for s in specs if s["kind"] == "fitting"]
        for i, a in enumerate(fit):
            for b in fit[i + 1:]:
                res.append(Result(f"tapa {which}: conectores {a['name']}/{b['name']}: paso",
                                  math.dist(a["xy"], b["xy"]), p.PORTS["fitting_pitch_min"]))
        res.append(Result(f"tapa {which}: ranura ↔ tornillos",
                          dist(lf["gland"], lf["screws"]), dry))
        res.append(Result(f"tapa {which}: tornillos ↔ exterior", dist(lf["screws"], lid_out.cut(
            cyl(200, lay["z_lid"][1], lay["z_lid"][1] + 50)).cut(
            cyl(200, lay["z_lid"][0] - 50, lay["z_lid"][0]))), dry))
        res.append(Result(f"tapa {which}: O-ring fuera de la cámara",
                          h["lid_oring_id"] / 2 - R, 0.5))
    # electrodos, tubos y agitación
    we = next(s for s in p.HCELL_WE_PORTS if s["name"] == "WE")
    res.append(Result("WE dentro de la cámara (al costado de la ventana)",
                      R - (we["xy"][0] + we["rod_d"] / 2), 1.0))
    res.append(Result("disco del WE bajo el nivel", h["z_fill"] - 0.0, 5.0))
    bar_l, bar_d = h["stir_bar"]
    res.append(Result("barra de agitación cabe en la cámara", h["chamber_d"] - bar_l, 4.0))
    res.append(Result("barra bajo el disco del WE", 0.0 - (h["z_floor"] + bar_d), 3.0))
    for which in ("we", "ce"):
        gas = next(s for s in m.lid_ports(p, which) if s["name"] == "gas_in")
        res.append(Result(f"tubo de gas ({which}) fuera del giro de la barra",
                          math.hypot(*gas["xy"]) - 0.8 - bar_l / 2, 1.0))
    res.append(Result("tubo de gas sobre la barra", h["gas_tube_z"] - (h["z_floor"] + bar_d), 1.0))
    res.append(Result("tubo de gas bajo el nivel", h["z_fill"] - h["gas_tube_z"], 5.0))
    # medias celdas y membrana sin interferencias
    a = m.build_half(p).val()
    b = m.other_side(a, p)
    res.append(Result("celda H: media celda ↔ media celda (−mm³)", -_vol(a, b), -1e-3))
    mem = m.membrane(p)
    res.append(Result("celda H: membrana ↔ medias celdas (−mm³)",
                      -(_vol(mem, a) + _vol(mem, b)), -1e-3))
    return res


def flow_checks(p=P) -> list[Result]:
    """Celda de flujo wall-jet: paredes, puertos, régimen del chorro, burbujas y collar."""
    from modules import flow_walljet as m

    fl, lay = p.FLOW, m.layout(p)
    wet, dry = p.CELL["min_wall"], p.CELL["min_wall_dry"]
    f = m.features(p)
    out = _outside_of(m.outline(p))
    g = m.we(p)
    a, R = fl["nozzle_d"], g["disk_d"] / 2
    res = [
        Result("flujo: cavidad ↔ exterior lateral", dist(f["cavity"], out.cut(
            cyl(200, lay["z_skirt"] - 50, lay["z_bottom"]))), wet),
        Result("flujo: cavidad ↔ cara superior",
               dist(f["cavity"], slab(p, fl["z_top"], True)), wet),
        Result("flujo: piso de la cavidad sobre el asiento del tubo",
               lay["z_floor"] - lay["z_seat"], wet),
        Result("flujo: asiento ↔ cara inferior (rosca completa)",
               lay["z_seat"] - lay["z_bottom"], p.PORTS["thread_depth"]),
    ]
    lat = {k: v for k, v in f.items() if k.startswith("port_") and k != "port_WE"}
    res += _ports_checks(p, lat, p.FLOW_PORTS, fl["z_top"], lay["z_bottom"], lay["z_floor"])
    for k, v in lat.items():      # sólo lo que atraviesa material (la cavidad está vacía)
        v = v.cut(f["cavity"])
        res.append(Result(f"flujo: {k} ↔ port_WE", dist(v, f["port_WE"]), dry))
        res.append(Result(f"flujo: {k} ↔ entrada", dist(v, f["inlet"]), dry))
    res.append(Result("flujo: ranura del WE ↔ cavidad", dist(f["port_WE"].cut(
        cyl(p.hole(g["body_d"]), fl["cavity_top"] - 1, fl["z_top"] + 1)), f["cavity"]), wet))
    # chorro (wall-jet de Albery y Brett: h del orden de a, R varias veces a, laminar)
    res += [
        Result("chorro: h/a ≥ 1", fl["jet_h"] / a, 1.0),
        Result("chorro: h/a ≤ 10 (−)", -fl["jet_h"] / a, -10.0),
        Result("chorro: R/a ≥ 3 (el disco recibe el abanico)", R / a, 3.0),
        Result("chorro: laminar (−Re ≥ −2000)", -m.nozzle_reynolds(p), -2000.0),
        Result("chorro: espacio radial alrededor del WE",
               fl["cavity_d"] / 2 - g["body_d"] / 2, 2.0),
    ]
    out_spec = next(s for s in p.FLOW_PORTS if s["name"] == "outlet")
    res.append(Result("salida en lo alto de la cavidad (−hueco sobre la boca)",
                      -(fl["cavity_top"] - (out_spec["z"] + p.hole(out_spec["through_d"]) / 2)),
                      -0.6))
    res.append(Result("patas: el conector de entrada cabe entre ellas",
                      fl["body_w"] - 2 * fl["skirt_wall"] - 8.0, 4.0))
    res.append(Result("patas: altura para conector y curva del tubo", fl["skirt_h"] - 10.0, 12.0))
    # collar y galga
    collar = m.build_collar(p).val()
    body = m.build(p).val()
    res.append(Result("collar ↔ cuerpo (−mm³)", -_vol(collar, body), -1e-3))
    res.append(Result("collar: pared para el inserto radial M3",
                      (fl["collar_od"] - p.hole(g["body_d"])) / 2, 4.0))
    res.append(Result("galga = distancia collar → disco (−diferencia)",
                      -abs(m.build_gauge(p).val().BoundingBox().zmax - lay["collar"][0]), -1e-6))
    return res


def _instrument(name: str) -> str:
    """Agrupa las piezas de un mismo electrodo por su prefijo (RE, CE, CS945, double knob)."""
    for key in ("RE", "CE", "CS945", "double knob", "GC"):
        if name.startswith(key):
            suffix = name[name.find("(pad"):] if "(pad" in name else ""
            return key + suffix
    return name


def electrode_checks(p=P, configs=None) -> list[Result]:
    """En cada ensamble, ningún electrodo invade una pieza ni otro electrodo."""
    from modules import assemblies as A

    res = []
    for key in configs or A.CONFIGS:
        parts = A.CONFIGS[key][1](p)
        elec = [(n, s) for n, s, _, g in parts if g == A.ELECTRODE and "Ag/AgCl" not in n]
        # obstáculos: celda, muestra y los tubos que entran al líquido
        others = [(n, s) for n, s, _, g in parts if g in (A.CELL, A.SAMPLE)
                  or (g == A.FLOW and n.startswith("tubo sumergido"))]
        worst = {}
        def overlap(a, b) -> float:
            # la distancia (BRepExtrema) es robusta; la intersección booleana de un alambre
            # fino contra un sólido complejo da falsos positivos: sólo se usa si se tocan
            if a.distance(b) > 1e-6:
                return 0.0
            return a.intersect(b).Volume()

        for en, es in elec:
            for on, os_ in others:
                v = overlap(es, os_)
                k = (_instrument(en), on)
                worst[k] = max(worst.get(k, 0.0), v)
            for en2, es2 in elec:
                if _instrument(en2) != _instrument(en) and en < en2:
                    v = overlap(es, es2)
                    k = (_instrument(en), _instrument(en2))
                    worst[k] = max(worst.get(k, 0.0), v)
        bad = {k: v for k, v in worst.items() if v > 1e-3}
        n_inst = len({_instrument(n) for n, _ in elec})
        res.append(Result(f"{key}: {n_inst} electrodos sin choques (−mm³ del peor)",
                          -max(bad.values(), default=0.0), -1e-3))
        for (a, b), v in sorted(bad.items(), key=lambda kv: -kv[1])[:4]:
            res.append(Result(f"{key}: {a} invade {b} (−mm³)", -v, -1e-3))
    return res


def footing_checks(p=P) -> list[Result]:
    """Cada celda asienta plana: ninguna pieza ni electrodo baja del fondo del cuerpo
    principal (la primera pieza de cada ensamble)."""
    from modules import assemblies as A

    res = []
    for key in A.CONFIGS:
        parts = A.CONFIGS[key][1](p)
        support = parts[0][1].BoundingBox().zmin
        lowest = min(((s.BoundingBox().zmin, n) for n, s, _, _ in parts))
        res.append(Result(f"{key}: nada baja del apoyo (z = {support:.1f}); "
                          f"lo más bajo: {lowest[1]}", lowest[0] - support, -1e-3))
    return res


def assembly_access_checks(p=P) -> list[Result]:
    """Montaje: cada electrodo puede LLEGAR a su posición, no sólo caber en ella.

    - Lo que entra por un puerto tiene que pasar por su agujero.
    - Un tornillo de pinza que se aprieta dentro de la celda tiene que mirar hacia el lado
      abierto durante el montaje (arriba) y quedar al alcance del destornillador.
    """
    from modules import chamber_xrd
    from modules._geom import clamp_extent, insertion_radius, port_bore_r

    e, c = p.ELECTRODES, p.WE_CLAMP
    coil_od = e["CE_coil_d"] + 0.5
    res = []
    from modules import tcell
    for cell, ports in (("Raman sellada", p.RING_PORTS), ("Raman pinza", p.CLAMP_PORTS),
                        ("XRD", p.XRD_PORTS), ("Celda T", tcell.port_specs(p)),
                        ("Flujo", p.FLOW_PORTS)):
        for s in ports:
            bore = 2 * port_bore_r(p, s)
            if s["name"] == "RE":
                res.append(Result(f"{cell}: la punta de la RE pasa por su agujero",
                                  bore - e["RE_tip_d"], 0.1))
            elif s["name"] == "CE" and s["kind"] == "radial":
                res.append(Result(f"{cell}: la espiral del CE pasa por su agujero",
                                  bore - coil_od, 0.2))
    # Raman con pinza: o se arma afuera y entra por un tapón, o el tornillo va arriba y libre
    we = next(s for s in p.CLAMP_PORTS if s["name"] == "WE")
    if we["kind"] == "plug":
        r_need = insertion_radius(p, c.get("screw", "down"))
        res.append(Result("Raman pinza: la pinza armada con la muestra pasa por el tapón",
                          port_bore_r(p, we) - r_need, 0.2))
    else:
        res.append(Result("Raman pinza: tornillo de la pinza accesible desde arriba "
                          "(1 = sí; con la tapa puesta no se llega)",
                          1.0 if c.get("screw") == "up" else 0.0, 1.0))
    # XRD: el tornillo se aprieta desde arriba con el bolsillo abierto
    s_xrd = p.WE_XRD
    res.append(Result("XRD: el tornillo de la pinza mira hacia el bolsillo abierto (1 = arriba)",
                      1.0 if s_xrd.get("clamp_screw") == "up" else 0.0, 1.0))
    lay = chamber_xrd.layout(p)
    k = p.CHAMBER_XRD
    y_screw = lay["y_jaw0"] + c["grip_l"] / 2
    hw = lay["pocket_hw"] + k["cover_ledge"]
    res.append(Result("XRD: el tornillo cae dentro de la abertura de la tapa del bolsillo",
                      min(hw - c.get("screw_head_d", 5.0) / 2,
                          y_screw - (lay["sy"] + k["cover_open"]),
                          lay["pocket_y1"] - y_screw), 0.3))
    we_x = next(s for s in p.XRD_PORTS if s["name"] == "WE")
    res.append(Result("XRD: la varilla (sin tornillo) pasa por su puerto",
                      2 * port_bore_r(p, we_x) - c["head_w"], 0.1))
    up, _ = clamp_extent(c, s_xrd.get("clamp_screw", "down"))
    res.append(Result("XRD: la cabeza del tornillo queda bajo la tapa del bolsillo",
                      -k["cover_t"] - (lay["z_clip"] + up), 0.3))
    # Celda H: todo entra desde arriba por su O-ring radial
    for s_ in p.HCELL_WE_PORTS + p.HCELL_CE_PORTS:
        if s_["name"] == "CE":
            res.append(Result("Celda H: la espiral del CE pasa por la tapa",
                              p.hole(s_["rod_d"]) - coil_od, 0.2))
        elif s_["name"] in ("WE", "RE"):
            res.append(Result(f"Celda H: el {s_['name']} pasa por la tapa",
                              p.hole(s_["rod_d"]) - s_["rod_d"], 0.1))
    # Chladni: la RE entra desde arriba por la tapa; el CE se coloca con la celda abierta
    res.append(Result("Chladni: la punta de la RE pasa por la tapa",
                      p.hole(e["RE_tip_d"]) - e["RE_tip_d"], 0.1))
    res.append(Result("Chladni: el alambre del CE (con funda) sale por su conector",
                      p.hole(p.CHLADNI_CE_PORT["through_d"]) - e["CE_sleeve_d"], 0.1))
    return res


def _box_outside(p, lay):
    """Todo lo que está fuera del cuerpo rectangular de la celda XRD, a lo ancho."""
    from modules.chamber_xrd import body_block
    big = cq.Solid.makeBox(400, 400, 100, cq.Vector(-200, -200, lay["z_bottom"] + 0.001))
    return big.cut(body_block(p, lay)).intersect(
        cq.Solid.makeBox(400, 400, -lay["z_bottom"] - 0.002,
                         cq.Vector(-200, -200, lay["z_bottom"] + 0.001)))


def run(p=P) -> list[Result]:
    return (ring_checks(p) + base_checks(p) + seal_checks(p) + assembly_checks(p)
            + lid_raman_checks(p) + clamp_checks(p) + lid_raman_checks(p, "clamp")
            + xrd_checks(p) + chladni_checks(p) + epd_checks(p) + electrode_checks(p)
            + assembly_access_checks(p) + footing_checks(p)
            + tcell_checks(p) + hcell_checks(p) + flow_checks(p))


if __name__ == "__main__":
    res = run()
    for r in res:
        print(r)
    bad = [r for r in res if not r.ok]
    print(f"\n{len(res) - len(bad)}/{len(res)} OK")
    sys.exit(1 if bad else 0)
