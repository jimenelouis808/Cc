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
    if spec["kind"] == "fitting":
        return p.hole(spec["through_d"]) / 2
    return p.hole(p.ELECTRODES["RE_body_d"]) / 2


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
    others = {k: f[k] for k in ("bolts", "pins", "lid_inserts") if k in f}
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
        res.append(Result(f"port_{s['name']}: boca sobre el piso", bottom - cavity_floor, 0.0))
    fittings = [s for s in specs if s["kind"] == "fitting"]
    for i, a in enumerate(fittings):
        for b in fittings[i + 1:]:
            if a["face"] == b["face"]:
                d = math.hypot(a["offset"] - b["offset"], a["z"] - b["z"])
                res.append(Result(f"conectores {a['name']}/{b['name']}: paso",
                                  d, p.PORTS["fitting_pitch_min"]))
    return res


def clamp_checks(p=P) -> list[Result]:
    """Cámara para pinza de Pt: paredes, puertos, pinza, muestra y pedestal."""
    import math

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

    # la pinza cerrada pasa por el agujero del puerto WE
    we = next(s for s in p.CLAMP_PORTS if s["name"] == "WE")
    diag = math.hypot(c["clip_w"], 2 * c["jaw_t"])
    res.append(Result("pinza cerrada cabe por el puerto WE (holgura)",
                      p.hole(we["through_d"]) - diag, 0.0))
    res.append(Result("cuerpo del portaelectrodo coincide con el puerto WE",
                      -abs(we["through_d"] - c["rod_d"]), 0.0))
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


def run(p=P) -> list[Result]:
    return (ring_checks(p) + base_checks(p) + seal_checks(p) + assembly_checks(p)
            + lid_raman_checks(p) + clamp_checks(p) + lid_raman_checks(p, "clamp"))


if __name__ == "__main__":
    res = run()
    for r in res:
        print(r)
    bad = [r for r in res if not r.ok]
    print(f"\n{len(res) - len(bad)}/{len(res)} OK")
    sys.exit(1 if bad else 0)
