"""Ensambles completos de cada celda, con los electrodos del usuario en sus puertos.

Para ver la configuración (STEP y GLB en out/) y para comprobar que ningún electrodo
choca con una pieza ni con otro electrodo (checks/walls.py: electrode_checks).

Profundidades de inserción, todas derivadas de params:
- RE: la punta Ø4 entra RE_protrusion en la cavidad; el hombro Ø9 apoya en el resalte.
- CE de Pt: el cuerpo Ø6 pasa justo el O-ring radial; la espiral queda en la boca.
- Pinza: su cara frontal donde la pone cada cámara (layout).
"""
import os
import sys

import cadquery as cq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402
from modules import electrodes as E  # noqa: E402
from modules._geom import FACE_ANGLE, cyl  # noqa: E402

C = {
    "body": (0.86, 0.83, 0.75, 1.0), "lid": (0.79, 0.64, 0.42, 1.0),
    "metal": (0.62, 0.66, 0.71, 1.0), "glass": (0.75, 0.88, 0.95, 0.40),
    "sample": (0.18, 0.18, 0.20, 1.0), "ti": (0.55, 0.57, 0.60, 1.0),
    "piezo": (0.91, 0.64, 0.24, 1.0), "film": (0.95, 0.72, 0.25, 0.35),
    "gasket": (0.10, 0.10, 0.10, 1.0), "gold": (0.83, 0.66, 0.25, 1.0),
    "fitting": (0.80, 0.72, 0.55, 1.0), "tube": (0.90, 0.90, 0.92, 0.7),
}

# grupos para el visor y para los checks
CELL, SAMPLE, ELECTRODE, FLOW = "celda", "muestra", "electrodo", "flujo"


def _wp(s):
    return s if isinstance(s, cq.Workplane) else cq.Workplane().add(s)


def side_fitting(spec, r_out) -> list:
    """Conector 1/4-28 y tubo en un puerto lateral (sólo para ver dónde va el flujo)."""
    import math
    a = math.radians(FACE_ANGLE[spec["face"]])
    body = cq.Solid.makeCylinder(4.0, 10, cq.Vector(r_out, spec["offset"], spec["z"]),
                                 cq.Vector(1, 0, 0))
    tube = cq.Solid.makeCylinder(0.8, 25, cq.Vector(r_out + 10, spec["offset"], spec["z"]),
                                 cq.Vector(1, 0, 0))
    rot = lambda s: s.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), math.degrees(a))  # noqa: E731
    return [(f"conector {spec['name']}", rot(body), C["fitting"], FLOW),
            (f"tubo {spec['name']}", rot(tube), C["tube"], FLOW)]


def top_fitting(name, x, y, z) -> list:
    return [(f"conector {name}", cyl(8.0, z, z + 10, x, y), C["fitting"], FLOW),
            (f"tubo {name}", cyl(1.6, z + 10, z + 35, x, y), C["tube"], FLOW)]


def _electrode(parts, group=ELECTRODE):
    return [(n, s, col, group) for n, s, col in parts]


def _ports(p, axes, clamp_front=None, clamp_z=None):
    """Electrodos y conectores de una cámara a partir de sus puertos.

    axes: [(spec, r_out, r_in)]. clamp_front: distancia del centro a la cara frontal de
    la pinza sobre el eje del puerto WE.
    """
    out = []
    g = p.oring_gland(p.ORING["re_cs"], "radial")
    coil_h, straight = E.ce_lengths(p)
    for spec, r_out, r_in in axes:
        ang = FACE_ANGLE[spec["face"]]
        if spec["name"] == "RE":
            r_tip = (r_in + 1.0) - p.ELECTRODES["RE_protrusion"]
            out += _electrode(E.place(E.reference(p), E.port_tip(spec, r_tip), ang))
        elif spec["name"] == "CE" and spec["kind"] == "radial":
            body_front = r_out - (p.ORING["re_groove_from_face"] + g["width"] + 0.5)
            r_tip = body_front - coil_h - straight
            out += _electrode(E.place(E.counter_pt(p), E.port_tip(spec, r_tip), ang))
        elif spec["name"] == "WE":
            parts = (E.clamp_cs945(dict(p.WE_CLAMP), p) if p.WE_CLAMP["model"] == "CS945"
                     else E.clamp_double_knob(dict(p.WE_CLAMP), p))
            tip = E.port_tip(dict(spec, z=clamp_z if clamp_z is not None else spec["z"]),
                             clamp_front)
            out += _electrode(E.place(parts, tip, ang))
        elif spec["kind"] == "fitting":
            out += side_fitting(spec, r_out)
    return out


# --- configuraciones -----------------------------------------------------------

def raman_sealed(p=P) -> list:
    from modules import base_flat, chamber_ring, lid_raman
    w = p.WE_FLAT
    g, t = p.CELL["thin_gap_raman"], p.RAMAN["window_t"]
    parts = [
        ("base", base_flat.build(p).val(), C["body"], CELL),
        ("muestra (WE plano)", cq.Solid.makeBox(w["sample_w"], w["sample_w"], w["sample_t"],
                                                cq.Vector(-w["sample_w"] / 2, -w["sample_w"] / 2,
                                                          -w["sample_t"])), C["sample"], SAMPLE),
        ("anillo-cámara", chamber_ring.build(p).val(), C["body"], CELL),
        ("tapa Raman", lid_raman.build(p).val(), C["lid"], CELL),
        ("ventana", cyl(p.LID_RAMAN["window_d"], g, g + t), C["glass"], CELL),
    ]
    return parts + _ports(p, chamber_ring.port_axes(p))


def raman_clamp(p=P) -> list:
    from modules import chamber_clamp, lid_raman
    lay = chamber_clamp.layout(p)
    g, t = p.CELL["thin_gap_raman"], p.RAMAN["window_t"]
    parts = [
        ("cámara para pinza", chamber_clamp.build(p).val(), C["body"], CELL),
        ("tapa Raman", lid_raman.build(p).val(), C["lid"], CELL),
        ("ventana", cyl(p.LID_RAMAN["window_d"], g, g + t), C["glass"], CELL),
        ("muestra (tira)", chamber_clamp.sample(p), C["sample"], SAMPLE),
    ]
    return parts + _ports(p, chamber_clamp.port_axes(p), clamp_front=-lay["x_jaw"],
                          clamp_z=lay["z_s"])


def xrd(p=P) -> list:
    from modules import chamber_xrd, lid_xrd
    lay = chamber_xrd.layout(p)
    s = p.WE_XRD
    lz = lay["lid"]
    path = chamber_xrd.tab_path(p)
    h = s["tab_t"] / 2
    pts = [(y, z + h) for y, z in path] + [(y, z - h) for y, z in reversed(path)]
    tab = cq.Workplane("YZ").polyline(pts).close().extrude(s["tab_w"] / 2, both=True).val()
    gx = lz["gx_in"] + lz["groove_w"] + 1.0
    film = cq.Solid.makeBox(2 * gx, lz["gy1_in"] - lz["gy0_in"] + 2 * (lz["groove_w"] + 1),
                            0.03, cq.Vector(-gx, lz["gy0_in"] - lz["groove_w"] - 1, 0.0))
    parts = [
        ("cuerpo XRD", chamber_xrd.build(p).val(), C["body"], CELL),
        ("marco tensor", lid_xrd.build(p).val().translate(cq.Vector(0, 0, 0.03)), C["metal"], CELL),
        ("tapa del bolsillo", lid_xrd.build_cover(p).val(), C["body"], CELL),
        ("película (Kapton/PP)", film, C["film"], CELL),
        ("muestra", chamber_xrd.sample(p), C["sample"], SAMPLE),
        ("lengüeta Pt/Ti", tab, C["ti"], SAMPLE),
    ]
    return parts + _ports(p, chamber_xrd.port_axes(p), clamp_front=lay["y_jaw0"],
                          clamp_z=lay["z_clip"])


def chladni(p=P, clamp: str = "double_knob") -> list:
    from modules import chladni as m
    lay, c = m.layout(p), p.CHLADNI
    parts = [
        ("marco inferior", m.build_bottom(p).val(), C["metal"], CELL),
        ("placa-WE (Ti)", m.plate(p), C["ti"], SAMPLE),
        ("piezo", m.piezo(p), C["piezo"], CELL),
        ("pared / cámara", m.build_wall(p).val(), C["body"], CELL),
        ("CE plano (espiral de Pt o malla)", m.ce_plate(p), E.COLORS["pt"], ELECTRODE),
        ("tapa", m.build_lid(p).val(), C["lid"], CELL),
    ]
    rx, ry = m.re_xy(p)
    parts += _electrode(E.place(E.reference(p), (rx, ry, c["luggin_h"]), 0, vertical="up"))
    cl = dict(p.CLAMPS[clamp])
    front = lay["outer"] + c["tab_out"] - cl["grip_l"]
    model = E.clamp_double_knob(cl, p) if clamp == "double_knob" else E.clamp_cs945(cl, p)
    parts += _electrode(E.place(model, (front, 0.0, -c["plate_h"] / 2), 0))
    parts += side_fitting(m.ce_port(p), lay["outer"])
    zl = lay["z_wall_top"] + c["lid_t"]
    for spec in p.CHLADNI_LID_PORTS:
        parts += top_fitting(spec["name"], *spec["xy"], zl)
    return parts


def epd(p=P) -> list:
    from modules import epd as m
    lay, e = m.layout(p), p.EPD
    parts = [
        ("base", m.build_base(p).val(), C["body"], CELL),
        ("chip (portaobjetos)", m.chip(p), C["glass"], CELL),
        ("electrodos interdigitados", m.electrodes(p, t=0.05), C["gold"], SAMPLE),
        ("junta (canal)", m.gasket(p), C["gasket"], CELL),
        ("vidrio superior", m.cover(p), C["glass"], CELL),
        ("marco", m.build_frame(p).val(), C["metal"], CELL),
    ]
    cl = dict(p.CLAMPS["CS945"])
    sx, sy, st = e["slide"]
    pad_y = e["pad_gap"] / 2 + e["pad"][1] / 2
    for sgn, name in ((1, "A"), (-1, "B")):
        model = [(f"{n} (pad {name})", s.rotate(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0), 180), col)
                 for n, s, col in E.clamp_cs945(cl, p)]
        parts += _electrode(E.place(model, (sx / 2 - cl["grip_l"], sgn * pad_y, -st / 2), 0))
    for spec in p.EPD_PORTS:
        parts += top_fitting(spec["name"], spec["x_sign"] * lay["hole_x"], 0.0, lay["z_frame_top"])
    return parts


CONFIGS = {
    "raman_sellada": ("Raman · muestra sellada (base + anillo + tapa)", raman_sealed),
    "raman_pinza": ("Raman · muestra en pinza CS945", raman_clamp),
    "xrd": ("XRD · película y pinza CS945", xrd),
    "chladni": ("Electrodeposición acústica (Chladni)", chladni),
    "epd": ("EPD / DEP · chip interdigitado", epd),
}


NOTES = {
    "raman_sellada": [
        "WE: muestra plana de 15 × 15 mm bajo el anillo; el O-ring define el área (Ø9.5).",
        "RE Gamry por −y: la punta Ø4 entra 1.5 mm en el pozo; el hombro Ø9 apoya en el resalte.",
        "CE de Pt por +y: el cuerpo Ø6 sella en su O-ring; la espiral de 35 mm queda en la boca.",
        "Entrada −x (abajo) y salida +x (arriba, purga burbujas).",
    ],
    "raman_pinza": [
        "WE: tira de 8 × 14 mm en la pinza CS945 (−x), cara en z = 0, apoyada en el pedestal.",
        "La cabeza de la pinza queda 0.5 mm fuera de la nariz de la tapa; tornillo hacia abajo.",
        "RE por −y (punta 1.5 mm dentro del pozo); CE de Pt por +y; entrada +y, salida +x.",
    ],
    "xrd": [
        "Haz a lo largo de x: nada sale por las caras ±x.",
        "WE: muestra bajo la película; la CS945 (+y) toma la lengüeta 6 mm más abajo, "
        "bajo una tapa.",
        "RE, CE, entrada y salida por −y, en el reservorio que se une a la capa delgada "
        "por una ranura.",
    ],
    "chladni": [
        "WE: placa de Ti empotrada; su lengüeta sale por +x y la double knob la toma en seco.",
        "RE vertical desde la tapa: la punta queda a 1.5 mm de la placa, en una esquina.",
        "CE plano (espiral de Pt) a 6 mm sobre la placa; su cable sale por −x.",
        "Piezo pegado bajo la placa en (−8.75, −8.75).",
    ],
    "epd": [
        "Electrodos interdigitados depositados en el chip; el gap baja de 500 a 100 µm hacia +x.",
        "Dos pinzas CS945 toman los pads en el extremo +x del chip, que sobresale de la base.",
        "Entrada y salida verticales sobre los agujeros del vidrio superior.",
    ],
}


LABELS = {"RE": "RE Ag/AgCl", "CE": "CE (Pt)", "CS945": "WE · pinza CS945",
          "double knob": "WE · pinza double knob"}
FLOW_LABELS = {"inlet": "entrada", "outlet": "salida", "vent": "venteo", "CE": "cable del CE"}


def _anchors(parts) -> dict:
    """Etiquetas del visor: electrodos, WE y conectores de flujo (coordenadas de la celda).

    Electrodos y conectores: centro de su caja, arriba. WE: centro de la cara de la muestra.
    """
    boxes = {}

    def grow(key, s, top=True):
        bb = s.BoundingBox()
        b = boxes.setdefault(key, [bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax, top])
        b[:6] = [min(b[0], bb.xmin), max(b[1], bb.xmax), min(b[2], bb.ymin), max(b[3], bb.ymax),
                 min(b[4], bb.zmin), max(b[5], bb.zmax)]

    for n, s, _, g in parts:
        if g == ELECTRODE:
            key = "double knob" if n.startswith("double knob") else n.split(" ")[0]
            key = LABELS.get(key, key)
            if "(pad" in n:
                key = "pinza " + n[n.find("(pad") + 1:-1]
            if n.startswith("CE plano"):
                key = LABELS["CE"]
            grow(key, s)
        elif g == SAMPLE and n.startswith("electrodos"):
            grow("WE: peines interdigitados A y B", s)
        elif g == SAMPLE and not n.startswith("lengüeta"):
            grow("WE (muestra)" if "placa" not in n else "WE (placa de Ti)", s)
        elif g == FLOW and n.startswith("conector"):
            grow(FLOW_LABELS.get(n.split(" ", 1)[1], n.split(" ", 1)[1]), s)
    return {k: [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, b[5]] for k, b in boxes.items()}


def export(name: str, p=P, out_dir=None) -> dict:
    """STEP + GLB del ensamble y un manifiesto (id, nombre, grupo, color) para el visor.

    Los nodos del GLB se llaman p00, p01…: el escritor de OpenCascade codifica mal los
    acentos, así que el nombre legible va en el manifiesto.
    """
    import json

    from export import OUT
    out_dir = out_dir or OUT
    os.makedirs(out_dir, exist_ok=True)
    title, fn = CONFIGS[name]
    parts = fn(p)
    assy = cq.Assembly(name=name)
    manifest = []
    for i, (n, s, col, group) in enumerate(parts):
        pid = f"p{i:02d}"
        assy.add(_wp(s), name=pid, color=cq.Color(*col))
        manifest.append({"id": pid, "name": n, "group": group, "color": col})
    assy.export(os.path.join(out_dir, f"ensamble_{name}.step"))
    assy.export(os.path.join(out_dir, f"ensamble_{name}.glb"), tolerance=0.05,
                angularTolerance=0.2)
    # el visor (docs/visor) lee el GLB como base64 en .txt: el servidor de artefactos no
    # sirve .glb, y un .txt se sirve en cualquier parte
    import base64
    with open(os.path.join(out_dir, f"ensamble_{name}.glb"), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    with open(os.path.join(out_dir, f"ensamble_{name}.glb.txt"), "w") as f:
        f.write(b64)
    meta = {"key": name, "title": title, "notes": NOTES.get(name, []), "parts": manifest,
            "labels": _anchors(parts)}
    with open(os.path.join(out_dir, f"ensamble_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


if __name__ == "__main__":
    for key in CONFIGS:
        meta = export(key)
        print(f"{key}: {len(meta['parts'])} piezas")
