"""Geometría compartida por base, anillo y tapas (misma huella y tornillería).

Todas las funciones reciben `p`, el módulo params (o una copia modificada, ver
`variant`). Aquí no hay dimensiones propias.
"""
import copy
import math
import types

import cadquery as cq

import params as P

FACE_ANGLE = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": 270.0}


def variant(p=P, **changes) -> types.SimpleNamespace:
    """Copia de params con cambios por diccionario: variant(CELL={"body_w": 60}).

    Las funciones (hole, oring_gland) se reconstruyen para que usen los dicts copiados.
    """
    ns = types.SimpleNamespace(**{k: copy.deepcopy(v) for k, v in vars(p).items()
                                  if k.isupper()})
    for name, upd in changes.items():
        cur = getattr(ns, name)
        if isinstance(cur, dict):
            cur.update(upd)
        else:
            setattr(ns, name, upd)
    ns.oring_gland = p.oring_gland
    ns.hole = lambda d: d + ns.FDM["hole_offset"]
    return ns


def body(p, z0: float, z1: float) -> cq.Solid:
    """Bloque cuadrado de la huella común, entre z0 y z1, con esquinas biseladas."""
    w = p.CELL["body_w"]
    wp = (cq.Workplane("XY").box(w, w, z1 - z0, centered=(True, True, False))
          .translate((0, 0, z0)).edges("|Z").chamfer(p.CELL["corner_chamfer"]))
    return wp.val()


def cyl(d: float, z0: float, z1: float, x: float = 0.0, y: float = 0.0) -> cq.Solid:
    return cq.Solid.makeCylinder(d / 2, z1 - z0, cq.Vector(x, y, z0))


def compound(solids) -> cq.Compound:
    return cq.Compound.makeCompound(list(solids))


def bolt_xy(p):
    a = p.CELL["body_w"] / 2 - p.CELL["bolt_inset"]
    return [(sx * a, sy * a) for sx in (1, -1) for sy in (1, -1)]


def lid_insert_xy(p):
    a = p.CELL["lid_insert_xy"]
    return [(sx * a, sy * a) for sx in (1, -1) for sy in (1, -1)]


def sample_seal(p):
    """Escalón del O-ring que sella sobre la muestra (cara inferior del anillo).

    El O-ring va en un escalón abierto hacia el centro, no en una ranura cerrada:
    su diámetro interior coincide con el agujero del piso y no queda rendija de
    electrolito entre la ranura y la muestra. Área expuesta = π·exposed_d²/4.
    """
    d = p.WE_FLAT["exposed_d"]
    g = p.oring_gland(p.ORING["sample_cs"], "face")
    return {"bore_d": d, "step_d": d + 2 * g["width"], "step_depth": g["depth"],
            "oring_od": d + 2 * p.ORING["sample_cs"],
            "area_mm2": math.pi * d**2 / 4}


def port_solid(p, spec: dict, r_outer: float, r_inner: float) -> cq.Shape:
    """Sólido a restar para un puerto lateral.

    Se construye a lo largo de +x (de la cara exterior r_outer hacia el centro,
    hasta r_inner, que debe caer dentro de la cavidad) y se gira a su cara.
    """
    y, z = spec["offset"], spec["z"]
    ax = cq.Vector(-1, 0, 0)

    def bar(d, x_from, x_to):
        return cq.Solid.makeCylinder(d / 2, x_from - x_to, cq.Vector(x_from, y, z), ax)

    if spec["kind"] == "fitting":
        x_seat = r_outer - p.PORTS["thread_depth"]
        s = bar(p.PORTS["tap_drill_d"], r_outer + 0.5, x_seat)
        s = s.fuse(bar(p.hole(spec["through_d"]), x_seat + 0.01, r_inner))
    elif spec["kind"] == "radial":
        rod = spec.get("rod_d", p.ELECTRODES["RE_body_d"])
        g = p.oring_gland(p.ORING["re_cs"], "radial")
        s = bar(p.hole(rod), r_outer + 0.5, r_inner)
        x_g = r_outer - p.ORING["re_groove_from_face"]
        s = s.fuse(bar(p.hole(rod + 2 * g["depth"]), x_g, x_g - g["width"]))
    elif spec["kind"] == "plug":
        s = bar(plug_bore_d(p), r_outer + 0.5, r_inner)
        c = p.CELL
        for side in (1, -1):
            s = s.fuse(cq.Solid.makeCylinder(
                c["insert_d"] / 2, c["insert_depth"] + 0.5,
                cq.Vector(r_outer + 0.5, y + side * p.PLUG["screw_off"], z), ax))
    elif spec["kind"] == "re_tip":
        e = p.ELECTRODES
        tip = e["RE_tip_d"]
        g = p.oring_gland(p.ORING["re_cs"], "radial")
        x_face = r_outer + re_boss_len(p, r_outer, r_inner)
        s = bar(p.hole(tip), x_face + 0.5, r_inner)
        x_g = x_face - p.ORING["re_groove_from_face"]
        s = s.fuse(bar(p.hole(tip + 2 * g["depth"]), x_g, x_g - g["width"]))
    else:
        raise ValueError(f"tipo de puerto desconocido: {spec['kind']}")
    s = s.clean()
    return s.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), FACE_ANGLE[spec["face"]])


STD_LENGTHS = (6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40)


def bolt_stack(p, z_head: float, z_bottom: float, tip_recess: float = 0.5) -> dict:
    """Tornillo M3 de cabeza en z_head (fondo del avellanado) a tuerca embebida.

    Elige la longitud estándar más larga cuya punta queda al menos `tip_recess`
    por encima de la cara inferior z_bottom (la celda asienta plana) y coloca la
    tuerca para que la rosca quede completa: tuerca entre z_end y z_end + nut_h.
    """
    room = z_head - z_bottom - tip_recess
    fits = [L for L in STD_LENGTHS if L <= room]
    if not fits:
        raise ValueError(f"no cabe ningún tornillo estándar (espacio {room:.1f} mm)")
    L = fits[-1]
    z_end = z_head - L
    return {"length": L, "z_end": z_end, "nut_top": z_end + p.CELL["nut_h"] + 0.2}


def re_boss_len(p, r_outer: float, r_inner: float) -> float:
    """Largo del resalte externo para que la punta de la RE entre RE_protrusion en la
    cavidad (r_inner + 1 es la pared de la cavidad sobre el eje del puerto)."""
    e = p.ELECTRODES
    wall = r_outer - (r_inner + 1.0)
    return max(0.0, e["RE_tip_l"] - e["RE_protrusion"] - wall)


def port_boss(p, spec: dict, r_outer: float, r_inner: float):
    """Resalte que se SUMA al cuerpo para un puerto re_tip (None para los demás)."""
    if spec["kind"] != "re_tip":
        return None
    L = re_boss_len(p, r_outer, r_inner)
    if L <= 0:
        return None
    s = cq.Solid.makeCylinder(p.ELECTRODES["RE_boss_d"] / 2, L + 0.01,
                              cq.Vector(r_outer - 0.01, spec["offset"], spec["z"]),
                              cq.Vector(1, 0, 0))
    return s.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), FACE_ANGLE[spec["face"]])


def clamp_extent(c: dict, screw: str = "down") -> tuple[float, float]:
    """(sobre, bajo) el plano medio de la muestra para la cabeza de la pinza.

    head_up / head_down del modelo valen con el tornillo (o las perillas) hacia abajo;
    con `screw` = "up" se intercambian.
    """
    if screw == "up":
        return c["head_down"], c["head_up"]
    return c["head_up"], c["head_down"]


def insertion_radius(p, screw: str = "down") -> float:
    """Radio que barre la pinza CON la muestra puesta y el tornillo apretado al entrar por un
    puerto: tira (ancho × espesor), varilla y cabeza del tornillo."""
    c = p.WE_CLAMP
    up, down = clamp_extent(c, screw)
    head_r = c.get("screw_head_d", c["head_w"]) / 2
    return max(math.hypot(c["sample_w"] / 2, c["sample_t"] / 2), c["rod_d"] / 2,
               math.hypot(head_r, max(up, down)))


def plug_bore_d(p) -> float:
    """Agujero del puerto "plug": deja pasar la pinza armada con su muestra."""
    return 2 * (insertion_radius(p, p.WE_CLAMP.get("screw", "down")) + p.PLUG["clearance"])


def plug_part(p, spec: dict, r_outer: float, r_inner: float) -> cq.Shape:
    """Tapón de PEEK del portaelectrodo desmontable, en su posición de montaje.

    Cuerpo dentro del agujero hasta la pared de la cavidad, brida con dos tornillos M3 sobre
    la cara, O-ring exterior (sella el agujero) e interior (sella la varilla).
    """
    P_, c = p.PLUG, p.CELL
    y, z = spec["offset"], spec["z"]
    ax = cq.Vector(1, 0, 0)
    od = plug_bore_d(p) - 2 * P_["fit"]
    x0, x1 = r_inner + 1.0, r_outer
    body = cq.Solid.makeCylinder(od / 2, x1 - x0, cq.Vector(x0, y, z), ax)
    body = body.fuse(cq.Solid.makeCylinder(P_["flange_d"] / 2, P_["flange_t"],
                                           cq.Vector(x1, y, z), ax))
    rod = spec.get("rod_d", p.WE_CLAMP["rod_d"])
    body = body.cut(cq.Solid.makeCylinder(p.hole(rod) / 2, x1 + P_["flange_t"] - x0 + 2,
                                          cq.Vector(x0 - 1, y, z), ax))
    gi = p.oring_gland(p.ORING["re_cs"], "radial")
    xg = x1 + P_["flange_t"] - p.ORING["re_groove_from_face"]
    body = body.cut(cq.Solid.makeCylinder(p.hole(rod + 2 * gi["depth"]) / 2, gi["width"],
                                          cq.Vector(xg - gi["width"], y, z), ax))
    go = p.oring_gland(P_["oring_cs"], "radial")
    xm = (x0 + x1) / 2
    ring = cq.Solid.makeCylinder(od / 2 + 1, go["width"], cq.Vector(xm, y, z), ax).cut(
        cq.Solid.makeCylinder(od / 2 - go["depth"], go["width"], cq.Vector(xm, y, z), ax))
    body = body.cut(ring)
    for side in (1, -1):
        body = body.cut(cq.Solid.makeCylinder(
            p.hole(c["bolt_clearance_d"] - p.FDM["hole_offset"]) / 2, P_["flange_t"] + 2,
            cq.Vector(x1 - 1, y + side * P_["screw_off"], z), ax))
    return body.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), FACE_ANGLE[spec["face"]])


def port_bore_r(p, spec) -> float:
    """Radio del agujero pasante de un puerto (lo que desemboca en la cavidad)."""
    if spec["kind"] == "fitting":
        return p.hole(spec["through_d"]) / 2
    if spec["kind"] == "re_tip":
        return p.hole(p.ELECTRODES["RE_tip_d"]) / 2
    if spec["kind"] == "plug":
        return plug_bore_d(p) / 2
    return p.hole(spec.get("rod_d", p.ELECTRODES["RE_body_d"])) / 2


def port_outer_r(p, spec) -> float:
    """Radio máximo del puerto: rosca del conector o ranura del O-ring radial."""
    if spec["kind"] == "fitting":
        return p.PORTS["tap_drill_d"] / 2
    if spec["kind"] == "plug":
        return plug_bore_d(p) / 2
    rod = p.ELECTRODES["RE_tip_d"] if spec["kind"] == "re_tip" else spec.get(
        "rod_d", p.ELECTRODES["RE_body_d"])
    return p.hole(rod + 2 * p.oring_gland(p.ORING["re_cs"], "radial")["depth"]) / 2
