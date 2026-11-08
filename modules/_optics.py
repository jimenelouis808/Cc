"""Geometría óptica del Raman: cono de luz enfocado en el WE y envolvente del objetivo.

Origen en el WE (z = 0). Entre el WE y el objetivo hay electrolito (gap) y ventana
(window_t). Cada capa plana desplaza el foco t·(1 − 1/n), así que la punta del
objetivo queda a  z_tip = WD + window_t·(1 − 1/n_w) + gap·(1 − 1/n_e).
"""
import math

import cadquery as cq


def _gap(p, gap=None):
    return p.CELL["thin_gap_raman"] if gap is None else gap


def z_tip(p, gap=None) -> float:
    r, g = p.RAMAN, _gap(p, gap)
    return (r["working_distance"] + r["window_t"] * (1 - 1 / r["n_window"])
            + g * (1 - 1 / r["n_electrolyte"]))


def _tan_in(n_a, n):
    s = n_a / n
    return s / math.sqrt(1 - s * s)


def light_radius(p, z: float, gap=None) -> float:
    """Radio del cono de luz (NA del objetivo) a la altura z, con refracción."""
    r, g = p.RAMAN, _gap(p, gap)
    na, t = r["numerical_aperture"], r["window_t"]
    te, tg, ta = _tan_in(na, r["n_electrolyte"]), _tan_in(na, r["n_window"]), _tan_in(na, 1.0)
    if z <= g:
        return z * te
    if z <= g + t:
        return g * te + (z - g) * tg
    return g * te + t * tg + (z - g - t) * ta


def _revolve(points) -> cq.Solid:
    """Sólido de revolución alrededor de z a partir de un perfil (r, z) cerrado en r = 0."""
    wp = cq.Workplane("XZ").polyline(points).close()
    return wp.revolve(360, (0, 0, 0), (0, 1, 0)).val()


def light_cone(p, gap=None, margin=None) -> cq.Solid:
    """Cono de luz desde el WE hasta la punta del objetivo, ensanchado `margin`."""
    g = _gap(p, gap)
    m = p.RAMAN["optical_margin"] if margin is None else margin
    zt = z_tip(p, g)
    zs = [g, g + p.RAMAN["window_t"], zt]
    base = [(0.0, 0.0)] if m <= 0 else [(0.0, -0.01), (m, -0.01)]
    return _revolve(base + [(light_radius(p, z, g) + m, z) for z in zs] + [(0.0, zt)])


def objective_envelope(p, gap=None, top: float = 120.0) -> cq.Solid:
    """Volumen que ocupa el objetivo (más clearance_margin), con nariz cónica."""
    r = p.RAMAN
    m = r["clearance_margin"]
    z0 = z_tip(p, gap) - m
    r0 = r["objective_tip_diameter"] / 2 + m
    rb = r["objective_barrel_d"] / 2 + m
    if rb <= r0:                                  # objetivo cilíndrico, sin nariz
        return _revolve([(0, z0), (r0, z0), (r0, top), (0, top)])
    z_b = z0 + (rb - r0) / math.tan(math.radians(r["objective_nose_angle_deg"]))
    return _revolve([(0, z0), (r0, z0), (rb, z_b), (rb, top), (0, top)])
