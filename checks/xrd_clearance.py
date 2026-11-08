"""Holgura del haz para XRD in situ en geometría Bragg-Brentano θ-θ.

Calcula:
  - longitud de la huella del haz sobre la muestra
  - altura máxima que puede tener cualquier parte de la celda (sobre el plano z=0)
    a una distancia d del centro, sin bloquear el haz incidente ni el difractado.
Supone geometría simétrica (mismo ángulo de apertura del lado del detector).
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from params import XRD  # noqa: E402


def _hit_x(phi, theta, R):
    """Posición x donde un rayo con ángulo phi toca la muestra (fuente en -x)."""
    return -R * math.cos(theta) + R * math.sin(theta) / math.tan(phi)


def footprint(two_theta_deg, R, div_deg):
    th = math.radians(two_theta_deg / 2)
    a = math.radians(div_deg)
    if th <= a / 2:
        return float("inf")
    return _hit_x(th - a / 2, th, R) - _hit_x(th + a / 2, th, R)


def min_two_theta_for_footprint(length, R, div_deg, tol=1e-4):
    """2θ (°) mínimo para que la huella completa quepa en `length` mm.

    Por debajo de ese ángulo el haz ilumina también el portamuestras / la celda
    (aparecen sus picos y cae la intensidad relativa de la muestra).
    """
    lo, hi = div_deg + 1e-6, 179.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if footprint(mid, R, div_deg) > length:
            lo = mid
        else:
            hi = mid
    return hi


def max_obstacle_height(d, two_theta_deg, R, div_deg, n=50):
    """Altura máxima permitida (mm) a distancia d (mm) del centro, en el eje del haz.
    Devuelve 0 si d cae dentro de la huella."""
    th = math.radians(two_theta_deg / 2)
    a = math.radians(div_deg)
    h_min = float("inf")
    for i in range(n + 1):
        phi = th - a / 2 + a * i / n
        x_hit = _hit_x(phi, th, R)
        if x_hit <= -d:          # el rayo toca la muestra antes del obstáculo
            return 0.0
        h_min = min(h_min, (x_hit + d) * math.tan(phi))
    return h_min


def allowed_height(p, d: float, two_thetas=None) -> float:
    """Altura máxima permitida a |x| = d dentro del corredor del haz, para todo 2θ
    desde two_theta_min hasta 40° (el peor caso suele ser el ángulo más bajo, pero
    se toma el mínimo para no suponerlo)."""
    x = p.XRD
    tts = two_thetas or [x["two_theta_min"] + 2 * i for i in range(16)]
    return min(max_obstacle_height(d, tt, x["goniometer_radius"], x["divergence_deg"])
               for tt in tts)


def forbidden_region(p, x_max: float = 80.0, step: float = 0.5, top: float = 60.0):
    """Sólido que ninguna pieza puede invadir: dentro del corredor del haz (|y| ≤
    semiancho) y por encima de la altura permitida a cada |x|. Una pieza que lo toca
    tapa el haz incidente o el difractado a algún 2θ ≥ two_theta_min."""
    import cadquery as cq

    from modules.chamber_xrd import corridor_half

    xs = [-x_max + i * step for i in range(int(2 * x_max / step) + 1)]
    pts = [(x, allowed_height(p, abs(x))) for x in xs]
    poly = [(-x_max, top)] + pts + [(x_max, top)]
    cy = corridor_half(p)
    # extrusión simétrica: cubre |y| ≤ cy
    return cq.Workplane("XZ").polyline(poly).close().extrude(cy, both=True).val()


def invasion(part, p) -> float:
    """mm³ de una pieza dentro de la región prohibida."""
    return part.val().intersect(forbidden_region(p)).Volume()


def report(two_thetas=(10, 13.7, 20, 26, 40), distances=(10, 15, 20, 25, 30)):
    R, div = XRD["goniometer_radius"], XRD["divergence_deg"]
    print(f"R = {R} mm, divergencia = {div}°\n")
    print("2θ (°) | huella (mm) | h_max (mm) a d = " + ", ".join(f"{d}" for d in distances))
    for tt in two_thetas:
        fp = footprint(tt, R, div)
        hs = ", ".join(f"{max_obstacle_height(d, tt, R, div):5.2f}" for d in distances)
        print(f"{tt:6.1f} | {fp:11.1f} | {hs}")
    from params import WE_FLAT
    w = WE_FLAT["sample_w"]
    print(f"\nLa huella cabe en la muestra ({w} mm) solo para 2θ ≥ "
          f"{min_two_theta_for_footprint(w, R, div):.1f}°")


if __name__ == "__main__":
    report()
