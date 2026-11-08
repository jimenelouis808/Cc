"""Modos de la placa-WE (Chladni) y cuánto los excita el piezo.

Placa cuadrada de lado L con borde empotrado.
- Frecuencias en seco: analíticas, con los valores λ² de Leissa.
- En líquido: orden de magnitud del fundamental (masa añadida, AVMI ≈ 0.67, fluido
  por un lado).
- Acoplamiento del piezo: con la aproximación de Warburton, cada modo es el producto
  de dos funciones de viga empotrada-empotrada, w_mn(x, y) = φ_m(x) φ_n(y). La fuerza
  modal de un piezo pegado es proporcional a la integral de la curvatura ∇²w sobre
  su área. Un acoplamiento cercano a cero significa que, en esa posición, el piezo
  no puede excitar ese modo por más que se le suba el voltaje.

Los valores reales dependen del piezo pegado, del adhesivo, de la altura de
electrolito y de la rigidez real del empotramiento: medirlos con un barrido de
impedancia del piezo, con la celda llena.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402

MATERIALS = {            # E (Pa), rho (kg/m3), nu
    "Ti grado 2": (105e9, 4510, 0.34),
    "Acero 316": (193e9, 8000, 0.29),
    "Ni 200": (204e9, 8890, 0.31),
}

# lambda^2 para placa cuadrada empotrada (Leissa), modos (m, n)
SQUARE_CLAMPED = [((1, 1), 35.99), ((2, 1), 73.41), ((2, 2), 108.27),
                  ((3, 1), 131.64), ((3, 2), 165.15), ((3, 3), 220.06)]

RHO_LIQ = 1050.0         # ~0.5 M H2SO4 / electrolitos acuosos diluidos
AVMI = 0.67

# βL de la viga empotrada-empotrada, modos 1..4
BEAM_BL = (4.730041, 7.853205, 10.995608, 14.137165)


def freqs(side_mm, h_mm, mat):
    E, rho, nu = MATERIALS[mat]
    L, h = side_mm / 1000, h_mm / 1000
    D = E * h**3 / (12 * (1 - nu**2))
    k = math.sqrt(D / (rho * h)) / L**2 / (2 * math.pi)
    dry = [(m, lam * k) for m, lam in SQUARE_CLAMPED]
    a_eq = L / 2
    wet_factor = 1 / math.sqrt(1 + AVMI * RHO_LIQ * a_eq / (rho * h))
    return dry, wet_factor


def beam(n: int, xi: np.ndarray) -> np.ndarray:
    """Modo n de una viga empotrada en ξ = 0 y ξ = 1."""
    b = BEAM_BL[n - 1]
    s = (math.cosh(b) - math.cos(b)) / (math.sinh(b) - math.sin(b))
    x = b * xi
    return np.cosh(x) - np.cos(x) - s * (np.sinh(x) - np.sin(x))


def coupling(m: int, n: int, center, diameter, L, grid: int = 400) -> float:
    """∫ ∇²w_mn sobre el disco del piezo, normalizado por max|∇²w| · área.

    ``center`` en mm respecto del centro de la placa. Devuelve un valor con signo
    en [-1, 1]: |valor| cerca de 0 = el piezo no excita ese modo.
    """
    xs = np.linspace(0, 1, grid)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    w = beam(m, X) * beam(n, Y)
    d = 1.0 / (grid - 1)
    lap = np.gradient(np.gradient(w, d, axis=0), d, axis=0) + np.gradient(
        np.gradient(w, d, axis=1), d, axis=1)
    cx, cy = (center[0] + L / 2) / L, (center[1] + L / 2) / L
    disk = (X - cx) ** 2 + (Y - cy) ** 2 <= (diameter / 2 / L) ** 2
    if not disk.any():
        return 0.0
    return float(lap[disk].mean() / np.abs(lap).max())


def mode_table(p=P) -> list[dict]:
    """Modos (m, n) y (n, m) con frecuencia seca, estimación húmeda y acoplamiento."""
    c = p.CHLADNI
    dry, wf = freqs(c["plate_L"], c["plate_h"], c["plate_material"])
    rows = []
    for (m, n), f in dry:
        for mm, nn in {(m, n), (n, m)}:
            rows.append({"modo": (mm, nn), "f_seco_hz": f, "f_liquido_hz": f * wf,
                         "acoplamiento": coupling(mm, nn, c["piezo_center"], c["piezo_d"],
                                                  c["plate_L"])})
    return rows


if __name__ == "__main__":
    c = P.CHLADNI
    print(f"Placa {c['plate_material']}, L = {c['plate_L']} mm, h = {c['plate_h']} mm; "
          f"piezo Ø{c['piezo_d']} en {c['piezo_center']}\n")
    print(f"{'modo':>6} {'seco (Hz)':>10} {'líquido (Hz)':>13} {'acoplamiento':>13}")
    for r in mode_table():
        m, n = r["modo"]
        print(f"  ({m},{n}) {r['f_seco_hz']:10.0f} {r['f_liquido_hz']:13.0f} "
              f"{r['acoplamiento']:+13.2f}")
    print("\nEl factor húmedo es exacto sólo para el fundamental; para los demás es una cota.")
    print("Tabla general por material y espesor:")
    for mat in MATERIALS:
        for side in (30, 40):
            for h in (0.2, 0.3, 0.5):
                dry, wf = freqs(side, h, mat)
                s = "  ".join(f"{m[0]}{m[1]}:{f/1000:5.2f}" for m, f in dry)
                print(f"{mat:10s} L={side} mm h={h} mm | seco kHz {s} | "
                      f"fundamental en líquido ≈ {dry[0][1]*wf/1000:4.2f} kHz")
