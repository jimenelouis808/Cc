"""Estimación rápida de frecuencias de modos para la placa-WE (Chladni).

Placa cuadrada de lado L con borde empotrado (valores de Leissa).
Frecuencias en seco: analíticas. En líquido: solo un orden de magnitud para el
modo fundamental (masa añadida, AVMI ~0.67, fluido por un lado). Los valores
reales dependen del piezo pegado, del adhesivo, de la altura de electrolito y de la
rigidez real del sujetador: medirlos con un barrido de impedancia del piezo.
"""
import math

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


def freqs(side_mm, h_mm, mat):
    E, rho, nu = MATERIALS[mat]
    L, h = side_mm / 1000, h_mm / 1000
    D = E * h**3 / (12 * (1 - nu**2))
    k = math.sqrt(D / (rho * h)) / L**2 / (2 * math.pi)
    dry = [(m, lam * k) for m, lam in SQUARE_CLAMPED]
    a_eq = L / 2
    wet_factor = 1 / math.sqrt(1 + AVMI * RHO_LIQ * a_eq / (rho * h))
    return dry, wet_factor


if __name__ == "__main__":
    for mat in MATERIALS:
        for side in (30, 40):
            for h in (0.2, 0.3, 0.5):
                dry, wf = freqs(side, h, mat)
                s = "  ".join(f"{m[0]}{m[1]}:{f/1000:5.2f}" for m, f in dry)
                print(f"{mat:10s} L={side} mm h={h} mm | seco kHz {s} | "
                      f"fundamental en líquido ≈ {dry[0][1]*wf/1000:4.2f} kHz")
