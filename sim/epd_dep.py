"""Estimación de captura DEP por segmento en la celda EPD (orden de magnitud).

Para cada segmento de gap g:
  flujo:   ū = Q / (W·h),  Re = ρ ū h / η,  tiempo sobre el segmento t = L_seg / ū
  campo:   |E| ~ V/g,  ∇|E|² ~ V²/g³   (escala de un IDE plano; el factor real sale del FEM)
  fuerza sobre un bastón alineado (Morgan y Green):
           F = (π d² l / 6) · ε0 εm · Re[K] · ∇|E|²
  arrastre lateral de un cuerpo esbelto:  γ = 4π η l / (ln(l/d) + 0.5)
  número de captura:  C = v_DEP · t / h,  con v_DEP = F/γ
C ≥ 1 quiere decir que, en lo que tarda en cruzar el segmento, la partícula recorre
la altura del canal hacia los electrodos: se captura ahí. Como el campo decae en una
altura ~g, el número es optimista lejos del piso. Sirve para elegir gaps, tensión y
caudal, no para predecir posiciones: eso es el paso 6 (FEM + trazado de partículas).
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import params as P  # noqa: E402

EPS0 = 8.8541878128e-12


def flow(p=P) -> dict:
    e, s = p.EPD, p.EPD_SIM
    from modules.epd import layout
    h = layout(p)["h_channel"] * 1e-3
    W = e["channel_w"] * 1e-3
    q = e["flow_ul_min"] * 1e-9 / 60
    u = q / (W * h)
    return {"u_mean_mm_s": u * 1e3, "reynolds": s["rho_medium"] * u * h / s["viscosity"],
            "h_m": h, "u_m_s": u}


def capture_table(p=P) -> list[dict]:
    e, s = p.EPD, p.EPD_SIM
    f = flow(p)
    rows = []
    for g in e["gaps"]:
        gm = g * 1e-3
        grad = s["v_rms"] ** 2 / gm ** 3
        t = e["segment_l"] * 1e-3 / f["u_m_s"]
        row = {"gap_um": g * 1000, "E_V_um": s["v_rms"] / (g * 1000), "t_s": t, "C": {}}
        L = s["particle_l_um"] * 1e-6
        for d_nm in s["particle_d_nm"]:
            d = d_nm * 1e-9
            force = math.pi * d * d * L / 6 * EPS0 * s["eps_r"] * s["re_k"] * grad
            gamma = 4 * math.pi * s["viscosity"] * L / (math.log(L / d) + 0.5)
            row["C"][d_nm] = force / gamma * t / f["h_m"]
        rows.append(row)
    return rows


def voltage_for_capture(p=P) -> list[dict]:
    """Tensión rms que haría C = 1 en cada segmento, para cada diámetro (C ∝ V²)."""
    s = p.EPD_SIM
    return [{"gap_um": r["gap_um"],
             "V": {d: s["v_rms"] / math.sqrt(r["C"][d]) for d in s["particle_d_nm"]}}
            for r in capture_table(p)]


def first_capture(p=P) -> dict:
    """Primer segmento (gap) en que cada diámetro llega a C ≥ 1; None si en ninguno."""
    out = {}
    for d in p.EPD_SIM["particle_d_nm"]:
        out[d] = next((r["gap_um"] for r in capture_table(p) if r["C"][d] >= 1.0), None)
    return out


if __name__ == "__main__":
    f, s = flow(), P.EPD_SIM
    print(f"Q = {P.EPD['flow_ul_min']} µL/min → ū = {f['u_mean_mm_s']:.2f} mm/s, "
          f"Re = {f['reynolds']:.2e} (laminar)")
    print(f"{s['medium']}, V = {s['v_rms']} V rms, Re[K] = {s['re_k']}, CNT de "
          f"{s['particle_l_um']} µm\n")
    ds = s["particle_d_nm"]
    print("gap (µm)  E (V/µm)  t (s)   " + "  ".join(f"C d={d:g} nm" for d in ds))
    for r in capture_table():
        print(f"{r['gap_um']:7.0f}  {r['E_V_um']:8.3f}  {r['t_s']:5.2f}   "
              + "  ".join(f"{r['C'][d]:11.2g}" for d in ds))
    print("\nPrimer gap con C ≥ 1:", {f"{d:g} nm": (f"{g:.0f} µm" if g else "ninguno")
                                      for d, g in first_capture().items()})
    print(f"\nTensión rms para C = 1 con Re[K] = {s['re_k']} (escala como 1/√Re[K]):")
    print("gap (µm)  " + "  ".join(f"d={d:g} nm" for d in ds))
    for r in voltage_for_capture():
        print(f"{r['gap_um']:7.0f}  " + "  ".join(f"{r['V'][d]:7.0f} V" for d in ds))
