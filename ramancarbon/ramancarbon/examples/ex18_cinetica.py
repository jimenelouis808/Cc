"""Cinética: cuatro experimentos, un coeficiente de difusión, y sus trampas.

    python -m ramancarbon.examples.ex18_cinetica
"""

from __future__ import annotations

import math

import numpy as np

from ramancarbon.echem.curve import Impedance
from ramancarbon.echem.eis import drt
from ramancarbon.echem.kinetics import (
    FARADAY,
    RANDLES_SEVCIK_25C,
    differential_capacity,
    fade_model,
    gitt,
    koutecky_levich,
    randles_sevcik,
    rate_capability,
    warburg_diffusion,
)
from ramancarbon.examples.demo_data import make_plateau_gcd_demo


def main() -> None:
    print("1. Randles-Ševčík: D de cómo crece el pico con √v\n")
    D, area, concentracion, n = 1e-8, 0.5, 5e-6, 1
    velocidades = np.array([0.005, 0.01, 0.02, 0.05, 0.1])
    picos = (RANDLES_SEVCIK_25C * n ** 1.5 * area * math.sqrt(D)
             * concentracion * np.sqrt(velocidades))
    resultado = randles_sevcik(velocidades, picos, concentracion, area, n)
    print(f"   puesto D = {D:.3g} cm²/s  →  {resultado.describe()}")
    print(f"   · {resultado.warnings[0][:76]}…\n")

    print("   Y lo que pasa si el proceso NO está limitado por difusión:")
    lineal = randles_sevcik(velocidades, 1e-4 * velocidades, concentracion, area)
    print(f"   {lineal.describe()[:90]}")
    print(f"   · {lineal.warnings[-1][:76]}…\n")

    print("2. GITT: D de un pulso y su relajación\n")
    pulso, Vm, masa, M, A = 600.0, 20.0, 0.002, 100.0, 1.0
    D_real = 5e-11
    razon = math.sqrt(D_real * math.pi * pulso / 4.0) * A / ((masa / M) * Vm)
    dEs = 0.05
    dEt = dEs / razon
    t = np.linspace(0.0, 2000.0, 400)
    E = np.where(t <= pulso, 3.0 - dEt * np.sqrt(np.maximum(t, 0.0) / pulso),
                 (3.0 - dEt) - (dEs - dEt)
                 * (1.0 - np.exp(-np.maximum(t - pulso, 0.0) / 60.0)))
    g = gitt(t, E, -1e-3, pulso, Vm, masa, M, A)
    print(f"   puesto D = {D_real:.3g} cm²/s  →  {g.describe()}")
    corto = gitt(t[t <= 700.0], E[t <= 700.0], -1e-3, pulso, Vm, masa, M, A)
    print("   El mismo experimento con el reposo cortado:")
    print(f"   · {[w for w in corto.warnings if 'equilibrio' in w][0][:76]}…\n")

    print("3. Warburg: D de la cola de la impedancia\n")
    f = np.logspace(3, -2, 60)
    w = 2.0 * math.pi * f
    sigma = 25.0
    espectro = Impedance(frequency=f, z=5.0 + sigma * w ** -0.5 * (1 - 1j),
                         name="warburg")
    wd = warburg_diffusion(espectro, concentracion, 1.0, n, 298.15, 1.0, 0.01)
    print(f"   σ = {sigma} Ω·s^-½  →  {wd.describe()}")
    print("   Los tres métodos dan D distintos sobre la misma muestra, y")
    print("   todos dividen por un área que casi nadie mide.\n")

    print("4. La DRT: separar lo que un Nyquist junta\n")
    frecuencia = np.logspace(5, -2, 70)
    omega = 2.0 * math.pi * frecuencia
    z = (5.0 + 10.0 / (1 + 1j * omega * 1e-3) + 25.0 / (1 + 1j * omega * 1e-1))
    resultado = drt(Impedance(frequency=frecuencia, z=z, name="dos procesos"))
    print("   puesto: τ = 1e-3 s (10 Ω) y τ = 1e-1 s (25 Ω), R_s = 5 Ω")
    print(f"   {resultado.describe()}")
    juntos = drt(Impedance(
        frequency=frecuencia,
        z=5.0 + 12.0 / (1 + 1j * omega * 1e-2) + 12.0 / (1 + 1j * omega * 2e-2),
        name="juntos"))
    print("   Dos procesos separados por un factor de dos en τ:")
    print(f"   {juntos.describe()}")
    print("   Salen como uno, que es la respuesta honrada: a esa distancia")
    print("   la separación dependería de λ más que de los datos.\n")

    print("5. dQ/dV: una curva plana convertida en picos\n")
    curva = make_plateau_gcd_demo()
    dc = differential_capacity(curva)
    print(f"   mesetas puestas: {curva.metadata['true_plateaus_v']} V")
    print(f"   {dc.describe()}")
    print(f"   · {dc.warnings[0][:76]}…\n")

    print("6. Velocidad y envejecimiento\n")
    rc = rate_capability([0.1, 0.2, 0.5, 1, 2, 5],
                         [120, 115, 105, 95, 80, 60], returned_capacity=115)
    print(f"   {rc.describe()}")
    print(f"   · {rc.warnings[0][:76]}…")
    danado = rate_capability([0.1, 0.2, 0.5, 1, 2, 5],
                             [120, 115, 105, 95, 80, 60], returned_capacity=75)
    print(f"   · {danado.warnings[0][:76]}…")

    ciclos = np.arange(1, 201)
    for nombre, capacidad in (
        ("raíz de ciclos", 100 * (1 - 0.02 * np.sqrt(ciclos))),
        ("lineal", 100 * (1 - 0.002 * ciclos)),
    ):
        fm = fade_model(ciclos, capacidad)
        print(f"   {nombre:16s} → {fm.describe()}")
        print(f"       · {fm.warnings[-1][:70]}…")

    print("\n7. Koutecky-Levich: cuántos electrones\n")
    Dd, nu, Cc, Ar, nn = 1.9e-5, 0.01, 1.2e-6, 0.196, 4
    rpm = np.array([400.0, 900.0, 1600.0, 2500.0])
    om = rpm * 2.0 * math.pi / 60.0
    j = (0.62 * nn * FARADAY * Dd ** (2 / 3) * nu ** (-1 / 6) * Cc * np.sqrt(om))
    kl = koutecky_levich(rpm, j * Ar, Ar, Cc, Dd, nu)
    print(f"   puesto n = {nn}  →  n = {kl['electrones']:.3f} "
          f"(R² = {kl['r2']:.5f})")
    print(f"   · {kl['avisos'][-1][:76]}…")


if __name__ == "__main__":
    main()
