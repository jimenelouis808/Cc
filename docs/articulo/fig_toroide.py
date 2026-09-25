"""Figura: el toroide, donde la regla de disclinaciones se ve.

Un toroide es genus 1, de modo que Sigma(6-n) = 0: cada pentagono
emparejado con un heptagono, exactamente. Su ecuador exterior tiene
curvatura gaussiana positiva y el interior negativa, asi que los
pentagonos van fuera y los heptagonos dentro, y no hay otro sitio.

Nadie coloca ninguno: salen del remallado.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.simplefilter("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import figuras as F
from nanocarbon_lab.builders.toroid import build_toroid

R, r = 20.0, 5.0
a = build_toroid(major_radius=R, minor_radius=r, place_curvature=True, seed=0)
print("toroide:", F.censo(a), flush=True)

pos = np.asarray(a.get_positions(), dtype=float)
anillos = a.info.get("rings") or []
pares = F.enlaces(a)

# La medida buena es la del propio paquete: el signo de la curvatura
# gaussiana local en cada anillo, no un radio. (Una prueba de radio
# escrita a mano falla en cuanto se olvida que el toroide lleva vacío
# alrededor y su centro NO esta en el origen.)
from nanocarbon_lab.analyse.curvature import disclination_check
chk = disclination_check(pos, anillos, pares)
print(f"acuerdo: {chk['n_correct']}/{chk['n_scored']} = {100*chk['agreement']:.0f}%")
for s_, d in chk["sizes"].items():
    print(f"  anillo {s_}: n={d['count']:4d}  signo medio de K = {d['mean_sign']:+.3f}")

# Confirmacion geometrica independiente: el radio cilindrico respecto
# del eje del toro, con las posiciones centradas.
centrado = pos - pos.mean(axis=0)
def radio(anillo):
    c = centrado[list(anillo)].mean(axis=0)
    return float(np.hypot(c[0], c[1]))
r5 = np.median([radio(x) for x in anillos if len(x) == 5])
r7 = np.median([radio(x) for x in anillos if len(x) == 7])
print(f"radio mediano: pentagonos {r5:.1f} A, heptagonos {r7:.1f} A (R = {R:.0f})",
      flush=True)

fig = plt.figure(figsize=(9.6, 5.3), dpi=280, facecolor="white")
for k, (resaltar, t) in enumerate([
        ({5}, "(a) Sólo los pentágonos: el ecuador exterior"),
        ({7}, "(b) Sólo los heptágonos: el interior del agujero")], start=1):
    ax = fig.add_subplot(1, 2, k, projection="3d", facecolor="white")
    F.panel(ax, a, caras=True, elev=76, azim=-60, corte_anillo=7.0,
            grosor=0.22, zoom=1.45, titulo=t, resaltar=resaltar)
fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.14, wspace=0.01)
fig.text(0.5, 0.085, F.censo(a), ha="center", fontsize=8.6, color="#333333")
fig.text(0.5, 0.045,
         f"El signo de la curvatura gaussiana local acierta en "
         f"{chk['n_correct']} de {chk['n_scored']} disclinaciones "
         f"({100*chk['agreement']:.0f} %).  Radio mediano: {r5:.1f} Å los "
         f"pentágonos, {r7:.1f} Å los heptágonos, con R = {R:.0f} Å.",
         ha="center", fontsize=8.0, color="#555555")
d = F.SALIDA / "fig3_toroide.png"
fig.savefig(d, facecolor="white")
print("escrito", d, flush=True)
