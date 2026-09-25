"""Figura 1: las cuatro etapas del método, sobre una unión Y.

Campo implícito -> marching cubes -> remallado isotrópico -> dual.
Los cuatro paneles son el mismo objeto en cuatro momentos del mismo
cálculo, no cuatro dibujos.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.simplefilter("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import figuras as F
from nanocarbon_lab.builders import implicit as im
from nanocarbon_lab.builders import remesh as rm
from nanocarbon_lab.builders.fullerene_mesh import dual_honeycomb
from nanocarbon_lab.utils.constants import CC_BOND

VISTA = dict(elev=24.0, azim=-62.0)
GRADO_COLOR = {4: "#8e44ad", 5: "#e4572e", 6: "#c3c9d1", 7: "#2e86ab",
               8: "#f2c14e", 9: "#16a085"}


def _encuadre(ax, pts, zoom=1.15):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = 0.5 * (lo + hi)
    ext = np.maximum(hi - lo, 1e-6)
    semi = 0.5 * ext + 0.04 * ext.max()
    for e, ci, si in zip("xyz", c, semi, strict=True):
        getattr(ax, f"set_{e}lim")(ci - si, ci + si)
    ax.set_box_aspect(tuple(np.maximum(semi / semi.max(), 0.18)), zoom=zoom)
    ax.view_init(**VISTA)
    ax.set_axis_off()


def panel_malla(ax, mesh, titulo, por_grado=True):
    """Triángulos, coloreados por el grado del vértice donde procede."""
    verts, faces = mesh
    grados = np.zeros(len(verts), dtype=int)
    for f in faces:
        for v in f:
            grados[v] += 1
    tris = verts[faces]
    if por_grado:
        # Cada triángulo toma el color del grado más "anómalo" de sus
        # tres vértices: es lo que se quiere ver.
        col = []
        for f in faces:
            g = [grados[v] for v in f]
            peor = max(g, key=lambda d: abs(d - 6))
            col.append(GRADO_COLOR.get(peor, "#c3c9d1"))
    else:
        col = ["#c3c9d1"] * len(faces)
    pc = Poly3DCollection(tris, facecolors=col, edgecolors="#6b7480",
                          linewidths=0.18, alpha=0.95)
    ax.add_collection3d(pc)
    _encuadre(ax, verts)
    ax.set_title(titulo, fontsize=9.5, pad=1)


def panel_atomos(ax, pos, anillos, titulo):
    polys, colores = [], []
    for a in anillos:
        esq = pos[list(a)]
        if float(np.ptp(esq, axis=0).max()) > 6.0:
            continue
        polys.append(esq)
        colores.append(F.COLOR_ANILLO.get(len(a), F.COLOR_ANILLO[6]))
    alfas = [0.16 if c == F.COLOR_ANILLO[6] else 0.8 for c in colores]
    pc = Poly3DCollection(polys, edgecolors="#5b6472", linewidths=0.25)
    pc.set_facecolor([(*F._rgb(c), a) for c, a in zip(colores, alfas, strict=True)])
    ax.add_collection3d(pc)
    _encuadre(ax, pos)
    ax.set_title(titulo, fontsize=9.5, pad=1)


def panel_campo(ax, campo, extent, titulo):
    """El campo: su conjunto de nivel cero, por marching cubes grueso,
    dibujado liso para que se lea como superficie y no como malla."""
    verts, faces = rm.marching_cubes_mesh(campo, extent, resolution=64)
    pc = Poly3DCollection(verts[faces], facecolors="#9fb3c8",
                          edgecolors="none", alpha=0.98)
    ax.add_collection3d(pc)
    _encuadre(ax, verts)
    ax.set_title(titulo, fontsize=9.5, pad=1)


def main():
    campo, extent = im.junction_field(kind="Y", tube_radius=6.0,
                                      arm_length=22.0, blend=4.0)
    cruda = rm.marching_cubes_mesh(campo, extent, resolution=70)
    fina = rm.isotropic_remesh(cruda, target_edge=CC_BOND * np.sqrt(3.0),
                               iterations=25, field=campo)
    pos, _, anillos = dual_honeycomb(fina)

    h_crudo = rm.degree_histogram(cruda)
    h_fino = rm.degree_histogram(fina)
    print("grados crudos :", dict(sorted(h_crudo.items())), flush=True)
    print("grados finos  :", dict(sorted(h_fino.items())), flush=True)
    censo = {}
    for a in anillos:
        censo[len(a)] = censo.get(len(a), 0) + 1
    print("censo anillos :", dict(sorted(censo.items())), flush=True)
    print("Sigma(6-n)    :", sum(6 - n for n, c in censo.items() for _ in range(c)),
          flush=True)

    fig = plt.figure(figsize=(13.6, 3.7), dpi=260, facecolor="white")
    ejes = [fig.add_subplot(1, 4, k, projection="3d", facecolor="white")
            for k in range(1, 5)]
    panel_campo(ejes[0], campo, extent, "(a) Campo implícito  $f(\\mathbf{x}) = 0$")
    panel_malla(ejes[1], cruda, "(b) Marching cubes: grados 4–9")
    panel_malla(ejes[2], fina, "(c) Remallado isotrópico: grados 5–7")
    panel_atomos(ejes[3], pos, anillos, "(d) Dual: la red de carbono")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.10,
                        wspace=0.01)
    fig.text(0.5, 0.035,
             "Vértice de malla de grado $d$  →  anillo de carbono de $d$ miembros.   "
             "Naranja: 5.   Azul: 7.   Morado: 4.   Amarillo: 8.   Verde: 9.",
             ha="center", fontsize=8.2, color="#444444")
    destino = F.SALIDA / "fig1_metodo.png"
    fig.savefig(destino, facecolor="white")
    print("escrito", destino, flush=True)


if __name__ == "__main__":
    main()
