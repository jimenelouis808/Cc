"""Figuras para el artículo: estructuras reales, renderizadas del paquete.

Nada aquí es un esquema dibujado a mano. Cada panel es una estructura que
`nanocarbon_lab` construye, con sus enlaces y su censo de anillos tal como
salen del constructor, de modo que las figuras y las cifras del texto no
pueden discrepar.

Uso:  OMP_NUM_THREADS=1 uv run python docs/articulo/figuras.py [nombre ...]
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

from nanocarbon_lab.analyse.rings import perceive_rings, ring_report  # noqa: E402
from nanocarbon_lab.utils.geometry import guess_bonds  # noqa: E402

SALIDA = Path(__file__).parent / "figuras"
SALIDA.mkdir(exist_ok=True)

#: Los mismos colores que usa la interfaz, para que la figura del artículo
#: y la pantalla digan lo mismo.
COLOR_ANILLO = {5: "#e4572e", 6: "#5b6472", 7: "#2e86ab", 8: "#f2c14e"}
COLOR_ENLACE = "#55606e"
FONDO = "white"


def _rgb(c: str) -> tuple[float, float, float]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def enlaces(atoms) -> np.ndarray:
    """Pares enlazados: los del constructor si los dejó, si no detectados."""
    pares = atoms.info.get("bonds")
    if pares is not None and len(pares):
        return np.asarray(pares, dtype=int)
    return np.asarray(guess_bonds(atoms), dtype=int)


def _segmentos(pos: np.ndarray, pares: np.ndarray, corte: float) -> list:
    """Sólo los enlaces cortos: uno que cruza la celda uniría dos lados
    opuestos de la figura con una línea recta que no es un enlace."""
    out = []
    for i, j in pares:
        a, b = pos[i], pos[j]
        if float(np.linalg.norm(b - a)) <= corte:
            out.append([a, b])
    return out


def _caras(pos: np.ndarray, anillos, corte: float, resaltar=None):
    """Polígonos de anillo, descartando los que cruzan la celda.

    `resaltar` limita el color a esos tamaños; el resto se dibuja como
    fondo. Sirve para separar una afirmación en dos paneles -- dónde
    están los pentágonos y dónde los heptágonos -- en lugar de pedir al
    lector que distinga dos colores entremezclados.
    """
    polys, colores = [], []
    for anillo in anillos:
        esquinas = pos[list(anillo)]
        if float(np.ptp(esquinas, axis=0).max()) > corte:
            continue
        tam = len(anillo)
        visible = resaltar is None or tam in resaltar
        polys.append(esquinas)
        colores.append(COLOR_ANILLO.get(tam, COLOR_ANILLO[6]) if visible
                       else COLOR_ANILLO[6])
    return polys, colores


def _traslaciones(atoms, repetir):
    """Los desplazamientos de red que hay que dibujar.

    Una celda periódica dibujada sola se lee como un fragmento roto: los
    enlaces que cruzan la frontera se cortan y el resultado parece flotar.
    Repetirla no añade información nueva, pero es lo que hace que una red
    se lea como una red.
    """
    if repetir is None or tuple(repetir) == (1, 1, 1):
        return [np.zeros(3)]
    celda = np.asarray(atoms.cell, dtype=float)
    nx, ny, nz = repetir
    out = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out.append(i * celda[0] + j * celda[1] + k * celda[2])
    return out


def panel(ax, atoms, *, caras=False, elev=22.0, azim=-60.0, corte=2.2,
          corte_anillo=6.0, grosor=0.55, titulo=None, atomos=False,
          zoom=1.0, resaltar=None, repetir=None):
    """Un panel 3D listo para publicación: sin ejes, proporción real."""
    pos = np.asarray(atoms.get_positions(), dtype=float)
    pares = enlaces(atoms)
    desplaz = _traslaciones(atoms, repetir)

    if caras:
        anillos = atoms.info.get("rings")
        if not anillos:
            anillos = perceive_rings(atoms, pares)
        base, colores = _caras(pos, anillos, corte_anillo, resaltar)
        polys = [p_ + d for d in desplaz for p_ in base]
        colores = colores * len(desplaz)
        if polys:
            # Los hexágonos son el fondo; los 5, 7 y 8 son el tema.
            alfas = [0.15 if c == COLOR_ANILLO[6] else 0.75 for c in colores]
            col = Poly3DCollection(polys, edgecolors="none")
            col.set_facecolor([(*_rgb(c), a)
                               for c, a in zip(colores, alfas, strict=True)])
            ax.add_collection3d(col)

    base_segs = _segmentos(pos, pares, corte)
    segs = [[a + d, b + d] for d in desplaz for a, b in base_segs]
    if segs:
        ax.add_collection3d(Line3DCollection(
            segs, colors=COLOR_ENLACE, linewidths=grosor, alpha=0.85))
    if atomos:
        ax.scatter(todo[:, 0], todo[:, 1], todo[:, 2], s=1.6,
                   c="#2b2b2b", depthshade=False, linewidths=0)

    # Proporciones reales por eje, no una caja cúbica: un tubo de 20 A de
    # largo y 8 de ancho metido en un cubo deja tres cuartas partes del
    # marco en blanco. Los límites son la extensión real de cada eje y la
    # relación de la caja es esa misma extensión, de modo que la figura
    # llena el marco SIN deformar la geometría.
    todo = np.vstack([pos + d for d in desplaz])
    minimo, maximo = todo.min(axis=0), todo.max(axis=0)
    centro = 0.5 * (minimo + maximo)
    extension = np.maximum(maximo - minimo, 1e-6)
    margen = 0.04 * float(extension.max())
    semi = 0.5 * extension + margen
    for eje, c, s_ in zip("xyz", centro, semi, strict=True):
        getattr(ax, f"set_{eje}lim")(c - s_, c + s_)
    # Un eje casi plano (una lámina) daría una relación degenerada.
    relacion = np.maximum(semi / semi.max(), 0.18)
    ax.set_box_aspect(tuple(relacion), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor(FONDO)
    if titulo:
        ax.set_title(titulo, fontsize=9, color="#1a1a1a", pad=2)


def figura(nombre, atoms, *, titulo=None, pie=None, **kw):
    """Guarda un panel único como PNG a 300 ppp."""
    fig = plt.figure(figsize=(4.2, 4.2), dpi=300, facecolor=FONDO)
    ax = fig.add_subplot(111, projection="3d", facecolor=FONDO)
    panel(ax, atoms, titulo=titulo, **kw)
    if pie:
        fig.text(0.5, 0.035, pie, ha="center", fontsize=7, color="#555555")
    fig.subplots_adjust(left=0, right=1, top=0.94 if titulo else 1.0,
                        bottom=0.07 if pie else 0)
    destino = SALIDA / f"{nombre}.png"
    fig.savefig(destino, facecolor=FONDO)
    plt.close(fig)
    return destino


def rejilla(nombre, paneles, *, ncols=3, pie=None):
    """Varias estructuras en una lámina."""
    n = len(paneles)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(3.5 * ncols, 3.95 * nrows), dpi=250, facecolor=FONDO)
    for k, (titulo, atoms, kw) in enumerate(paneles, start=1):
        ax = fig.add_subplot(nrows, ncols, k, projection="3d", facecolor=FONDO)
        panel(ax, atoms, titulo=titulo, **kw)
    if pie:
        fig.text(0.5, 0.012, pie, ha="center", fontsize=8, color="#555555")
    # Los títulos llevan dos líneas (nombre y censo), así que necesitan
    # sitio: con top=0.97 la fila de arriba se corta.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88,
                        bottom=0.055 if pie else 0.02, wspace=0.02, hspace=0.20)
    destino = SALIDA / f"{nombre}.png"
    fig.savefig(destino, facecolor=FONDO)
    plt.close(fig)
    return destino


def censo(atoms) -> str:
    """Una línea con el censo de anillos y el déficit de Euler."""
    pares = enlaces(atoms)
    rep = ring_report(atoms, pares)
    c = rep["counts"]
    partes = [f"{c[k]}×{k}" for k in sorted(c) if c[k]]
    return f"{len(atoms)} átomos · {' '.join(partes)} · Σ(6−n) = {rep['euler_deficit']:+d}"


CACHE = SALIDA.parent / "cache"
CACHE.mkdir(exist_ok=True)


def construir(nombre, fn):
    """Construye una estructura, o la lee del caché si ya se construyó.

    Una super-fcc tarda nueve minutos. Ajustar el ángulo de la cámara no
    debería costar nueve minutos, así que la estructura se guarda con sus
    enlaces y su censo -- que es lo que el dibujo necesita y lo que un
    .xyz corriente pierde.
    """
    import pickle
    ruta = CACHE / f"{nombre}.pkl"
    if ruta.exists():
        with open(ruta, "rb") as fh:
            return pickle.load(fh)
    atoms = fn()
    with open(ruta, "wb") as fh:
        pickle.dump(atoms, fh)
    return atoms
