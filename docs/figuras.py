"""Figuras 2D de cada celda con los electrodos: cortes y vista superior.

    python docs/figuras.py [--vistas3d DIR]

Escribe out/figuras/<celda>.png. Cada figura tiene dos cortes que pasan por los
electrodos y una vista superior en la que cada electrodo se corta a la altura de su
propio eje (están a distintas alturas). Con --vistas3d, el primer panel es la
captura del visor 3D (DIR/stage_<celda>.png, la toma docs/visor/captura.js).
"""
import argparse
import math
import os
import sys

import cadquery as cq
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules import assemblies as A  # noqa: E402

# planos de corte: (tipo, valor, título). tipo "x"/"y" = plano vertical; "d" = diagonal x = y
PANELS = {
    "raman_sellada": [("x", 0.0, "Corte x = 0: RE (−y) y CE (+y)"),
                      ("y", 0.0, "Corte y = 0: entrada (−x) y salida (+x)")],
    "raman_pinza": [("y", 0.0, "Corte y = 0: pinza CS945 (−x) y salida (+x)"),
                    ("x", 0.0, "Corte x = 0: RE (−y)")],
    "xrd": [("x", 0.0, "Corte x = 0: pinza CS945 (+y), lengüeta y muestra"),
            ("x", -6.0, "Corte x = −6: RE en el reservorio (−y)")],
    "chladni": [("d", 0.0, "Corte diagonal x = y: RE (esquina) y piezo"),
                ("y", 0.0, "Corte y = 0: cable del CE (−x) y pinza en la lengüeta (+x)")],
    "epd": [("y", 0.0, "Corte y = 0: canal, entrada y salida"),
            ("y", 5.0, "Corte y = +5: pinza del pad A")],
}


def _wire_pts(w, proj):
    from OCP.BRepTools import BRepTools_WireExplorer
    ex = BRepTools_WireExplorer(w.wrapped)
    out = []
    while ex.More():
        e = cq.Edge(ex.Current())
        ts = np.linspace(0, 1, 2 if e.geomType() == "LINE" else 24)
        if ex.Orientation().name == "TopAbs_REVERSED":
            ts = ts[::-1]
        out += [proj(e.positionAt(t)) for t in ts]
        ex.Next()
    return out


def _cutter(kind, value):
    """Losa delgada en el plano de corte y la proyección a 2D."""
    if kind == "x":
        return (cq.Solid.makeBox(0.002, 400, 400, cq.Vector(value - 0.001, -200, -200)),
                lambda v: (v.y, v.z), "x")
    if kind == "y":
        return (cq.Solid.makeBox(400, 0.002, 400, cq.Vector(-200, value - 0.001, -200)),
                lambda v: (v.x, v.z), "y")
    if kind == "z":
        return (cq.Solid.makeBox(400, 400, 0.002, cq.Vector(-200, -200, value - 0.001)),
                lambda v: (v.x, v.y), "z")
    slab = cq.Solid.makeBox(400, 0.002, 400, cq.Vector(-200, -0.001, -200)).rotate(
        cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 45)
    return slab, lambda v: ((v.x + v.y) / math.sqrt(2), v.z), "d"


def _draw(ax, solid, color, cut, edge="k", lw=0.4, alpha=None):
    slab, proj, normal = cut
    try:
        sec = solid.intersect(slab)
    except Exception:  # noqa: BLE001 (un sólido que no corta el plano)
        return None
    pts_all = []
    for f in sec.Faces():
        n = f.normalAt()
        big = {"x": abs(n.x), "y": abs(n.y), "z": abs(n.z),
               "d": abs(n.y - n.x) / math.sqrt(2)}[normal]   # normal del plano x = y
        if big < 0.9:
            continue
        a = color[3] if alpha is None else alpha
        poly = _wire_pts(f.outerWire(), proj)
        ax.add_patch(Polygon(poly, fc=color[:3], ec=edge, lw=lw, alpha=max(a, 0.35)))
        for w in f.innerWires():
            ax.add_patch(Polygon(_wire_pts(w, proj), fc="white", ec=edge, lw=lw))
        pts_all += poly
    return np.array(pts_all) if pts_all else None


def _instrument(name):
    for key, label in (("RE", "RE"), ("CE", "CE"), ("CS945", "CS945"),
                       ("double knob", "double knob")):
        if name.startswith(key):
            return label + (" " + name[name.find("(pad"):] if "(pad" in name else "")
    return name


def _anchor(ax, pts, kind, parts, reach=20.0):
    """Punto de la etiqueta: el centro de la parte del electrodo que queda en el encuadre."""
    bb = _cell_box(parts)
    r = max(abs(bb[0]), abs(bb[1]), abs(bb[2]), abs(bb[3]))
    h = {"x": (bb[2], bb[3]), "y": (bb[0], bb[1]), "d": (-r, r), "z": (bb[0], bb[1])}[kind]
    v = (bb[2], bb[3]) if kind == "z" else (bb[4], bb[5])
    vr = reach if kind == "z" else 4.0
    inside = pts[(pts[:, 0] > h[0] - reach + 2) & (pts[:, 0] < h[1] + reach - 2)
                 & (pts[:, 1] > v[0] - vr) & (pts[:, 1] < v[1] + vr - 4)]
    q = inside if len(inside) else pts
    return q[:, 0].mean(), q[:, 1].max()


def _label(ax, text, xy, color):
    ax.annotate(text, xy, xytext=(0, 14), textcoords="offset points", ha="center",
                fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1.2),
                arrowprops=dict(arrowstyle="-", color=color, lw=1))


def _cell_box(parts):
    """Caja de las piezas de la celda (sin electrodos ni conectores)."""
    bb = None
    for n, s, _, g in parts:
        if g in (A.CELL, A.SAMPLE):
            b = s.BoundingBox()
            bb = [b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax] if bb is None else [
                min(bb[0], b.xmin), max(bb[1], b.xmax), min(bb[2], b.ymin),
                max(bb[3], b.ymax), min(bb[4], b.zmin), max(bb[5], b.zmax)]
    return bb


def _crop(ax, kind, bb, reach=20.0, pad=4.0):
    """Encuadra la celda más `reach` mm de electrodo a cada lado: el resto sigue afuera."""
    r = max(abs(bb[0]), abs(bb[1]), abs(bb[2]), abs(bb[3]))
    h = {"x": (bb[2], bb[3]), "y": (bb[0], bb[1]), "d": (-r, r), "z": (bb[0], bb[1])}[kind]
    v = (bb[2], bb[3]) if kind == "z" else (bb[4], bb[5])
    ax.set_xlim(h[0] - reach, h[1] + reach)
    ax.set_ylim(v[0] - (reach if kind == "z" else pad), v[1] + (reach if kind == "z" else pad + 6))


def section_panel(ax, parts, kind, value, title):
    cut = _cutter(kind, value)
    labels = {}
    for n, s, col, g in parts:
        edge = "k" if g != A.ELECTRODE else "#7a1f1f"
        pts = _draw(ax, s, col, cut, edge=edge, lw=0.4 if g != A.ELECTRODE else 0.9)
        if pts is not None and g == A.ELECTRODE:
            key = _instrument(n)
            labels.setdefault(key, []).append(pts)
    for key, ptss in labels.items():
        pts = np.vstack(ptss)
        _label(ax, key, _anchor(ax, pts, kind, parts), "#b03a2e")
    ax.set_aspect("equal")
    _crop(ax, kind, _cell_box(parts))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel({"x": "y (mm)", "y": "x (mm)", "d": "a lo largo de x = y (mm)"}[kind])
    ax.set_ylabel("z (mm)")


def top_panel(ax, parts, z_cell):
    """Celda cortada a z_cell; cada electrodo y conector, a la altura de su eje."""
    cell = _cutter("z", z_cell)
    # primero la celda y encima la muestra (en EPD la junta opaca taparía los peines)
    for group in (A.CELL, A.SAMPLE):
        for n, s, col, g in parts:
            if g == group:
                _draw(ax, s, col, cell, alpha=0.95 if g == A.SAMPLE else None)
    labels = {}
    for n, s, col, g in parts:
        if g not in (A.ELECTRODE, A.FLOW):
            continue
        bb = s.BoundingBox()
        vertical = (bb.zmax - bb.zmin) > max(bb.xmax - bb.xmin, bb.ymax - bb.ymin)
        z = (bb.zmin + 1.0) if vertical else (bb.zmin + bb.zmax) / 2
        pts = _draw(ax, s, col, _cutter("z", z), edge="#7a1f1f" if g == A.ELECTRODE else "k",
                    lw=0.9 if g == A.ELECTRODE else 0.4, alpha=1.0)
        if pts is not None and g == A.ELECTRODE:
            labels.setdefault(_instrument(n), []).append(pts)
    for key, ptss in labels.items():
        pts = np.vstack(ptss)
        _label(ax, key, _anchor(ax, pts, "z", parts), "#b03a2e")
    ax.set_aspect("equal")
    _crop(ax, "z", _cell_box(parts))
    ax.set_title(f"Vista superior: celda cortada a z = {z_cell:g}; "
                 "electrodos a la altura de su eje", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")


TOP_Z = {"raman_sellada": 7.0, "raman_pinza": 6.5, "xrd": -8.0, "chladni": 4.0, "epd": 0.03}


def figure(key, vistas3d=None, out_dir=None):
    from export import OUT
    out_dir = out_dir or os.path.join(OUT, "figuras")
    os.makedirs(out_dir, exist_ok=True)
    title, fn = A.CONFIGS[key]
    parts = fn()
    img = os.path.join(vistas3d, f"stage_{key.replace('_', '-')}.png") if vistas3d else None
    has3d = bool(img and os.path.exists(img))
    fig = plt.figure(figsize=(16, 12), dpi=120)
    gs = fig.add_gridspec(2, 2)
    slots = [gs[0, 1], gs[1, 0], gs[1, 1]]
    if has3d:
        ax = fig.add_subplot(gs[0, 0])
        im = plt.imread(img)
        # recorta el fondo liso de la captura
        bg = im[40, 40, :3]                       # dentro del marco de la captura
        diff = np.abs(im[..., :3] - bg).sum(axis=2) > 0.04
        rows, cols = np.where(diff[10:-10, 10:-10])
        if len(rows):
            m = 20
            im = im[max(rows.min() - m, 0):rows.max() + m + 20,
                    max(cols.min() - m, 0):cols.max() + m + 20]
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title("Vista 3D (celda transparente, cortada por la mitad)", fontsize=10)
    else:
        slots = [gs[0, 0], gs[0, 1], gs[1, :]]
    for i, (kind, value, t) in enumerate(PANELS[key]):
        section_panel(fig.add_subplot(slots[i]), parts, kind, value, t)
    top_panel(fig.add_subplot(slots[2]), parts, TOP_Z[key])
    fig.suptitle(title + " — electrodos con borde rojo", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{key}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vistas3d", default=None)
    ap.add_argument("celdas", nargs="*")
    args = ap.parse_args()
    for key in args.celdas or A.CONFIGS:
        print(figure(key, args.vistas3d))
