"""Every model on one sheet, at ONE scale, so they can be compared.

Usage::

    uv run python modelos_SEM/contacto.py

The per-model pictures in ``img/`` each autoscale to their own subject,
so a 30 Å particle and a 300 Å anemone fill the same frame and look the
same size. That is the right choice for looking at one structure and
the wrong one for deciding whether the rod is long enough -- which is
the decision these are for. Here every panel is drawn at the **same
Ångström per inch**, with a shared scale bar, and the panel that does
not fit is the one that is bigger.

Drawn as an orthographic projection rather than through ``mplot3d``:
a contact sheet of twenty structures and sixty thousand atoms has to be
cheap enough to regenerate after every change, and the perspective and
the box aspect of a 3D axes are two more things that would differ
between panels and make the comparison wrong again.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

from nanocarbon_lab.blender import styles as style_defs  # noqa: E402
from nanocarbon_lab.viz.plot import ELEMENT_COLOURS  # noqa: E402

HERE = Path(__file__).resolve().parent
XYZ = HERE / "xyz"
IMG = HERE / "img"

#: Ångström across one panel. Every panel gets the same, which is the
#: whole point; a structure wider than this is drawn overflowing rather
#: than shrunk, because shrinking it is what hides the difference.
PANEL_ANGSTROM = 130.0

#: And the anemone, which is three times that. It gets its own sheet at
#: its own scale rather than forcing every other panel down to where a
#: nanotube is four pixels.
WIDE_ANGSTROM = 380.0

#: Panels per row.
COLUMNS = 4

#: Points per Ångström² for an atom's marker. Scatter sizes are in
#: points² and the panel's size in points is fixed, so this keeps an
#: atom the same physical size in every panel.
MARKER_SCALE = 7.0

#: Elements drawn at their van der Waals radius; everything else at the
#: smaller display radius. The SAME convention as the Blender render
#: (`--model auto --spacefill-element Se`): a metal or a chalcogenide
#: core is a solid, and the carbon around it is open.
#:
#: It is not cosmetic here either. Drawn at vdW the shell's carbon
#: occludes the core completely and every coated model came out as one
#: black blob -- which is exactly the comparison these sheets exist to
#: make possible.
SOLID = frozenset({"Fe", "Se"})

#: How far the shell's carbon is faded, so the core reads through it.
SHELL_ALPHA = 0.55


def projected(atoms) -> tuple[np.ndarray, np.ndarray]:
    """Positions in the structure's own two widest directions.

    Principal axes rather than x and y, so a rod lies along the panel
    instead of pointing at the reader -- the one thing a fixed camera
    would get wrong on exactly the models whose length is in question.
    Returns all three principal coordinates: which two make the picture
    is the caller's choice, because a plate seen face-on does not show
    its thickness and a plate seen edge-on does not show its habit.
    """
    positions = np.asarray(atoms.positions, dtype=float)
    centred = positions - positions.mean(axis=0)
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    return centred @ axes.T


def draw(axis, path: Path, span: float,
         view: tuple[int, int] = (0, 1)) -> tuple[int, float, float]:
    """One panel. Returns the atom count and the two measured extents."""
    atoms = read(path)
    principal = projected(atoms)
    flat = principal[:, view]
    depth = principal[:, 3 - sum(view)]
    symbols = atoms.get_chemical_symbols()
    radii = np.array([style_defs.vdw_radius(s) if s in SOLID
                      else style_defs.element_radius(s) for s in symbols])
    colours = [ELEMENT_COLOURS.get(s, "#888888") for s in symbols]
    alphas = np.array([1.0 if s in SOLID else SHELL_ALPHA for s in symbols])
    # Far atoms first, so the near ones are drawn on top. Without this
    # the core shows through its own shell wherever the shell happened
    # to be written first, which on a core-shell particle is most of it.
    order = np.argsort(depth)
    axis.scatter(flat[order, 0], flat[order, 1],
                 s=MARKER_SCALE * radii[order] ** 2,
                 c=[colours[i] for i in order],
                 linewidths=0.0, alpha=alphas[order])
    axis.set_xlim(-span / 2, span / 2)
    axis.set_ylim(-span / 2, span / 2)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for side in axis.spines.values():
        side.set_color("#cccccc")
    extent = np.ptp(flat, axis=0)
    return len(atoms), float(extent[0]), float(extent[1])


def scale_bar(axis, span: float, length: float) -> None:
    """A bar of a known length, so the sheet is readable off the page."""
    x0 = -span / 2 + 0.06 * span
    y0 = -span / 2 + 0.07 * span
    axis.plot([x0, x0 + length], [y0, y0], "-", color="#222222", lw=2.2)
    axis.text(x0 + length / 2, y0 + 0.025 * span, f"{length:g} Å",
              ha="center", va="bottom", fontsize=7, color="#222222")


def sheet(names: list[str], span: float, bar: float, out: Path,
          title: str, view: tuple[int, int] = (0, 1)) -> None:
    """One contact sheet."""
    rows = (len(names) + COLUMNS - 1) // COLUMNS
    size = 3.1
    figure, axes = plt.subplots(rows, COLUMNS,
                                figsize=(COLUMNS * size, rows * size + 0.5))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[len(names):]:
        axis.axis("off")
    for axis, name in zip(axes[:len(names)], names, strict=True):
        path = XYZ / f"{name}.xyz"
        if not path.exists():
            axis.axis("off")
            continue
        count, wide, tall = draw(axis, path, span, view)
        scale_bar(axis, span, bar)
        ratio = max(wide, tall) / max(min(wide, tall), 1e-6)
        axis.set_title(f"{name}\n{count} átomos · {wide:.0f} × {tall:.0f} Å"
                       f" · {ratio:.1f}:1",
                       fontsize=8, color="#222222", pad=4)
    figure.suptitle(f"{title}  —  todos a la misma escala "
                    f"({span:g} Å de lado)", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=130, facecolor="white")
    plt.close(figure)
    print(f"escrito {out}")


def main() -> None:
    rows = json.loads((HERE / "modelos.json").read_text(encoding="utf-8"))
    names = [row["modelo"] for row in rows]
    wide = [n for n in names if n.startswith("escena")]
    normal = [n for n in names if n not in wide]
    sheet(normal, PANEL_ANGSTROM, 25.0, IMG / "_contacto.png",
          "Modelos FeSe@C — de frente")
    # The same models edge-on. A plate seen face-on shows its habit and
    # hides its thickness, which is the number under discussion.
    sheet(normal, PANEL_ANGSTROM, 25.0, IMG / "_contacto_perfil.png",
          "Modelos FeSe@C — de perfil", view=(0, 2))
    if wide:
        sheet(wide + ["brazo_particula_dentro"], WIDE_ANGSTROM, 100.0,
              IMG / "_contacto_escenas.png", "Escenas")


if __name__ == "__main__":
    main()
