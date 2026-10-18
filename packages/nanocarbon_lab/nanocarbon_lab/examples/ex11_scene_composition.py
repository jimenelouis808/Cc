"""Example 11 — three structures in one figure, ready for Blender.

Run: ``python -m nanocarbon_lab.examples.ex11_scene_composition``

What a composition is for: a figure that shows a cage, a tube and a
coated particle side by side, which no builder produces because it is not
a structure. It is an **illustration** -- the pieces are placed where
they look right, there is no relaxed interface between any two of them,
and nothing about the gaps is physics.

Four things in the output are worth watching, and each is something a
composition can get wrong silently:

1. **No bond crosses between two items.** The bonds are each item's own,
   re-indexed. A bond search over the whole scene would join two
   structures that merely sit 8 Å apart, which is the one claim a
   composition must not make.
2. **The achieved gap against the requested one.** Spacing is measured
   between the items' *oriented* bounding boxes, so a rotated item still
   gets the gap asked for.
3. **Bonds dropped at a periodic seam.** A periodic item's bond list
   records the pair and not the cell, and a composition is finite -- so
   the far end of such a bond is not in the scene and the bond goes.
4. **The item tag per atom**, which is all ``--colour-by structure``
   reads: each piece gets its own hue with nothing else to configure.
"""

from pathlib import Path

from nanocarbon_lab import render as render_support
from nanocarbon_lab import scene
from nanocarbon_lab.builders import build_cnt
from nanocarbon_lab.builders.fullerene import build_fullerene
from nanocarbon_lab.builders.knee import build_knee_junction

OUT = Path("out/ex11_scene")
GAP = 10.0


def main() -> int:
    items = [
        scene.Item(atoms=build_fullerene(freq=1), label="C60"),
        # Turned across the row on purpose: the spacing has to follow the
        # rotation, or the gap it leaves is the gap of a different shape.
        scene.Item(atoms=build_cnt(5, 5, length=20.0), label="CNT(5,5)",
                   rotation=(0.0, 90.0, 0.0)),
        scene.Item(atoms=build_knee_junction(kind="y"), label="Y junction"),
    ]

    items = scene.arrange(items, layout="row", gap=GAP)
    atoms, report = scene.compose(items)

    print(report.summary())
    print()

    tags = atoms.get_tags()
    crossing = sum(1 for i, j in atoms.info["bonds"] if tags[i] != tags[j])
    print(f"bonds between two items     : {crossing}   (must be 0)")

    achieved = min(distance for _, _, distance in report.clearances)
    print(f"requested gap / achieved    : {GAP:.2f} / {achieved:.2f} A")

    carried = sum(len(item.atoms.info.get("bonds", [])) for item in items)
    print(f"bonds carried / recorded    : {report.bonds} / {carried}"
          f"   (the difference left a periodic cell)")
    print(f"items per atom              : {sorted({int(tag) for tag in tags})}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    paths = scene.write_scene(atoms, OUT, report)
    for kind, path in paths.items():
        print(f"wrote {kind:6s} {path}")

    # The command that turns it into an image. Printed rather than run:
    # whether Blender is installed is not this example's business, and
    # the construction is the part worth seeing.
    print()
    try:
        command = render_support.render_command(
            paths["xyz"], OUT.with_suffix(".png"), json_path=paths["json"],
            style="paper_figure", colour_by="structure",
            background="studio", blender="blender")
    except RuntimeError as exc:                 # pragma: no cover
        print(exc)
    else:
        print("to render it:")
        print("  " + " ".join(str(token) for token in command))
        print("or, without remembering any of that:")
        print(f"  nanocarbon render {paths['xyz']} --style paper_figure "
              f"--colour-by structure --background studio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
