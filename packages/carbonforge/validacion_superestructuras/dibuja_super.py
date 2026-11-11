"""Render the saved GCMC snapshots of the three superstructures.

One figure per (framework, pressure), so the pictures can be read as a
series: the same framework at rising pressure shows where the hydrogen
goes first and where it only goes once forced.  The framework is drawn
with its bonds and the adsorbate in a contrasting colour, which is what
`adsorption.snapshot` is for.

The view is chosen per framework rather than globally: a sheet is
pointless seen face-on and a cage is pointless seen edge-on.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ase.io import read

from carbonforge.adsorption.snapshot import (
    ADSORBATE_TAG,
    FRAMEWORK_TAG,
    plot_snapshot,
)

# A sheet is read along the stacking direction's perpendicular; the two
# big hollow objects are read down an axis, so the cavity is visible.
VIEWS = {"supergrafeno": "y", "supertubo": "z", "superfulereno": "z"}


def render(source: Path, destination: Path, view: str) -> None:
    """Draw one snapshot: framework with bonds, adsorbate as spheres."""
    atoms = read(source)
    tags = atoms.get_tags()
    framework = atoms[tags == FRAMEWORK_TAG]
    sites = atoms.get_positions()[tags == ADSORBATE_TAG]
    figure = plot_snapshot(framework, sites, view=view, figsize=(5.0, 5.0))
    figure.savefig(destination, dpi=150, bbox_inches="tight",
                   facecolor=figure.get_facecolor())
    print(f"{destination.name}: {len(framework)} C, {len(sites)} H2")


def main() -> int:
    """Render every snapshot each isotherm folder holds."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    out = root / "img"
    out.mkdir(exist_ok=True)
    for name, view in VIEWS.items():
        folder = root / f"iso_{name}" / "instantaneas"
        if not folder.is_dir():
            continue
        for snapshot in sorted(folder.glob("*.extxyz")):
            pressure = snapshot.stem.split("_")[0]
            render(snapshot, out / f"{name}_{pressure}.png", view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
