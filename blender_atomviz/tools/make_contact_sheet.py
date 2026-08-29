"""Tile the per-look previews into one labelled contact sheet.

Run after ``render_look_sheet.py``::

    python tools/make_contact_sheet.py --in /tmp/looks --out /tmp/looks/sheet.png

Requires Pillow (``pip install pillow``); it is a developer convenience, not a
dependency of the add-on itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def build(source: Path, out: Path, columns: int = 3, label_height: int = 34) -> Path:
    """Tile every PNG in *source* into a grid and write it to *out*.

    Raises:
        FileNotFoundError: when *source* holds no PNG files.
    """
    images = sorted(p for p in source.glob("*.png") if p.resolve() != out.resolve())
    if not images:
        raise FileNotFoundError(f"no PNG files in {source}")

    tiles = [(path.stem, Image.open(path).convert("RGB")) for path in images]
    width, height = tiles[0][1].size
    rows = (len(tiles) + columns - 1) // columns
    gap = 10
    sheet = Image.new(
        "RGB",
        (columns * width + (columns + 1) * gap, rows * (height + label_height) + (rows + 1) * gap),
        (18, 18, 20),
    )
    draw = ImageDraw.Draw(sheet)

    for index, (name, image) in enumerate(tiles):
        row, column = divmod(index, columns)
        x = gap + column * (width + gap)
        y = gap + row * (height + label_height + gap)
        sheet.paste(image.resize((width, height)), (x, y))
        draw.text((x + 4, y + height + 9), name, fill=(230, 230, 235))

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main() -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Tile look previews into a contact sheet.")
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()

    written = build(args.source.resolve(), args.out.resolve(), args.columns)
    print(f"wrote {written} ({written.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
