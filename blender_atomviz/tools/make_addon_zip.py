"""Package the add-on into an installable Blender zip.

Usage::

    python tools/make_addon_zip.py [--out dist/atomviz_studio.zip] [--with-tests]

The zip contains a single top-level ``atomviz_studio/`` folder, which is what
``Preferences > Add-ons > Install...`` expects. Tests, caches and editor files
are excluded by default so the installed add-on stays small.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PACKAGE = "atomviz_studio"
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".swp"}


def iter_files(root: Path, with_tests: bool):
    """Yield the files that belong in the add-on zip."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.relative_to(root).parts)
        if parts & EXCLUDED_DIRS:
            continue
        if not with_tests and "tests" in parts:
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        yield path


def build(out_path: Path, with_tests: bool = False) -> Path:
    """Write the zip and return its path.

    Raises:
        FileNotFoundError: if the package directory cannot be found.
    """
    project_root = Path(__file__).resolve().parents[1]
    package_root = project_root / PACKAGE
    if not package_root.is_dir():
        raise FileNotFoundError(f"{package_root} not found")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in iter_files(package_root, with_tests):
            archive.write(path, Path(PACKAGE) / path.relative_to(package_root))
    return out_path


def main() -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Build the AtomViz Studio add-on zip.")
    parser.add_argument("--out", type=Path, default=Path("dist/atomviz_studio.zip"))
    parser.add_argument("--with-tests", action="store_true", help="Include the test suite")
    args = parser.parse_args()

    written = build(args.out.resolve(), args.with_tests)
    size_kb = written.stat().st_size / 1024
    print(f"wrote {written} ({size_kb:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
