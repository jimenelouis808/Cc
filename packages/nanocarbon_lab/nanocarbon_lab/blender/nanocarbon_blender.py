#!/usr/bin/env blender --background --python
"""Run the whole nanocarbon_lab suite from inside Blender.

Blender ships its own Python, and that interpreter is where ``bpy``
lives -- so the render half of this package has always run there while
the building half ran outside, with a pair of files passed between them.
This script closes that gap: given a structure to build it builds it,
bundles it and renders it **in one process**, so a figure is one command
instead of three.

    blender -b -P nanocarbon_blender.py -- --check
    blender -b -P nanocarbon_blender.py -- \\
        --build "fullerene:freq=2" --style cosmos --out cover.png
    blender -b -P nanocarbon_blender.py -- \\
        --scene a.xyz b.xyz --layout row --style plasmon --out fig.png

Run it without ``-b`` to end up in a live Blender window with the scene
already assembled.

**What Blender's Python does and does not have.** It has ``numpy``. It
does not have ``ase``, ``scipy``, ``networkx`` or ``scikit-image``, and
this package needs them to *build* a structure -- so ``--build`` and
``--scene`` need them installed into Blender's own interpreter:

    <blender>/python/bin/python3.x -m pip install ase scipy networkx scikit-image

``--check`` reports exactly what is missing and prints that command with
the right path filled in, which is the whole reason it exists: the
failure is otherwise an ImportError from inside a 3D application and
gives no clue that the fix is one pip install.

Rendering an already-built bundle (``--xyz``) needs **neither**: that
path is numpy-only by design, so the common case works in a stock
Blender with nothing installed. :mod:`render_cnt` is the script for it
and this one simply forwards to it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: What each entry point needs beyond Blender itself. Read off the
#: package's own dependency list rather than guessed: the first version
#: named ase and scipy, and a build died on networkx several seconds in.
#: ``scikit-image`` is imported as ``skimage`` and only by the
#: implicit-surface builders, so it is listed for them and the hint
#: spells the install name.
BUILD_REQUIREMENTS = ("ase", "scipy", "networkx", "skimage")
#: Import name -> what pip calls it, where the two differ.
PIP_NAMES = {"skimage": "scikit-image"}
REQUIREMENTS = {
    "render": (),                    # numpy only, which Blender ships
    "build": BUILD_REQUIREMENTS,
    "scene": ("ase", "scipy", "networkx"),
}


def _missing(names) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def _package_root() -> Path | None:
    """Where ``nanocarbon_lab`` is, if this file is still inside it.

    The render scripts are package *data*, so in a normal install they
    sit under the package and the root is two directories up. Shipped
    anywhere else -- copied next to a .blend, say -- this returns None
    and only the numpy-only render path is available.
    """
    candidate = HERE.parent.parent
    return candidate if (candidate / "nanocarbon_lab").is_dir() else None


def _pip_hint(names) -> str:
    """The exact command to install into *this* Blender's Python."""
    wanted = " ".join(PIP_NAMES.get(name, name) for name in names)
    return (f"  {sys.executable} -m pip install {wanted}\n"
            "(that is Blender's own interpreter, not the system one -- "
            "installing into the system Python will not help)")


def check() -> int:
    """Report what this Blender can and cannot do."""
    import bpy

    print(f"Blender      : {bpy.app.version_string}")
    print(f"Python       : {sys.version.split()[0]} at {sys.executable}")
    root = _package_root()
    print(f"package root : {root or 'not found (render-only)'}")
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for name in ("numpy", *BUILD_REQUIREMENTS):
        spec = importlib.util.find_spec(name)
        print(f"  {name:8s} {'yes' if spec else 'NO'}")
    for job, needs in REQUIREMENTS.items():
        missing = _missing(needs)
        print(f"  {job:8s} {'ready' if not missing else 'needs ' + ', '.join(missing)}")
    missing = _missing(BUILD_REQUIREMENTS)
    if missing:
        print("\nTo build structures inside Blender:")
        print(_pip_hint(missing))
    return 0


def _ensure_package():
    """Put ``nanocarbon_lab`` on the path, or explain why it cannot be."""
    root = _package_root()
    if root is None:
        raise SystemExit(
            "This script is not inside a nanocarbon_lab checkout, so it "
            "can only render an existing bundle (--xyz). Run it from "
            "nanocarbon_lab/blender/, or pass --xyz."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    missing = _missing(BUILD_REQUIREMENTS)
    if missing:
        raise SystemExit(
            "Building a structure needs " + ", ".join(missing)
            + " inside Blender's Python:\n" + _pip_hint(missing)
        )


def list_modes() -> int:
    """Every mode ``--build`` takes, with the parameters it accepts.

    Printed rather than left to a TypeError from inside the builder: the
    mode names here are the job names (``"capped tube"``), not the CLI's
    sub-command names (``cnt-cap``), and the parameter names are the
    builder's own. Neither is guessable, and both are already written
    down in :mod:`nanocarbon_lab.jobs`.
    """
    _ensure_package()
    from nanocarbon_lab import jobs

    for mode in jobs.MODES:
        names = ", ".join(jobs.parameter_names(mode))
        print(f"{mode}\n    {names}")
    print("\nQuote a mode with a space in it: "
          "--build \"capped tube:n_body_rings=14,freq=3\"")
    return 0


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="Report what this Blender can do, and exit.")
    p.add_argument("--list-modes", action="store_true", dest="list_modes",
                   help="Print every --build mode with its parameters, "
                        "and exit.")
    p.add_argument("--build", default=None, metavar="MODE:K=V,...",
                   help="Build a structure, e.g. 'fullerene:freq=2' or "
                        "'cnt:n=6,m=6,length=20'. Needs ase and scipy "
                        "inside Blender.")
    p.add_argument("--scene", nargs="*", default=None, metavar="FILE",
                   help="Compose these structure files into one "
                        "illustration before rendering.")
    p.add_argument("--layout", default="row",
                   help="Layout for --scene.")
    p.add_argument("--gap", type=float, default=8.0)
    p.add_argument("--xyz", default=None,
                   help="Render an existing bundle. Needs nothing but "
                        "Blender.")
    p.add_argument("--work", default=None,
                   help="Where to write the intermediate bundle. "
                        "Default: beside --out.")
    args, rest = p.parse_known_args(argv)
    args.passthrough = rest
    return args


def _value(text: str):
    """A parameter value: int if it is written as one, else float, else text.

    The distinction is not cosmetic. ``freq=2`` reaching a builder as
    ``2.0`` fails inside ``range()`` several seconds into a build, with
    a TypeError that names neither the parameter nor the command line.
    A number written without a decimal point is an integer, which is
    also how a person reading ``freq=2`` understands it.
    """
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _build_structure(spec: str, work: Path) -> Path:
    """Build one structure by mode name and write its render bundle."""
    _ensure_package()
    from nanocarbon_lab import jobs
    from nanocarbon_lab.exports.xyz import write_render_bundle

    mode, _, rest = spec.partition(":")
    params: dict = {}
    for chunk in rest.split(",") if rest.strip() else []:
        key, sep, value = chunk.partition("=")
        if not sep:
            raise SystemExit(f"--build: expected key=value, got {chunk!r}.")
        params[key.strip()] = _value(value.strip())
    # Through jobs.Job, so a mode means here exactly what it means to the
    # command line and to the window -- one mapping from "what to build"
    # to builder arguments, which is the rule the rest of the package
    # already follows.
    mode = mode.strip()
    known = jobs.parameter_names(mode)
    unknown = [key for key in params if key not in known]
    if unknown:
        # The builder's own TypeError names one bad keyword and nothing
        # else; this names all of them and what the mode does take.
        raise SystemExit(
            f"--build {mode!r}: unknown parameter(s) "
            f"{', '.join(unknown)}.\n  Takes: {', '.join(known)}"
        )
    job = jobs.Job(mode=mode, params=params)
    atoms = jobs.build(job)
    work.parent.mkdir(parents=True, exist_ok=True)
    xyz, _ = write_render_bundle(atoms, work)
    print(f"built {len(atoms)} atoms -> {xyz}")
    return xyz


def _build_scene(paths, layout: str, gap: float, work: Path) -> Path:
    _ensure_package()
    from nanocarbon_lab import scene as scene_mod

    items = scene_mod.items_from_files(paths)
    items = scene_mod.arrange(items, layout=layout, gap=gap)
    atoms, report = scene_mod.compose(items)
    print(report.summary())
    written = scene_mod.write_scene(atoms, work, report)
    return written["xyz"]


def main() -> int:
    args = _parse_args()
    if args.check:
        return check()
    if args.list_modes:
        return list_modes()

    given = [name for name, value in
             (("--build", args.build), ("--scene", args.scene),
              ("--xyz", args.xyz)) if value]
    if len(given) != 1:
        raise SystemExit(
            "Give exactly one of --build, --scene or --xyz "
            f"(got {', '.join(given) or 'none'}). --check reports what "
            "this Blender can do."
        )

    out = Path(_option(args.passthrough, "--out") or "render.png")
    work = Path(args.work) if args.work else out.with_suffix("")
    if args.build:
        xyz = _build_structure(args.build, work)
    elif args.scene:
        xyz = _build_scene(args.scene, args.layout, args.gap, work)
    else:
        xyz = Path(args.xyz)

    # Forward to the render script rather than reimplementing it: it is
    # the one place the world, lights, camera and effects are assembled,
    # and a second copy of that would drift.
    sys.argv = ["render_cnt", "--",
                "--xyz", str(xyz), "--json", str(Path(xyz).with_suffix(".json")),
                "--out", str(out), *args.passthrough]
    import render_cnt

    return render_cnt.main()


def _option(tokens, name: str):
    """Read ``--name value`` out of a passthrough list, without eating it."""
    if name in tokens:
        index = tokens.index(name)
        if index + 1 < len(tokens):
            return tokens[index + 1]
    return None


if __name__ == "__main__":
    sys.exit(main())
