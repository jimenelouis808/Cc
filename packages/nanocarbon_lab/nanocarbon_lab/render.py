"""Find Blender, and build the command line that drives the render script.

The render pipeline itself lives in ``nanocarbon_lab/blender/`` and is
package *data*, not a subpackage: those modules import ``bpy``, which
exists only inside Blender. This module is the part that runs in an
ordinary interpreter -- where Blender is, which script to hand it, and
what arguments to pass -- so the GUI and the CLI reach it the same way
and neither owns the knowledge.

It used to live in :mod:`nanocarbon_lab.gui.app`, which cannot be
imported without tkinter. That made the whole render pipeline
GUI-only: ``nanocarbon render`` could not exist, and none of the command
construction could be tested on a headless machine. Nothing here imports
tkinter, matplotlib or ``bpy``.

Two ways to reach the same script, and the difference is worth knowing:

* a **Blender application**, given ``-b -P script`` (headless) or
  ``-P script`` (a window, so the scene can be adjusted by hand);
* the **``bpy`` PyPI wheel**, which is a full Blender build importable
  from a plain interpreter, so the script runs as an ordinary Python
  program. ``pip install bpy`` makes rendering work on a machine with no
  3D application installed at all, which removes the pipeline's single
  most common failure.

An application is preferred when both exist, so there is no doubt which
Blender produced an image -- and an interactive session needs one, since
the wheel has no window to open.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import shutil
import sys
from pathlib import Path

__all__ = ["BLENDER_HELP", "find_blender", "has_bpy", "render_command",
           "camera_module", "effect_catalogue", "script_path",
           "style_catalogue"]

#: What to tell a user who has neither. Shared, so the GUI dialog and
#: the CLI's refusal cannot drift apart.
BLENDER_HELP = (
    "Could not find Blender.\n"
    "  Easiest fix : pip install bpy  (Blender as a Python module, no "
    "separate application needed)\n"
    "  Or          : install Blender and set the BLENDER environment "
    "variable to its executable\n"
    "  Windows     : usually C:\\Program Files\\Blender Foundation\\"
    "Blender 4.x\\blender.exe"
)


def has_bpy() -> bool:
    """Is Blender importable as a Python module in this interpreter?

    Checked with ``find_spec`` rather than an import: importing Blender
    costs hundreds of megabytes of process memory, and this is called
    while merely drawing a label or deciding what to print.
    """
    try:
        return importlib.util.find_spec("bpy") is not None
    except (ImportError, ValueError):
        return False


def find_blender() -> str | None:
    """Locate a Blender executable across platforms.

    ``PATH`` alone is not enough: the Windows installer does not add
    Blender to ``PATH``, and the macOS build lives inside an ``.app``
    bundle, so on both ``shutil.which`` finds nothing even though Blender
    is installed. Standard install locations are searched too, newest
    version first.

    Set the ``BLENDER`` environment variable to override the search
    entirely, which is what a portable, Steam or Flatpak installation
    needs.

    Returns
    -------
    str or None
        Path to the executable, or ``None`` if nothing was found.
    """
    override = os.environ.get("BLENDER")
    if override and os.path.exists(override):
        return override

    for name in ("blender", "blender.exe"):
        found = shutil.which(name)
        if found:
            return found

    # Built with os.path.join, not pathlib: these are glob patterns
    # (plain strings), and pathlib would refuse to model a Windows path
    # on POSIX, which would also make this function untestable
    # off-Windows.
    patterns: list[str] = []
    if os.name == "nt":
        for root in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ):
            patterns.append(os.path.join(root, "Blender Foundation",
                                         "Blender*", "blender.exe"))
            patterns.append(os.path.join(root, "Blender*", "blender.exe"))
    elif sys.platform == "darwin":
        patterns += [
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender/Blender.app/Contents/MacOS/Blender",
            os.path.expanduser(
                "~/Applications/Blender.app/Contents/MacOS/Blender"),
        ]
    else:
        patterns += [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/snap/bin/blender",
            "/var/lib/flatpak/exports/bin/org.blender.Blender",
            os.path.expanduser(
                "~/.local/share/flatpak/exports/bin/org.blender.Blender"),
            "/opt/blender*/blender",
        ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    # Newest-looking install first ("Blender 4.2" sorts above "Blender 3.6").
    candidates.sort(reverse=True)
    return candidates[0]


def script_path(name: str = "render_cnt") -> Path:
    """Where a render script is, inside the installed package.

    Resolved against this module rather than against a repository root,
    which does not exist once the package is installed -- that was the
    bug that left the GUI's render button dead for every installed copy.
    """
    return Path(__file__).resolve().parent / "blender" / f"{name}.py"


def render_command(
    xyz: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    json_path: str | os.PathLike[str] | None = None,
    style: str = "nature_dark",
    mode: str = "ballstick",
    colour_by: str | None = None,
    background: str | None = None,
    finish: str | None = None,
    model: str | None = None,
    detail: str | None = None,
    engine: str | None = None,
    floor: bool | None = None,
    resolution: tuple[int, int] | None = None,
    quality: str | None = None,
    samples: int | None = None,
    bond_finish: str | None = None,
    bond_colour: str | None = None,
    atom_scale: float = 1.0,
    bond_scale: float = 1.0,
    view: str | None = None,
    camera_fill: float | None = None,
    lens: float | None = None,
    effects: list[str] | None = None,
    style_effects: bool = True,
    transparent: bool = False,
    save_blend: str | os.PathLike[str] | None = None,
    render: bool = True,
    blender: str | None = None,
    interactive: bool = False,
    script: str | os.PathLike[str] | None = None,
) -> list[str]:
    """The argv for one render, ready for :func:`subprocess.run`.

    Parameters that are ``None`` are left off the command line entirely
    rather than passed as a default, so the render script's own defaults
    -- and through them each style's ``default_colour_by`` and
    ``background`` -- stay the single source of truth. A GUI that sent
    ``--colour-by ring`` because that is what its combobox happened to
    show would silently override every style's own choice.

    Raises
    ------
    RuntimeError
        If no Blender can be reached at all, or if an interactive session
        was asked for and only the ``bpy`` module is available: the wheel
        has no window to open.
    """
    script = Path(script) if script is not None else script_path()
    if not Path(script).exists():
        raise RuntimeError(f"render script not found at {script}.")

    blender = blender or find_blender()
    if blender:
        # -b is background/headless; leaving it off opens a window with
        # the scene already assembled, which is what "adjust it in
        # Blender" means.
        launcher = [blender, "-P", str(script)] if interactive else \
            [blender, "-b", "-P", str(script)]
    elif interactive:
        raise RuntimeError(
            "Opening Blender needs the Blender application; the bpy module "
            "has no window.\n" + BLENDER_HELP
        )
    elif has_bpy():
        launcher = [sys.executable, str(script)]
    else:
        raise RuntimeError(BLENDER_HELP)

    cmd = [*launcher, "--", "--xyz", str(xyz), "--out", str(out),
           "--style", style, "--mode", mode]
    if json_path is not None:
        cmd += ["--json", str(json_path)]
    if colour_by is not None:
        cmd += ["--colour-by", colour_by]
    if background is not None:
        cmd += ["--background", background]
    if finish is not None and finish != "style":
        cmd += ["--finish", finish]
    if model is not None and model != "style":
        cmd += ["--model", model]
    if detail is not None and detail != "auto":
        cmd += ["--detail", detail]
    if engine is not None:
        cmd += ["--engine", engine]
    if floor is not None:
        cmd += ["--floor"] if floor else ["--no-floor"]
    if resolution is not None:
        cmd += ["--resolution", str(int(resolution[0])), str(int(resolution[1]))]
    if quality is not None:
        cmd += ["--quality", quality]
    if samples is not None:
        cmd += ["--samples", str(int(samples))]
    if bond_finish is not None and bond_finish != "style":
        cmd += ["--bond-finish", bond_finish]
    if bond_colour is not None and bond_colour != "auto":
        cmd += ["--bond-colour", bond_colour]
    if atom_scale != 1.0:
        cmd += ["--atom-scale", f"{atom_scale:g}"]
    if bond_scale != 1.0:
        cmd += ["--bond-scale", f"{bond_scale:g}"]
    if view is not None:
        cmd += ["--view", view]
    if camera_fill is not None:
        cmd += ["--camera-fill", f"{camera_fill:g}"]
    if lens is not None:
        cmd += ["--lens", f"{lens:g}"]
    for effect in effects or []:
        cmd += ["--effect", str(effect)]
    if not style_effects:
        cmd += ["--no-style-effects"]
    if transparent:
        cmd += ["--transparent-background"]
    if save_blend is not None:
        cmd += ["--save-blend", str(save_blend)]
    if not render:
        cmd += ["--no-render"]
    return cmd


_STYLES = None
_EFFECTS = None
_CAMERA = None


def effect_catalogue():
    """Import ``blender/effects.py`` and return it as a module.

    Same reason as :func:`style_catalogue`: it imports no ``bpy`` at
    module level, so the catalogue, the parameter defaults and the spec
    parser are all reachable from the window and the command line
    without Blender installed.
    """
    global _EFFECTS
    if _EFFECTS is None:
        spec = importlib.util.spec_from_file_location(
            "nanocarbon_effects", script_path("effects"))
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("nanocarbon_effects", module)
        spec.loader.exec_module(module)
        _EFFECTS = module
    return _EFFECTS


def camera_module():
    """Import ``blender/camera.py``: the view choice, without Blender."""
    global _CAMERA
    if _CAMERA is None:
        spec = importlib.util.spec_from_file_location(
            "nanocarbon_camera", script_path("camera"))
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("nanocarbon_camera", module)
        spec.loader.exec_module(module)
        _CAMERA = module
    return _CAMERA


def style_catalogue():
    """Import ``blender/styles.py`` and return it as a module.

    By path, because it is package data rather than a subpackage -- its
    siblings import ``bpy``, so an importable ``nanocarbon_lab.blender``
    would put an unimportable module in the tree. ``styles.py`` itself
    imports no ``bpy``, which is the whole reason it is a separate file:
    the catalogue can be listed, and a menu filled from it, on a machine
    with no Blender at all.

    This exists so nothing keeps its own copy of the style names. The
    GUI did, as a hardcoded list of five, and it had been wrong since
    the sixth style was written -- a menu that silently omits half the
    catalogue is the kind of drift that only a shared source of truth
    prevents.
    """
    global _STYLES
    if _STYLES is None:
        spec = importlib.util.spec_from_file_location(
            "nanocarbon_styles", script_path("styles"))
        module = importlib.util.module_from_spec(spec)
        # Registered under a private name: ``mesh_builder`` and
        # ``render_cnt`` do ``import styles``, and they run inside
        # Blender where this process's sys.modules is irrelevant -- so
        # the plain name is deliberately left free rather than shadowed.
        sys.modules.setdefault("nanocarbon_styles", module)
        spec.loader.exec_module(module)
        _STYLES = module
    return _STYLES
