"""AtomViz Studio — cover-grade shading and VFX for Atomic Blender structures.

Install this folder as a Blender add-on (Edit > Preferences > Add-ons >
Install..., pointing at the zipped ``atomviz_studio`` directory) and the panel
appears in the 3D View sidebar under the "AtomViz" tab.

The add-on never touches atomic coordinates: it only assigns materials,
rescales the instanced spheres, and adds lights, world shaders, arcs, beams
and compositor nodes around them. Your structure stays exactly as imported.
"""

bl_info = {
    "name": "AtomViz Studio (Atomic Blender covers & VFX)",
    "author": "nanocarbon_lab contributors",
    "version": (1, 0, 0),
    "blender": (3, 3, 0),
    "location": "View3D > Sidebar (N) > AtomViz",
    "description": (
        "Journal-cover shading, palettes, backgrounds, electricity and laser "
        "effects for structures imported with Atomic Blender (PDB/XYZ)."
    ),
    "warning": "",
    "doc_url": "",
    "category": "Material",
}

import importlib  # noqa: E402  (bl_info must be the first statement of the file)
import sys  # noqa: E402

try:  # The bpy-free half (core.elements, core.palettes, ...) is unit tested
    import bpy  # noqa: F401  outside Blender, where importing bpy must not fail.

    HAS_BPY = True
except ModuleNotFoundError:  # pragma: no cover - only outside Blender
    HAS_BPY = False

# Reload every submodule when the add-on is re-registered from the text editor,
# otherwise Blender keeps the first import for the whole session.
_SUBMODULES = (
    "core.colors",
    "core.elements",
    "core.mathutil",
    "core.palettes",
    "core.presets",
    "core.compat",
    "core.nodes",
    "core.detect",
    "materials.styles",
    "materials.apply",
    "effects.backgrounds",
    "effects.lighting",
    "effects.electricity",
    "effects.lasers",
    "effects.postfx",
    "scene.camera",
    "scene.render",
    "looks.apply",
    "ui.props",
    "ui.operators",
    "ui.panels",
)

if HAS_BPY:
    for _name in _SUBMODULES:
        _full = f"{__name__}.{_name}"
        if _full in sys.modules:
            importlib.reload(sys.modules[_full])

    from .ui import operators, panels, props  # noqa: E402  (must follow the reload loop)

    def register():
        """Register properties, operators and panels."""
        props.register()
        operators.register()
        panels.register()

    def unregister():
        """Unregister everything, newest first."""
        panels.unregister()
        operators.unregister()
        props.unregister()

else:  # pragma: no cover - importing the package outside Blender

    def register():
        """Raise: the add-on can only be registered inside Blender."""
        raise RuntimeError("AtomViz Studio requires Blender (bpy is not importable)")

    def unregister():
        """Raise: the add-on can only be unregistered inside Blender."""
        raise RuntimeError("AtomViz Studio requires Blender (bpy is not importable)")


if __name__ == "__main__":  # pragma: no cover - "Run Script" in the text editor
    register()
