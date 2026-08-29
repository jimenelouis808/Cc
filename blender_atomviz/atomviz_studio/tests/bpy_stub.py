"""A minimal fake ``bpy`` / ``mathutils`` used by the import smoke test.

It is nowhere near a Blender emulator: it only provides what the add-on
touches **at import time** (property declarations, base classes, the version
tuple), which is enough to prove that every module imports cleanly and that
every name it references at class-body level actually exists.
"""

from __future__ import annotations

import sys
import types


class _Deferred:
    """Stand-in for the tuple Blender's property functions return."""

    def __init__(self, kind, kwargs):
        self.kind = kind
        self.kwargs = kwargs


def _property_factory(name):
    """Build a fake ``bpy.props.<name>Property`` callable."""

    def factory(*args, **kwargs):
        return _Deferred(name, kwargs)

    factory.__name__ = name
    return factory


class _Base:
    """Base class for the fake ``bpy.types`` classes."""

    bl_rna = None

    @classmethod
    def register(cls):  # pragma: no cover - never called by the smoke test
        return None


def _make_bpy() -> tuple[types.ModuleType, list[str]]:
    """Create the fake ``bpy`` package. Returns the module and its sys.modules keys."""
    bpy = types.ModuleType("bpy")

    props = types.ModuleType("bpy.props")
    for name in (
        "BoolProperty",
        "EnumProperty",
        "FloatProperty",
        "FloatVectorProperty",
        "IntProperty",
        "StringProperty",
        "PointerProperty",
        "CollectionProperty",
    ):
        setattr(props, name, _property_factory(name))

    types_module = types.ModuleType("bpy.types")
    for name in (
        "PropertyGroup",
        "Operator",
        "Panel",
        "Object",
        "Scene",
        "Context",
        "Collection",
        "Material",
        "World",
        "Node",
        "NodeTree",
        "NodeSocket",
        "NodeLink",
        "Light",
        "Mesh",
        "Curve",
    ):
        setattr(types_module, name, type(name, (_Base,), {}))

    utils = types.ModuleType("bpy.utils")
    utils.register_class = lambda cls: None
    utils.unregister_class = lambda cls: None

    app = types.ModuleType("bpy.app")
    app.version = (4, 2, 0)

    path = types.ModuleType("bpy.path")
    path.abspath = lambda value: str(value)

    bpy.props = props
    bpy.types = types_module
    bpy.utils = utils
    bpy.app = app
    bpy.path = path
    bpy.data = types.SimpleNamespace(
        materials={}, objects={}, meshes={}, curves={}, lights={}, collections={}, worlds={}
    )
    bpy.context = types.SimpleNamespace(scene=None)
    bpy.ops = types.SimpleNamespace()

    mathutils = types.ModuleType("mathutils")

    class Vector(tuple):
        """Just enough of ``mathutils.Vector`` for import-time use."""

        def __new__(cls, values=(0.0, 0.0, 0.0)):
            return super().__new__(cls, tuple(values))

    class Euler(tuple):
        """Just enough of ``mathutils.Euler`` for import-time use."""

        def __new__(cls, values=(0.0, 0.0, 0.0), order="XYZ"):
            return super().__new__(cls, tuple(values))

    mathutils.Vector = Vector
    mathutils.Euler = Euler
    mathutils.Matrix = type("Matrix", (), {})

    modules = {
        "bpy": bpy,
        "bpy.props": props,
        "bpy.types": types_module,
        "bpy.utils": utils,
        "bpy.app": app,
        "bpy.path": path,
        "mathutils": mathutils,
    }
    for key, module in modules.items():
        sys.modules[key] = module
    return bpy, list(modules)


def install() -> dict[str, object]:
    """Install the stub into :data:`sys.modules`.

    A real ``bpy`` may already be importable (the ``bpy`` pip module, or a test
    run from inside Blender). It is shadowed, not destroyed: whatever was in
    ``sys.modules`` is handed back so :func:`uninstall` can put it back
    untouched — re-importing the real ``bpy`` would try to register Blender's
    UI classes a second time and fail.

    Returns:
        The previous ``sys.modules`` entries, to pass to :func:`uninstall`.
    """
    keys = ("bpy", "bpy.props", "bpy.types", "bpy.utils", "bpy.app", "bpy.path", "mathutils")
    saved = {key: sys.modules.get(key) for key in keys}
    _make_bpy()
    return saved


def uninstall(saved: dict[str, object]) -> None:
    """Restore whatever was in :data:`sys.modules` before :func:`install`."""
    for key, module in saved.items():
        if module is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = module  # type: ignore[assignment]
