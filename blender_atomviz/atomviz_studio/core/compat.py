"""Blender API compatibility layer (3.3 LTS ... 4.5).

The add-on targets a wide Blender range because labs rarely upgrade in sync.
Every API that moved between versions is funnelled through this module:

* Principled BSDF socket names ("Transmission" -> "Transmission Weight", ...).
* EEVEE vs EEVEE Next engine identifier and the removal of legacy bloom.
* ``AgX`` (4.0+) vs ``Filmic`` (3.x) view transforms.
* Material blend/transparency settings.

Nothing here raises when a feature is missing: unsupported settings are
skipped and reported, so a scene built on 4.x still opens on 3.6.
"""

from __future__ import annotations

import bpy

#: Blender version currently running, e.g. ``(4, 2, 1)``.
VERSION: tuple[int, int, int] = tuple(bpy.app.version)  # type: ignore[assignment]

IS_4X: bool = VERSION >= (4, 0, 0)
IS_EEVEE_NEXT: bool = VERSION >= (4, 2, 0)

#: Collection that holds everything the add-on creates.
ROOT_COLLECTION = "AtomViz Studio"


# --------------------------------------------------------------------------
# Collections and objects
# --------------------------------------------------------------------------
def ensure_collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
    """Return the collection *name*, creating and linking it when missing.

    Args:
        name: Collection name.
        parent: Parent collection; defaults to the current scene collection.

    Returns:
        The existing or newly created collection.
    """
    parent = parent or bpy.context.scene.collection
    existing = bpy.data.collections.get(name)
    if existing is None:
        existing = bpy.data.collections.new(name)
    if existing.name not in parent.children:
        try:
            parent.children.link(existing)
        except RuntimeError:
            # Already linked somewhere else in the scene graph: fine.
            pass
    return existing


def ensure_group(name: str) -> bpy.types.Collection:
    """Return a sub-collection of :data:`ROOT_COLLECTION` named *name*."""
    return ensure_collection(name, ensure_collection(ROOT_COLLECTION))


def link_object(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    """Link *obj* to *collection* and unlink it from every other collection."""
    for other in list(obj.users_collection):
        if other is not collection:
            other.objects.unlink(obj)
    if obj.name not in collection.objects:
        collection.objects.link(obj)


def delete_objects(objects: object) -> int:
    """Delete objects, leaving their (now orphaned) data to Blender's GC.

    Returns:
        How many objects were actually removed.
    """
    removed = 0
    for obj in list(objects):  # type: ignore[arg-type]
        if obj is None or obj.name not in bpy.data.objects:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
        removed += 1
    return removed


# --------------------------------------------------------------------------
# Sockets / Principled BSDF
# --------------------------------------------------------------------------
#: Logical parameter -> candidate socket names, newest naming first.
PRINCIPLED_ALIASES: dict[str, tuple[str, ...]] = {
    "base_color": ("Base Color",),
    "metallic": ("Metallic",),
    "roughness": ("Roughness",),
    "ior": ("IOR",),
    "alpha": ("Alpha",),
    "normal": ("Normal",),
    "specular": ("Specular IOR Level", "Specular"),
    "transmission": ("Transmission Weight", "Transmission"),
    "transmission_roughness": ("Transmission Roughness",),
    "subsurface": ("Subsurface Weight", "Subsurface"),
    "subsurface_radius": ("Subsurface Radius",),
    "subsurface_color": ("Subsurface Color",),
    "coat": ("Coat Weight", "Clearcoat"),
    "coat_roughness": ("Coat Roughness", "Clearcoat Roughness"),
    "sheen": ("Sheen Weight", "Sheen"),
    "anisotropic": ("Anisotropic",),
    "emission_color": ("Emission Color", "Emission"),
    "emission_strength": ("Emission Strength",),
}


def find_input(node: bpy.types.Node, *names: str) -> bpy.types.NodeSocket | None:
    """Return the first input socket of *node* matching any of *names*."""
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    return None


def _coerce(socket: bpy.types.NodeSocket, value: object) -> object:
    """Adapt *value* to the socket type (scalar <-> colour <-> vector)."""
    socket_type = getattr(socket, "type", "VALUE")
    if socket_type == "RGBA":
        if isinstance(value, (int, float)):
            return (float(value), float(value), float(value), 1.0)
        seq = tuple(float(v) for v in value)  # type: ignore[arg-type]
        return seq if len(seq) == 4 else (*seq[:3], 1.0)
    if socket_type == "VECTOR":
        if isinstance(value, (int, float)):
            return (float(value),) * 3
        return tuple(float(v) for v in value)[:3]  # type: ignore[arg-type]
    if isinstance(value, (list, tuple)):
        return float(sum(float(v) for v in value[:3]) / 3.0)
    return float(value)  # type: ignore[arg-type]


def set_input(node: bpy.types.Node, name: str, value: object) -> bool:
    """Set an input socket by exact name. Returns ``False`` when absent."""
    socket = node.inputs.get(name)
    if socket is None:
        return False
    socket.default_value = _coerce(socket, value)  # type: ignore[assignment]
    return True


def set_principled(node: bpy.types.Node, **params: object) -> list[str]:
    """Set Principled BSDF inputs by *logical* name, across Blender versions.

    Example::

        set_principled(bsdf, base_color=(1, 0, 0, 1), transmission=1.0, coat=0.4)

    Unknown or version-removed parameters are skipped.

    Returns:
        The list of parameter names that could not be applied.
    """
    skipped: list[str] = []
    for key, value in params.items():
        names = PRINCIPLED_ALIASES.get(key, (key.replace("_", " ").title(),))
        socket = find_input(node, *names)
        if socket is None:
            skipped.append(key)
            continue
        socket.default_value = _coerce(socket, value)  # type: ignore[assignment]
    return skipped


# --------------------------------------------------------------------------
# Render engine / colour management
# --------------------------------------------------------------------------
def engine_candidates(engine: str) -> tuple[str, ...]:
    """Engine identifiers to try, best first.

    The EEVEE identifier moved twice: ``BLENDER_EEVEE`` up to 4.1,
    ``BLENDER_EEVEE_NEXT`` in 4.2-4.5, and back to ``BLENDER_EEVEE`` in 5.0.
    Rather than mapping versions, every plausible name is tried in turn.
    """
    if engine.upper().startswith("CYCLES"):
        return ("CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
    if IS_EEVEE_NEXT:
        return ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES")
    return ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES")


def engine_id(engine: str) -> str:
    """Return the first engine identifier this build actually accepts."""
    try:
        available = {
            item.identifier
            for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        }
    except (KeyError, AttributeError):  # pragma: no cover - defensive
        available = set()
    for candidate in engine_candidates(engine):
        if not available or candidate in available:
            return candidate
    return next(iter(available)) if available else "BLENDER_EEVEE"


def set_engine(scene: bpy.types.Scene, engine: str, samples: int) -> None:
    """Select the render engine and its sample count.

    The engine enum is populated by registered engines, so the identifier is
    assigned by trial: an unavailable one raises ``TypeError`` and the next
    candidate is tried.
    """
    for candidate in engine_candidates(engine):
        try:
            scene.render.engine = candidate
            break
        except TypeError:
            continue
    if scene.render.engine == "CYCLES" and hasattr(scene, "cycles"):
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        # Transparent + volumetric heavy scenes need a few extra bounces.
        scene.cycles.transparent_max_bounces = max(scene.cycles.transparent_max_bounces, 16)
        scene.cycles.volume_bounces = max(scene.cycles.volume_bounces, 2)
    else:
        eevee = scene.eevee
        eevee.taa_render_samples = samples
        for attr, value in (
            ("use_gtao", True),  # 3.x/4.0-4.1 ambient occlusion
            ("use_ssr", True),
            ("use_ssr_refraction", True),
            ("use_raytracing", True),  # EEVEE Next
        ):
            if hasattr(eevee, attr):
                try:
                    setattr(eevee, attr, value)
                except (AttributeError, TypeError):
                    pass


def iter_fcurves(target: object):
    """Yield the F-curves animating *target* (an object, light, material, ...).

    Blender 4.4 introduced slotted actions and 5.0 removed the flat
    ``action.fcurves`` accessor, so the curves now live in
    ``action.layers[].strips[].channelbags[]``. Both shapes are handled.
    """
    anim = getattr(target, "animation_data", None)
    action = getattr(anim, "action", None)
    if action is None:
        return
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy
        return
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                yield from channelbag.fcurves


def set_keyframe_interpolation(target: object, interpolation: str = "CONSTANT") -> int:
    """Set the interpolation of every keyframe animating *target*.

    Returns:
        How many keyframes were changed.
    """
    changed = 0
    for fcurve in iter_fcurves(target):
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = interpolation
            changed += 1
    return changed


def set_view_transform(scene: bpy.types.Scene, transform: str, look: str = "None") -> None:
    """Apply a view transform, falling back when the build lacks it."""
    view = scene.view_settings
    candidates = [transform, "AgX", "Filmic", "Standard"] if IS_4X else [transform, "Filmic", "Standard"]
    for name in candidates:
        try:
            view.view_transform = name
            break
        except TypeError:
            continue
    for name in (look, "None"):
        try:
            view.look = name
            break
        except TypeError:
            continue


def set_volumetrics(scene: bpy.types.Scene, enabled: bool, distance: float = 200.0) -> None:
    """Enable EEVEE volumetric lighting (no-op on Cycles, which is always on)."""
    eevee = getattr(scene, "eevee", None)
    if eevee is None:
        return
    for attr, value in (
        ("use_volumetric_lights", enabled),
        ("use_volumetric_shadows", enabled),
        ("volumetric_end", distance),
        ("volumetric_tile_size", "4"),
    ):
        if hasattr(eevee, attr):
            try:
                setattr(eevee, attr, value)
            except (AttributeError, TypeError):
                pass


def set_bloom(scene: bpy.types.Scene, enabled: bool, threshold: float = 0.8) -> bool:
    """Enable legacy EEVEE bloom. Returns ``False`` on 4.2+ (use the glare node).

    EEVEE Next removed the render-time bloom pass; the compositor Glare node in
    :mod:`atomviz_studio.effects.postfx` covers both engines.
    """
    eevee = getattr(scene, "eevee", None)
    if eevee is None or not hasattr(eevee, "use_bloom"):
        return False
    eevee.use_bloom = enabled
    if enabled and hasattr(eevee, "bloom_threshold"):
        eevee.bloom_threshold = threshold
        eevee.bloom_intensity = 0.05
        eevee.bloom_radius = 6.5
    return True


def set_material_blend(material: bpy.types.Material, transparent: bool) -> None:
    """Configure alpha blending for EEVEE across the 4.2 render-method change."""
    if transparent:
        if hasattr(material, "surface_render_method"):  # 4.2+
            material.surface_render_method = "BLENDED"
        for attr, value in (("blend_method", "BLEND"), ("shadow_method", "HASHED")):
            if hasattr(material, attr):
                try:
                    setattr(material, attr, value)
                except (AttributeError, TypeError):
                    pass
        if hasattr(material, "use_backface_culling"):
            material.use_backface_culling = False
    else:
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        if hasattr(material, "blend_method"):
            try:
                material.blend_method = "OPAQUE"
            except (AttributeError, TypeError):
                pass
