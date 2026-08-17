"""Light rigs and volumetric haze.

Rigs are built around the structure's bounding sphere, so the same preset
works for a 20-atom molecule and a 5000-atom foam: positions scale with the
radius and light power scales with the square of the distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import bpy
from mathutils import Euler, Vector

from ..core import nodes as N
from ..core.colors import hex_to_linear
from ..core.compat import ensure_group, link_object
from ..core.mathutil import Vec3, look_at_euler, orbit_position

LIGHT_COLLECTION = "AV_Lights"
HAZE_NAME = "AV_Haze"


def clear_lights() -> int:
    """Remove every light previously created by the add-on."""
    collection = bpy.data.collections.get(LIGHT_COLLECTION)
    if collection is None:
        return 0
    from ..core.compat import delete_objects

    return delete_objects(list(collection.objects))


def _light(
    name: str,
    light_type: str,
    energy: float,
    color: str,
    location: Vec3,
    target: Vec3,
    size: float = 1.0,
    spot_size: float = 0.9,
) -> bpy.types.Object:
    """Create one light aimed at *target* and put it in the add-on collection."""
    data = bpy.data.lights.new(name, type=light_type)
    data.energy = energy
    data.color = hex_to_linear(color)[:3]
    if light_type == "AREA":
        data.size = size
        data.shape = "DISK"
    elif light_type in {"POINT", "SPOT"}:
        data.shadow_soft_size = size * 0.25
    if light_type == "SPOT":
        data.spot_size = spot_size
        data.spot_blend = 0.35
    if light_type == "SUN":
        data.angle = 0.12

    obj = bpy.data.objects.new(name, data)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(look_at_euler(location, target), "XYZ")
    link_object(obj, ensure_group(LIGHT_COLLECTION))
    return obj


def _power(base: float, distance: float) -> float:
    """Scale a nominal wattage to the rig distance (inverse-square)."""
    return base * max(0.25, (distance / 4.0) ** 2)


def rig_three_point(center: Vec3, radius: float, params: dict) -> list[bpy.types.Object]:
    """Classic key / fill / rim setup. Neutral and always defensible."""
    warm = str(params.get("warm", "#fff2e0"))
    cool = str(params.get("cool", "#dbe9ff"))
    base = float(params.get("energy", 1000.0))
    d = radius * 3.2
    return [
        _light("AV_Key", "AREA", _power(base, d), warm,
               orbit_position(center, d, -35.0, 28.0), center, size=radius * 1.8),
        _light("AV_Fill", "AREA", _power(base * 0.35, d), cool,
               orbit_position(center, d * 1.15, 55.0, 8.0), center, size=radius * 2.6),
        _light("AV_Rim", "AREA", _power(base * 0.8, d), "#ffffff",
               orbit_position(center, d, 165.0, 35.0), center, size=radius * 1.2),
    ]


def rig_studio_soft(center: Vec3, radius: float, params: dict) -> list[bpy.types.Object]:
    """Big soft boxes all around — shadowless, product-photography look."""
    warm = str(params.get("warm", "#ffffff"))
    cool = str(params.get("cool", "#eef4ff"))
    base = float(params.get("energy", 800.0))
    d = radius * 3.6
    lights = [
        _light("AV_Soft_Top", "AREA", _power(base, d), warm,
               orbit_position(center, d, 0.0, 70.0), center, size=radius * 4.0),
        _light("AV_Soft_Left", "AREA", _power(base * 0.6, d), cool,
               orbit_position(center, d, -70.0, 10.0), center, size=radius * 3.2),
        _light("AV_Soft_Right", "AREA", _power(base * 0.6, d), warm,
               orbit_position(center, d, 70.0, 10.0), center, size=radius * 3.2),
        _light("AV_Soft_Back", "AREA", _power(base * 0.4, d), cool,
               orbit_position(center, d, 180.0, 25.0), center, size=radius * 3.0),
    ]
    return lights


def rig_neon_rim(center: Vec3, radius: float, params: dict) -> list[bpy.types.Object]:
    """Two saturated rim lights plus a dim key — the dark-cover workhorse."""
    color_a = str(params.get("color_a", "#22d3ee"))
    color_b = str(params.get("color_b", "#f472b6"))
    base = float(params.get("energy", 600.0))
    d = radius * 3.0
    return [
        _light("AV_Rim_A", "AREA", _power(base, d), color_a,
               orbit_position(center, d, -115.0, 22.0), center, size=radius * 1.6),
        _light("AV_Rim_B", "AREA", _power(base, d), color_b,
               orbit_position(center, d, 115.0, 18.0), center, size=radius * 1.6),
        _light("AV_Key_Dim", "AREA", _power(base * 0.22, d), "#ffffff",
               orbit_position(center, d * 1.2, -20.0, 42.0), center, size=radius * 2.4),
    ]


def rig_dramatic_top(center: Vec3, radius: float, params: dict) -> list[bpy.types.Object]:
    """Single hard top spot with a coloured kicker. Deep shadows, high drama."""
    color_a = str(params.get("color_a", "#ffffff"))
    color_b = str(params.get("color_b", "#3b82f6"))
    base = float(params.get("energy", 900.0))
    d = radius * 3.4
    return [
        _light("AV_Top_Spot", "SPOT", _power(base * 2.0, d), color_a,
               orbit_position(center, d, 10.0, 72.0), center, size=radius * 0.5, spot_size=1.1),
        _light("AV_Kicker", "AREA", _power(base * 0.45, d), color_b,
               orbit_position(center, d, 190.0, -8.0), center, size=radius * 1.4),
    ]


def rig_godrays(center: Vec3, radius: float, params: dict) -> list[bpy.types.Object]:
    """Narrow spots angled through the structure — visible only with haze on."""
    color_a = str(params.get("color_a", "#dbeafe"))
    color_b = str(params.get("color_b", "#a5f3fc"))
    base = float(params.get("energy", 1200.0))
    d = radius * 4.5
    return [
        _light("AV_Ray_A", "SPOT", _power(base, d), color_a,
               orbit_position(center, d, -45.0, 58.0), center, size=radius * 0.2, spot_size=0.45),
        _light("AV_Ray_B", "SPOT", _power(base * 0.7, d), color_b,
               orbit_position(center, d, 60.0, 44.0), center, size=radius * 0.2, spot_size=0.35),
        _light("AV_Ray_Fill", "AREA", _power(base * 0.15, d), "#94a3b8",
               orbit_position(center, d, 150.0, 5.0), center, size=radius * 2.0),
    ]


@dataclass(frozen=True)
class LightRig:
    """A registered lighting rig."""

    key: str
    label: str
    description: str
    builder: object


_RIG_LIST: tuple[LightRig, ...] = (
    LightRig("three_point", "Three Point", "Key + fill + rim. Neutral default.", rig_three_point),
    LightRig("studio_soft", "Studio Soft", "Large soft boxes, almost shadowless.", rig_studio_soft),
    LightRig("neon_rim", "Neon Rim", "Two saturated rims for dark covers.", rig_neon_rim),
    LightRig("dramatic_top", "Dramatic Top", "Hard top spot, deep shadows.", rig_dramatic_top),
    LightRig("godrays", "God Rays", "Narrow spots for volumetric shafts.", rig_godrays),
)

#: Rig key -> :class:`LightRig`.
LIGHT_RIGS: dict[str, LightRig] = {r.key: r for r in _RIG_LIST}

#: Default rig key.
DEFAULT_RIG = "three_point"


def list_rigs() -> list[tuple[str, str, str]]:
    """Return ``(key, label, description)`` triples for UI enum construction."""
    return [(r.key, r.label, r.description) for r in _RIG_LIST]


def apply_rig(
    center: Vec3,
    radius: float,
    key: str = DEFAULT_RIG,
    params: dict | None = None,
    replace: bool = True,
) -> list[bpy.types.Object]:
    """Build a lighting rig around a structure.

    Args:
        center: Structure centre in world space.
        radius: Structure bounding radius.
        key: Rig key.
        params: Rig options (``energy``, ``warm``, ``cool``, ``color_a``, ``color_b``).
        replace: Delete previously created add-on lights first.

    Returns:
        The created light objects.
    """
    if replace:
        clear_lights()
    rig = LIGHT_RIGS.get(key, LIGHT_RIGS[DEFAULT_RIG])
    return rig.builder(center, radius, params or {})  # type: ignore[operator]


def add_haze(
    center: Vec3,
    radius: float,
    density: float = 0.015,
    color: str = "#9fb8d6",
    anisotropy: float = 0.35,
    name: str = HAZE_NAME,
) -> bpy.types.Object | None:
    """Fill a box around the structure with a thin participating medium.

    This is what turns lights into visible shafts and lasers into beams. The
    domain is deliberately a box around the subject rather than a world volume:
    it renders far faster and keeps the background clean.

    Args:
        center: Structure centre.
        radius: Structure bounding radius.
        density: Scatter density; ``0`` removes the haze.
        color: Scatter tint.
        anisotropy: Forward-scattering factor (``0`` isotropic, ``0.7`` beamy).
        name: Object name (reused across calls).

    Returns:
        The haze object, or ``None`` when *density* is zero.
    """
    from ..core.compat import delete_objects

    existing = bpy.data.objects.get(name)
    if existing is not None:
        delete_objects([existing])
    if density <= 0.0:
        return None

    size = radius * 9.0
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    h = size / 2.0
    verts = [
        (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
        (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
    ]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(center)
    obj.display_type = "WIRE"
    obj.hide_select = True
    link_object(obj, ensure_group("AV_Scene"))

    material, tree = N.new_material(f"{name}_mat")
    volume = N.new(tree, "ShaderNodeVolumePrincipled", (0, 0))
    volume.inputs["Color"].default_value = hex_to_linear(color)
    volume.inputs["Density"].default_value = density
    if "Anisotropy" in volume.inputs:
        volume.inputs["Anisotropy"].default_value = anisotropy
    N.link(tree, volume, N.material_output(tree), 0, "Volume")
    mesh.materials.append(material)
    return obj


def sun_from_camera(center: Vec3, radius: float, color: str = "#ffffff", energy: float = 2.0) -> bpy.types.Object:
    """Add a sun light for a crisp, directional key. Useful for outdoor looks."""
    location = orbit_position(center, radius * 6.0, -30.0, 55.0)
    data = bpy.data.lights.new("AV_Sun", type="SUN")
    data.energy = energy
    data.color = hex_to_linear(color)[:3]
    data.angle = 0.08
    obj = bpy.data.objects.new("AV_Sun", data)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(look_at_euler(location, center), "XYZ")
    link_object(obj, ensure_group(LIGHT_COLLECTION))
    return obj
