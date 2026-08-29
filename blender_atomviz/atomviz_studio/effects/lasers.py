"""Laser beams, volumetric shafts and impact glows.

A convincing beam is three things stacked: a hot thin core, a soft halo that
fades at grazing angles, and — crucially — a participating medium for the
light to scatter in (see :func:`atomviz_studio.effects.lighting.add_haze`).
The impact point gets its own emissive bead plus a real light so the structure
is actually lit by the beam it is being hit with.
"""

from __future__ import annotations

import random

import bpy
from mathutils import Euler, Vector

from ..core import nodes as N
from ..core.colors import hex_to_linear, lighten
from ..core.compat import delete_objects, ensure_group, link_object, set_material_blend
from ..core.mathutil import (
    Vec3,
    add,
    fibonacci_sphere,
    length,
    look_at_euler,
    normalize,
    scale,
    sub,
)

LASER_COLLECTION = "AV_Lasers"


def clear_lasers() -> int:
    """Delete every laser object previously created by the add-on."""
    collection = bpy.data.collections.get(LASER_COLLECTION)
    if collection is None:
        return 0
    return delete_objects(list(collection.objects))


def beam_material(name: str, color: str, strength: float, halo: bool = False) -> bpy.types.Material:
    """Emissive material for a beam core (``halo=False``) or its sleeve."""
    material, tree = N.new_material(name)
    if not halo:
        emission = N.emission_shader(tree, lighten(color, 0.55), strength, (-200, 0))
        N.link(tree, emission, N.material_output(tree), 0, "Surface")
        set_material_blend(material, False)
    else:
        emission = N.emission_shader(tree, color, strength * 0.12, (-220, -80))
        transparent = N.new(tree, "ShaderNodeBsdfTransparent", (-220, 120))
        layer = N.new(tree, "ShaderNodeLayerWeight", (-620, 0), Blend=0.5)
        falloff = N.new(tree, "ShaderNodeValToRGB", (-420, 0))
        N.ramp(falloff, [(0.0, "#000000"), (0.9, "#777777")])
        mix = N.new(tree, "ShaderNodeMixShader", (40, 0))
        N.link(tree, layer, falloff, "Facing", "Fac")
        N.link(tree, falloff, mix, "Color", "Fac")
        N.link(tree, transparent, mix, 0, 1)
        N.link(tree, emission, mix, 0, 2)
        N.link(tree, mix, N.material_output(tree), 0, "Surface")
        set_material_blend(material, True)
        if hasattr(material, "shadow_method"):
            try:
                material.shadow_method = "NONE"
            except (AttributeError, TypeError):
                pass
    material.diffuse_color = hex_to_linear(color)
    return material


def _beam_curve(name: str, start: Vec3, end: Vec3, radius: float, material: bpy.types.Material) -> bpy.types.Object:
    """Build a straight bevelled curve between two points."""
    curve = bpy.data.curves.new(f"{name}_curve", type="CURVE")
    curve.dimensions = "3D"
    curve.fill_mode = "FULL"
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    curve.use_fill_caps = True
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (*start, 1.0)
    spline.points[1].co = (*end, 1.0)
    curve.materials.append(material)

    obj = bpy.data.objects.new(name, curve)
    link_object(obj, ensure_group(LASER_COLLECTION))
    return obj


def _impact(name: str, position: Vec3, color: str, radius: float, strength: float) -> list[bpy.types.Object]:
    """Emissive bead plus a point light where the beam hits the structure."""
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    # Low-poly icosphere-ish blob: a cube subdivided by a Subsurf modifier keeps
    # the data tiny while rendering perfectly round at bevel scale.
    r = radius
    verts = [
        (-r, -r, -r), (r, -r, -r), (r, r, -r), (-r, r, -r),
        (-r, -r, r), (r, -r, r), (r, r, r), (-r, r, r),
    ]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    bead = bpy.data.objects.new(name, mesh)
    bead.location = Vector(position)
    modifier = bead.modifiers.new("AV_Round", "SUBSURF")
    modifier.levels = 2
    modifier.render_levels = 3
    mesh.materials.append(beam_material(f"{name}_mat", lighten(color, 0.6), strength * 2.0))
    link_object(bead, ensure_group(LASER_COLLECTION))

    light_data = bpy.data.lights.new(f"{name}_light", type="POINT")
    light_data.energy = max(10.0, strength * radius * 25.0)
    light_data.color = hex_to_linear(color)[:3]
    light_data.shadow_soft_size = radius * 2.0
    light = bpy.data.objects.new(f"{name}_light", light_data)
    light.location = Vector(position)
    link_object(light, ensure_group(LASER_COLLECTION))
    return [bead, light]


def build_laser(
    start: Vec3,
    end: Vec3,
    color: str = "#ff2d2d",
    radius: float = 0.05,
    strength: float = 90.0,
    halo: bool = True,
    impact: bool = True,
    overshoot: float = 1.0,
    name: str = "AV_Laser",
) -> list[bpy.types.Object]:
    """Create one laser beam.

    Args:
        start: Emitter position in world space.
        end: Target position (usually a point on/inside the structure).
        color: Beam colour.
        radius: Core radius in scene units.
        strength: Core emission strength.
        halo: Add the soft outer sleeve.
        impact: Add an emissive bead and a point light at *end*.
        overshoot: Extend the beam past *end* by this fraction of its length
            (``1.0`` = stops at the target, ``1.4`` = shoots through).
        name: Base object name.

    Returns:
        Every created object.
    """
    direction = sub(end, start)
    span = max(1e-6, length(direction))
    tip = add(start, scale(normalize(direction), span * max(0.05, overshoot)))

    created = [_beam_curve(name, start, tip, radius, beam_material(f"{name}_core_mat", color, strength))]
    if halo:
        sleeve = _beam_curve(
            f"{name}_halo",
            start,
            tip,
            radius * 3.5,
            beam_material(f"{name}_halo_mat", color, strength, halo=True),
        )
        # The sleeve is pure glow: it must not block light or cast shadows.
        for attr in ("visible_shadow", "visible_diffuse"):
            if hasattr(sleeve, attr):
                setattr(sleeve, attr, False)
        created.append(sleeve)
    if impact:
        created.extend(_impact(f"{name}_impact", end, color, radius * 2.2, strength))
    return created


def laser_rig(
    center: Vec3,
    radius: float,
    count: int = 3,
    color: str = "#ff2d2d",
    beam_radius: float = 0.05,
    strength: float = 90.0,
    distance: float = 6.0,
    spread: float = 0.35,
    impact: bool = True,
    halo: bool = True,
    seed: int = 0,
    replace: bool = True,
) -> list[bpy.types.Object]:
    """Aim *count* beams at the structure from evenly spread directions.

    Args:
        center: Structure centre.
        radius: Structure bounding radius.
        count: Number of beams.
        color: Beam colour.
        beam_radius: Core radius; scaled with the structure size.
        strength: Emission strength.
        distance: Emitter distance as a multiple of *radius*.
        spread: How far off-centre each beam lands, as a fraction of *radius*.
        impact: Add impact beads and lights.
        halo: Add halo sleeves.
        seed: Reproducibility seed.
        replace: Delete previous lasers first.

    Returns:
        Every created object.
    """
    if replace:
        clear_lasers()
    if count <= 0:
        return []

    rng = random.Random(seed)
    created: list[bpy.types.Object] = []
    for i, unit in enumerate(fibonacci_sphere(count, 1.0)):
        origin = add(center, scale(unit, radius * distance))
        target = add(
            center,
            (
                rng.uniform(-spread, spread) * radius,
                rng.uniform(-spread, spread) * radius,
                rng.uniform(-spread, spread) * radius,
            ),
        )
        created.extend(
            build_laser(
                origin,
                target,
                color=color,
                radius=beam_radius,
                strength=strength,
                halo=halo,
                impact=impact,
                name=f"AV_Laser_{i:02d}",
            )
        )
    return created


def add_emitter_gizmo(position: Vec3, target: Vec3, size: float, name: str = "AV_Emitter") -> bpy.types.Object:
    """Add an empty marking where a beam comes from, aimed at *target*.

    Purely an authoring aid: move the empty and rebuild to re-aim a beam.
    """
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "SINGLE_ARROW"
    obj.empty_display_size = size
    obj.location = Vector(position)
    obj.rotation_euler = Euler(look_at_euler(position, target), "XYZ")
    link_object(obj, ensure_group(LASER_COLLECTION))
    return obj


def sweep_animation(
    beams: list[bpy.types.Object],
    center: Vec3,
    frame_start: int = 1,
    frame_end: int = 120,
    degrees: float = 25.0,
    name: str = "AV_Laser_Pivot",
) -> bpy.types.Object:
    """Sweep the beams around the structure by parenting them to a pivot.

    The beams' geometry lives in world space, so rotating each object in place
    would swing it around the scene origin. Instead an empty is created at the
    structure centre, the beams are parented to it (keeping their transform)
    and the empty is keyframed.

    Only useful for animated covers / video abstracts; still renders ignore it.

    Returns:
        The pivot empty.
    """
    import math

    existing = bpy.data.objects.get(name)
    if existing is not None:
        delete_objects([existing])

    pivot = bpy.data.objects.new(name, None)
    pivot.empty_display_type = "PLAIN_AXES"
    pivot.location = Vector(center)
    link_object(pivot, ensure_group(LASER_COLLECTION))

    for obj in beams:
        world = obj.matrix_world.copy()
        obj.parent = pivot
        obj.matrix_parent_inverse = pivot.matrix_world.inverted()
        obj.matrix_world = world

    pivot.rotation_mode = "XYZ"
    for frame, angle in ((frame_start, 0.0), (frame_end, math.radians(degrees))):
        pivot.rotation_euler.z = angle
        pivot.keyframe_insert("rotation_euler", frame=frame)
    return pivot
