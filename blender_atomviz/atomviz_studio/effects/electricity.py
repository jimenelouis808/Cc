"""Electric arcs, discharges and plasma cages.

Arcs are built as bevelled poly curves whose vertices come from the
midpoint-displacement generator in :mod:`atomviz_studio.core.mathutil`. Every
arc is a hot thin core plus a soft halo, which is what makes the discharge
read as light rather than as a grey noodle.

Everything is seeded: the same seed always rebuilds the same lightning.
"""

from __future__ import annotations

import math
import random

import bpy

from ..core import nodes as N
from ..core.colors import hex_to_linear, lighten
from ..core.compat import delete_objects, ensure_group, link_object, set_material_blend
from ..core.mathutil import (
    Vec3,
    add,
    branch_points,
    fibonacci_sphere,
    fractal_path,
    length,
    normalize,
    scale,
    sub,
)

ARC_COLLECTION = "AV_Electricity"


def clear_arcs() -> int:
    """Delete every arc previously created by the add-on."""
    collection = bpy.data.collections.get(ARC_COLLECTION)
    if collection is None:
        return 0
    return delete_objects(list(collection.objects))


def arc_material(name: str, color: str, strength: float, halo: bool = False) -> bpy.types.Material:
    """Build the emissive material for an arc core or its halo.

    Args:
        name: Material name (reused on rebuild).
        color: Arc colour.
        strength: Emission strength of the core; the halo uses a fraction.
        halo: When ``True`` the material fades out at grazing angles and is
            alpha-blended, producing a soft glow sleeve around the core.
    """
    material, tree = N.new_material(name)
    if not halo:
        # White-hot centre, coloured at the edges: a Fresnel-driven blend.
        hot = N.emission_shader(tree, lighten(color, 0.75), strength, (-260, 120))
        tint = N.emission_shader(tree, color, strength * 0.6, (-260, -80))
        fresnel = N.new(tree, "ShaderNodeFresnel", (-260, 300), IOR=1.2)
        mix = N.new(tree, "ShaderNodeMixShader", (40, 0))
        N.link(tree, fresnel, mix, 0, "Fac")
        N.link(tree, hot, mix, 0, 1)
        N.link(tree, tint, mix, 0, 2)
        N.link(tree, mix, N.material_output(tree), 0, "Surface")
        set_material_blend(material, False)
    else:
        emission = N.emission_shader(tree, color, strength * 0.25, (-260, -80))
        transparent = N.new(tree, "ShaderNodeBsdfTransparent", (-260, 120))
        layer = N.new(tree, "ShaderNodeLayerWeight", (-640, 0), Blend=0.35)
        falloff = N.new(tree, "ShaderNodeValToRGB", (-440, 0))
        N.ramp(falloff, [(0.0, "#000000"), (0.85, "#8f8f8f")])
        mix = N.new(tree, "ShaderNodeMixShader", (40, 0))
        N.link(tree, layer, falloff, "Facing", "Fac")
        N.link(tree, falloff, mix, "Color", "Fac")
        N.link(tree, transparent, mix, 0, 1)
        N.link(tree, emission, mix, 0, 2)
        N.link(tree, mix, N.material_output(tree), 0, "Surface")
        set_material_blend(material, True)
        # A glow sleeve must never cast a shadow (removed in EEVEE Next).
        if hasattr(material, "shadow_method"):
            try:
                material.shadow_method = "NONE"
            except (AttributeError, TypeError):
                pass
    material.diffuse_color = hex_to_linear(color)
    return material


def _curve_object(
    name: str,
    paths: list[list[Vec3]],
    thickness: float,
    material: bpy.types.Material,
    taper: bool = True,
) -> bpy.types.Object:
    """Create one curve object holding several polyline splines."""
    curve = bpy.data.curves.new(f"{name}_curve", type="CURVE")
    curve.dimensions = "3D"
    curve.fill_mode = "FULL"
    curve.bevel_depth = thickness
    curve.bevel_resolution = 2
    curve.use_fill_caps = True
    curve.resolution_u = 2

    for path in paths:
        if len(path) < 2:
            continue
        spline = curve.splines.new("POLY")
        spline.points.add(len(path) - 1)
        for i, point in enumerate(path):
            spline.points[i].co = (point[0], point[1], point[2], 1.0)
            if taper:
                # Thin at both ends, full thickness in the middle.
                t = i / max(1, len(path) - 1)
                spline.points[i].radius = 0.25 + 0.75 * math.sin(math.pi * t) ** 0.5

    obj = bpy.data.objects.new(name, curve)
    curve.materials.append(material)
    link_object(obj, ensure_group(ARC_COLLECTION))
    return obj


def build_arc(
    start: Vec3,
    end: Vec3,
    color: str = "#7dd3fc",
    thickness: float = 0.03,
    strength: float = 40.0,
    chaos: float = 0.18,
    subdivisions: int = 5,
    branches: int = 2,
    halo: bool = True,
    seed: int = 0,
    name: str = "AV_Arc",
) -> list[bpy.types.Object]:
    """Create one electric arc (plus optional halo) between two points.

    Args:
        start: Arc origin in world space.
        end: Arc target in world space.
        color: Emission colour.
        thickness: Core bevel radius in scene units.
        strength: Core emission strength.
        chaos: Lateral wander as a fraction of the arc length.
        subdivisions: Midpoint-displacement depth (detail).
        branches: Number of secondary forks.
        halo: Add a fatter, softer sleeve around the core.
        seed: Reproducibility seed.
        name: Base object name.

    Returns:
        The created objects (core first, halo second when requested).
    """
    rng = random.Random(seed)
    main = fractal_path(start, end, subdivisions=subdivisions, chaos=chaos, rng=rng)
    paths = [main]

    span = max(1e-6, length(sub(end, start)))
    for index in branch_points(main, branches, rng):
        origin = main[index]
        direction = normalize(sub(main[min(index + 3, len(main) - 1)], origin))
        wander = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
        tip = add(origin, scale(normalize(add(direction, wander)), span * rng.uniform(0.15, 0.4)))
        paths.append(fractal_path(origin, tip, subdivisions=max(2, subdivisions - 2), chaos=chaos * 1.4, rng=rng))

    core_mat = arc_material(f"AV_arc_core_{name}", color, strength, halo=False)
    created = [_curve_object(name, paths, thickness, core_mat)]
    if halo:
        halo_mat = arc_material(f"AV_arc_halo_{name}", color, strength, halo=True)
        sleeve = _curve_object(f"{name}_halo", paths, thickness * 4.0, halo_mat, taper=False)
        if hasattr(sleeve, "visible_shadow"):  # Cycles ray visibility
            sleeve.visible_shadow = False
        created.append(sleeve)
    return created


def arcs_between_atoms(
    positions: list[Vec3],
    count: int = 6,
    color: str = "#7dd3fc",
    thickness: float = 0.03,
    strength: float = 40.0,
    chaos: float = 0.18,
    branches: int = 2,
    min_span: float = 0.25,
    max_span: float = 1.3,
    seed: int = 0,
    replace: bool = True,
) -> list[bpy.types.Object]:
    """Fire arcs between randomly chosen atoms of a structure.

    Args:
        positions: World-space atom positions (see
            :meth:`~atomviz_studio.core.detect.Structure.atom_positions`).
        count: Number of arcs.
        color: Arc colour.
        thickness: Core radius.
        strength: Emission strength.
        chaos: Lateral wander fraction.
        branches: Forks per arc.
        min_span: Minimum arc length as a fraction of the structure size.
        max_span: Maximum arc length as a fraction of the structure size.
        seed: Reproducibility seed.
        replace: Delete previous arcs first.

    Returns:
        Every created object.
    """
    if replace:
        clear_arcs()
    if len(positions) < 2 or count <= 0:
        return []

    rng = random.Random(seed)
    extent = 0.0
    reference = positions[0]
    for point in positions[:: max(1, len(positions) // 200)]:
        extent = max(extent, length(sub(point, reference)))
    extent = max(extent, 1e-3)

    created: list[bpy.types.Object] = []
    for i in range(count):
        for _ in range(24):  # rejection sampling on the arc length
            a, b = rng.choice(positions), rng.choice(positions)
            span = length(sub(b, a))
            if min_span * extent <= span <= max_span * extent:
                break
        created.extend(
            build_arc(
                a,
                b,
                color=color,
                thickness=thickness,
                strength=strength,
                chaos=chaos,
                branches=branches,
                seed=seed * 1000 + i,
                name=f"AV_Arc_{i:02d}",
            )
        )
    return created


def cage_arcs(
    center: Vec3,
    radius: float,
    count: int = 10,
    color: str = "#c084fc",
    thickness: float = 0.03,
    strength: float = 45.0,
    chaos: float = 0.22,
    branches: int = 2,
    bulge: float = 1.25,
    seed: int = 0,
    replace: bool = False,
) -> list[bpy.types.Object]:
    """Wrap the structure in arcs that travel across its bounding sphere.

    Args:
        center: Structure centre.
        radius: Structure bounding radius.
        count: Number of arcs.
        color: Arc colour.
        thickness: Core radius.
        strength: Emission strength.
        chaos: Lateral wander fraction.
        branches: Forks per arc.
        bulge: How far outside the bounding sphere the arcs bow.
        seed: Reproducibility seed.
        replace: Delete previous arcs first.

    Returns:
        Every created object.
    """
    if replace:
        clear_arcs()
    rng = random.Random(seed)
    anchors = [add(center, scale(p, radius * bulge)) for p in fibonacci_sphere(max(4, count * 2), 1.0)]
    created: list[bpy.types.Object] = []
    for i in range(count):
        start = anchors[rng.randrange(len(anchors))]
        end = anchors[rng.randrange(len(anchors))]
        if length(sub(end, start)) < radius * 0.6:
            end = add(center, scale(normalize(sub(center, start)), radius * bulge))
        created.extend(
            build_arc(
                start,
                end,
                color=color,
                thickness=thickness,
                strength=strength,
                chaos=chaos,
                branches=branches,
                seed=seed * 977 + i,
                name=f"AV_Cage_{i:02d}",
            )
        )
    return created


def discharge_to_space(
    positions: list[Vec3],
    center: Vec3,
    radius: float,
    count: int = 4,
    color: str = "#a5f3fc",
    thickness: float = 0.025,
    strength: float = 35.0,
    reach: float = 2.4,
    seed: int = 0,
) -> list[bpy.types.Object]:
    """Shoot arcs from surface atoms outwards, as if the structure discharges."""
    if not positions:
        return []
    rng = random.Random(seed)
    # Prefer atoms far from the centre: they look like emission points.
    ranked = sorted(positions, key=lambda p: -length(sub(p, center)))
    surface = ranked[: max(4, len(ranked) // 5)]
    created: list[bpy.types.Object] = []
    for i in range(count):
        start = surface[rng.randrange(len(surface))]
        direction = normalize(sub(start, center))
        tip = add(start, scale(direction, radius * reach * rng.uniform(0.6, 1.0)))
        created.extend(
            build_arc(
                start,
                tip,
                color=color,
                thickness=thickness,
                strength=strength,
                chaos=0.14,
                branches=1,
                seed=seed * 613 + i,
                name=f"AV_Discharge_{i:02d}",
            )
        )
    return created


def add_flicker(
    objects: list[bpy.types.Object],
    frame_start: int = 1,
    frame_end: int = 120,
    step: int = 2,
    on_probability: float = 0.55,
    seed: int = 0,
) -> int:
    """Keyframe arc visibility so the discharge crackles over time.

    Keys use constant interpolation, so arcs pop in and out instead of fading.
    Harmless for still covers — it only matters when you render an animation.

    Args:
        objects: Arc objects to animate.
        frame_start: First frame.
        frame_end: Last frame.
        step: Frames between decisions.
        on_probability: Chance an arc is visible at each step.
        seed: Reproducibility seed.

    Returns:
        Number of objects animated.
    """
    rng = random.Random(seed)
    step = max(1, step)
    for obj in objects:
        for frame in range(frame_start, frame_end + 1, step):
            visible = rng.random() < on_probability
            obj.hide_viewport = not visible
            obj.hide_render = not visible
            obj.keyframe_insert("hide_viewport", frame=frame)
            obj.keyframe_insert("hide_render", frame=frame)
        obj.hide_viewport = False
        obj.hide_render = False
        action = getattr(getattr(obj, "animation_data", None), "action", None)
        if action is not None:
            for fcurve in action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "CONSTANT"
    return len(objects)


def arc_positions_hint(positions: list[Vec3], center: Vec3) -> Vec3:
    """Return the atom furthest from *center* — a good manual arc anchor."""
    if not positions:
        return center
    return max(positions, key=lambda p: length(sub(p, center)))


def bridge(a: bpy.types.Object, b: bpy.types.Object, **kwargs: object) -> list[bpy.types.Object]:
    """Convenience wrapper: arc between the origins of two objects."""
    start = tuple(a.matrix_world.translation)
    end = tuple(b.matrix_world.translation)
    return build_arc(start, end, **kwargs)  # type: ignore[arg-type]
