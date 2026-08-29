"""Camera framing and depth of field."""

from __future__ import annotations

import bpy
from mathutils import Euler, Vector

from ..core.compat import ensure_group, link_object, set_keyframe_interpolation
from ..core.detect import Structure
from ..core.mathutil import Vec3, frame_camera

CAMERA_NAME = "AV_Camera"


def ensure_camera(scene: bpy.types.Scene, name: str = CAMERA_NAME) -> bpy.types.Object:
    """Return the add-on camera, creating it and making it active if needed."""
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "CAMERA":
        data = bpy.data.cameras.new(name)
        obj = bpy.data.objects.new(name, data)
        link_object(obj, ensure_group("AV_Scene"))
    if obj.name not in scene.objects:
        link_object(obj, ensure_group("AV_Scene"))
    scene.camera = obj
    return obj


def frame_structure(
    scene: bpy.types.Scene,
    structure: Structure,
    focal_mm: float = 50.0,
    azimuth_deg: float = 35.0,
    elevation_deg: float = 18.0,
    margin: float = 1.25,
    dof: bool = True,
    fstop: float = 2.8,
    use_existing: bool = False,
) -> tuple[bpy.types.Object, Vec3, float]:
    """Place the camera so *structure* fills the frame.

    The framing accounts for the render aspect ratio, which matters for
    portrait covers: at 2480x3508 the horizontal field of view is the limiting
    one, so the camera has to pull back further than a square render needs.

    Args:
        scene: Target scene (its resolution drives the aspect ratio).
        structure: Structure to frame.
        focal_mm: Focal length. Long lenses (70-100 mm) flatten perspective and
            look editorial; short ones (28-40 mm) exaggerate depth.
        azimuth_deg: Orbit angle; ``0`` is Blender's front view.
        elevation_deg: Height above the horizon.
        margin: Framing margin; ``1.0`` touches the frame edges.
        dof: Enable depth of field focused on the structure centre.
        fstop: Aperture; smaller values blur the background more.
        use_existing: Reuse the scene camera instead of the add-on one.

    Returns:
        ``(camera_object, center, radius)``.

    Raises:
        ValueError: when the structure has no measurable geometry.
    """
    points = structure.atom_positions()
    if not points:
        raise ValueError("structure has no geometry to frame")

    render = scene.render
    aspect = (render.resolution_x * render.pixel_aspect_x) / max(
        1e-6, render.resolution_y * render.pixel_aspect_y
    )

    camera = scene.camera if (use_existing and scene.camera) else ensure_camera(scene)
    sensor = getattr(camera.data, "sensor_width", 36.0)
    location, rotation, center, distance = frame_camera(
        points,
        focal_mm=focal_mm,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        margin=margin,
        aspect=aspect,
        sensor_mm=sensor,
    )

    camera.location = Vector(location)
    camera.rotation_mode = "XYZ"
    camera.rotation_euler = Euler(rotation, "XYZ")
    camera.data.lens = focal_mm
    camera.data.clip_start = max(0.01, distance * 0.01)
    camera.data.clip_end = max(1000.0, distance * 20.0)

    camera.data.dof.use_dof = dof
    if dof:
        camera.data.dof.focus_object = None
        camera.data.dof.focus_distance = distance
        camera.data.dof.aperture_fstop = max(0.1, fstop)

    radius = max(1e-3, structure.bounds()[1])
    return camera, center, radius


def orbit_keyframes(
    camera: bpy.types.Object,
    center: Vec3,
    frame_start: int = 1,
    frame_end: int = 120,
    turns: float = 1.0,
    name: str = "AV_Camera_Pivot",
) -> bpy.types.Object:
    """Parent the camera to an empty at *center* and spin it for a turntable.

    Returns:
        The pivot empty holding the animation.
    """
    import math

    from ..core.compat import delete_objects

    existing = bpy.data.objects.get(name)
    if existing is not None:
        delete_objects([existing])

    pivot = bpy.data.objects.new(name, None)
    pivot.empty_display_type = "PLAIN_AXES"
    pivot.location = Vector(center)
    link_object(pivot, ensure_group("AV_Scene"))

    world = camera.matrix_world.copy()
    camera.parent = pivot
    camera.matrix_parent_inverse = pivot.matrix_world.inverted()
    camera.matrix_world = world

    pivot.rotation_mode = "XYZ"
    for frame, angle in ((frame_start, 0.0), (frame_end, math.tau * turns)):
        pivot.rotation_euler.z = angle
        pivot.keyframe_insert("rotation_euler", frame=frame)
    set_keyframe_interpolation(pivot, "LINEAR")
    return pivot
