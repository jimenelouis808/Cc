"""Render settings: formats, engine, colour management and output paths."""

from __future__ import annotations

from pathlib import Path

import bpy

from ..core.compat import set_engine, set_view_transform, set_volumetrics
from ..core.presets import DEFAULT_RENDER_PRESET, RENDER_PRESETS, RenderPreset


def apply_render_preset(
    scene: bpy.types.Scene,
    key: str = DEFAULT_RENDER_PRESET,
    percentage: int = 100,
    volumetrics: bool = False,
) -> RenderPreset:
    """Apply a render preset to *scene*.

    Args:
        scene: Target scene.
        key: Preset key (see :data:`~atomviz_studio.core.presets.RENDER_PRESETS`).
        percentage: Resolution percentage, handy for quick test frames.
        volumetrics: Enable EEVEE volumetric lighting (Cycles ignores it).

    Returns:
        The preset that was applied.
    """
    preset = RENDER_PRESETS.get(key, RENDER_PRESETS[DEFAULT_RENDER_PRESET])
    render = scene.render
    render.resolution_x = preset.width
    render.resolution_y = preset.height
    render.resolution_percentage = max(1, min(400, percentage))
    render.film_transparent = preset.film_transparent
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA" if preset.film_transparent else "RGB"
    render.image_settings.color_depth = "16"
    render.image_settings.compression = 15

    set_engine(scene, preset.engine, preset.samples)
    set_view_transform(scene, preset.view_transform, preset.look)
    set_volumetrics(scene, volumetrics)
    return preset


def set_volumetrics_hint(scene: bpy.types.Scene, enabled: bool) -> None:
    """Turn EEVEE volumetrics on/off to match the presence of haze.

    Cycles renders volumes unconditionally, so this is a no-op there.
    """
    set_volumetrics(scene, enabled)


def set_output(scene: bpy.types.Scene, path: str | Path, file_format: str = "PNG") -> Path:
    """Point the render output at *path* and return the resolved path.

    Args:
        scene: Target scene.
        path: Output file (still) or directory (animation).
        file_format: Blender image format identifier.

    Returns:
        The absolute output path.
    """
    resolved = Path(bpy.path.abspath(str(path))).expanduser()
    if resolved.suffix:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    else:
        resolved.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(resolved)
    scene.render.image_settings.file_format = file_format
    return resolved


def render_still(scene: bpy.types.Scene, path: str | Path | None = None) -> Path | None:
    """Render the current frame, optionally writing it to *path*.

    Returns:
        The written path, or ``None`` when rendering to the image editor only.
    """
    target = set_output(scene, path) if path is not None else None
    bpy.ops.render.render(write_still=target is not None)
    return target


def print_report(scene: bpy.types.Scene) -> list[str]:
    """Return a short human readable summary of the current render settings."""
    render = scene.render
    width = render.resolution_x * render.resolution_percentage // 100
    height = render.resolution_y * render.resolution_percentage // 100
    engine = render.engine.replace("BLENDER_", "").replace("_", " ").title()
    samples = (
        scene.cycles.samples
        if render.engine == "CYCLES" and hasattr(scene, "cycles")
        else scene.eevee.taa_render_samples
    )
    return [
        f"{width} x {height} px ({width / 300:.1f} x {height / 300:.1f} in @ 300 dpi)",
        f"engine: {engine}, samples: {samples}",
        f"view transform: {scene.view_settings.view_transform} / {scene.view_settings.look}",
        f"film: {'transparent' if render.film_transparent else 'opaque'}",
    ]
