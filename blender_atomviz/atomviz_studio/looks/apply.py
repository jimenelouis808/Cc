"""One-click cover looks: apply a whole :class:`~atomviz_studio.core.presets.Look`.

A look is applied in the same order a human would work: shading, then world,
then lights, then atmosphere, then effects, then camera, then render settings
and post-processing. Every step is the same function the individual operators
call, so nothing is hidden and any step can be re-tweaked afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import bpy

from ..core.detect import Structure
from ..core.presets import Look, get_look
from ..effects import backgrounds, electricity, lasers, lighting, postfx
from ..materials.apply import apply_style
from ..materials.styles import StyleParams
from ..scene import camera as camera_module
from ..scene import render as render_module


@dataclass
class LookReport:
    """What applying a look actually did."""

    look: str
    structure: str
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line message suitable for ``operator.report``."""
        return f"Look '{self.look}' applied to {self.structure}: " + ", ".join(self.steps)


def apply_look(
    scene: bpy.types.Scene,
    structure: Structure,
    key: str,
    seed: int = 0,
    with_camera: bool = True,
    with_render_preset: bool = True,
) -> LookReport:
    """Apply a complete cover look to *structure*.

    Args:
        scene: Target scene.
        structure: Structure to dress up.
        key: Look key (see :data:`~atomviz_studio.core.presets.LOOKS`).
        seed: Seed for the stochastic effects (arcs, laser jitter).
        with_camera: Also frame the camera.
        with_render_preset: Also apply the look's render preset.

    Returns:
        A :class:`LookReport` listing the steps that ran.
    """
    look: Look = get_look(key)
    report = LookReport(look=look.key, structure=structure.name)

    try:
        center, radius = structure.bounds()
    except ValueError as exc:
        report.warnings.append(str(exc))
        center, radius = (0.0, 0.0, 0.0), 5.0

    # 1. Shading -----------------------------------------------------------
    params = StyleParams().merged(look.style_params)
    style_report = apply_style(
        structure,
        style_key=look.style,
        palette_key=look.palette,
        params=params,
        accent=look.accent,
        stick_style=look.stick_style,
        stick_color=look.stick_color,
    )
    report.steps.append(f"materials ({style_report.objects} objects)")

    # 2. World -------------------------------------------------------------
    bg = look.background_params
    backgrounds.apply_background(
        scene,
        look.background,
        top=str(bg.get("top", "#0d1b2a")),
        bottom=str(bg.get("bottom", "#05070f")),
        accent=str(bg.get("accent", look.accent)),
        strength=float(bg.get("strength", 1.0)),  # type: ignore[arg-type]
    )
    report.steps.append(f"background {look.background}")

    # 3. Lights ------------------------------------------------------------
    lighting.apply_rig(center, radius, look.lighting, dict(look.lighting_params), replace=True)
    report.steps.append(f"lighting {look.lighting}")

    # 4. Atmosphere --------------------------------------------------------
    if look.volumetrics > 0.0:
        lighting.add_haze(center, radius, density=look.volumetrics, color=look.accent)
        report.steps.append("haze")
    else:
        lighting.add_haze(center, radius, density=0.0)

    # 5. Electricity -------------------------------------------------------
    electricity.clear_arcs()
    if look.electricity:
        spec = look.electricity
        arcs = electricity.arcs_between_atoms(
            structure.atom_positions(),
            count=int(spec.get("count", 6)),  # type: ignore[arg-type]
            color=str(spec.get("color", look.accent)),
            thickness=float(spec.get("thickness", 0.03)) * max(1.0, radius / 8.0),  # type: ignore[arg-type]
            strength=float(spec.get("strength", 40.0)),  # type: ignore[arg-type]
            chaos=float(spec.get("chaos", 0.18)),  # type: ignore[arg-type]
            branches=int(spec.get("branches", 2)),  # type: ignore[arg-type]
            seed=seed,
            replace=False,
        )
        if spec.get("cage"):
            arcs += electricity.cage_arcs(
                center,
                radius,
                count=max(4, int(spec.get("count", 6)) // 2),  # type: ignore[arg-type]
                color=str(spec.get("color", look.accent)),
                thickness=float(spec.get("thickness", 0.03)) * max(1.0, radius / 8.0),  # type: ignore[arg-type]
                strength=float(spec.get("strength", 40.0)),  # type: ignore[arg-type]
                seed=seed + 7,
            )
        if spec.get("flicker") and scene.frame_end > scene.frame_start:
            electricity.add_flicker(arcs, scene.frame_start, scene.frame_end, seed=seed)
        report.steps.append(f"electricity ({len(arcs)} objects)")

    # 6. Lasers ------------------------------------------------------------
    lasers.clear_lasers()
    if look.lasers:
        spec = look.lasers
        beams = lasers.laser_rig(
            center,
            radius,
            count=int(spec.get("count", 3)),  # type: ignore[arg-type]
            color=str(spec.get("color", "#ff2d2d")),
            beam_radius=float(spec.get("radius", 0.05)) * max(1.0, radius / 8.0),  # type: ignore[arg-type]
            strength=float(spec.get("strength", 90.0)),  # type: ignore[arg-type]
            impact=bool(spec.get("impact", True)),
            seed=seed,
            replace=False,
        )
        report.steps.append(f"lasers ({len(beams)} objects)")
        if spec.get("haze") and look.volumetrics <= 0.0:
            lighting.add_haze(center, radius, density=0.02, color=str(spec.get("color", "#ff2d2d")))
            report.steps.append("beam haze")

    # 7. Render settings ---------------------------------------------------
    if with_render_preset:
        preset = render_module.apply_render_preset(
            scene, look.render_preset, volumetrics=look.volumetrics > 0.0
        )
        report.steps.append(f"render {preset.key}")

    # 8. Camera (after the render preset: framing depends on the aspect) ----
    if with_camera:
        cam = look.camera
        try:
            camera_module.frame_structure(
                scene,
                structure,
                focal_mm=float(cam.get("focal_mm", 50.0)),  # type: ignore[arg-type]
                azimuth_deg=float(cam.get("azimuth_deg", 35.0)),  # type: ignore[arg-type]
                elevation_deg=float(cam.get("elevation_deg", 15.0)),  # type: ignore[arg-type]
                margin=float(cam.get("margin", 1.25)),  # type: ignore[arg-type]
                dof=bool(cam.get("dof", True)),
                fstop=float(cam.get("fstop", 2.8)),  # type: ignore[arg-type]
            )
            report.steps.append("camera")
        except ValueError as exc:
            report.warnings.append(f"camera not framed: {exc}")

    # 9. Post-processing ---------------------------------------------------
    fx = look.postfx
    postfx.setup(
        scene,
        glare=str(fx.get("glare", "FOG_GLOW")),
        threshold=float(fx.get("glare_threshold", 0.7)),  # type: ignore[arg-type]
        streaks=bool(fx.get("streaks", False)),
        vignette=float(fx.get("vignette", 0.3)),  # type: ignore[arg-type]
        contrast=float(fx.get("contrast", 0.0)),  # type: ignore[arg-type]
        saturation=float(fx.get("saturation", 1.0)),  # type: ignore[arg-type]
        tint=str(fx["tint"]) if fx.get("tint") else None,
    )
    report.steps.append("post-fx")
    return report


def clear_all() -> dict[str, int]:
    """Remove every object the add-on generated (lights, arcs, lasers, haze).

    Materials are left alone: they are cheap, and rebuilding a look reuses
    them by name.

    Returns:
        ``{"lights": n, "arcs": n, "lasers": n, "scene": n}``.
    """
    from ..core.compat import delete_objects

    removed = {
        "lights": lighting.clear_lights(),
        "arcs": electricity.clear_arcs(),
        "lasers": lasers.clear_lasers(),
    }
    scene_collection = bpy.data.collections.get("AV_Scene")
    generated = []
    if scene_collection is not None:
        generated = [obj for obj in scene_collection.objects if obj.name.startswith(("AV_Haze", "AV_Backdrop"))]
    removed["scene"] = delete_objects(generated)
    return removed
