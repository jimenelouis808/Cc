"""Operators — one per action in the AtomViz Studio panel.

Like :mod:`atomviz_studio.ui.props`, this module must not use postponed
annotation evaluation: Blender reads operator properties from evaluated class
annotations.
"""

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator

from ..core.detect import active_structure, detect_structures
from ..effects import backgrounds, electricity, lasers, lighting, postfx
from ..looks.apply import apply_look, clear_all
from ..materials.apply import apply_radii, apply_style, describe
from ..materials.styles import StyleParams
from ..scene import camera as camera_module
from ..scene import render as render_module
from .props import hex_of

#: Message shown when the XYZ importer cannot be found.
ATOMIC_BLENDER_HINT = (
    "Atomic Blender (XYZ importer) not found. Blender 4.1 and older ship it: "
    "enable 'Import-Export: Atomic Blender PDB/XYZ' in Preferences > Add-ons. "
    "Blender 4.2+ moved it to the extensions platform: Preferences > Get "
    "Extensions, search 'Atomic Blender', install, then try again."
)


def _structure(operator, context):
    """Return the active structure, reporting an error when there is none."""
    props = context.scene.atomviz
    structure = active_structure(context, props.scope)
    if structure is None:
        operator.report({"ERROR"}, "No atomic structure found. Import an XYZ file first.")
    return structure


def _style_params(props):
    """Build :class:`StyleParams` from the scene properties."""
    return StyleParams(
        roughness=props.roughness,
        metallic=props.metallic,
        ior=props.ior,
        alpha=props.alpha,
        emission_strength=props.emission_strength,
        clearcoat=props.clearcoat,
        subsurface=props.subsurface,
        steps=props.toon_steps,
        scanlines=props.scanlines,
        emissive_dopants=props.emissive_dopants,
    )


def _bounds(operator, structure):
    """Return ``(center, radius)``, reporting a warning on empty structures."""
    try:
        return structure.bounds()
    except ValueError as exc:
        operator.report({"WARNING"}, str(exc))
        return (0.0, 0.0, 0.0), 5.0


class ATOMVIZ_OT_detect(Operator):
    """List the atomic structures AtomViz can see in the scene"""

    bl_idname = "atomviz.detect"
    bl_label = "Detect structures"
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = context.scene.atomviz
        structures = detect_structures(context, props.scope)
        if not structures:
            self.report({"WARNING"}, "No structure detected in the current scope")
            return {"CANCELLED"}
        for structure in structures:
            print(f"[AtomViz] {structure.name}: {structure.atom_count} atoms")
            for line in describe(structure):
                print(f"  - {line}")
        top = structures[0]
        self.report(
            {"INFO"},
            f"{len(structures)} structure(s); largest: {top.name} "
            f"({top.atom_count} atoms, {len(top.groups)} element(s)) - details in the console",
        )
        return {"FINISHED"}


class ATOMVIZ_OT_apply_materials(Operator):
    """Build and assign materials for every element of the structure"""

    bl_idname = "atomviz.apply_materials"
    bl_label = "Apply style & palette"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.atomviz
        structure = _structure(self, context)
        if structure is None:
            return {"CANCELLED"}
        report = apply_style(
            structure,
            style_key=props.style,
            palette_key=props.palette,
            params=_style_params(props),
            accent=hex_of(props.accent),
            stick_style=props.stick_style,
            stick_color=hex_of(props.stick_color),
        )
        if props.radius_mode != "KEEP":
            apply_radii(structure, props.radius_mode.lower(), props.radius_factor)
        self.report({"INFO"}, report.summary())
        return {"FINISHED"}


class ATOMVIZ_OT_apply_radii(Operator):
    """Rescale the atom spheres by element radius"""

    bl_idname = "atomviz.apply_radii"
    bl_label = "Apply atom radii"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.atomviz
        structure = _structure(self, context)
        if structure is None:
            return {"CANCELLED"}
        mode = "vdw" if props.radius_mode == "KEEP" else props.radius_mode.lower()
        count = apply_radii(structure, mode, props.radius_factor)
        self.report({"INFO"}, f"Rescaled {count} atom object(s) using {mode} radii")
        return {"FINISHED"}


class ATOMVIZ_OT_background(Operator):
    """Build the procedural world background"""

    bl_idname = "atomviz.background"
    bl_label = "Apply background"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.atomviz
        backgrounds.apply_background(
            context.scene,
            props.background,
            top=hex_of(props.bg_top),
            bottom=hex_of(props.bg_bottom),
            accent=hex_of(props.bg_accent),
            strength=props.bg_strength,
        )
        self.report({"INFO"}, f"Background '{props.background}' applied")
        return {"FINISHED"}


class ATOMVIZ_OT_lighting(Operator):
    """Build a lighting rig scaled to the structure"""

    bl_idname = "atomviz.lighting"
    bl_label = "Build light rig"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.atomviz
        structure = _structure(self, context)
        if structure is None:
            return {"CANCELLED"}
        center, radius = _bounds(self, structure)
        lights = lighting.apply_rig(
            center,
            radius,
            props.light_rig,
            {
                "energy": props.light_energy,
                "color_a": hex_of(props.light_color_a),
                "color_b": hex_of(props.light_color_b),
                "warm": hex_of(props.light_color_a),
                "cool": hex_of(props.light_color_b),
            },
        )
        lighting.add_haze(center, radius, density=props.haze_density, color=hex_of(props.light_color_a))
        render_module.set_volumetrics_hint(context.scene, props.haze_density > 0.0)
        self.report({"INFO"}, f"Rig '{props.light_rig}': {len(lights)} light(s)")
        return {"FINISHED"}


class ATOMVIZ_OT_electricity(Operator):
    """Generate electric arcs on or around the structure"""

    bl_idname = "atomviz.electricity"
    bl_label = "Generate arcs"
    bl_options = {"REGISTER", "UNDO"}

    replace: BoolProperty(name="Replace existing", default=True)

    def execute(self, context):
        props = context.scene.atomviz
        structure = _structure(self, context)
        if structure is None:
            return {"CANCELLED"}
        center, radius = _bounds(self, structure)
        color = hex_of(props.arc_color)
        common = {
            "color": color,
            "thickness": props.arc_thickness,
            "strength": props.arc_strength,
            "seed": props.seed,
        }
        if self.replace:
            electricity.clear_arcs()

        if props.arc_mode == "CAGE":
            arcs = electricity.cage_arcs(
                center, radius, count=props.arc_count, chaos=props.arc_chaos,
                branches=props.arc_branches, **common
            )
        elif props.arc_mode == "DISCHARGE":
            arcs = electricity.discharge_to_space(
                structure.atom_positions(), center, radius, count=props.arc_count, **common
            )
        else:
            arcs = electricity.arcs_between_atoms(
                structure.atom_positions(), count=props.arc_count, chaos=props.arc_chaos,
                branches=props.arc_branches, replace=False, **common
            )

        if props.arc_flicker:
            scene = context.scene
            electricity.add_flicker(arcs, scene.frame_start, scene.frame_end, seed=props.seed)
        self.report({"INFO"}, f"{len(arcs)} arc object(s) created ({props.arc_mode.lower()})")
        return {"FINISHED"}


class ATOMVIZ_OT_lasers(Operator):
    """Aim laser beams at the structure"""

    bl_idname = "atomviz.lasers"
    bl_label = "Generate lasers"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.atomviz
        structure = _structure(self, context)
        if structure is None:
            return {"CANCELLED"}
        center, radius = _bounds(self, structure)
        color = hex_of(props.laser_color)
        beams = lasers.laser_rig(
            center,
            radius,
            count=props.laser_count,
            color=color,
            beam_radius=props.laser_radius,
            strength=props.laser_strength,
            distance=props.laser_distance,
            impact=props.laser_impact,
            seed=props.seed,
        )
        if props.laser_auto_haze and props.haze_density <= 0.0:
            lighting.add_haze(center, radius, density=0.02, color=color)
            render_module.set_volumetrics_hint(context.scene, True)
        self.report({"INFO"}, f"{len(beams)} laser object(s) created")
        return {"FINISHED"}


class ATOMVIZ_OT_camera(Operator):
    """Frame the structure with the AtomViz camera"""

    bl_idname = "atomviz.camera"
    bl_label = "Frame structure"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.atomviz
        structure = _structure(self, context)
        if structure is None:
            return {"CANCELLED"}
        try:
            camera, _, radius = camera_module.frame_structure(
                context.scene,
                structure,
                focal_mm=props.focal_mm,
                azimuth_deg=props.azimuth_deg,
                elevation_deg=props.elevation_deg,
                margin=props.frame_margin,
                dof=props.use_dof,
                fstop=props.fstop,
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"{camera.name} framed on a {radius:.1f} unit structure")
        return {"FINISHED"}


class ATOMVIZ_OT_render_preset(Operator):
    """Apply the output format, engine and colour management"""

    bl_idname = "atomviz.render_preset"
    bl_label = "Apply render format"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.atomviz
        preset = render_module.apply_render_preset(
            context.scene,
            props.render_preset,
            percentage=props.resolution_percentage,
            volumetrics=props.haze_density > 0.0,
        )
        for line in render_module.print_report(context.scene):
            print(f"[AtomViz] {line}")
        if preset.note:
            self.report({"INFO"}, preset.note)
        else:
            self.report({"INFO"}, f"{preset.label}: {preset.width}x{preset.height}")
        return {"FINISHED"}


class ATOMVIZ_OT_postfx(Operator):
    """Build the compositor glow / vignette / grade chain"""

    bl_idname = "atomviz.postfx"
    bl_label = "Apply post-processing"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.atomviz
        postfx.setup(
            context.scene,
            glare=props.glare,
            threshold=props.glare_threshold,
            streaks=props.glare_streaks,
            vignette=props.vignette,
            contrast=props.contrast,
            saturation=props.saturation,
        )
        self.report({"INFO"}, "Post-processing chain rebuilt")
        return {"FINISHED"}


class ATOMVIZ_OT_apply_look(Operator):
    """Apply a complete cover look in one click"""

    bl_idname = "atomviz.apply_look"
    bl_label = "Apply cover look"
    bl_options = {"REGISTER", "UNDO"}

    look: StringProperty(name="Look", default="")

    def execute(self, context):
        props = context.scene.atomviz
        structure = _structure(self, context)
        if structure is None:
            return {"CANCELLED"}
        key = self.look or props.look
        report = apply_look(context.scene, structure, key, seed=props.seed)
        for warning in report.warnings:
            self.report({"WARNING"}, warning)
        self.report({"INFO"}, report.summary())
        return {"FINISHED"}


class ATOMVIZ_OT_clear(Operator):
    """Delete every light, arc, laser and haze volume created by AtomViz"""

    bl_idname = "atomviz.clear"
    bl_label = "Clear AtomViz objects"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        removed = clear_all()
        total = sum(removed.values())
        self.report({"INFO"}, f"Removed {total} object(s): {removed}")
        return {"FINISHED"}


class ATOMVIZ_OT_import_xyz(Operator):
    """Import an XYZ file with Atomic Blender (enabling the add-on if needed)"""

    bl_idname = "atomviz.import_xyz"
    bl_label = "Import XYZ (Atomic Blender)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not enable_atomic_blender():
            self.report({"ERROR"}, ATOMIC_BLENDER_HINT)
            print(f"[AtomViz] {ATOMIC_BLENDER_HINT}")
            return {"CANCELLED"}
        bpy.ops.import_mesh.xyz("INVOKE_DEFAULT")
        return {"FINISHED"}


class ATOMVIZ_OT_render(Operator):
    """Render the current frame to the output path"""

    bl_idname = "atomviz.render"
    bl_label = "Render cover"
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = context.scene.atomviz
        if context.scene.camera is None:
            self.report({"ERROR"}, "No active camera; run 'Frame structure' first")
            return {"CANCELLED"}
        path = render_module.render_still(context.scene, props.output_path)
        self.report({"INFO"}, f"Rendered to {path}")
        return {"FINISHED"}


def atomic_blender_ready():
    """Return ``True`` when the XYZ import operator is actually registered.

    ``hasattr(bpy.ops.import_mesh, "xyz")`` is **not** a valid check: ``bpy.ops``
    resolves attributes lazily and answers ``True`` for operators that do not
    exist. The registered operator class is the reliable probe.
    """
    return hasattr(bpy.types, "IMPORT_MESH_OT_xyz")


def enable_atomic_blender():
    """Enable Atomic Blender, whether it is a bundled add-on or an extension.

    Returns:
        ``True`` when the XYZ import operator is available afterwards.
    """
    if atomic_blender_ready():
        return True

    candidates = [
        "io_mesh_atomic",
        "bl_ext.blender_org.atomic_blender_pdb_xyz",
        "bl_ext.user_default.atomic_blender_pdb_xyz",
    ]
    try:  # Discover it wherever the user installed it.
        import addon_utils

        for module in addon_utils.modules():
            info = getattr(module, "bl_info", None) or {}
            label = str(info.get("name", "")).lower()
            if "atomic" in module.__name__.lower() or "atomic blender" in label:
                candidates.append(module.__name__)
    except Exception:  # noqa: BLE001 - discovery is best effort
        pass

    for module in dict.fromkeys(candidates):
        try:
            bpy.ops.preferences.addon_enable(module=module)
        except (RuntimeError, TypeError):
            continue
        if atomic_blender_ready():
            return True
    return atomic_blender_ready()


CLASSES = (
    ATOMVIZ_OT_detect,
    ATOMVIZ_OT_apply_materials,
    ATOMVIZ_OT_apply_radii,
    ATOMVIZ_OT_background,
    ATOMVIZ_OT_lighting,
    ATOMVIZ_OT_electricity,
    ATOMVIZ_OT_lasers,
    ATOMVIZ_OT_camera,
    ATOMVIZ_OT_render_preset,
    ATOMVIZ_OT_postfx,
    ATOMVIZ_OT_apply_look,
    ATOMVIZ_OT_clear,
    ATOMVIZ_OT_import_xyz,
    ATOMVIZ_OT_render,
)


def register():
    """Register every operator."""
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    """Unregister every operator."""
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
