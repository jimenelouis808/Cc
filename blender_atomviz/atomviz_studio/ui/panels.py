"""The AtomViz Studio side panel (3D View > N > AtomViz)."""

import bpy
from bpy.types import Panel

from ..core.detect import detect_structures
from ..core.presets import RENDER_PRESETS, get_look
from ..materials.apply import describe

CATEGORY = "AtomViz"


class _Base:
    """Shared panel configuration."""

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY


class ATOMVIZ_PT_main(_Base, Panel):
    """Structure detection and one-click cover looks."""

    bl_idname = "ATOMVIZ_PT_main"
    bl_label = "AtomViz Studio"

    def draw(self, context):
        layout = self.layout
        props = context.scene.atomviz

        row = layout.row(align=True)
        row.prop(props, "scope", text="")
        row.operator("atomviz.detect", text="", icon="VIEWZOOM")

        box = layout.box()
        # Detection runs on every redraw; skip the live preview in very heavy
        # scenes and let the user trigger it explicitly instead.
        if len(context.scene.objects) > 2000:
            box.label(text=f"{len(context.scene.objects)} objects in scene", icon="INFO")
            box.label(text="Press the magnifier to list structures")
        else:
            structures = detect_structures(context, props.scope)
            if structures:
                top = structures[0]
                box.label(text=f"{top.name} - {top.atom_count} atoms", icon="OUTLINER_OB_MESH")
                for line in describe(top)[:6]:
                    box.label(text=line, icon="DOT")
                if len(structures) > 1:
                    box.label(text=f"+{len(structures) - 1} more structure(s)")
            else:
                box.label(text="No structure detected", icon="ERROR")
                box.operator("atomviz.import_xyz", icon="IMPORT")

        layout.separator()
        layout.label(text="Cover look:")
        layout.prop(props, "look", text="")
        layout.label(text=get_look(props.look).description, icon="INFO")
        row = layout.row(align=True)
        row.prop(props, "seed")
        layout.operator("atomviz.apply_look", icon="SHADERFX")
        layout.operator("atomviz.clear", icon="TRASH")


class ATOMVIZ_PT_materials(_Base, Panel):
    """Palette, shading style and atom radii."""

    bl_idname = "ATOMVIZ_PT_materials"
    bl_parent_id = "ATOMVIZ_PT_main"
    bl_label = "Shading & palette"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.atomviz

        layout.prop(props, "palette")
        layout.prop(props, "accent")
        layout.prop(props, "style")

        box = layout.box()
        box.label(text="Style tuning")
        col = box.column(align=True)
        col.prop(props, "roughness")
        col.prop(props, "metallic")
        col.prop(props, "clearcoat")
        col.prop(props, "ior")
        col.prop(props, "alpha")
        col.prop(props, "subsurface")
        col.prop(props, "emission_strength")
        col.prop(props, "emissive_dopants")
        if props.style == "toon_cel":
            col.prop(props, "toon_steps")
        if props.style == "hologram":
            col.prop(props, "scanlines")

        box = layout.box()
        box.label(text="Bonds")
        box.prop(props, "stick_style", text="")
        box.prop(props, "stick_color")

        box = layout.box()
        box.label(text="Atom radii")
        box.prop(props, "radius_mode", text="")
        sub = box.row()
        sub.enabled = props.radius_mode != "KEEP"
        sub.prop(props, "radius_factor")

        layout.operator("atomviz.apply_materials", icon="MATERIAL")


class ATOMVIZ_PT_world(_Base, Panel):
    """World background and lighting."""

    bl_idname = "ATOMVIZ_PT_world"
    bl_parent_id = "ATOMVIZ_PT_main"
    bl_label = "Background & light"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.atomviz

        box = layout.box()
        box.label(text="Background")
        box.prop(props, "background", text="")
        col = box.column(align=True)
        col.prop(props, "bg_top")
        col.prop(props, "bg_bottom")
        col.prop(props, "bg_accent")
        col.prop(props, "bg_strength")
        box.operator("atomviz.background", icon="WORLD")

        box = layout.box()
        box.label(text="Lighting")
        box.prop(props, "light_rig", text="")
        col = box.column(align=True)
        col.prop(props, "light_energy")
        col.prop(props, "light_color_a")
        col.prop(props, "light_color_b")
        col.prop(props, "haze_density")
        box.operator("atomviz.lighting", icon="LIGHT")


class ATOMVIZ_PT_effects(_Base, Panel):
    """Electricity and laser generators."""

    bl_idname = "ATOMVIZ_PT_effects"
    bl_parent_id = "ATOMVIZ_PT_main"
    bl_label = "Effects: electricity & lasers"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.atomviz

        box = layout.box()
        box.label(text="Electricity", icon="LIGHT_AREA")
        box.prop(props, "arc_mode", text="")
        col = box.column(align=True)
        col.prop(props, "arc_count")
        col.prop(props, "arc_color")
        col.prop(props, "arc_thickness")
        col.prop(props, "arc_strength")
        col.prop(props, "arc_chaos")
        col.prop(props, "arc_branches")
        col.prop(props, "arc_flicker")
        box.operator("atomviz.electricity", icon="FORCE_ELECTRIC")

        box = layout.box()
        box.label(text="Lasers", icon="LIGHT_SPOT")
        col = box.column(align=True)
        col.prop(props, "laser_count")
        col.prop(props, "laser_color")
        col.prop(props, "laser_radius")
        col.prop(props, "laser_strength")
        col.prop(props, "laser_distance")
        col.prop(props, "laser_impact")
        col.prop(props, "laser_auto_haze")
        box.operator("atomviz.lasers", icon="LIGHT_SUN")


class ATOMVIZ_PT_camera(_Base, Panel):
    """Camera framing and depth of field."""

    bl_idname = "ATOMVIZ_PT_camera"
    bl_parent_id = "ATOMVIZ_PT_main"
    bl_label = "Camera"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.atomviz
        col = layout.column(align=True)
        col.prop(props, "focal_mm")
        col.prop(props, "azimuth_deg")
        col.prop(props, "elevation_deg")
        col.prop(props, "frame_margin")
        col.prop(props, "use_dof")
        sub = col.row()
        sub.enabled = props.use_dof
        sub.prop(props, "fstop")
        layout.operator("atomviz.camera", icon="CAMERA_DATA")


class ATOMVIZ_PT_output(_Base, Panel):
    """Render format and post-processing."""

    bl_idname = "ATOMVIZ_PT_output"
    bl_parent_id = "ATOMVIZ_PT_main"
    bl_label = "Render & post"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.atomviz

        box = layout.box()
        box.label(text="Format")
        box.prop(props, "render_preset", text="")
        preset = RENDER_PRESETS.get(props.render_preset)
        if preset is not None:
            box.label(text=f"{preset.width} x {preset.height} px, {preset.engine}", icon="INFO")
            if preset.note:
                for chunk in _wrap(preset.note, 34):
                    box.label(text=chunk)
        box.prop(props, "resolution_percentage")
        box.operator("atomviz.render_preset", icon="OUTPUT")

        box = layout.box()
        box.label(text="Post-processing")
        box.prop(props, "glare", text="")
        col = box.column(align=True)
        col.enabled = props.glare != "NONE" or props.vignette > 0.0
        col.prop(props, "glare_threshold")
        col.prop(props, "glare_streaks")
        col.prop(props, "vignette")
        col.prop(props, "contrast")
        col.prop(props, "saturation")
        box.operator("atomviz.postfx", icon="NODE_COMPOSITING")

        layout.separator()
        layout.prop(props, "output_path")
        layout.operator("atomviz.render", icon="RENDER_STILL")


def _wrap(text, width):
    """Split *text* into lines of at most *width* characters (panel labels)."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


CLASSES = (
    ATOMVIZ_PT_main,
    ATOMVIZ_PT_materials,
    ATOMVIZ_PT_world,
    ATOMVIZ_PT_effects,
    ATOMVIZ_PT_camera,
    ATOMVIZ_PT_output,
)


def register():
    """Register every panel."""
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    """Unregister every panel."""
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
