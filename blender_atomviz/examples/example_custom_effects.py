"""Hand-built cover: custom palette, targeted arcs, one laser, tuned camera.

This is the "no presets" path — every step is an explicit call, so you can
cherry-pick what you need for a figure that has to look exactly one way.
Paste into Blender's Text Editor and press Run Script.
"""

import bpy

from atomviz_studio.core.detect import active_structure
from atomviz_studio.effects import backgrounds, electricity, lasers, lighting, postfx
from atomviz_studio.materials.apply import apply_radii, apply_style
from atomviz_studio.materials.styles import StyleParams
from atomviz_studio.scene.camera import frame_structure
from atomviz_studio.scene.render import apply_render_preset, render_still

scene = bpy.context.scene
structure = active_structure(bpy.context, "SCENE")
if structure is None:
    raise SystemExit("No atomic structure found — import an XYZ file first")

center, radius = structure.bounds()
print(f"{structure.name}: {structure.atom_count} atoms, elements {structure.symbols}")

# 1. Shading: graphite framework, nitrogen forced to a custom cyan, glowing dopants.
apply_style(
    structure,
    style_key="polished_metal",
    palette_key="mono_accent",
    params=StyleParams(roughness=0.18, emissive_dopants=8.0),
    accent="#22d3ee",
    stick_style="polished_metal",
    stick_color="#2b303b",
    overrides={"N": "#00fff2"},
)
apply_radii(structure, mode="vdw", factor=0.85)

# 2. World and light.
backgrounds.apply_background(scene, "nebula", top="#05070f", bottom="#0d1b2a", accent="#1b3b6f")
lighting.apply_rig(center, radius, "neon_rim", {"color_a": "#22d3ee", "color_b": "#f472b6"})
lighting.add_haze(center, radius, density=0.015, color="#22d3ee")

# 3. Effects: a cage of arcs, discharges to space, and one laser through the centre.
electricity.clear_arcs()
electricity.cage_arcs(center, radius, count=8, color="#7dd3fc", strength=45.0, seed=3)
electricity.discharge_to_space(structure.atom_positions(), center, radius, count=3, seed=3)
lasers.laser_rig(center, radius, count=1, color="#ff2d2d", strength=120.0, seed=3)

# 4. Camera, format and post.
apply_render_preset(scene, "cover_a4_300", volumetrics=True)
frame_structure(scene, structure, focal_mm=85.0, azimuth_deg=25.0, elevation_deg=12.0, fstop=2.4)
postfx.setup(scene, glare="FOG_GLOW", threshold=0.5, streaks=True, vignette=0.4, saturation=1.1)

render_still(scene, "//cover_custom.png")
