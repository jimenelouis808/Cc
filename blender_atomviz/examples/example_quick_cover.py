"""Cover in six lines — paste into Blender's Text Editor and press Run Script.

Assumes an XYZ structure is already imported (Atomic Blender). Change the look
key to try the others: clean_journal, neon_lab, plasma_storm, laser_lab,
crystal_ice, gold_catalysis, xray_schematic, toon_outreach, hologram.
"""

import bpy

from atomviz_studio.core.detect import active_structure
from atomviz_studio.looks.apply import apply_look
from atomviz_studio.scene.render import render_still

LOOK = "neon_lab"
OUTPUT = "//cover_neon.png"

structure = active_structure(bpy.context, "SCENE")
if structure is None:
    raise SystemExit("No atomic structure found — import an XYZ file first")

report = apply_look(bpy.context.scene, structure, LOOK, seed=7)
print(report.summary())

# Comment this out to look around in the viewport before committing to a render.
render_still(bpy.context.scene, OUTPUT)
