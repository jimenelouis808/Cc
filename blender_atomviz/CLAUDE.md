# CLAUDE.md — AtomViz Studio

Instructions for Claude Code (or any assistant) working on this sub-project.

## Scope
`atomviz_studio` is a **Blender add-on** that gives cover-quality shading and
visual effects (electricity, volumetric light, lasers) to atomic structures
imported with **Atomic Blender** (`io_mesh_atomic`, the bundled PDB/XYZ
importer). It is the rendering front-end for structures produced by
`nanocarbon_lab`.

It must never modify atomic coordinates. Materials, object scale, lights,
world, effect objects and compositor nodes only.

## Layout
```
atomviz_studio/
├── core/        # elements, colours, palettes, geometry, presets (no bpy) + compat/nodes/detect (bpy)
├── materials/   # style builders + per-element assignment
├── effects/     # backgrounds, light rigs, haze, electricity, lasers, post-fx
├── scene/       # camera framing, render presets
├── looks/       # complete one-click cover recipes
├── ui/          # PropertyGroup, operators, N-panel
├── cli/         # headless render entry point
└── tests/       # pytest, runs without Blender
```

## Golden rules
1. **Python >= 3.10**, type hints and docstrings on every public function.
2. Keep `core/elements.py`, `core/colors.py`, `core/palettes.py`,
   `core/mathutil.py` and `core/presets.py` **free of `bpy`** — they are the
   tested half. Put Blender calls in the other modules.
3. `ui/props.py`, `ui/operators.py` and `ui/panels.py` must **not** use
   `from __future__ import annotations`: Blender registers properties by
   evaluating class annotations.
4. Every Blender API that changed between 3.3 and 4.5 goes through
   `core/compat.py`. Never call a version-specific socket name directly; use
   `set_principled(node, logical_name=value)`.
5. Only use node types available in **every** supported release
   (no `ShaderNodeTexMusgrave`, removed in 4.1; use `nodes.mix_rgb()` instead
   of `ShaderNodeMixRGB`).
6. Anything stochastic (arcs, laser jitter) takes a `seed` and is reproducible.
7. Objects the add-on creates live in `AV_*` collections and are named `AV_*`
   so `looks.apply.clear_all()` can remove them without touching user data.
8. New style / palette / background / rig / look → add it to its registry dict
   and it appears in the UI, the CLI and the tests automatically.
9. New feature ships with a test in `tests/`; Blender-facing modules must at
   least stay importable under `tests/bpy_stub.py`.

## Test commands
```bash
cd blender_atomviz
pytest -q                    # 85 tests, no Blender needed
ruff check .
python tools/make_addon_zip.py           # installable zip
blender -b -P atomviz_studio/cli/render_cover.py -- --list
```

## What not to do
- Do **not** move atoms, edit meshes or change the importer's coordinates.
- Do **not** add third-party Python dependencies: the add-on must run on a
  stock Blender install.
- Do **not** rely on EEVEE-only nodes (`ShaderNodeShaderToRGB`) without saying
  so in the style description.
- Do **not** run structure detection in a panel `draw()` for heavy scenes; the
  panel already guards at 2000 objects.
