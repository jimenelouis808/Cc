#!/usr/bin/env bash
# Cover renders. Needs a Python with `bpy` (the wheel) or a Blender app:
#
#   uv venv bpyenv && VIRTUAL_ENV=bpyenv uv pip install bpy scipy
#   BPY=bpyenv/bin/python bash modelos_SEM/renderizar.sh
#
# or, with Blender installed:
#
#   blender -b -P nanocarbon_lab/blender/render_cnt.py -- <los mismos flags>
#
# `bpy` pins numpy 1.26 and the workspace needs numpy 2, so it goes in a
# venv of its own and never in the project's -- the root pyproject.toml
# declares the conflict for exactly this reason.
set -euo pipefail
BPY="${BPY:-python}"
R="nanocarbon_lab/blender/render_cnt.py"
IN="modelos_SEM/bundle"
OUT="modelos_SEM/render"
mkdir -p "$OUT"

# The look, in three flags:
#   --model auto              metals space-filling, carbon ball-and-stick
#   --spacefill-element Se    ...and the chalcogen too, because an FeSe
#                             core is a solid and Se is not a metal
#   --atom-scale/--bond-scale thin balls and thin sticks for the cage
COMMON=(--style clusters_on_sheet --model auto --colour-by element
        --atom-scale 0.55 --bond-scale 0.5 --engine cycles)

render () {        # render <name> <resolution> <samples> [extra flags...]
  local name="$1" res="$2" samples="$3"; shift 3
  echo "=== $name (${res}px, ${samples} muestras)"
  "$BPY" "$R" -- --xyz "$IN/$name.xyz" --json "$IN/$name.json" \
    "${COMMON[@]}" "$@" --resolution "$res" "$res" --samples "$samples" \
    --out "$OUT/$name.png"
}

# The pure-carbon versions carry Se only in the core, so naming it as
# space-filling is unambiguous there.
render placa_H_C              1400 64 --spacefill-element Se
render barra_T_C              1400 64 --spacefill-element Se
render particula_Fe3C_C       1400 64
render brazo_particula_dentro 1400 64
render escena_anemona         1600 48

# The doped one WITHOUT the flag: its selenium is in two places at once
# -- in the FeSe core and substituted into the carbon around it -- and a
# per-element rule cannot tell them apart, so here it stays open.
render placa_H_NSeO           1400 64
