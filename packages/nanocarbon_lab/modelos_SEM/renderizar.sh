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
# BPY may be a whole command, not just a binary:
#   BPY="blender -b -P"   (a Blender application)
#   BPY=bpyenv/bin/python (the bpy wheel)
# so it is split into an array. Quoting it as one word made
# the first form look for a program literally called
# "blender -b -P", which is not an obvious error message.
read -r -a BPY <<< "${BPY:-python}"
R="nanocarbon_lab/blender/render_cnt.py"
IN="modelos_SEM/bundle"
OUT="modelos_SEM/render"
mkdir -p "$OUT"

# The look, in three flags:
#   --model auto              metals space-filling, carbon ball-and-stick
#   --spacefill-element Se    ...and the chalcogen too, because an FeSe
#                             core is a solid and Se is not a metal
#   --atom-scale/--bond-scale thin balls and thin sticks for the cage
# One variable, because the preset is the thing most likely to
# change: `bash modelos_SEM/renderizar.sh` after
# `ESTILO=core_shell_cover`. See modelos_SEM/pruebas_estilo.sh for
# the cheap way to choose one.
ESTILO="${ESTILO:-core_shell_cover}"
COMMON=(--style "$ESTILO" --model auto --colour-by element
        --atom-scale 0.55 --bond-scale 0.5 --engine cycles)

render () {        # render <name> <resolution> <samples> [extra flags...]
  local name="$1" res="$2" samples="$3"; shift 3
  echo "=== $name (${res}px, ${samples} muestras)"
  "${BPY[@]}" "$R" -- --xyz "$IN/$name.xyz" --json "$IN/$name.json" \
    "${COMMON[@]}" "$@" --resolution "$res" "$res" --samples "$samples" \
    --out "$OUT/$name.png"
}

# The pure-carbon versions carry Se only in the core, so naming it as
# space-filling is unambiguous there.
render placa_H_C              1400 64 --spacefill-element Se
render barra_T_C              1400 64 --spacefill-element Se
render particula_Fe3C_C       1400 64
render brazo_particula_dentro 1400 64

# The cover scenes, PORTRAIT and with the same framing the preview
# (`modelos_SEM/camara.py`) reports -- same lens, same fill, same view
# name -- so what was approved on screen is what gets rendered. A fill
# above 1 crops on purpose: the composition is supposed to run off the
# page. `escena_profundidad` is the one that must NOT use `auto`: the
# largest silhouette of three items on a line puts them side by side,
# which is the one thing a depth composition is not for.
scene () {         # scene <name> <samples> <fill> [view]
  local name="$1" samples="$2" fill="$3" view="${4:-auto}"
  echo "=== $name (1600x2000, ${samples} muestras, relleno $fill, vista $view)"
  "${BPY[@]}" "$R" -- --xyz "$IN/$name.xyz" --json "$IN/$name.json" \
    "${COMMON[@]}" --resolution 1600 2000 --samples "$samples" \
    --camera-fill "$fill" --view "$view" --out "$OUT/$name.png"
}

scene escena_barras           48 1.18
scene escena_profundidad      48 1.10 depth
scene escena_anemona          48 0.95
scene escena_anemonas         40 1.12
scene escena_anemona_decorada 48 1.00

# The doped one WITHOUT the flag: its selenium is in two places at once
# -- in the FeSe core and substituted into the carbon around it -- and a
# per-element rule cannot tell them apart, so here it stays open.
render placa_H_NSeO           1400 64
