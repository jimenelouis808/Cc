#!/usr/bin/env bash
# Style tests: ONE subject through several presets, small and fast, so
# the look can be chosen before anything is rendered at cover size.
#
#   BPY=<python with bpy> bash modelos_SEM/pruebas_estilo.sh
#   BPY=... bash modelos_SEM/pruebas_estilo.sh escena_profundidad
#
# Why a small subject and not the scene itself: the mesh is built one
# icosphere and one cylinder at a time in Python, so the cost is in the
# ATOM COUNT and barely in the samples or the resolution. Measured in
# this container, a 10 353-atom scene takes about a quarter of an hour
# whatever it is asked to render, and a 55 000-atom one is hours. A
# 3565-atom arm carries the same three materials -- graphitic shell,
# iron beads on it, the particle in the tip -- and comes back in
# minutes, which is what makes a comparison across presets possible at
# all. Pick the style here; spend the hours once.
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
OUT="modelos_SEM/render/pruebas"
SUBJECT="${1:-brazo_decorado}"
RES="${RES:-800 1000}"
SAMPLES="${SAMPLES:-32}"
mkdir -p "$OUT"

# The shortlist. `core_shell_cover` is the one written for this sample
# (dark graphitic shell, bright metal, glare only over the metal); the
# rest are catalogue presets worth seeing it against.
STYLES=(core_shell_cover clusters_on_sheet nature_nano science_cover
        advanced_materials electron_micrograph acs_nano_vivid)

for style in "${STYLES[@]}"; do
  echo "=== $SUBJECT · $style"
  # --spacefill-element Se: the AUTO model asks the element table, and
  # that table holds METALS, because a metal has no directional bonds
  # to draw. Selenium in an FeSe core is part of a solid and is not a
  # metal, so it has to be named. Safe on the pure-carbon versions,
  # where the only selenium there is sits in the core.
  "${BPY[@]}" "$R" -- --xyz "$IN/$SUBJECT.xyz" --json "$IN/$SUBJECT.json" \
    --style "$style" --model auto --colour-by element \
    --atom-scale 0.55 --bond-scale 0.5 --engine cycles \
    --spacefill-element Se \
    --resolution $RES --samples "$SAMPLES" \
    --out "$OUT/${SUBJECT}__${style}.png"
done

echo
echo "escritas en $OUT"
