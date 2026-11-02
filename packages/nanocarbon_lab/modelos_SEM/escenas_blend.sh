#!/usr/bin/env bash
# The five cover scenes as .blend files, to open and adjust by hand.
#
#   BPY=<python con bpy> bash modelos_SEM/escenas_blend.sh
#   BPY=... bash modelos_SEM/escenas_blend.sh escena_barras
#
# `--no-render` builds the scene and stops: materials, lights, camera
# and all the geometry are in the file, and nothing is sampled. That
# is the whole cost saved -- what is left is the mesh build, which is
# where the time goes anyway (one icosphere and one cylinder at a
# time, in Python), so a 55 000-atom scene is still long.
#
# The camera in the file is the one the preview reported: same lens,
# same fill, same view name. Move it; the rest of the scene does not
# depend on it.
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
OUT="modelos_SEM/blend"
ESTILO="${ESTILO:-core_shell_cover}"
# One Blender object per piece of the composition, named after it, so
# each plate, rod or arm can be moved and turned by hand. PARTIR=0 goes
# back to the two merged meshes, which draw faster and cannot be taken
# apart.
PARTIR="${PARTIR:-1}"
SPLIT=()
[ "$PARTIR" = "1" ] && SPLIT=(--split-items)
mkdir -p "$OUT"

escena () {        # escena <nombre> <relleno> [vista]
  local name="$1" fill="$2" view="${3:-auto}"
  echo "=== $name (relleno $fill, vista $view)"
  "${BPY[@]}" "$R" -- --xyz "$IN/$name.xyz" --json "$IN/$name.json" \
    --style "$ESTILO" --model auto --colour-by element \
    --atom-scale 0.55 --bond-scale 0.5 --engine cycles \
    --spacefill-element Se --resolution 1600 2000 \
    --camera-fill "$fill" --view "$view" "${SPLIT[@]}" \
    --no-render --save-blend "$OUT/$name.blend"
}

# Lightest first, so the first file lands soonest.
todo="${1:-}"
if [ -n "$todo" ]; then
  case "$todo" in
    escena_profundidad)      escena escena_profundidad 1.10 depth ;;
    escena_anemona)          escena escena_anemona 0.95 ;;
    escena_anemona_decorada) escena escena_anemona_decorada 1.00 ;;
    escena_barras)           escena escena_barras 1.18 ;;
    escena_anemonas)         escena escena_anemonas 1.12 ;;
    *) echo "escena desconocida: $todo" >&2; exit 2 ;;
  esac
else
  escena escena_profundidad      1.10 depth
  escena escena_anemona          0.95
  escena escena_anemona_decorada 1.00
  escena escena_barras           1.18
  escena escena_anemonas         1.12
fi

echo
echo "escritos en $OUT"
