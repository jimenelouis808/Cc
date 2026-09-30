#!/usr/bin/env bash
# Supervisor of the GPAW stage of crystal_validation: one worker per crystal,
# JOBS at a time, each relaunched if it dies, until every crystal has its
# `done` marker. Everything a worker finished is on disk (see the recipe), so
# a relaunch -- by this loop, or by hand after a container restart -- only
# repeats the SCF that was running.
#
#   PYTHON=/path/to/python JOBS=3 tbkit/recipes/run_crystals.sh WORKDIR [crystal ...]
#
# WORKDIR/supervisor.log gets one line per launch, death and finish;
# WORKDIR/ALL_DONE appears at the end (a monitor waits for it).
set -u
WORKDIR=${1:?WORKDIR}
shift
PYTHON=${PYTHON:-python}
JOBS=${JOBS:-3}
MAX_RESTARTS=${MAX_RESTARTS:-20}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if [ $# -gt 0 ]; then
    CRYSTALS=("$@")
else
    mapfile -t CRYSTALS < <("$PYTHON" -c \
        "from tbkit.recipes.crystal_validation import CRYSTALS; print('\n'.join(CRYSTALS))")
fi
mkdir -p "$WORKDIR"
LOG="$WORKDIR/supervisor.log"
rm -f "$WORKDIR/ALL_DONE"
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }

worker() {   # one crystal, relaunched until its marker exists
    local name=$1 tries=0
    while [ ! -e "$WORKDIR/$name/done" ]; do
        if [ "$tries" -ge "$MAX_RESTARTS" ]; then
            say "$name: ABANDONADO tras $tries intentos"
            return 1
        fi
        tries=$((tries + 1))
        say "$name: lanzado (intento $tries)"
        "$PYTHON" -m tbkit.recipes.crystal_validation gpaw "$WORKDIR" --systems "$name" \
            >> "$WORKDIR/$name.out" 2>&1
        status=$?
        [ -e "$WORKDIR/$name/done" ] || { say "$name: murió (código $status)"; sleep 5; }
    done
    say "$name: HECHO"
}

running=0
for name in "${CRYSTALS[@]}"; do
    worker "$name" &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait
missing=0
for name in "${CRYSTALS[@]}"; do [ -e "$WORKDIR/$name/done" ] || missing=$((missing + 1)); done
if [ "$missing" -eq 0 ]; then
    say "TODOS TERMINADOS"
    touch "$WORKDIR/ALL_DONE"
else
    say "FALTAN $missing"
fi
