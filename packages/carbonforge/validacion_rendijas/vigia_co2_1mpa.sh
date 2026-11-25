#!/usr/bin/env bash
# corre_co2_1mpa.sh under the watchdog: done at 14 PNGs. TRACKED background
# task; it leaves by itself after 110 min ("renovar") and is started again.
cd "$(dirname "$0")"
exec ./vigia.sh co2_1mpa "corre_co2_1mpa.sh" \
  "test \$(ls galeria/*__co2_1mpa.png 2>/dev/null | wc -l) -ge 14" \
  "./corre_co2_1mpa.sh"
