#!/usr/bin/env bash
# Watchdog for a resumable calculation. Usage:
#   vigia.sh <name> <pattern> <done-command> <worker-command>
# Every 60 s: stop if <done-command> succeeds; otherwise, if no process
# matches <pattern>, relaunch <worker-command> detached (setsid), so it
# survives the end of this watchdog. Run the watchdog itself as a TRACKED
# background task: that keeps the session busy so the container is not
# reclaimed. The worker must skip finished work on a rerun.
# A tracked task is killed at its time limit (2 h), and that kill took a
# worker the watchdog had started with it once. So the watchdog leaves on its
# own first, after VIGIA_MAX_MIN minutes (default 110), printing "renovar":
# whoever started it starts it again; the worker is not touched.
name="$1"; pattern="$2"; done_cmd="$3"; worker="$4"
cd "$(dirname "$0")"
log="vigia-$name.log"
deadline=$(( $(date +%s) + ${VIGIA_MAX_MIN:-110} * 60 ))
while true; do
  if bash -c "$done_cmd" >/dev/null 2>&1; then
    echo "$(date +%H:%M) $name terminado" >> "$log"; echo "todo terminado"; exit 0
  fi
  if ! pgrep -f "$pattern" >/dev/null; then
    echo "$(date +%H:%M) $name no corre: relanzo" >> "$log"
    setsid nohup bash -c "$worker" >> "$name.worker.log" 2>&1 < /dev/null &
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "$(date +%H:%M) $name: renovar el vigía" >> "$log"; echo "renovar"; exit 0
  fi
  sleep 60
done
