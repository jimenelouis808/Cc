#!/usr/bin/env bash
# Watchdog for a resumable calculation. Usage:
#   vigia.sh <name> <pattern> <done-command> <worker-command>
# Every 60 s: stop if <done-command> succeeds; otherwise, if no process
# matches <pattern>, relaunch <worker-command> detached (setsid), so it
# survives the end of this watchdog. Run the watchdog itself as a TRACKED
# background task: that keeps the session busy so the container is not
# reclaimed. The worker must skip finished work on a rerun.
name="$1"; pattern="$2"; done_cmd="$3"; worker="$4"
cd "$(dirname "$0")"
log="vigia-$name.log"
while true; do
  if bash -c "$done_cmd" >/dev/null 2>&1; then
    echo "$(date +%H:%M) $name terminado" >> "$log"; echo "todo terminado"; exit 0
  fi
  if ! pgrep -f "$pattern" >/dev/null; then
    echo "$(date +%H:%M) $name no corre: relanzo" >> "$log"
    setsid nohup bash -c "$worker" >> "$name.worker.log" 2>&1 < /dev/null &
  fi
  sleep 60
done
