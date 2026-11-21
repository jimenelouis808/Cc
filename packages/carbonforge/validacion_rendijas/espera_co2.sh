#!/usr/bin/env bash
# Keep the CO2 isotherms alive until all three are finished.
#
# Run this as a TRACKED background task of the session, not with nohup:
# a cloud session's container is reclaimed shortly after it goes idle,
# and processes the harness does not know about die with it. That is
# why the hourly check alone advanced about one point an hour although a
# point takes 3-4 minutes. While this loop runs the session is busy, the
# container stays up, and a run that dies is relaunched within a minute.
cd "$(dirname "$0")"
while true; do
  out=$(./lanza_co2.sh)
  if [ "$(grep -c terminado <<< "$out")" -eq 3 ]; then
    echo "todas terminadas"; exit 0
  fi
  sleep 60
done
