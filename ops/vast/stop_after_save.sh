#!/bin/bash
# Hand the run over to another box: stop training right after a checkpoint
# that is whole on disk and loads, and stop the backups. Starts nothing.
set -u
RUN=/root/Mortal/mortal/logs/v4
CK=$RUN/mortal.pth
log() { echo "$(date -u +%H:%M:%S) $*"; }

old=$(stat -c %Y "$CK")
log "waiting for the next save"
while [ "$(stat -c %Y "$CK")" = "$old" ]; do sleep 2; done
prev=""; while true; do cur=$(stat -c "%s %Y" "$CK"); [ "$cur" = "$prev" ] && break; prev=$cur; sleep 5; done
steps=$(/root/venv/bin/python -c "import torch; print(torch.load(\"$CK\", weights_only=True, map_location=\"cpu\")[\"steps\"])") \
    || { log "checkpoint does not load; training left running"; exit 1; }

pkill -TERM -f "[t]orchrun --standalone"
for i in $(seq 60); do pgrep -f "[t]rain.py" > /dev/null || break; sleep 1; done
pgrep -f "[t]rain.py" > /dev/null && { log "still running after 60 s, killing"; pkill -KILL -f "[t]rain.py"; sleep 3; }
pkill -f "[b]ackup_hf.py"
log "stopped at step $steps; nothing restarted here"
