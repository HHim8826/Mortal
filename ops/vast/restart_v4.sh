#!/bin/bash
# Restart the v4 run between two checkpoint saves, with the new libriichi.
set -u
RUN=/root/Mortal/mortal/logs/v4
CK=$RUN/mortal.pth
log() { echo "$(date -u +%H:%M:%S) $*"; }
old=$(stat -c %Y "$CK")
log "waiting for the next save (mortal.pth mtime $old)"
while [ "$(stat -c %Y "$CK")" = "$old" ]; do sleep 2; done
# torch.save takes a few seconds; wait until the file holds still, then prove it loads.
prev=""; while true; do cur=$(stat -c "%s %Y" "$CK"); [ "$cur" = "$prev" ] && break; prev=$cur; sleep 5; done
steps=$(/root/venv/bin/python -c "import torch; print(torch.load(\"$CK\", weights_only=True, map_location=\"cpu\")[\"steps\"])") || { log "checkpoint does not load; not restarting"; exit 1; }
cp "$CK" /root/mortal.pth.pre-restart
log "checkpoint at step $steps saved and loads; copy kept at /root/mortal.pth.pre-restart"
pkill -TERM -f "[t]orchrun --standalone"
for i in $(seq 60); do pgrep -f "[t]rain.py" > /dev/null || break; sleep 1; done
if pgrep -f "[t]rain.py" > /dev/null; then log "still running after 60 s, killing"; pkill -KILL -f "[t]rain.py"; sleep 3; fi
log "training stopped"
cp /root/Mortal/target/release/libriichi.so /root/Mortal/mortal/libriichi.so.new && mv /root/Mortal/mortal/libriichi.so.new /root/Mortal/mortal/libriichi.so
log "libriichi.so replaced ($(md5sum < /root/Mortal/mortal/libriichi.so | cut -c1-12))"
bash /root/launch_v4.sh
