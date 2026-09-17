#!/bin/bash
# One look at the run. Prints the lines worth waking somebody for and nothing
# else, so it can be called on a timer.
#
# It keeps its place in each log in /root/.watch_state, so every call prints
# only what has appeared since the last one. Silence means the run is going.
set -uo pipefail

STATE=/root/.watch_state
MORTAL=/root/Mortal/mortal
PHASE=$(tr -cd 'a-z' < /root/phase 2>/dev/null || echo offline)
case "$PHASE" in offline|online) ;; *) PHASE=offline ;; esac

if [ "$PHASE" = online ]; then
    RUN=$MORTAL/logs/v4o
    TRAIN_LOG=$RUN/trainer.log
    SAVE_STALL=2700          # a save every ~4 min, an evaluation every ~20
else
    RUN=$MORTAL/logs/v4
    TRAIN_LOG=$RUN/train.log
    SAVE_STALL=1200          # a save every ~45 s, an evaluation every ~9 min
fi

mkdir -p "$STATE"

# Lines of $1 that have appeared since the last call, by byte offset. A file
# that shrank (a new run, a rotation) is read from the start.
fresh() {
    local file=$1 key=$2 mark=$STATE/$2.offset size from
    [ -e "$file" ] || return 0
    size=$(stat -c %s "$file")
    from=$(cat "$mark" 2>/dev/null || echo 0)
    [ "$from" -le "$size" ] 2>/dev/null || from=0
    echo "$size" > "$mark"
    [ "$from" -lt "$size" ] || return 0
    tail -c +$((from + 1)) "$file" | head -c 2000000 | tr '\r' '\n'
}

# What the training log is asked for: the measurements, and the ways it fails.
fresh "$TRAIN_LOG" train | grep -aE \
    'avg rank|avg pt|progress since|a new record|a new ema record|training is complete|Traceback|Error|error:|RuntimeError|CUDA|out of memory|Killed' \
    | sed 's/.*INFO *//; s/^/train: /' | head -40

# The watchdog says little and all of it matters.
fresh /root/watchdog.log watchdog | grep -a . | sed 's/^/watchdog: /' | head -20

# And the things no log mentions: a run that is simply gone, or one that has
# stopped writing checkpoints without anyone noticing yet.
gate=$(tr -cd '0-9' < /root/stop_at_step 2>/dev/null)
step=$(tail -c 400000 "$TRAIN_LOG" 2>/dev/null | tr '\r' '\n' \
       | grep -a 'total steps:' | tail -1 | sed 's/.*total steps: //; s/ .*//; s/,//g')
held=''
if [ -n "$gate" ] && [ -n "$step" ] && [ "$step" -gt "$gate" ] 2>/dev/null; then
    held=1      # stopped on purpose at the gate; not a fault
fi

if [ -z "$held" ]; then
    if ! pgrep -f "[t]rain.py" >/dev/null; then
        echo "ALERT: no trainer process, phase $PHASE, last step ${step:-?}"
    elif [ -e "$RUN/mortal.pth" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$RUN/mortal.pth") ))
        [ "$age" -gt "$SAVE_STALL" ] && \
            echo "ALERT: nothing saved for $age s, phase $PHASE, last step ${step:-?}"
    fi
fi
if ! pgrep -f "[w]atchdog.sh" >/dev/null; then
    echo "ALERT: the watchdog is gone; nothing will restart the run"
fi

free=$(df -BG --output=avail /root | tail -1 | tr -cd '0-9')
[ "${free:-99}" -lt 5 ] && echo "ALERT: only ${free}G of disk left"

exit 0
