#!/bin/bash
# Keep whichever run is current going while nobody is watching.
#
#     setsid nohup bash /root/watchdog.sh > /dev/null 2>&1 < /dev/null &
#
# /root/phase names the run: `offline` (the human corpus, two ranks under
# torchrun) or `online` (self-play: a server, a trainer and N workers). A
# missing file means offline. /root/switch_to_online.sh writes it.
#
# /root/stop_at_step, if it holds a number, is where the offline run ends: the
# watchdog stops it once the log passes that step and then leaves it stopped,
# across reboots too, until the phase changes. Offline's cosine is flat long
# before its nominal end, so the run is meant to be cut short on purpose rather
# than left to finish.
#
# `touch /root/watchdog.off` to stop it doing anything at all; remove the file
# to hand control back.
set -uo pipefail

# Overridable so the decision logic can be exercised against a tree of made-up
# logs, which is the only way to see the step gate fire without waiting for the
# run to reach it.
LOG=${WATCHDOG_LOG:-/root/watchdog.log}
PHASE_FILE=${WATCHDOG_PHASE:-/root/phase}
STOP_AT_FILE=${WATCHDOG_STOP_AT:-/root/stop_at_step}
MORTAL=${WATCHDOG_MORTAL:-/root/Mortal/mortal}
PIDFILE=${WATCHDOG_PIDFILE:-/root/watchdog.pid}
PY=/root/venv/bin/python
CHECK_EVERY=60

say() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"
}

# Only one of these at a time. onstart.sh brings one up on every container
# start and a person may start another by hand; two loops would each see the
# other restarting training, kill it, and start it again, and the run would
# never get past compiling.
#
# A pid, not an flock: children inherit the lock's file descriptor and go on
# holding it after the parent is killed, and this script is asleep in a `sleep`
# almost all of the time, so a `pkill` left the lock held by an orphan and the
# replacement refused to start. A pid can be checked against the process.
if [ -e "$PIDFILE" ]; then
    other=$(cat "$PIDFILE" 2>/dev/null)
    if [ -n "${other:-}" ] && [ "$other" != "$$" ] && kill -0 "$other" 2>/dev/null \
        && tr '\0' ' ' < "/proc/$other/cmdline" 2>/dev/null | grep -q 'watchdog.sh'; then
        say "another watchdog is running (pid $other); exiting"
        exit 0
    fi
fi
echo $$ > "$PIDFILE"

# ---------------------------------------------------------------- the phases

phase() {
    local p=offline
    [ -s "$PHASE_FILE" ] && p=$(tr -cd 'a-z' < "$PHASE_FILE")
    case "$p" in
        offline|online) echo "$p" ;;
        *) echo offline ;;
    esac
}

run_dir() {
    [ "$1" = online ] && echo "$MORTAL/logs/v4o" || echo "$MORTAL/logs/v4"
}

launcher() {
    [ "$1" = online ] && echo /root/launch_online.sh || echo /root/launch_v4.sh
}

# The trainer's own log, where the step counter and the completion line are.
main_log() {
    [ "$1" = online ] && echo "$(run_dir online)/trainer.log" || echo "$(run_dir offline)/train.log"
}

# Long enough that an evaluation is not mistaken for a hang. Offline saves
# every ~45 s and stops for ~9 minutes every 40,000 steps; online saves every
# 400 steps at ~1.5 steps/s and its evaluations take ~20 minutes, because the
# workers hold most of the CPUs while the trainer plays.
stall() {
    [ "$1" = online ] && echo 3600 || echo 1500
}

# ------------------------------------------------------------------ the state

alive() {
    if [ "$1" = offline ]; then
        pgrep -f "[t]rain.py" >/dev/null
        return
    fi
    # Online is several processes and the run is only healthy with all of them:
    # the trainer starves without workers, and the workers have nowhere to send
    # games without the server. launch_online.sh starts back just the ones that
    # are gone, so treating a partial failure as "not running" is cheap.
    local run f pid
    run=$(run_dir online)
    shopt -s nullglob
    local files=("$run"/*.pid)
    shopt -u nullglob
    [ ${#files[@]} -gt 0 ] || return 1
    for f in "${files[@]}"; do
        pid=$(cat "$f" 2>/dev/null) || return 1
        kill -0 "$pid" 2>/dev/null || return 1
    done
    return 0
}

# The newest step the trainer wrote down. Read from the tail of the log rather
# than from the checkpoint: loading 174 MB of weights once a minute to find one
# integer is a strange way to ask the question.
last_step() {
    tail -c 400000 "$(main_log "$1")" 2>/dev/null | tr '\r' '\n' \
        | grep -a 'total steps:' | tail -1 \
        | sed 's/.*total steps: //; s/ .*//; s/,//g'
}

# The last line of a finished run, written by train.py once the epoch is done.
# Without this a completed run would be restarted for ever.
training_complete() {
    tail -c 200000 "$(main_log "$1")" 2>/dev/null | tr '\r' '\n' | grep -q 'training is complete'
}

stop_at() {
    [ -s "$STOP_AT_FILE" ] || return 1
    local n
    n=$(tr -cd '0-9' < "$STOP_AT_FILE")
    [ -n "$n" ] || return 1
    echo "$n"
}

# Past the target, not at it. The step line is written before the evaluation
# that step runs, and the evaluations are the only measurement this run
# produces: stopping the moment the number appears would kill the last one
# half-played. One more window -- 400 steps, about 45 seconds -- and the
# evaluation is in the log.
reached_target() {
    [ "$1" = offline ] || return 1
    local target step
    target=$(stop_at) || return 1
    step=$(last_step offline)
    [ -n "$step" ] || return 1
    [ "$step" -gt "$target" ] 2>/dev/null
}

# ----------------------------------------------------------------- the levers

stop_run() {
    if [ "$1" = offline ]; then
        pkill -f "[t]orchrun --standalone"
        pkill -f "[t]rain.py"
    else
        local run f pid
        run=$(run_dir online)
        shopt -s nullglob
        for f in "$run"/*.pid; do
            pid=$(cat "$f" 2>/dev/null) && kill "$pid" 2>/dev/null
        done
        shopt -u nullglob
        # The online trainer runs its steps in a child of the process the pid
        # file names, and the workers and the server have no pid but their own.
        pkill -f "[t]rain.py"
        pkill -f "[c]lient.py"
        pkill -f "[s]erver.py"
    fi
    for _ in $(seq 30); do
        pgrep -f "[t]rain.py" >/dev/null || return
        sleep 2
    done
    say "it would not exit; killing it"
    pkill -9 -f "[t]rain.py"
    sleep 5
}

fails=0
first_fail=0

start_run() {
    local now
    now=$(date +%s)
    # Five restarts inside an hour is a run that cannot get going, not a run
    # with bad luck. Keep trying, but slowly, so the log stays readable and a
    # broken state is obvious in the morning.
    if [ "$fails" -eq 0 ] || [ $((now - first_fail)) -gt 3600 ]; then
        fails=0
        first_fail=$now
    fi
    fails=$((fails + 1))
    if [ "$fails" -ge 5 ]; then
        say "restart #$fails within the hour: something is wrong, waiting 15 min first"
        sleep 900
    fi
    say "starting the $1 run (restart #$fails)"
    bash "$(launcher "$1")" >> "$LOG" 2>&1 || say "$(launcher "$1") failed"
    # Compiling takes a couple of minutes; do not judge it before then.
    sleep 240
}

# ------------------------------------------------------------------- the loop

current=$(phase)
say "watchdog up (pid $$), phase $current$( stop_at >/dev/null && echo ", stopping at step $(stop_at)")"
gated=''

while true; do
    sleep "$CHECK_EVERY"
    [ -e /root/watchdog.off ] && continue

    p=$(phase)
    if [ "$p" != "$current" ]; then
        say "phase changed: $current -> $p"
        current=$p
        fails=0
        gated=''
    fi

    # This box is not the run's alone: it serves Jupyter over the whole
    # filesystem, and Jupyter leaves a `.ipynb_checkpoints` directory wherever
    # it has been. One landed in the server's drain directory and took the run
    # down in a way no restart could fix -- os.remove raised on the directory,
    # the handler thread died with the connection, and the trainer read that as
    # an unexpected EOF, every time, until a person deleted a folder. server.py
    # skips non-files now; this sweeps them regardless, because the cost is one
    # `find` a minute and the failure it prevents is total.
    find "$MORTAL/online" -maxdepth 2 -name '.ipynb_checkpoints' -type d \
        -exec rm -rf {} + 2>/dev/null

    if ! pgrep -f "[b]ackup_hf.py --loop" >/dev/null; then
        say "backup loop is gone; starting it"
        setsid nohup env MORTAL_RUN="$(run_dir "$p")" \
            "$PY" /root/backup_hf.py --loop >> /root/backup.log 2>&1 < /dev/null &
    fi

    # The offline run has gone as far as it was meant to. Stop it, and keep it
    # stopped: this branch is reached again after a reboot, so nothing brings
    # it back until the phase changes or the file goes away.
    if reached_target "$p"; then
        if alive "$p"; then
            say "step $(last_step "$p") is past the $(stop_at) it was to stop at; stopping the offline run"
            stop_run "$p"
        fi
        if [ -z "$gated" ]; then
            say "offline is finished; holding. /root/switch_to_online.sh goes on, or remove $STOP_AT_FILE to keep training"
            gated=1
        fi
        continue
    fi
    gated=''

    if alive "$p"; then
        ckpt="$(run_dir "$p")/mortal.pth"
        [ -e "$ckpt" ] || continue
        age=$(( $(date +%s) - $(stat -c %Y "$ckpt") ))
        if [ "$age" -gt "$(stall "$p")" ]; then
            say "nothing saved for $age s: the $p run is stuck, restarting it"
            stop_run "$p"
            start_run "$p"
        fi
        continue
    fi

    if training_complete "$p"; then
        say "the $p run finished; nothing left to watch"
        exit 0
    fi
    say "the $p run is not running; it crashed or was killed"
    tr '\r' '\n' < "$(main_log "$p")" 2>/dev/null \
        | grep -aE 'Error|error:|Traceback' | tail -3 >> "$LOG"
    start_run "$p"
done
