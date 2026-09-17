#!/bin/bash
# Exercise the watchdog's decisions against a tree of made-up logs.
#
# Only the part that decides is run: everything up to the main loop is sourced,
# so nothing here can start or stop a real process. The step gate is the point
# -- it fires once, at a step this box will not reach for hours, and there is
# no other way to watch it happen.
set -uo pipefail

T=/tmp/wdtest
rm -rf "$T"
mkdir -p "$T/logs/v4" "$T/logs/v4o"

export WATCHDOG_LOG=$T/watchdog.log
export WATCHDOG_PHASE=$T/phase
export WATCHDOG_STOP_AT=$T/stop_at_step
export WATCHDOG_MORTAL=$T
export WATCHDOG_PIDFILE=$T/watchdog.pid

# Everything above the loop: the functions, none of the doing. The copy beside
# this script, so it tests the same file whether that is /root or the repo.
WATCHDOG=${WATCHDOG:-$(dirname "$0")/watchdog.sh}
sed '/^# -*  *the loop/,$d' "$WATCHDOG" > "$T/lib.sh"
# shellcheck disable=SC1090
source "$T/lib.sh"

pass=0
fail=0
check() { # what, expected, actual
    if [ "$2" = "$3" ]; then
        pass=$((pass + 1))
        printf '  ok   %-42s %s\n' "$1" "$3"
    else
        fail=$((fail + 1))
        printf '  FAIL %-42s got %-18s want %s\n' "$1" "$3" "$2"
    fi
}
yesno() { if "$@"; then echo yes; else echo no; fi; }

echo 'phase, with no file:'
check 'phase' offline "$(phase)"
check 'run_dir' "$T/logs/v4" "$(run_dir "$(phase)")"
check 'main_log' "$T/logs/v4/train.log" "$(main_log offline)"
check 'stall' 1500 "$(stall offline)"

echo 'phase online:'
echo online > "$WATCHDOG_PHASE"
check 'phase' online "$(phase)"
check 'run_dir' "$T/logs/v4o" "$(run_dir "$(phase)")"
check 'main_log' "$T/logs/v4o/trainer.log" "$(main_log online)"
check 'stall' 3600 "$(stall online)"

echo 'a phase file with nonsense in it falls back to offline:'
echo 'sideways' > "$WATCHDOG_PHASE"
check 'phase' offline "$(phase)"
echo offline > "$WATCHDOG_PHASE"

echo 'the step, read from the log:'
check 'no log at all' '' "$(last_step offline)"
{
    echo '2026-09-12 17:00:00     INFO     train.py:489  total steps: 799,200 (~800)'
    echo '2026-09-12 17:00:43     INFO     train.py:489  total steps: 799,600 (~400)'
} > "$T/logs/v4/train.log"
check 'last_step' 799600 "$(last_step offline)"

echo 'the gate:'
check 'no stop_at_step file' no "$(yesno reached_target offline)"
echo 800000 > "$WATCHDOG_STOP_AT"
check 'short of the target' no "$(yesno reached_target offline)"
# The evaluation this step runs is logged after this line and before the next
# one, so the gate must wait for the window after it.
echo '2026-09-12 17:01:26     INFO     train.py:489  total steps: 800,000 (~0)' >> "$T/logs/v4/train.log"
check 'at the target, evaluation still to come' no "$(yesno reached_target offline)"
echo '2026-09-12 17:02:09     INFO     train.py:489  total steps: 800,400 (~39,600)' >> "$T/logs/v4/train.log"
check 'past the target' yes "$(yesno reached_target offline)"
check 'the gate is offline-only' no "$(yesno reached_target online)"
check 'stop_at' 800000 "$(stop_at)"

echo 'the completion line:'
check 'not there' no "$(yesno training_complete offline)"
echo 'training is complete after 1,004,000 steps' >> "$T/logs/v4/train.log"
check 'there' yes "$(yesno training_complete offline)"

echo 'alive(), for online, wants every pid file to name a live process:'
check 'no pid files' no "$(yesno alive online)"
echo 1 > "$T/logs/v4o/server.pid"          # pid 1 is always there
check 'one live' yes "$(yesno alive online)"
echo 999999 > "$T/logs/v4o/worker0.pid"    # and this one is not
check 'one live, one dead' no "$(yesno alive online)"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
