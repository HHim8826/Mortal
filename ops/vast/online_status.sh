#!/bin/bash
# How the online run is doing, on one screen.
#
# The number to watch is the workers' `trainee rankings`: the trainee plays one
# seat against three frozen copies of itself, so 2.5 is dead even with the
# model it started as and anything below it is the trainee ahead. It arrives
# every session -- about ten minutes -- where test play against the v3 ruler
# arrives every 10,000 steps, which is hours.
set -uo pipefail

RUN=/root/Mortal/mortal/logs/v4o

echo "== phase: $(cat /root/phase 2>/dev/null || echo offline)   $(date -u '+%Y-%m-%d %H:%M UTC')"

echo '== processes'
shopt -s nullglob
for f in "$RUN"/*.pid; do
    name=$(basename "$f" .pid)
    pid=$(cat "$f" 2>/dev/null)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        printf '   %-9s pid %-7s up %s\n' "$name" "$pid" \
            "$(ps -o etime= -p "$pid" | tr -d ' ')"
    else
        printf '   %-9s DOWN\n' "$name"
    fi
done
shopt -u nullglob

echo '== trainer'
if [ -e "$RUN/trainer.log" ]; then
    tr '\r' '\n' < "$RUN/trainer.log" | grep -aE 'total steps|steps in|avg rank|avg pt|progress since|a new record|param has been submitted' \
        | sed 's/.*INFO *//' | tail -6 | sed 's/^/   /'
fi
if [ -e "$RUN/mortal.pth" ]; then
    echo "   last save: $(( $(date +%s) - $(stat -c %Y "$RUN/mortal.pth") )) s ago"
fi

echo '== server (buffer)'
[ -e "$RUN/server.log" ] && tr '\r' '\n' < "$RUN/server.log" \
    | grep -aE 'buffer size|transferred' | sed 's/.*INFO *//' | tail -4 | sed 's/^/   /'

echo '== workers (2.5 = level with the frozen copy of itself; lower is the trainee ahead)'
shopt -s nullglob
for f in "$RUN"/worker*.log; do
    echo "   $(basename "$f" .log):"
    tr '\r' '\n' < "$f" | grep -a 'sessions:' | sed 's/.*INFO *//' | tail -2 | sed 's/^/     /'
done
shopt -u nullglob

echo '== gpu'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/   /'
echo "== disk: $(df -h /root | tail -1 | awk '{print $4}') free"

echo '== anything that went wrong'
shopt -s nullglob
for f in "$RUN"/*.log; do
    out=$(tr '\r' '\n' < "$f" | grep -aE 'Traceback|Error|error:|CUDA|Killed' | tail -2)
    [ -n "$out" ] && { echo "   $(basename "$f"):"; echo "$out" | sed 's/^/     /'; }
done
shopt -u nullglob
exit 0
