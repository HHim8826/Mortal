#!/bin/bash
# Start a phase-3 policy-gradient run on this box: the parameter/replay server,
# the PPO trainer on GPU 0, and the self-play workers on GPU 1.
#
#     VARIANT=kl         /root/launch_ppo.sh     # KL to the distilled start
#     VARIANT=kl-refresh /root/launch_ppo.sh     # the anchor follows, every 10 rounds
#     VARIANT=plain      /root/launch_ppo.sh     # clipping alone, no anchor
#     WORKERS=4 VARIANT=kl /root/launch_ppo.sh
#
# The three share one trainer and differ only in two flags, so a difference
# between their curves is the anchor and nothing else. Run them one at a time:
# each wants the whole box.
#
# The trainee and its three opponents all start as the same distilled policy,
# so the self-play average starts at 2.5 by construction and any move in it is
# this run's doing. That number is the early warning -- it caught the v4
# degradation hours before the 10,000-step evaluations could.
set -euo pipefail
cd /root/Mortal/mortal

CFG=${MORTAL_CFG_PPO:-config.ppo.toml}
VARIANT=${VARIANT:-kl}
WORKERS=${WORKERS:-3}
START=${START:-logs/policy/policy-t0.05.pth}
PY=/root/venv/bin/python
RUN=logs/ppo/$VARIANT

case $VARIANT in
    kl)         FLAGS="--kl-coef 0.1 --ref-refresh 0" ;;
    kl-refresh) FLAGS="--kl-coef 0.1 --ref-refresh 10" ;;
    plain)      FLAGS="--kl-coef 0" ;;
    *) echo "unknown VARIANT $VARIANT: expected kl, kl-refresh or plain" >&2; exit 1 ;;
esac

if [ ! -e "$START" ]; then
    echo "no $START to start from: run train_policy.py, then sharpen_policy.py" >&2
    exit 1
fi
# The head must already carry its play temperature: the workers sample it raw
# (epsilon 1, temperature 1), and the teacher's own scale plays at 88% fourths.
$PY -c "
import sys, torch
s = torch.load('$START', weights_only=True, map_location='cpu')
t = s.get('play_temperature')
if not t:
    sys.exit('$START has no play temperature folded in; run sharpen_policy.py')
print('starting from a policy sharpened to %g' % t)
"

eval "$($PY -c "
import toml
c = toml.load('$CFG')
print('PORT=%d' % c['online']['remote']['port'])
print('CHAMPION=' + c['baseline']['train']['state_file'])
print('HEAD=' + c['online'].get('head', 'dqn'))
")"
if [ "$HEAD" != policy ]; then
    echo "$CFG publishes a $HEAD head; the workers would play the wrong thing" >&2
    exit 1
fi

CPUS=$(nproc)
LOADER_RAYON=${LOADER_RAYON:-$(( CPUS / 4 / 6 ))}
WORKER_RAYON=${WORKER_RAYON:-$(( CPUS * 3 / 4 / WORKERS ))}
[ "$LOADER_RAYON" -lt 1 ] && LOADER_RAYON=1
[ "$WORKER_RAYON" -lt 1 ] && WORKER_RAYON=1
echo "$CPUS cpus: the trainer's loaders x $LOADER_RAYON threads, $WORKERS workers x $WORKER_RAYON threads"

mkdir -p "$RUN" online
# Frozen on purpose, and frozen at the start rather than at the best: the
# opponent is the ruler for this run, and a ruler that moves measures nothing.
[ -e "$CHAMPION" ] || cp "$START" "$CHAMPION"

# Tracked by pid file, one per run directory, so two variants' processes can
# never find each other and nothing that tells them apart has to survive onto a
# command line to grep for.
start() { # name, env=value... cmd...
    local name=$1; shift
    local pidfile=$RUN/$name.pid log=$RUN/$name.log pid
    if [ -e "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q python; then
            echo "$name is already running (pid $pid)"
            return
        fi
    fi
    echo "==== launched $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >> "$log"
    setsid nohup env MORTAL_CFG=$CFG "$@" >> "$log" 2>&1 < /dev/null &
    echo $! > "$pidfile"
    echo "$name started, pid $!"
}

start server $PY server.py
for _ in $(seq 30); do
    sleep 1
    $PY -c "import socket,sys; sys.exit(socket.socket().connect_ex(('127.0.0.1', $PORT)))" && break
done

# The trainer publishes the first parameters, and the workers wait for them, so
# it goes first.
start trainer MORTAL_DEVICE=cuda:0 MORTAL_LOADER_RAYON_THREADS=$LOADER_RAYON \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY train_ppo.py --from "$START" --out "$RUN" $FLAGS

for i in $(seq 0 $((WORKERS - 1))); do
    start "worker$i" MORTAL_DEVICE=cuda:1 MORTAL_WORKER=$i \
        RAYON_NUM_THREADS=$WORKER_RAYON $PY client.py
done

echo
echo "watch it with:  tail -f $RUN/trainer.log $RUN/worker0.log"
echo "what to watch:  'ratio 1 within' on every round (the denominator is right),"
echo "                clipped% (how hard the brake is working), kl to ref (drift),"
echo "                and the workers' 'last N sessions' average against the start."
