#!/bin/bash
# Start the online run on this box: the parameter/replay server, the trainer on
# GPU 0, and the self-play workers on GPU 1.
#
#     /root/launch_online.sh              # 3 workers
#     WORKERS=4 /root/launch_online.sh
#     MORTAL_CFG_ONLINE=config.online-smoke.toml WORKERS=1 SEED_FROM= /root/launch_online.sh
#
# The first launch seeds the run from offline: the trainee starts as the best
# offline model, and its opponent is a frozen copy of that same model, so from
# here it has to beat itself rather than the v3 bot. After that the run
# directory is the run, and this starts whatever is not already up.
#
# Offline must be stopped first. They cannot share the box: each side wants
# every CPU. /root/watchdog.sh restarts *offline* when it finds no trainer, so
# stop it (touch /root/watchdog.off) before switching, and point it at this
# script afterwards.
set -euo pipefail
cd /root/Mortal/mortal

CFG=${MORTAL_CFG_ONLINE:-config.online.toml}
WORKERS=${WORKERS:-3}
# The offline model the run starts from, and the copy of it the workers play
# against. Empty to start from random weights, which only a smoke test wants.
#
# The averaged weights, not best.pth: over eleven paired evaluations on the
# same walls the average was ahead by 0.0172 +- 0.0053 of avg_rank, which is
# the 2 SE agreed for preferring it. It is a whole checkpoint, so aux_net comes
# with it; the optimizer and the schedule inside it are ignored, because
# train.py takes those from the online config when an online run starts from an
# offline checkpoint.
SEED_FROM=${SEED_FROM-logs/v4/best_ema.pth}
PY=/root/venv/bin/python

# Wherever the chosen config keeps its run: a smoke config points somewhere
# else entirely, and nothing below should have to know which is which.
eval "$($PY -c "
import os, toml
c = toml.load('$CFG')
print('RUN=' + os.path.dirname(c['control']['state_file']))
print('PORT=%d' % c['online']['remote']['port'])
print('CHAMPION=' + c['baseline']['train']['state_file'])
print('BEST_EMA=' + os.path.splitext(c['control']['best_state_file'])[0] + '_ema.pth')
")"

# nproc reports the host's CPUs, not the ones this container may use: 256
# against a cgroup quota of 61 on the box this was written for. Sizing the
# thread pools from it oversubscribes by four times -- 3 workers x 64 rayon
# threads on 61 CPUs -- so take the quota when there is one.
CPUS=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    if [ "$quota" != max ] && [ "$period" -gt 0 ]; then
        CPUS=$(( quota / period ))
        [ "$CPUS" -lt 1 ] && CPUS=1
    fi
fi
LOADER_RAYON=${LOADER_RAYON:-$(( CPUS / 4 / 6 ))}
WORKER_RAYON=${WORKER_RAYON:-$(( CPUS * 3 / 4 / WORKERS ))}
[ "$LOADER_RAYON" -lt 1 ] && LOADER_RAYON=1
[ "$WORKER_RAYON" -lt 1 ] && WORKER_RAYON=1
echo "$CPUS cpus: the trainer's loaders x $LOADER_RAYON threads, $WORKERS workers x $WORKER_RAYON threads"

mkdir -p "$RUN" online
if [ -e "$RUN/mortal.pth" ]; then
    step=$($PY -c "import torch; print(torch.load('$RUN/mortal.pth', weights_only=True, map_location='cpu')['steps'])")
    echo "resuming the online run from step $step"
elif [ -n "$SEED_FROM" ]; then
    if [ ! -e "$SEED_FROM" ]; then
        echo "no $SEED_FROM to start from" >&2
        exit 1
    fi
    step=$($PY -c "import torch; print(torch.load('$SEED_FROM', weights_only=True, map_location='cpu')['steps'])")
    echo "seeding the online run from $SEED_FROM (step $step)"
    cp "$SEED_FROM" "$RUN/mortal.pth"
    # The opponent, frozen here on purpose: refresh it by hand, between
    # sessions, or one buffer will hold games played against two of them.
    [ -e "$CHAMPION" ] || cp "$SEED_FROM" "$CHAMPION"
    # And the gate's champion, which is a different file: the model the first
    # evaluation has to beat. Without it that evaluation finds no champion and
    # crowns whatever it measured, so the seed is never a bar at all.
    [ -e "$BEST_EMA" ] || cp "$SEED_FROM" "$BEST_EMA"
else
    echo "starting from random weights"
fi

# `env A=b cmd` execs cmd, so nothing that tells these processes apart -- the
# config, the GPU, the worker number -- survives onto a command line to grep
# for. They are tracked by pid file instead, one per run directory, which also
# keeps two configs' runs from finding each other.
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

# The workers ask for parameters as soon as they are up, and the trainer is
# what puts the first set there, so it goes before them.
# The trainer shares one GPU between a 1024 batch and, every 10,000 steps, a
# 4,000-game evaluation. Expandable segments let the allocator hand the
# evaluation memory the training steps have finished with, rather than keeping
# it in fixed blocks of the wrong size.
start trainer MORTAL_DEVICE=cuda:0 MORTAL_LOADER_RAYON_THREADS=$LOADER_RAYON \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train.py

for i in $(seq 0 $((WORKERS - 1))); do
    start "worker$i" MORTAL_DEVICE=cuda:1 MORTAL_WORKER=$i \
        RAYON_NUM_THREADS=$WORKER_RAYON $PY client.py
done

# One TensorBoard for both runs, bound to localhost: reachable only through the
# SSH tunnel. It watches the parent of logs/v4 and logs/v4o, so the offline and
# online runs appear side by side and neither launcher has to care which of
# them started it. Anything already on the port watching something narrower is
# replaced -- that is the bug this replaced.
TB_LOGDIR=/root/Mortal/mortal/logs
if ! pgrep -f "[t]ensorboard --logdir $TB_LOGDIR --host" >/dev/null; then
    pkill -f "[t]ensorboard.*--port 6007"
    sleep 1
    setsid nohup /root/venv/bin/tensorboard --logdir "$TB_LOGDIR" \
        --host 127.0.0.1 --port 6007 > /root/tensorboard.log 2>&1 < /dev/null &
    echo "tensorboard started on 127.0.0.1:6007 over $TB_LOGDIR"
fi
