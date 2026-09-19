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
# Which GPUs the self-play workers use, round-robin. Both, by default, and for
# a measured reason: five workers on one A4000 pinned it at 100% while the
# trainer's card sat at 0% and the container used 36 of its 61 CPUs. The
# trainer only works in bursts, after a drain, so its card is nearly free the
# rest of the time. WORKER_GPUS=1 keeps them off it.
# The trainer already sits on cuda:0, so the first worker goes to the other
# card and an odd number of them leaves the busier side away from it.
WORKER_GPUS=${WORKER_GPUS:-1,0}
START=${START:-logs/policy/policy-t0.05.pth}
PY=/root/venv/bin/python
# SUFFIX keeps a second run of the same variant apart from the first, so one
# knob can be changed without overwriting what it is being compared against.
RUN=logs/ppo/$VARIANT${SUFFIX:-}

# 0.01, not 0.1. Measured on this box: at 0.1 the policy settled at a KL of
# 0.0011 from its reference by step 750 and was still at 0.0011 after 6,150 --
# the anchor's pull cancels the gradient's almost immediately, which is a
# freeze rather than a constraint, and a variant that cannot move teaches
# nothing about whether moving helps.
case $VARIANT in
    kl)         FLAGS="--kl-coef 0.01 --ref-refresh 0" ;;
    kl-refresh) FLAGS="--kl-coef 0.01 --ref-refresh 10" ;;
    plain)      FLAGS="--kl-coef 0" ;;
    # A positive control, not a candidate. The policy is played by sampling at
    # the temperature folded into its head, and phase 2 measured what that
    # costs against its own argmax: +0.023 of rank, 1.1 pt. That is the largest
    # certain gain in front of this run, and the entropy bonus is what stops it
    # being taken -- so with the bonus off and nothing anchoring, a working
    # pipeline should sharpen the policy and walk the self-play average from
    # 2.52 towards 2.50. If it cannot do that, the problem is not a
    # hyperparameter.
    sharp)      FLAGS="--kl-coef 0 --ent-coef 0" ;;
    *) echo "unknown VARIANT $VARIANT: expected kl, kl-refresh or plain" >&2; exit 1 ;;
esac

# The learning rate is a knob rather than a variant: the arithmetic says the
# policy moves about 0.12 of a logit in 1,500 updates at 3e-4, which changes
# the entropy by 0.005 and nothing measurable in strength. Raising it scales
# signal and noise alike, so the trajectory's quality is unchanged and it
# simply arrives sooner.
[ -n "${LR:-}" ] && FLAGS="$FLAGS --lr $LR"

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
IFS=',' read -r -a GPUS <<< "$WORKER_GPUS"
echo "$CPUS cpus: the trainer's loaders x $LOADER_RAYON threads, $WORKERS workers x $WORKER_RAYON threads"
echo "workers on gpu(s) $WORKER_GPUS, trainer on gpu 0"

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
    start "worker$i" MORTAL_DEVICE=cuda:${GPUS[$(( i % ${#GPUS[@]} ))]} MORTAL_WORKER=$i MORTAL_TB_DIR=$RUN/tb \
        RAYON_NUM_THREADS=$WORKER_RAYON $PY client.py
done

# Every variant under one logdir, so the three read against each other, and
# bound to localhost: reachable only through the SSH tunnel
# (ssh -L 6007:localhost:6007). The trainer writes the PPO diagnostics, the
# workers the self-play average -- which is the number that moves first.
TB_LOGDIR=/root/Mortal/mortal/logs/ppo
if ! pgrep -f "[t]ensorboard --logdir $TB_LOGDIR --host" >/dev/null; then
    pkill -f "[t]ensorboard.*--port 6007" || true
    sleep 1
    setsid nohup /root/venv/bin/tensorboard --logdir "$TB_LOGDIR"         --host 127.0.0.1 --port 6007 > /root/tensorboard.log 2>&1 < /dev/null &
    echo "tensorboard started on 127.0.0.1:6007 over $TB_LOGDIR"
fi

# Two-hourly copy of the checkpoint, the logs and the events into the private
# repo, under ppo/<variant>. It refuses to upload if that repo is not private.
if ! pgrep -f "[b]ackup_ppo_hf.py" >/dev/null; then
    setsid nohup env MORTAL_RUN=/root/Mortal/mortal/$RUN         MORTAL_CFG_PATH=/root/Mortal/mortal/$CFG         $PY /root/Mortal/ops/vast/backup_ppo_hf.py --loop         > "$RUN/backup.log" 2>&1 < /dev/null &
    echo "hugging face backup started, every 2 h into ppo/$VARIANT"
fi

echo
echo "watch it with:  tail -f $RUN/trainer.log $RUN/worker0.log"
echo "                and ssh -L 6007:localhost:6007, then http://localhost:6007"
echo "what to watch:  'ratio 1 within' on every round (the denominator is right),"
echo "                clipped% (how hard the brake is working), kl to ref (drift),"
echo "                and the workers' 'last N sessions' average against the start."
