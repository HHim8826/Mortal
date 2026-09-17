#!/bin/bash
# Hand the box over from the offline run to the online one.
#
#     /root/switch_to_online.sh --dry-run     # say what it would do
#     /root/switch_to_online.sh               # do it
#
# The offline run learns to imitate the corpus and is capped by it; online is
# where the model plays itself and can pass the humans it learned from. This
# stops the first, seeds the second from the best weights the first produced,
# and points the watchdog at it.
#
# What it starts from is `best_ema.pth`, the averaged weights, not `best.pth`:
# over eleven paired evaluations on the same walls the average was ahead by
# 0.0172 +- 0.0053 of avg_rank, which is the 2 SE agreed for preferring it. It
# is a whole checkpoint, so the online trainer gets aux_net with the weights.
# The optimizer and the schedule inside it are NOT used -- train.py takes those
# fresh from the online config when an online run starts from an offline
# checkpoint -- so the run begins with a 200-step warm-up to a flat 1e-5.
set -euo pipefail

MORTAL=/root/Mortal/mortal
PY=/root/venv/bin/python
SEED=${SEED:-$MORTAL/logs/v4/best_ema.pth}
OFFLINE=$MORTAL/logs/v4
ONLINE=$MORTAL/logs/v4o
WORKERS=${WORKERS:-3}

dry=''
force=''
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry=1 ;;
        --force) force=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

step_of() {
    "$PY" - "$1" <<'PY'
import sys, torch
s = torch.load(sys.argv[1], weights_only=True, map_location='cpu')
best = s.get('best_perf') or {}
print(f"step {s['steps']:,}"
      + (f", test play avg_rank {best['avg_rank']:.4f} / avg_pt {best['avg_pt']:+.3f}"
         if 'avg_rank' in best else ''))
PY
}

say() { echo "==> $*"; }

# ---------------------------------------------------------------- the checks

[ -e "$SEED" ] || { echo "no $SEED to start from" >&2; exit 1; }
say "starting weights: $SEED"
say "  $(step_of "$SEED")"

free_gb=$(df -BG --output=avail /root | tail -1 | tr -cd '0-9')
say "disk free: ${free_gb}G"
if [ "$free_gb" -lt 6 ]; then
    echo "under 6G free: the workers' game logs will fill it" >&2
    [ -n "$force" ] || exit 1
fi

if pgrep -f "[t]rain.py" >/dev/null && [ -z "$force" ]; then
    # Only stop a run that has done what it was asked to. Cutting one short by
    # accident costs hours, and this script is meant to be safe to run twice.
    target=$(tr -cd '0-9' < /root/stop_at_step 2>/dev/null || true)
    now=$(tail -c 400000 "$OFFLINE/train.log" | tr '\r' '\n' \
          | grep -a 'total steps:' | tail -1 | sed 's/.*total steps: //; s/ .*//; s/,//g')
    if [ -z "$target" ]; then
        echo "the offline run is going and no /root/stop_at_step says where it ends;" >&2
        echo "pass --force to stop it here (step ${now:-?})" >&2
        exit 1
    fi
    if [ -z "$now" ] || [ "$now" -lt "$target" ]; then
        echo "the offline run is at step ${now:-?}, short of the $target it stops at;" >&2
        echo "pass --force to cut it short" >&2
        exit 1
    fi
    say "offline reached step $now of $target; stopping it"
fi

if [ -n "$dry" ]; then
    say "dry run; nothing changed"
    say "would: stop offline, write /root/phase=online, remove /root/stop_at_step,"
    say "       seed $ONLINE from $SEED, launch the server, the trainer and $WORKERS workers"
    exit 0
fi

# ----------------------------------------------------------------- the switch

# Hands off while this runs: the watchdog would otherwise start the online run
# from its own loop, halfway through this one. Lifted however this exits --
# left behind by a failure it would quietly stop the watchdog looking after
# anything at all, which is the one state nobody would notice.
touch /root/watchdog.off
trap 'rm -f /root/watchdog.off' EXIT

say "stopping the offline run"
pkill -f "[t]orchrun --standalone" 2>/dev/null || true
pkill -f "[t]rain.py" 2>/dev/null || true
for _ in $(seq 60); do
    pgrep -f "[t]rain.py" >/dev/null || break
    sleep 2
done
pkill -9 -f "[t]rain.py" 2>/dev/null || true
sleep 3

# The offline checkpoint the online run starts from, kept where a later look
# back can find it: best_ema.pth itself keeps being overwritten by whatever
# run owns logs/v4.
mkdir -p "$ONLINE"
cp "$SEED" "$ONLINE/seeded_from.pth"

say "phase -> online"
echo online > /root/phase
rm -f /root/stop_at_step

# It is watching logs/v4; the watchdog starts it again against logs/v4o.
pkill -f "[b]ackup_hf.py" 2>/dev/null || true

say "launching"
SEED_FROM="$SEED" WORKERS="$WORKERS" bash /root/launch_online.sh

say "watchdog is back on, phase online"
say "logs: $ONLINE/{server,trainer,worker*}.log"
say "to go back: echo offline > /root/phase  (and the offline run resumes from its own checkpoint)"
