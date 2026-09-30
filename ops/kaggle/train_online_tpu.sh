#!/bin/bash
# Online v4 training on a Kaggle TPU v5e-8 (`tpu.online`), from a notebook committed with
# "Save Version". One cell, as for train_tpu.sh:
#
#   from kaggle_secrets import UserSecretsClient
#   import os, subprocess
#   os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
#   subprocess.run('rm -rf /root/Mortal && git clone -q --depth 1 -b train-parquet https://github.com/HHim8826/Mortal.git /root/Mortal'
#                  ' && bash /root/Mortal/ops/kaggle/train_online_tpu.sh', shell=True, check=True,
#                  env=dict(os.environ, RUN_PATH='online-tpu60'))
#
# rm -rf first so the cell can run again in the same session; an interactive session
# that has run for a while needs HOURS below the 9 h it has left (it is 8 by default).
#
# The algorithm is config.online.toml's (see config.online.tpu.toml); it starts from INIT
# and plays against OPPONENT, both .npz in the model repo -- by default the 192x60 offline
# run's EMA (tpu-run/weights_ema.npz, step 1,354,328), the trainee against a frozen copy
# of where it started. The gate's first champion is that same start.
#
# Before the run, a smoke run of a few steps on the chips plays self-play, trains, saves
# and gates once, on a config shrunk to minutes, so a fault in any of those shows here and
# not at the first real gate an hour or two in.
#
# Output goes to this run's own folder under /dev/shm/runs (named by RUN_REPO/RUN_PATH, so
# another run in the same session never finds its state) and is uploaded to
# RUN_REPO/RUN_PATH every BACKUP_MIN minutes and once more after, from a snapshot of one
# save (backup.py); a later version with the same RUN_PATH resumes it, gate and all, from
# its own folder if it ran earlier in the session and from the Hub if not. A state
# already at RUN_PATH is always resumed, so a new run needs a RUN_PATH of its own. The
# gate stopping the run (two evaluations without a new champion, exit 3) is
# how it is meant to end: the script then exits 0 after the upload. The model to use is
# champion.npz.
set -euo pipefail
MODEL_REPO=${MODEL_REPO:-hhim8826/mortal4-0911}
INIT=${INIT:-tpu-run/weights_ema.npz}
OPPONENT=${OPPONENT:-$INIT}
RUN_REPO=${RUN_REPO:-hhim8826/mortal4-0911}
RUN_PATH=${RUN_PATH:-online-tpu60}
STEPS=${STEPS:-0}
BACKUP_MIN=${BACKUP_MIN:-30}
# Under the 9 h a session gets, with room for the smoke run and the upload after.
HOURS=${HOURS:-8}
CONFIG=${CONFIG:-config.online.tpu.toml}
SMOKE=${SMOKE:-1}
# This run's own folder, named by where it is backed up (#48).
OUT=/dev/shm/runs/$(printf '%s' "$RUN_REPO/$RUN_PATH" | tr '/' '_')

cd /root
echo "== libriichi"
[ -x ~/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
(cd /root/Mortal && git log --oneline -1 &&
 PYO3_PYTHON=$(command -v python3) ~/.cargo/bin/cargo build -q -p libriichi --release --lib &&
 cp target/release/libriichi.so mortal/libriichi.so)
pip install -q toml

echo "== nets and state"
python3 - <<EOF
import os
import shutil
from huggingface_hub import hf_hub_download, snapshot_download
try:
    from huggingface_hub import errors
except ImportError:
    from huggingface_hub import utils as errors
# The loader's reward net, which is not in git.
hf_hub_download('${MODEL_REPO}', 'grp/grp.pth', local_dir='/root/grp')
os.makedirs('/root/Mortal/mortal/grp_v2', exist_ok=True)
os.replace('/root/grp/grp/grp.pth', '/root/Mortal/mortal/grp_v2/grp.pth')
for name in {'${INIT}', '${OPPONENT}'}:
    hf_hub_download('${MODEL_REPO}', name, local_dir='/root/nets')
# What the gate scores against (sha256 475f95ab).
hf_hub_download('${MODEL_REPO}', 'baseline/baseline.pth', local_dir='/root/nets')
# Where to resume from. This run's own folder first: it has a state only if this run ran
# earlier in this session, and then it is at least as new as anything it backed up.
# Otherwise a previous version's, from the Hub. Only a state that the Hub says is not
# there starts afresh: a download that failed for any other reason stops the script here,
# or a fresh run would be uploaded over the one it could not fetch.
out, resume = '${OUT}', '${OUT}.resume'
if os.path.exists(f'{out}/state.msgpack'):
    print(f'resuming from {out}, left by this run earlier in this session')
else:
    try:
        hf_hub_download('${RUN_REPO}', '${RUN_PATH}/state.msgpack', local_dir=resume)
    except errors.LocalEntryNotFoundError:
        raise
    except getattr(errors, 'RemoteEntryNotFoundError', errors.EntryNotFoundError):
        print('no state at ${RUN_REPO}/${RUN_PATH}; starting fresh')
    else:
        snapshot_download('${RUN_REPO}', local_dir=resume, allow_patterns=[
            '${RUN_PATH}/gate.json', '${RUN_PATH}/gate.jsonl', '${RUN_PATH}/champion.npz'])
        # Fetched beside OUT, in /dev/shm, so os.replace can move it there.
        os.makedirs(out, exist_ok=True)
        for name in os.listdir(f'{resume}/${RUN_PATH}'):
            os.replace(f'{resume}/${RUN_PATH}/{name}', f'{out}/{name}')
        print('resuming from ${RUN_REPO}/${RUN_PATH}:', sorted(os.listdir(out)))
    shutil.rmtree(resume, ignore_errors=True)
EOF
cd /root/Mortal/mortal
python3 -m tpu.convert export /root/nets/baseline/baseline.pth /root/nets/baseline.npz
ARGS=(--init "/root/nets/$INIT" --opponent "/root/nets/$OPPONENT" --baseline /root/nets/baseline.npz)
export MORTAL_CFG=$CONFIG MORTAL_LOADER_RAYON_THREADS=3

if [ "$SMOKE" = 1 ]; then
    echo "== smoke run: self-play, training, a save and a gate, in minutes"
    python3 - <<EOF
import toml
c = toml.load('$CONFIG')
c['control'].update(batch_size=64, save_every=2, test_every=2, submit_every=1)
c['test_play'].update(games=8, gate_patience=0)
c['tpu_online'].update(arenas=1, walls=1, gate_arenas=1)
c['online']['server']['capacity'] = 50
toml.dump(c, open('/root/cfg_smoke.toml', 'w'))
EOF
    rm -rf /dev/shm/online-smoke
    # To a file, not through a pipe: see the training run below.
    smoke=0
    MORTAL_CFG=/root/cfg_smoke.toml python3 -m tpu.online --out /dev/shm/online-smoke "${ARGS[@]}" \
        --steps 4 --log-every 1 --remat > /root/online-smoke.log 2>&1 || smoke=$?
    tail -20 /root/online-smoke.log
    [ "$smoke" = 0 ] || { echo "smoke run: exit $smoke"; exit 1; }
    grep -q '"steps": 4' /dev/shm/online-smoke/gate.jsonl || { echo "smoke run: no gate at step 4"; exit 1; }
    rm -rf /dev/shm/online-smoke
    # The chips take a while to come free after a process that held them exits.
    sleep 30
fi

# From a snapshot of one save that holds still while it uploads (#49): see backup.py.
backup() {
    python3 /root/Mortal/ops/kaggle/backup.py --out "$OUT" --repo "$RUN_REPO" --path "$RUN_PATH" \
        --message "online run: $1"
}

echo "== train"
mkdir -p "$OUT"
( while sleep $((BACKUP_MIN * 60)); do
      [ -f "$OUT/state.msgpack" ] && { backup "backup during training" || echo "backup failed; will retry"; }
  done ) &
BACKUP_PID=$!
status=0
# The trainer is waited for by itself, not as `| tee`: a pipeline ends only once everything
# holding its write end has, and a process a killed trainer left behind held it for good (#53).
python3 -m tpu.online --out "$OUT" "${ARGS[@]}" --steps "$STEPS" --hours "$HOURS" --remat \
    > >(tee -a "$OUT/train.log") 2>&1 || status=$?
TEE_PID=$!
# The log's last lines, before it is uploaded; not for ever, for the same reason.
for _ in $(seq 60); do kill -0 $TEE_PID 2>/dev/null || break; sleep 1; done
pkill -P $BACKUP_PID 2>/dev/null || true      # an upload under way, or the sleep
kill $BACKUP_PID 2>/dev/null || true

echo "== upload"
[ -f "$OUT/state.msgpack" ] && backup "after the run (exit $status)"
if [ "$status" = 3 ]; then
    echo "the gate stopped the run, as it is meant to end; the champion is $RUN_PATH/champion.npz"
    status=0
fi
exit $status
