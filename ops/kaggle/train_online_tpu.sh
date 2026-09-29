#!/bin/bash
# Online v4 training on a Kaggle TPU v5e-8 (`tpu.online`), from a notebook committed with
# "Save Version". One cell, as for train_tpu.sh:
#
#   from kaggle_secrets import UserSecretsClient
#   import os, subprocess
#   os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
#   subprocess.run('git clone -q --depth 1 -b train-parquet https://github.com/HHim8826/Mortal.git /root/Mortal'
#                  ' && bash /root/Mortal/ops/kaggle/train_online_tpu.sh', shell=True, check=True,
#                  env=dict(os.environ, RUN_PATH='online-tpu60'))
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
# Output goes to /dev/shm/online and is uploaded to RUN_REPO/RUN_PATH every BACKUP_MIN
# minutes and once more after; a later version with the same RUN_PATH resumes it, gate
# and all. A state already at RUN_PATH is always resumed, so a new run needs a RUN_PATH of
# its own. The gate stopping the run (two evaluations without a new champion, exit 3) is
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
OUT=/dev/shm/online

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
# A previous version's state, to resume from. Only a state that the Hub says is not there
# starts afresh: a download that failed for any other reason stops the script here, or a
# fresh run would be uploaded over the one it could not fetch. Fetched into /dev/shm,
# beside OUT, so os.replace can move it there.
resume = '/dev/shm/resume'
try:
    hf_hub_download('${RUN_REPO}', '${RUN_PATH}/state.msgpack', local_dir=resume)
except errors.LocalEntryNotFoundError:
    raise
except getattr(errors, 'RemoteEntryNotFoundError', errors.EntryNotFoundError):
    print('no state at ${RUN_REPO}/${RUN_PATH}; starting fresh')
else:
    snapshot_download('${RUN_REPO}', local_dir=resume, allow_patterns=[
        '${RUN_PATH}/gate.json', '${RUN_PATH}/gate.jsonl', '${RUN_PATH}/champion.npz'])
    os.makedirs('${OUT}', exist_ok=True)
    for name in os.listdir(f'{resume}/${RUN_PATH}'):
        os.replace(f'{resume}/${RUN_PATH}/{name}', f'${OUT}/{name}')
    print('resuming from ${RUN_REPO}/${RUN_PATH}:', sorted(os.listdir('${OUT}')))
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
    MORTAL_CFG=/root/cfg_smoke.toml python3 -m tpu.online --out /dev/shm/online-smoke "${ARGS[@]}" \
        --steps 4 --log-every 1 --remat 2>&1 | tail -20
    grep -q '"steps": 4' /dev/shm/online-smoke/gate.jsonl || { echo "smoke run: no gate at step 4"; exit 1; }
    rm -rf /dev/shm/online-smoke
    # The chips take a while to come free after a process that held them exits.
    sleep 30
fi

backup() {
    python3 - <<EOF
from huggingface_hub import HfApi
api = HfApi()
if not api.repo_info('${RUN_REPO}').private:
    raise SystemExit('${RUN_REPO} is public; refusing to upload checkpoints to it')
# Every file is written under a temporary name and renamed, so each one read here is whole.
api.upload_folder(folder_path='${OUT}', repo_id='${RUN_REPO}', path_in_repo='${RUN_PATH}',
                  ignore_patterns=['*.tmp', '*.tmp.npz'], commit_message='online run: ${1}')
EOF
}

echo "== train"
mkdir -p "$OUT"
( while sleep $((BACKUP_MIN * 60)); do
      [ -f "$OUT/state.msgpack" ] && { backup "backup during training" || echo "backup failed; will retry"; }
  done ) &
BACKUP_PID=$!
status=0
python3 -m tpu.online --out "$OUT" "${ARGS[@]}" --steps "$STEPS" --hours "$HOURS" --remat 2>&1 \
    | tee -a "$OUT/train.log" || status=$?
pkill -P $BACKUP_PID 2>/dev/null || true      # an upload under way, or the sleep
kill $BACKUP_PID 2>/dev/null || true

echo "== upload"
[ -f "$OUT/state.msgpack" ] && backup "after the run (exit $status)"
if [ "$status" = 3 ]; then
    echo "the gate stopped the run, as it is meant to end; the champion is $RUN_PATH/champion.npz"
    status=0
fi
exit $status
