#!/bin/bash
# Offline v4 training on a Kaggle TPU v5e-8, from a notebook committed with "Save Version".
#
# In the notebook, one cell (a committed notebook has no idle cutoff; an
# interactive one is stopped when nobody touches it, ssh or not):
#
#   from kaggle_secrets import UserSecretsClient
#   import os, subprocess
#   os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
#   subprocess.run('git clone -q --depth 1 -b train-parquet https://github.com/HHim8826/Mortal.git /root/Mortal'
#                  ' && bash /root/Mortal/ops/kaggle/train_tpu.sh', shell=True, check=True,
#                  env=dict(os.environ, GROW_TO='60', MAX_STEPS='1390000'))
#
# That is the 192x60 run: the offline 800k grown to 60 blocks, one epoch of the
# 2.12M-game corpus (~1.39M steps of 1,024 at ~671 decisions a game), a fresh
# warm-up and cosine over exactly that epoch. At ~80,000 samples/s the data runs
# out after ~4.9 h, inside HOURS.
#
# The cell runs in the notebook kernel, so the TPU environment (PJRT_DEVICE,
# TPU_SKIP_MDS_QUERY, ...) is already set; an ssh shell has to source it.
#
# INIT is a checkpoint in the private model repo, exported as its EMA weights;
# GROW_TO deepens it first. Output goes to /dev/shm/run and is uploaded to
# RUN_REPO/RUN_PATH every BACKUP_MIN minutes while it trains and once more after,
# so the next version can resume, and a session that dies loses at most that
# much. (The first run, from an interactive session over ssh, was cut 25 minutes
# in, before its first backup, and left nothing.)
#
# The corpus and the output live in /dev/shm (164 GB of RAM), not on disk: the
# host's disk has been throttled to ~1 MB/s for reads the page cache does not
# hold (measured 2026-09-24), which starves the loader and stalls every save.
set -euo pipefail
# mortal-800k.pth at the repo root is the offline run's save at step 800,000 (sha256 98e40026),
# the checkpoint the gate plays as the baseline; --ema takes its average.
INIT=${INIT:-mortal-800k.pth}
INIT_EMA=${INIT_EMA:---ema}
GROW_TO=${GROW_TO:-}
STEPS=${STEPS:-0}
# The cosine's length; empty keeps config.tpu.toml's.
MAX_STEPS=${MAX_STEPS:-}
BACKUP_MIN=${BACKUP_MIN:-30}
# Under the 9 h a session gets, with room for the upload after it.
HOURS=${HOURS:-8}
MODEL_REPO=${MODEL_REPO:-hhim8826/mortal4-0911}
RUN_REPO=${RUN_REPO:-hhim8826/mortal4-0911}
RUN_PATH=${RUN_PATH:-tpu-run}
OUT=/dev/shm/run

cd /root
echo "== libriichi"
[ -x ~/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
(cd /root/Mortal && git log --oneline -1 &&
 PYO3_PYTHON=$(command -v python3) ~/.cargo/bin/cargo build -q -p libriichi --release --lib &&
 cp target/release/libriichi.so mortal/libriichi.so)
pip install -q toml

echo "== corpus and starting point"
python3 - <<EOF
import os
from huggingface_hub import hf_hub_download, snapshot_download
try:
    from huggingface_hub import errors
except ImportError:
    from huggingface_hub import utils as errors
snapshot_download('hhim8826/tenhou-houou-mjai', repo_type='dataset', local_dir='/dev/shm/hf-dataset')
# The loader's reward net, which is not in git.
hf_hub_download('${MODEL_REPO}', 'grp/grp.pth', local_dir='/root/grp')
os.makedirs('/root/Mortal/mortal/grp_v2', exist_ok=True)
os.replace('/root/grp/grp/grp.pth', '/root/Mortal/mortal/grp_v2/grp.pth')
if '${INIT}':
    hf_hub_download('${MODEL_REPO}', '${INIT}', local_dir='/root/init')
# A previous version's state, to resume from. Only a state that the Hub says is not
# there starts the run afresh: a download that failed for any other reason stops the
# script here, or a fresh run would be uploaded over the one it could not fetch.
try:
    hf_hub_download('${RUN_REPO}', '${RUN_PATH}/state.msgpack', local_dir='/root/resume')
except errors.LocalEntryNotFoundError:
    # The Hub was never asked: a dropped connection with nothing cached. It is a
    # subclass of EntryNotFoundError, so it has to go through before that is caught.
    raise
except getattr(errors, 'RemoteEntryNotFoundError', errors.EntryNotFoundError):
    print('no state at ${RUN_REPO}/${RUN_PATH}; starting fresh')
else:
    os.makedirs('${OUT}', exist_ok=True)
    os.replace('/root/resume/${RUN_PATH}/state.msgpack', '${OUT}/state.msgpack')
    print('resuming from ${RUN_REPO}/${RUN_PATH}/state.msgpack')
EOF
cd /root/Mortal/mortal
INIT_ARGS=()
if [ -n "$INIT" ]; then
    python3 -m tpu.convert export "/root/init/$INIT" /root/init.npz $INIT_EMA
    INIT_ARGS=(--init /root/init.npz)
    [ -n "$GROW_TO" ] && INIT_ARGS+=(--grow-to "$GROW_TO")
fi

export MORTAL_CFG=config.tpu.toml MORTAL_LOADER_RAYON_THREADS=3
if [ -n "$MAX_STEPS" ]; then
    python3 -c "import toml; c = toml.load('config.tpu.toml'); c['optim']['scheduler']['max_steps'] = $MAX_STEPS; toml.dump(c, open('/root/cfg_run.toml', 'w'))"
    export MORTAL_CFG=/root/cfg_run.toml
fi
echo "== one batch through the loader, before the chips are touched"
python3 - <<EOF
import copy
from config import config
from tpu.run import build_file_list, loader
cfg = copy.deepcopy(config)
cfg['dataset']['num_workers'] = 0
data = loader(cfg, build_file_list(cfg['dataset'], seed=0)[:1], 8, 0)
batch = next(iter(data))
print('loader ok:', [tuple(t.shape) for t in batch])
# In this process the dataset's decode thread is already a batch ahead, inside Rust,
# and it keeps its own generator alive, so nothing would stop it: at exit it comes
# back for the GIL and aborts the process ("FATAL: exception not rethrown", exit
# 134), which is how this check once ended a run before it started. Closing the
# generator runs decoded_ahead's finally, which waits for that decode to finish.
data.dataset.iterator.close()
EOF

backup() {
    python3 - <<EOF
from huggingface_hub import HfApi
api = HfApi()
if not api.repo_info('${RUN_REPO}').private:
    raise SystemExit('${RUN_REPO} is public; refusing to upload checkpoints to it')
# run.py writes every file under a temporary name and renames it, so each file
# read here is a whole save; the temporaries themselves are left out.
api.upload_folder(folder_path='${OUT}', repo_id='${RUN_REPO}', path_in_repo='${RUN_PATH}',
                  ignore_patterns=['*.tmp', '*.tmp.npz'], commit_message='tpu run: ${1}')
EOF
}

echo "== train"
mkdir -p "$OUT"
( while sleep $((BACKUP_MIN * 60)); do
      [ -f "$OUT/state.msgpack" ] && { backup "backup during training" || echo "backup failed; will retry"; }
  done ) &
BACKUP_PID=$!
# --remat: faster on the v5e, not only smaller. A run that fails still has its last
# save uploaded before the script exits with its status.
status=0
python3 -m tpu.run --out "$OUT" "${INIT_ARGS[@]}" --steps "$STEPS" --hours "$HOURS" --remat 2>&1     | tee "$OUT/train.log" || status=$?
pkill -P $BACKUP_PID 2>/dev/null || true      # an upload under way, or the sleep
kill $BACKUP_PID 2>/dev/null || true

echo "== upload"
[ -f "$OUT/state.msgpack" ] && backup "state and weights after the run (exit $status)"
exit $status
