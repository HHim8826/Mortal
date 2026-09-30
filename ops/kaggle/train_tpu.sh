#!/bin/bash
# Offline v4 training on a Kaggle TPU v5e-8, from a notebook committed with "Save Version".
#
# In the notebook, one cell (a committed notebook has no idle cutoff; an
# interactive one is stopped when nobody touches it, ssh or not):
#
#   from kaggle_secrets import UserSecretsClient
#   import os, subprocess
#   os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
#   subprocess.run('rm -rf /root/Mortal && git clone -q --depth 1 -b train-parquet https://github.com/HHim8826/Mortal.git /root/Mortal'
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
# GROW_TO deepens it first. Output goes to /dev/shm/runs (below) and is uploaded to
# RUN_REPO/RUN_PATH every BACKUP_MIN minutes while it trains and once more after,
# so the next version can resume, and a session that dies loses at most that
# much. (The first run, from an interactive session over ssh, was cut 25 minutes
# in, before its first backup, and left nothing.)
#
# A state already at RUN_REPO/RUN_PATH is resumed, and reads on from where it
# was in the data (#55): one that has read all of it ends at once, so a new run
# needs a RUN_PATH of its own. `tpu-run` holds the 192x60 run, done at step
# 1,354,328 on 2026-09-27; it was saved before the position was kept, and
# tpu.run refuses to run the data out on it again unless STEPS says how far.
# Locally each run has its own folder under /dev/shm/runs, named by
# RUN_REPO/RUN_PATH, so a second run in one session never finds the first's
# state; a run that ran earlier in the session resumes from its own folder.
#
# Every TEST_EVERY steps, and at the start and the end, the EMA plays the first
# TEST_WALLS dev walls against the v3 baseline (baseline/baseline.pth in the model
# repo) beside training; the results go to test_play.jsonl and are backed up with
# the rest, and a resumed run keeps pairing its tests against its first. 0 plays none.
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
TEST_EVERY=${TEST_EVERY:-100000}
TEST_WALLS=${TEST_WALLS:-500}
# The run's config; another only to try the script on a smaller box.
CONFIG=${CONFIG:-config.tpu.toml}
# This run's own folder, named by where it is backed up: another run later in the same
# session, under another RUN_PATH, must not find this one's state and resume it (#48).
OUT=/dev/shm/runs/$(printf '%s' "$RUN_REPO/$RUN_PATH" | tr '/' '_')

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
import shutil
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
if int('${TEST_EVERY}'):
    # What test play scores against (sha256 475f95ab), the champion of every evaluation.
    hf_hub_download('${MODEL_REPO}', 'baseline/baseline.pth', local_dir='/root/baseline')
# Where to resume from. This run's own folder first: it has a state only if this run
# ran earlier in this session, and then that state is at least as new as anything it
# backed up. Otherwise a previous version's, from the Hub; only a state that the Hub
# says is not there starts the run afresh: a download that failed for any other reason
# stops the script here, or a fresh run would be uploaded over the one it could not fetch.
out, resume = '${OUT}', '${OUT}.resume'
if os.path.exists(f'{out}/state.msgpack'):
    print(f'resuming from {out}, left by this run earlier in this session')
else:
    try:
        hf_hub_download('${RUN_REPO}', '${RUN_PATH}/state.msgpack', local_dir=resume)
    except errors.LocalEntryNotFoundError:
        # The Hub was never asked: a dropped connection with nothing cached. It is a
        # subclass of EntryNotFoundError, so it has to go through before that is caught.
        raise
    except getattr(errors, 'RemoteEntryNotFoundError', errors.EntryNotFoundError):
        print('no state at ${RUN_REPO}/${RUN_PATH}; starting fresh')
    else:
        # Its test play so far, so the tests to come are paired against their series' first.
        snapshot_download('${RUN_REPO}', local_dir=resume,
                          allow_patterns=['${RUN_PATH}/test_play.jsonl', '${RUN_PATH}/test_play/*'])
        # Fetched beside OUT, in /dev/shm: os.replace cannot move a file from the disk
        # into RAM, and from /root every resume stopped here ("Invalid cross-device link").
        os.makedirs(out, exist_ok=True)
        kept = f'{resume}/${RUN_PATH}'
        for name in ('state.msgpack', 'test_play.jsonl', 'test_play'):
            if os.path.exists(f'{kept}/{name}'):
                if os.path.isdir(f'{out}/{name}'):
                    shutil.rmtree(f'{out}/{name}')      # what the Hub holds is this run's record
                os.replace(f'{kept}/{name}', f'{out}/{name}')
        print('resuming from ${RUN_REPO}/${RUN_PATH}:', sorted(os.listdir(out)))
    shutil.rmtree(resume, ignore_errors=True)
EOF
cd /root/Mortal/mortal
INIT_ARGS=()
if [ -n "$INIT" ]; then
    python3 -m tpu.convert export "/root/init/$INIT" /root/init.npz $INIT_EMA
    INIT_ARGS=(--init /root/init.npz)
    [ -n "$GROW_TO" ] && INIT_ARGS+=(--grow-to "$GROW_TO")
fi
TEST_ARGS=()
if [ "$TEST_EVERY" -gt 0 ]; then
    python3 -m tpu.convert export /root/baseline/baseline/baseline.pth /root/baseline.npz
    TEST_ARGS=(--test-every "$TEST_EVERY" --test-baseline /root/baseline.npz --test-walls "$TEST_WALLS")
fi

export MORTAL_CFG=$CONFIG MORTAL_LOADER_RAYON_THREADS=3
if [ -n "$MAX_STEPS" ]; then
    python3 -c "import toml; c = toml.load('$CONFIG'); c['optim']['scheduler']['max_steps'] = $MAX_STEPS; toml.dump(c, open('/root/cfg_run.toml', 'w'))"
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

# From a snapshot of one save that holds still while it uploads, not from the folder the
# trainer keeps renaming files into: see backup.py (#49).
backup() {
    python3 /root/Mortal/ops/kaggle/backup.py --out "$OUT" --repo "$RUN_REPO" --path "$RUN_PATH" \
        --message "tpu run: $1"
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
# The trainer is waited for by itself, not as `| tee`: a pipeline ends only once everything
# holding its write end has, and anything a killed trainer left behind would hold it (#53).
python3 -m tpu.run --out "$OUT" "${INIT_ARGS[@]}" "${TEST_ARGS[@]}" --steps "$STEPS" --hours "$HOURS" --remat \
    > >(tee -a "$OUT/train.log") 2>&1 || status=$?
TEE_PID=$!
# The log's last lines, before it is uploaded; not for ever, for the same reason.
for _ in $(seq 60); do kill -0 $TEE_PID 2>/dev/null || break; sleep 1; done
pkill -P $BACKUP_PID 2>/dev/null || true      # an upload under way, or the sleep
kill $BACKUP_PID 2>/dev/null || true

echo "== upload"
[ -f "$OUT/state.msgpack" ] && backup "state and weights after the run (exit $status)"
exit $status
