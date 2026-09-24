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
#                  ' && bash /root/Mortal/ops/kaggle/train_tpu.sh', shell=True, check=True)
#
# The cell runs in the notebook kernel, so the TPU environment (PJRT_DEVICE,
# TPU_SKIP_MDS_QUERY, ...) is already set; an ssh shell has to source it.
#
# INIT is a checkpoint in the private model repo, exported as its EMA weights;
# GROW_TO deepens it first. Output goes to /kaggle/working/run (20 GB) and is
# uploaded to RUN_REPO/RUN_PATH after the run, so the next version can resume.
set -euo pipefail
# mortal.pth at the repo root is the offline run's last save, step 800,400; --ema takes its average.
INIT=${INIT:-mortal.pth}
INIT_EMA=${INIT_EMA:---ema}
GROW_TO=${GROW_TO:-}
STEPS=${STEPS:-0}
# Under the 9 h a session gets, with room for the upload after it.
HOURS=${HOURS:-8}
MODEL_REPO=${MODEL_REPO:-hhim8826/mortal4-0911}
RUN_REPO=${RUN_REPO:-hhim8826/mortal4-0911}
RUN_PATH=${RUN_PATH:-tpu-run}
OUT=/kaggle/working/run

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
snapshot_download('hhim8826/tenhou-houou-mjai', repo_type='dataset', local_dir='/root/hf-dataset')
if '${INIT}':
    hf_hub_download('${MODEL_REPO}', '${INIT}', local_dir='/root/init')
# A previous version's state, to resume from.
try:
    hf_hub_download('${RUN_REPO}', '${RUN_PATH}/state.msgpack', local_dir='/root/resume')
    os.makedirs('${OUT}', exist_ok=True)
    os.replace('/root/resume/${RUN_PATH}/state.msgpack', '${OUT}/state.msgpack')
    print('resuming from ${RUN_REPO}/${RUN_PATH}/state.msgpack')
except Exception as exc:
    print(f'no state to resume ({type(exc).__name__}); starting fresh')
EOF
cd /root/Mortal/mortal
INIT_ARGS=()
if [ -n "$INIT" ]; then
    python3 -m tpu.convert export "/root/init/$INIT" /root/init.npz $INIT_EMA
    INIT_ARGS=(--init /root/init.npz)
    [ -n "$GROW_TO" ] && INIT_ARGS+=(--grow-to "$GROW_TO")
fi

echo "== train"
mkdir -p "$OUT"
export MORTAL_CFG=config.tpu.toml MORTAL_LOADER_RAYON_THREADS=3
python3 -m tpu.run --out "$OUT" "${INIT_ARGS[@]}" --steps "$STEPS" --hours "$HOURS" 2>&1 | tee "$OUT/train.log"

echo "== upload"
python3 - <<EOF
from huggingface_hub import HfApi
api = HfApi()
if not api.repo_info('${RUN_REPO}').private:
    raise SystemExit('${RUN_REPO} is public; refusing to upload checkpoints to it')
api.upload_folder(folder_path='${OUT}', repo_id='${RUN_REPO}', path_in_repo='${RUN_PATH}',
                  commit_message='tpu run: state and weights')
EOF
