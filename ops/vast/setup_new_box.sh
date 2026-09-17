#!/bin/bash
# One-time setup of a rented box for the v4 run. Idempotent: safe to run again.
#   bash setup_new_box.sh
set -euo pipefail
exec > >(tee -a /root/setup.log) 2>&1
echo "=== setup started $(date -u)"

echo "=== machine"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader
nvidia-smi topo -m 2>/dev/null | head -4 || true
echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)   nproc: $(nproc)"
echo "cgroup memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"
lscpu | grep -E '^(Model name|Socket|NUMA node|Thread|Core)' || true
free -g | head -2
df -h / | tail -1
PY=$(command -v python3)
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), f"need Python >= 3.10 (upstream uses match), have {sys.version}"; print("python", sys.version.split()[0])'

echo "=== system packages"
need=""
command -v git >/dev/null || need="$need git"
command -v cc >/dev/null || need="$need build-essential"
"$PY" -m venv --help >/dev/null 2>&1 || need="$need python3-venv"
if [ -n "$need" ]; then apt-get update -qq && apt-get install -y -qq $need; fi

echo "=== python env"
[ -x /root/venv/bin/python ] || "$PY" -m venv /root/venv
/root/venv/bin/pip install -q --no-cache-dir --upgrade pip
# The same torch as the first box: compile, fused AdamW and NCCL behave alike.
/root/venv/bin/pip install -q --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
/root/venv/bin/pip install -q --no-cache-dir numpy tqdm toml tensorboard pyarrow huggingface_hub
/root/venv/bin/python - <<'EOF'
import torch
n = torch.cuda.device_count()
print('torch', torch.__version__, '| gpus', n, [torch.cuda.get_device_name(i) for i in range(n)])
if n >= 2:
    print('peer-to-peer 0<->1:', torch.cuda.can_device_access_peer(0, 1))
EOF

echo "=== code"
[ -d /root/Mortal/.git ] || git clone -q -b train-parquet https://github.com/HHim8826/Mortal.git /root/Mortal
cd /root/Mortal
git fetch -q origin train-parquet && git checkout -q train-parquet && git merge -q --ff-only origin/train-parquet
git remote | grep -qx upstream || git remote add upstream https://github.com/Equim-chan/Mortal.git
git log --oneline -1

echo "=== rust + libriichi"
[ -x /root/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
/root/.cargo/bin/cargo --version
PYO3_PYTHON=/root/venv/bin/python /root/.cargo/bin/cargo build -p libriichi --lib --release 2>&1 | tail -2
cp target/release/libriichi.so mortal/libriichi.so.new && mv mortal/libriichi.so.new mortal/libriichi.so
cd mortal && /root/venv/bin/python -c "from libriichi.consts import obs_shape; print('libriichi ok, v4 obs', obs_shape(4))" && cd ..

echo "=== dataset"
/root/venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download('hhim8826/tenhou-houou-mjai', repo_type='dataset',
                  allow_patterns=['data/*.parquet'], local_dir='/root/hf-dataset', max_workers=8)
EOF
echo "$(ls /root/hf-dataset/data/*.parquet | wc -l) shards, $(du -sh /root/hf-dataset/data | cut -f1)"

mkdir -p /root/Mortal/mortal/grp_v2 /root/Mortal/mortal/logs/v4
echo "=== setup done $(date -u)"
df -h / | tail -1
