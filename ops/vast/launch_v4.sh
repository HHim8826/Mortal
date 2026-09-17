#!/bin/bash
# Start the v4 run and TensorBoard on the box. Both detach from the shell, so
# closing SSH leaves them running.
set -euo pipefail
cd /root/Mortal/mortal

if pgrep -f "[t]rain.py" >/dev/null; then
    echo "training is already running; not starting another" >&2
    exit 1
fi

# train.py resumes from logs/v4/mortal.pth when it exists, so after an
# interruption this carries on from the last save (at most 400 steps back).
mkdir -p logs/v4
if [ -e logs/v4/mortal.pth ]; then
    step=$(/root/venv/bin/python -c "import torch; print(torch.load('logs/v4/mortal.pth', weights_only=True, map_location='cpu')['steps'])")
    echo "resuming from step $step"
else
    echo "starting fresh"
fi

# Appended, not truncated: a restart keeps the earlier runs' log.
echo "==== launched $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >> logs/v4/train.log
# NCCL's LL protocol polls host memory in small pieces, and with the loaders
# busy on the CPU that made the gradient all-reduce the slow part of a step.
setsid nohup env MORTAL_CFG=config.vast.toml NCCL_PROTO=Simple \
    /root/venv/bin/torchrun --standalone --nproc_per_node=2 train.py \
    >> logs/v4/train.log 2>&1 < /dev/null &
echo "training started, pid $!"

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
