#!/bin/bash
# Online v4 training on a Kaggle TPU v5e-8 (`tpu.online`), from a notebook committed with
# "Save Version". One cell, as for train_tpu.sh:
#
#   from kaggle_secrets import UserSecretsClient
#   import os, subprocess
#   os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
#   subprocess.run('rm -rf /root/Mortal && git clone -q --depth 1 -b train-parquet https://github.com/HHim8826/Mortal.git /root/Mortal'
#                  ' && bash /root/Mortal/ops/kaggle/train_online_tpu.sh', shell=True, check=True,
#                  env=dict(os.environ, RUN_PATH='online-tpu60-full',
#                           CONFIG_SET='freeze.trainable_blocks=0 control.test_every=20000 test_play.gate_metric=pt'))
#
# rm -rf first so the cell can run again in the same session; an interactive session
# that has run for a while needs HOURS below the 9 h it has left (it is 8 by default).
#
# That is the run after online-tpu60, which trained the last 4 of 60 blocks and was stopped
# by the gate at 60,000 steps with nothing gained: the whole net trains, as in the one online
# phase that measured a gain (520k -> 560k, +2.40 pt on dev, every block training), with the
# gate every 20,000 steps and deciding on pt. CONFIG_SET is "section.key=value ..." over
# CONFIG, each a key it already has and a value of the same type.
#
# A run keeps its settings: its first session writes them into run.json, with the init,
# opponent and baseline it had, and a session resuming it trains with those whatever its
# cell says -- only how the work is spread may differ (tpu.runid.PLACEMENT) -- and prints
# each of its own settings that it does not use (#61). A run that changed algorithm or
# ruler half way would be two runs under one name. To change one anyway, from here on,
# ACCEPT_CHANGES=1; run.json records it.
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
CONFIG_SET=${CONFIG_SET:-}
ACCEPT_CHANGES=${ACCEPT_CHANGES:-0}
SMOKE=${SMOKE:-1}
# This run's own folder, named by where it is backed up (#48): RUN_REPO/RUN_PATH
# percent-encoded, which decodes back to it (`/` -> `_` gave online/tpu60 and online_tpu60
# one folder). A stray / would give one Hub folder two local ones, so it is refused.
case "$RUN_PATH" in ''|/*|*/|*//*) echo "RUN_PATH '$RUN_PATH': no leading, trailing or double /"; exit 1;; esac
OUT=/dev/shm/runs/$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' \
                    "$RUN_REPO/$RUN_PATH")

cd /root
echo "== libriichi"
[ -x ~/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
(cd /root/Mortal && git log --oneline -1 &&
 PYO3_PYTHON=$(command -v python3) ~/.cargo/bin/cargo build -q -p libriichi --release --lib &&
 cp target/release/libriichi.so mortal/libriichi.so)
pip install -q toml

if [ -n "$CONFIG_SET" ]; then
    echo "== config: $CONFIG with $CONFIG_SET"
    # Here, before anything is fetched: a key that is not in the config stops the script.
    (cd /root/Mortal/mortal && python3 - "$CONFIG" "$CONFIG_SET" /root/cfg_run.toml <<'EOF'
import ast
import sys
import toml
src, sets, dst = sys.argv[1:]
c = toml.load(src)
for item in sets.split():
    key, eq, text = item.partition('=')
    *sections, name = key.split('.')
    d = c
    for section in sections:
        d = d.get(section) if isinstance(d, dict) else None
    if not eq or not sections or not isinstance(d, dict) or name not in d:
        raise SystemExit(f'CONFIG_SET: {item!r} is not section.key=value for a key in {src}')
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        value = {'true': True, 'false': False}.get(text, text)     # TOML's booleans, or a bare word
    old = d[name]
    if type(value) is not type(old) and not (type(old) is float and type(value) is int):
        raise SystemExit(f'CONFIG_SET: {key} is {type(old).__name__} ({old!r}), not {value!r}')
    d[name] = value
    print(f'  {key}: {old!r} -> {value!r}')
with open(dst, 'w') as f:
    toml.dump(c, f)
EOF
    )
    CONFIG=/root/cfg_run.toml
fi

echo "== nets and state"
python3 - <<EOF
import os
import shutil
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
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
    # Every file from one commit: two downloads of the head could each find another backup's,
    # a state with a later gate (#58). A backup is one save's files in one commit (#49).
    revision = HfApi().model_info('${RUN_REPO}').sha
    try:
        hf_hub_download('${RUN_REPO}', '${RUN_PATH}/state.msgpack', local_dir=resume, revision=revision)
    except errors.LocalEntryNotFoundError:
        raise
    except getattr(errors, 'RemoteEntryNotFoundError', errors.EntryNotFoundError):
        print('no state at ${RUN_REPO}/${RUN_PATH}; starting fresh')
    else:
        snapshot_download('${RUN_REPO}', local_dir=resume, revision=revision, allow_patterns=[
            '${RUN_PATH}/gate.json', '${RUN_PATH}/gate.jsonl', '${RUN_PATH}/champion.npz', '${RUN_PATH}/run.json'])
        # Fetched beside OUT, in /dev/shm, so os.replace can move it there.
        os.makedirs(out, exist_ok=True)
        for name in os.listdir(f'{resume}/${RUN_PATH}'):
            os.replace(f'{resume}/${RUN_PATH}/{name}', f'{out}/{name}')
        print(f'resuming from ${RUN_REPO}/${RUN_PATH} at commit {revision[:8]}:', sorted(os.listdir(out)))
    shutil.rmtree(resume, ignore_errors=True)
EOF
cd /root/Mortal/mortal
python3 -m tpu.convert export /root/nets/baseline/baseline.pth /root/nets/baseline.npz
ARGS=(--init "/root/nets/$INIT" --opponent "/root/nets/$OPPONENT" --baseline /root/nets/baseline.npz)
ACCEPT=()
[ "$ACCEPT_CHANGES" = 1 ] && ACCEPT=(--accept-changes)

echo "== settings"
# A resumed run's own, from its run.json, but for how the work is spread; a new run's are
# this session's (#61). The trainer checks them again, and the opponent, before it starts.
python3 -m tpu.runid --out "$OUT" --config "$CONFIG" --write /root/cfg_resolved.toml "${ACCEPT[@]}"
CONFIG=/root/cfg_resolved.toml
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
echo "session: CONFIG_SET='$CONFIG_SET' ACCEPT_CHANGES=$ACCEPT_CHANGES; the run's settings are in run.json" \
    >> "$OUT/train.log"
( while sleep $((BACKUP_MIN * 60)); do
      [ -f "$OUT/state.msgpack" ] && { backup "backup during training" || echo "backup failed; will retry"; }
  done ) &
BACKUP_PID=$!
status=0
# The trainer is waited for by itself, not as `| tee`: a pipeline ends only once everything
# holding its write end has, and a process a killed trainer left behind held it for good (#53).
python3 -m tpu.online --out "$OUT" "${ARGS[@]}" "${ACCEPT[@]}" --steps "$STEPS" --hours "$HOURS" --remat \
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
