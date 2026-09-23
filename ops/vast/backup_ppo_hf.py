"""Copy a phase-3 policy-gradient run to the private Hugging Face repo.

    MORTAL_RUN=/root/Mortal/mortal/logs/ppo/kl python backup_ppo_hf.py
    MORTAL_RUN=... python backup_ppo_hf.py --loop

`backup_hf.py` is for the Q-learning runs: it wants `mortal.pth`, `best.pth`
and a `best_perf` written by test play. A PPO run has none of those. It writes
one `policy.pth` every few rounds, its opponent never changes, and what says
whether it is working is the self-play average in the TensorBoard events and
the round lines in the trainer's log -- so those are what this uploads, beside
the checkpoint.

The trainer rewrites `policy.pth` in place, so it is copied first, the copy is
kept only if the file did not change underneath it, and it must load before
anything is uploaded. A torn file never reaches the repo.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi

REPO = 'hhim8826/mortal4-0911'
RUN = Path(os.environ.get('MORTAL_RUN', '/root/Mortal/mortal/logs/ppo/kl'))
# Each variant keeps its own place in the repo, so the three cannot overwrite
# each other and none of them can touch the offline weights at the root.
PATH_IN_REPO = f'ppo/{RUN.name}'
CONFIG = Path(os.environ.get('MORTAL_CFG_PATH', '/root/Mortal/mortal/config.ppo.toml'))
STAGE = Path(f'/root/hf-backup-stage-ppo-{RUN.name}')
LAST = Path(f'/root/hf-backup-last-ppo-{RUN.name}.json')
BACKUP_EVERY = 2 * 3600


def log(msg):
    print(f'{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S} {msg}', flush=True)


def stable_copy(src, dst, tries=5):
    """Copy `src` and return its state, or None if it kept changing or will not load."""
    for _ in range(tries):
        before = src.stat()
        shutil.copyfile(src, dst)
        after = src.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            time.sleep(5)
            continue
        try:
            return torch.load(dst, weights_only=True, map_location='cpu')
        except Exception as exc:
            log(f'{src.name}: copy does not load ({exc!r}), retrying')
            time.sleep(5)
    return None


def info_log(src, dst):
    """A log without the progress bars, which are most of its bytes."""
    keep = re.compile(r' (INFO|WARNING|ERROR|CRITICAL) |Traceback|Error')
    with open(src, encoding='utf-8', errors='replace') as f, open(dst, 'w', encoding='utf-8') as out:
        for line in f.read().replace('\r', '\n').splitlines():
            if keep.search(line):
                out.write(line + '\n')


def self_play(run):
    """The last self-play average each worker logged, which is the run's own verdict."""
    out = {}
    for log_file in sorted(run.glob('worker*.log')):
        last = None
        with open(log_file, encoding='utf-8', errors='replace') as f:
            for line in f.read().replace('\r', '\n').splitlines():
                if 'last ' in line and 'sessions:' in line:
                    last = line
        if last:
            found = re.search(r'\(([\d.]+), ([-\d.]+)pt\)', last)
            if found:
                out[log_file.stem] = (float(found.group(1)), float(found.group(2)))
    return out


def backup(api):
    checkpoint = RUN / 'policy.pth'
    if not checkpoint.exists():
        log(f'{checkpoint} does not exist yet; nothing to back up')
        return
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    state = stable_copy(checkpoint, STAGE / 'policy.pth')
    if state is None:
        log('policy.pth never held still long enough to copy; skipping this round')
        return
    rounds, steps = state.get('rounds', 0), state.get('steps', 0)
    started_from = state.get('started_from', 'unknown')
    temperature = state.get('play_temperature')
    flags = state.get('args', {})
    del state

    last = json.loads(LAST.read_text()) if LAST.exists() else {}
    if last.get('steps') == steps:
        log(f'still at step {steps}; nothing new to back up')
        return

    if CONFIG.exists():
        shutil.copyfile(CONFIG, STAGE / CONFIG.name)
    if (RUN / 'trainer.log').exists():
        info_log(RUN / 'trainer.log', STAGE / 'trainer_info.log')
    for log_file in sorted(RUN.glob('worker*.log')):
        info_log(log_file, STAGE / f'{log_file.stem}_info.log')
    tb = STAGE / 'tensorboard'
    tb.mkdir()
    for f in RUN.rglob('events.out.tfevents.*'):
        dst = tb / f.relative_to(RUN)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst)

    play = self_play(RUN)
    verdict = '\n'.join(f'  - `{k}`: avg rank {r:.4f}, avg pt {p:+.2f}'
                        for k, (r, p) in sorted(play.items())) or '  - no session finished yet'
    (STAGE / 'README.md').write_text(
        f'# Mortal v4, policy gradient ({RUN.name})\n\n'
        f'Private backup of a phase-3 run in progress on the '
        f'[`train-parquet`](https://github.com/HHim8826/Mortal/tree/train-parquet) '
        f'branch. The policy was distilled from the 560k online EMA, sharpened to a '
        f'play temperature of {temperature}, and is being improved by PPO against a '
        f'frozen copy of itself.\n\n'
        f'- rounds: **{rounds:,}**, optimizer steps: **{steps:,}**\n'
        f'- started from: `{started_from}`\n'
        f'- clip {flags.get("clip")}, kl coefficient {flags.get("kl_coef")}, '
        f'reference refresh {flags.get("ref_refresh")}, lr {flags.get("lr")}, '
        f'target kl {flags.get("target_kl")}\n'
        f'- self-play against the frozen start (2.5 is level, lower is better):\n{verdict}\n'
        f'- `policy.pth` plays with `python -m evaluation.evaluate --model x=policy.pth#policy`\n'
        f'- backed up {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC\n',
        encoding='utf-8')

    info = api.repo_info(REPO, repo_type='model')
    if not info.private:
        raise SystemExit(f'{REPO} is public; refusing to upload a run to it')
    api.upload_folder(repo_id=REPO, folder_path=str(STAGE), path_in_repo=PATH_IN_REPO,
                      commit_message=f'{RUN.name}: round {rounds:,}, step {steps:,}')
    LAST.write_text(json.dumps({'steps': steps, 'rounds': rounds,
                                'at': datetime.datetime.now(datetime.timezone.utc).isoformat()}))
    log(f'uploaded {PATH_IN_REPO}: round {rounds:,}, step {steps:,}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loop', action='store_true')
    args = ap.parse_args()
    api = HfApi()
    while True:
        try:
            backup(api)
        except SystemExit:
            raise
        except Exception as exc:
            log(f'backup failed: {exc!r}')
        if not args.loop:
            return
        time.sleep(BACKUP_EVERY)


if __name__ == '__main__':
    main()
