"""Copy the v4 run's checkpoints to a private Hugging Face repo.

    python backup_hf.py            # one backup
    python backup_hf.py --loop     # one every BACKUP_EVERY seconds, for good

The trainer rewrites mortal.pth every 400 steps, so a checkpoint is copied
first, the copy is kept only if the file did not change underneath it, and it
must load before anything is uploaded. A torn file never reaches the repo.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi

REPO = 'hhim8826/mortal4-0911'
# Which run: the offline one by default, the online one when the watchdog says
# so. They keep separate checkpoints and go to separate places in the repo, so
# the online run cannot overwrite the offline weights it started from.
RUN = Path(os.environ.get('MORTAL_RUN', '/root/Mortal/mortal/logs/v4'))
ONLINE = RUN.name == 'v4o'
PATH_IN_REPO = 'online' if ONLINE else None
CONFIG = Path('/root/Mortal/mortal/'
              + ('config.online.toml' if ONLINE else 'config.vast.toml'))
STAGE = Path('/root/hf-backup-stage')
LAST = Path('/root/hf-backup-last-online.json' if ONLINE
            else '/root/hf-backup-last.json')
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
            time.sleep(5)       # caught mid-save; the next save is ~90 s away
            continue
        try:
            state = torch.load(dst, weights_only=True, map_location='cpu')
        except Exception as exc:
            log(f'{src.name}: copy does not load ({exc!r}), retrying')
            time.sleep(5)
            continue
        return state
    return None


def info_log(src, dst):
    """The trainer's log without the progress bars, which are most of its bytes."""
    keep = re.compile(r' (INFO|WARNING|ERROR|CRITICAL) |Traceback|Error')
    with open(src, encoding='utf-8', errors='replace') as f, open(dst, 'w', encoding='utf-8') as out:
        for line in f.read().replace('\r', '\n').splitlines():
            if keep.search(line):
                out.write(line + '\n')


def backup(api):
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    state = stable_copy(RUN / 'mortal.pth', STAGE / 'mortal.pth')
    if state is None:
        log('mortal.pth never held still long enough to copy; skipping this round')
        return
    steps, best = state['steps'], state['best_perf']
    del state

    last = json.loads(LAST.read_text()) if LAST.exists() else {}
    if last.get('steps') == steps:
        log(f'still at step {steps}; nothing new to back up')
        return

    have_best = (RUN / 'best.pth').exists()
    if have_best and stable_copy(RUN / 'best.pth', STAGE / 'best.pth') is None:
        log('best.pth would not copy cleanly; uploading without it this round')
        (STAGE / 'best.pth').unlink(missing_ok=True)
        have_best = False
    # The weight average's own best, a candidate until it beats best.pth.
    ema = state_ema = None
    if (RUN / 'best_ema.pth').exists():
        state_ema = stable_copy(RUN / 'best_ema.pth', STAGE / 'best_ema.pth')
        if state_ema is None:
            log('best_ema.pth would not copy cleanly; uploading without it this round')
            (STAGE / 'best_ema.pth').unlink(missing_ok=True)
        else:
            ema = state_ema['best_perf']
            del state_ema

    shutil.copyfile(CONFIG, STAGE / CONFIG.name)
    for name in ('train.log', 'trainer.log'):
        if (RUN / name).exists():
            info_log(RUN / name, STAGE / 'train_info.log')
            break
    tb = STAGE / 'tensorboard'
    tb.mkdir()
    for f in RUN.rglob('events.out.tfevents.*'):
        dst = tb / f.relative_to(RUN)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst)

    (STAGE / 'README.md').write_text(
        f'# Mortal v4, trained from scratch\n\n'
        f'Private backup of an {"online (self-play)" if ONLINE else "offline"} '
        f'training run in progress on the '
        f'[`train-parquet`](https://github.com/HHim8826/Mortal/tree/train-parquet) '
        f'branch.\n\n'
        f'- latest step: **{steps:,}**\n'
        f'- best test play so far (vs the v3 baseline, 2.5 = even): '
        f'avg rank {best["avg_rank"]:.4f}, avg pt {best["avg_pt"]:.3f}\n'
        f'- `mortal.pth` is the latest checkpoint; `best.pth` the best by test play'
        f'{"" if have_best else " (none yet: the first evaluation is at step 40,000)"}\n'
        + (f'- `best_ema.pth`: the best of the weight average (EMA), test-played on the '
           f'same walls: avg rank {ema["avg_rank"]:.4f}, avg pt {ema["avg_pt"]:.3f}; a '
           f'candidate until it beats `best.pth`\n' if ema else '') +
        f'- backed up {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC\n',
        encoding='utf-8')

    api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type='model',
                      path_in_repo=PATH_IN_REPO,
                      commit_message=f'{"online " if ONLINE else ""}step {steps:,}')
    LAST.write_text(json.dumps({'steps': steps, 'best_perf': best}))
    log(f'backed up step {steps:,} (best {best}){" with best.pth" if have_best else ""}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()

    api = HfApi()
    info = api.repo_info(REPO, repo_type='model')
    if not info.private:
        sys.exit(f'{REPO} is public; refusing to upload checkpoints to it')

    while True:
        try:
            backup(api)
        except Exception as exc:
            # A failed round must not end the loop; the next one retries.
            log(f'backup failed: {exc!r}')
        if not args.loop:
            return
        time.sleep(BACKUP_EVERY)


if __name__ == '__main__':
    main()
