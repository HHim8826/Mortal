"""Upload a run's output folder to the Hub from a snapshot that holds still while it goes up.

    python3 backup.py --out /dev/shm/runs/... --repo hhim8826/mortal4-0911 --path tpu-run --message '...'

The trainer keeps saving while a backup runs. Every file is written under a temporary
name and renamed, so one open reads a whole file -- but the Hub's uploader opens each
file twice, once to hash it and again to send it, and a save in between swaps the
file under it: the bytes sent are not the bytes hashed, and the commit fails or
retries (issue #49). So the folder is first copied into a snapshot beside it:

- state and weights (*.msgpack, *.npz) are hard links. A save renames a new file over
  the name and never writes into the old one, so the link keeps the old contents for
  as long as the upload takes, at no cost in memory.
- everything else is copied: logs and .jsonl files are appended to in place.

And the snapshot has to be of one save. The trainer bumps manifest.json around each
group of writes (`tpu.run.Manifest`): a snapshot taken with the same generation,
complete, before and after it is one save's files, not a state from one save and
weights from the next. A snapshot that caught a save is taken again.
"""
import argparse
import json
import os
import shutil
import sys
import time
from os import path

LINKED = ('.msgpack', '.npz')
SKIPPED = ('.tmp', '.tmp.npz')


def manifest(out):
    try:
        with open(path.join(out, 'manifest.json'), encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def snapshot(out, snap, tries=40, wait=3.):
    """Copy `out` into `snap` between two saves; the manifest it was taken under."""
    for _ in range(tries):
        before = manifest(out)
        if before is not None and not before.get('complete', True):
            time.sleep(wait)
            continue
        shutil.rmtree(snap, ignore_errors=True)
        for base, _, names in os.walk(out):
            into = path.join(snap, path.relpath(base, out))
            os.makedirs(into, exist_ok=True)
            for name in names:
                if name.endswith(SKIPPED):
                    continue
                src, dst = path.join(base, name), path.join(into, name)
                try:
                    if name.endswith(LINKED):
                        os.link(src, dst)
                    else:
                        shutil.copy2(src, dst)
                except FileNotFoundError:
                    pass                      # renamed away since the listing; retried below
        if manifest(out) == before:
            return before
        time.sleep(wait)
    raise SystemExit(f'no quiet moment between saves in {out} to copy it; not backed up this time')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--path', required=True)
    ap.add_argument('--message', required=True)
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    if not api.repo_info(args.repo).private:
        raise SystemExit(f'{args.repo} is public; refusing to upload checkpoints to it')
    snap = args.out.rstrip('/') + '.snapshot'
    try:
        taken = snapshot(args.out, snap)
        api.upload_folder(folder_path=snap, repo_id=args.repo, path_in_repo=args.path,
                          commit_message=args.message)
        print(f'backed up {args.out} -> {args.repo}/{args.path}'
              + (f' (save {taken["generation"]}, step {taken["steps"]:,})' if taken else ''), flush=True)
    finally:
        shutil.rmtree(snap, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
