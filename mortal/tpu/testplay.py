"""Test play beside training: the EMA against the v3 baseline on fixed walls, in a thread.

A TPU belongs to one process, and `tpu.run` is it, so the games are played there,
in a thread, while training goes on: libriichi's arena plays them on its own CPU
threads and asks a `JaxEngine` on one device for every move. A test plays the first
`walls` walls of one of `evaluation.evaluate`'s sets, four games a wall, so its
numbers sit beside every evaluation made on the same walls with PyTorch, and
writes one line to test_play.jsonl in the run's output: the average pt, rank and
fourths with their error over walls, and the difference from the run's first test,
paired wall by wall. The ranks of every test are kept in test_play/, for pairing
any two of them later.

Nothing is chosen by it. A best-so-far picked from these numbers would be the
largest of several noisy ones -- 4,000 games carry 1.3 pt of error -- so a run's
checkpoints are compared afterwards, on walls none of them was chosen on.
"""
import glob
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from os import path


class TestPlay:
    def __init__(self, baseline, out, *, wall_set='dev', walls=500, device=None, threads=8):
        from evaluation.evaluate import WALL_SETS
        from tpu.engine import JaxEngine
        # The arena runs on libriichi's rayon pool in this process, which is sized the
        # first time it is used; the loader's workers are other processes with their own.
        os.environ.setdefault('RAYON_NUM_THREADS', str(threads))
        self.wall_set = WALL_SETS[wall_set]
        self.walls = walls
        self.device = device
        self.champion = JaxEngine.from_npz(baseline, device=device, name='baseline')
        self.challenger = None
        self.thread = None
        self.last = None
        self.log_file = path.join(out, 'test_play.jsonl')
        self.ranks_dir = path.join(out, 'test_play')
        os.makedirs(self.ranks_dir, exist_ok=True)
        # The game logs are only read back for the ranks: RAM, where there is some,
        # since the Kaggle host's disk can be throttled to 1 MB/s.
        self.scratch = '/dev/shm' if path.isdir('/dev/shm') else None

    def start(self, steps, variables, channels, blocks):
        """Test `variables` (host arrays) in the background, unless a test is still playing."""
        if self.busy():
            logging.info(f'test play at step {steps:,} skipped: the one at step {self.last:,} is still playing')
            return
        self.last = steps
        # Not a daemon: one killed at exit inside the arena aborts the process, the way
        # the loader's decode thread did ("FATAL: exception not rethrown").
        self.thread = threading.Thread(target=self._play, args=(steps, variables, channels, blocks),
                                       name='test_play')
        self.thread.start()

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def finish(self):
        if self.busy():
            logging.info(f'waiting for the test play at step {self.last:,}')
            self.thread.join()

    def _play(self, steps, variables, channels, blocks):
        try:
            self._test(steps, variables, channels, blocks)
        except Exception:
            # A failed test costs its numbers, never the run.
            logging.exception(f'test play at step {steps:,} failed')

    def _test(self, steps, variables, channels, blocks):
        from evaluation.evaluate import paired, summarize, summarize_logs, walls_of
        from libriichi.arena import OneVsThree
        from tpu.engine import JaxEngine
        started = time.time()
        if self.challenger is None:
            self.challenger = JaxEngine(variables, version=4, conv_channels=channels, num_blocks=blocks,
                                        device=self.device, name='mortal')
        else:
            self.challenger.set_variables(variables)
        logs = tempfile.mkdtemp(prefix='test_play_', dir=self.scratch)
        try:
            OneVsThree(disable_progress_bar=True, log_dir=logs).py_vs_py(
                challenger=self.challenger, champion=self.champion,
                seed_start=(self.wall_set.first_seed, self.wall_set.key), seed_count=self.walls)
            ranks = summarize_logs(logs, 'mortal')
        finally:
            shutil.rmtree(logs, ignore_errors=True)

        # The first test of the run is what the others are paired against; a run that
        # resumes finds it among the kept ranks.
        kept = sorted(glob.glob(path.join(self.ranks_dir, '*.json')))
        first = None
        if kept:
            with open(kept[0], encoding='utf-8') as f:
                first = (int(path.basename(kept[0])[:-len('.json')]), json.load(f))
        tmp = path.join(self.ranks_dir, f'{steps:010d}.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(ranks, f)
        os.replace(tmp, tmp[:-len('.tmp')])

        as_games = lambda r: {(int(k.split('_')[0]), k.split('_')[1]): v for k, v in r.items()}
        walls = walls_of(as_games(ranks))
        seeds = sorted(walls)
        record = {'steps': steps, 'set': self.wall_set.name, 'walls': len(seeds), 'games': 4 * len(seeds),
                  'seconds': round(time.time() - started, 1)}
        for name in ('pt', 'rank', 'fourth'):
            record[name] = summarize(walls, seeds, name)
        line = (f'test play at step {steps:,}: {record["pt"]["mean"]:+.2f} ± {record["pt"]["se"]:.2f} pt, '
                f'rank {record["rank"]["mean"]:.4f} ± {record["rank"]["se"]:.4f}')
        if first is not None and first[0] != steps:
            base = walls_of(as_games(first[1]))
            common = sorted(set(seeds) & set(base))
            record['vs_first'] = {'steps': first[0],
                                  **{name: paired(walls, base, common, name, reps=0) for name in ('pt', 'rank')}}
            d = record['vs_first']['pt']
            line += f'; against step {first[0]:,}: {d["diff"]:+.2f} ± {d["se"]:.2f} pt'
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        logging.info(f'{line} ({record["games"]:,} games in {record["seconds"]:.0f} s)')
