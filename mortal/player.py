import torch
import numpy as np
import os
import gzip
import json
import shutil
import secrets
import logging
import threading
from collections import defaultdict
from glob import glob
from os import path
from model import Brain, head_for, head_state
from engine import MortalEngine
from libriichi.stat import Stat
from libriichi.arena import OneVsThree
from config import config

class TestPlayer:
    def __init__(self, device=None, rank=0, world_size=1):
        baseline_cfg = config['baseline']['test']
        # Under DDP each rank plays its share of test games on its own GPU, so
        # the champion lives there too rather than on the configured device.
        device = device or torch.device(baseline_cfg['device'])

        state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        # The opponent seats play whichever head the run is about: a policy
        # gradient wants its three champions to be the policy it started from,
        # not that checkpoint's Q, or the self-play number measures the 0.4%
        # of decisions where the two disagree as well as the learning.
        kind = baseline_cfg.get('head', 'dqn')
        stable_dqn = head_for(kind, version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(head_state(kind, state))
        if baseline_cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()

        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle = False,
            version = version,
            device = device,
            enable_amp = True,
            enable_rule_based_agari_guard = True,
            name = 'baseline',
        )
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(config['test_play']['log_dir'])
        self.rank = rank
        self.world_size = world_size

    def test_play(self, seed_count, mortal, dqn, device):
        self.clear()
        self.play(seed_count, mortal, dqn, device)
        return self.collect()

    def track_dir(self, track=None):
        """Where a track's games go: `log_dir` itself, or `log_dir_<track>` beside it.

        Tracks play the same seeds, so a second model can be compared with
        the first game by game over the same walls.
        """
        return self.log_dir if track is None else f'{self.log_dir}_{track}'

    def clear(self, track=None):
        if path.isdir(self.track_dir(track)):
            shutil.rmtree(self.track_dir(track))

    def seeds(self, seed_count):
        """This rank's contiguous slice of [10000, 10000 + seed_count).

        The slices tile the range exactly, so every world size plays the same
        games and the numbers stay comparable across runs.
        """
        per_rank = seed_count // self.world_size
        first = self.rank * per_rank
        last = seed_count if self.rank == self.world_size - 1 else first + per_rank
        return 10000 + first, last - first

    def play(self, seed_count, mortal, dqn, device, track=None):
        self.play_all(seed_count, [(mortal, dqn, track)], device)

    def play_all(self, seed_count, jobs, device):
        """Play this rank's seeds with each (mortal, dqn, track), all at once.

        The tracks are separate games that only share seeds, and libriichi
        releases the GIL while it plays them, so threads here really do run at
        the same time. It is worth the trouble because one arena leaves most of
        the box idle: it advances every game in flight on a single thread, then
        encodes the ones that must act in parallel, then runs one forward, and
        through the first and last of those the cores have nothing to do. An
        evaluation measured 14 of 56 cores busy against 51 while training. Two
        arenas fall into each other's gaps.

        They share the champion, which holds nothing between calls, so it stays
        one copy of one model on the GPU.
        """
        torch.backends.cudnn.benchmark = False
        failures = []

        def run(job, quiet):
            try:
                self.play_one(seed_count, *job, device=device, quiet=quiet)
            except BaseException as exc:
                failures.append(exc)

        try:
            # Only the first track draws a progress bar; two of them writing to
            # one terminal produce a log nobody can read.
            threads = [threading.Thread(target=run, args=(job, i > 0))
                       for i, job in enumerate(jobs)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if failures:
                raise failures[0]
        finally:
            torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']

    def play_one(self, seed_count, mortal, dqn, track=None, device=None, quiet=False):
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            device = device,
            enable_amp = True,
            # The champion has this on, and so does the released bot. Leaving
            # it off here measured a player nobody ever runs. Upstream #114.
            enable_rule_based_agari_guard = True,
            name = 'mortal',
        )

        first, count = self.seeds(seed_count)
        # One directory per rank under log_dir, which `collect` reads whole.
        log_dir = self.track_dir(track)
        if self.world_size > 1:
            log_dir = path.join(log_dir, f'rank{self.rank}')
        if count > 0:
            env = OneVsThree(
                disable_progress_bar = quiet or self.rank != 0,
                log_dir = log_dir,
            )
            env.py_vs_py(
                challenger = engine_chal,
                champion = self.baseline_engine,
                seed_start = (first, 0x2000),
                seed_count = count,
            )

    def collect(self, track=None):
        # Globs **/*.json.gz, so it picks up every rank's directory at once.
        return Stat.from_dir(self.track_dir(track), 'mortal')

    def paired(self, track, against=None):
        """The challenger's rank on `track` minus on `against`, by seed.

        Both tracks deal the same walls from the same seats, so the luck of the
        deal largely cancels and the difference is far tighter than two
        separate averages.

        The error is taken over walls, not games. A seed is played four times,
        once from each seat, off one wall, and those four differences are not
        four independent draws. In practice they are very nearly independent --
        on the 280,000-step evaluation this widened the interval from 0.0184 to
        0.0188, because pairing has already cancelled the wall and what is left
        is each seat's own decisions -- but averaging a wall's four first costs
        nothing and does not rest on that holding.

        Returns (mean difference, its standard error, games, seeds).
        """
        def rank_in(file):
            with gzip.open(file, 'rt') as f:
                log = f.read()
            names = json.loads(log.split('\n', 1)[0])['names']
            return Stat.from_log(log, names.index('mortal')).avg_rank

        base = self.track_dir(against)
        by_seed = defaultdict(list)
        for main in glob(path.join(base, '**', '*.json.gz'), recursive=True):
            other = path.join(self.track_dir(track), path.relpath(main, base))
            if path.exists(other):
                # libriichi names them <seed>_<key>_<a|b|c|d>.json.gz, one
                # letter per seat of the same wall.
                seed = path.basename(main).rsplit('_', 1)[0]
                by_seed[seed].append(rank_in(other) - rank_in(main))
        games = sum(len(v) for v in by_seed.values())
        if len(by_seed) < 2:
            return float('nan'), float('nan'), games, len(by_seed)
        d = np.array([np.mean(v) for v in by_seed.values()])
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), games, len(d)

class TrainPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['train']
        # The opponents play on the worker's own GPU, not on whichever one the
        # config names. Three of the four seats are the champion, so it is most
        # of the forward work in a session: with it pinned to one card, workers
        # handed the other still sent three quarters of their play back to the
        # first, which sat at 100% while the other did nothing. `MORTAL_DEVICE`
        # is what the launcher uses to spread them; for a run that keeps every
        # worker on one card this is the same device it always was.
        device = torch.device(os.environ.get('MORTAL_DEVICE') or baseline_cfg['device'])

        state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        # The opponent seats play whichever head the run is about: a policy
        # gradient wants its three champions to be the policy it started from,
        # not that checkpoint's Q, or the self-play number measures the 0.4%
        # of decisions where the two disagree as well as the learning.
        kind = baseline_cfg.get('head', 'dqn')
        stable_dqn = head_for(kind, version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(head_state(kind, state))
        if baseline_cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()

        self.baseline_engine_args = dict(
            is_oracle = False,
            version = version,
            device = device,
            enable_amp = True,
            name = 'baseline',
        )
        self.baseline_parts = (stable_mortal, stable_dqn)

        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        logging.info(f'using profile {profile}')
        cfg = config['train_play'][profile]
        # The rule-based agari guard replaces the action the policy sampled
        # when it wants a win that cannot lift it out of fourth in the last
        # hand. For a policy gradient that is not a small detail: two sampled
        # actions then lead to the one that was played, so its real probability
        # is p(alternative) + p(agari), while the trainer -- which only sees
        # what was played -- uses p(alternative). The two disagree as soon as
        # the policy moves, and the error goes straight into the ratio.
        # Off for a run that learns from these games; on everywhere else, which
        # is what every evaluation so far has measured.
        self.agari_guard = cfg.get('enable_rule_based_agari_guard', True)
        self.baseline_engine = self.build_baseline()
        self.chal_version = config['control']['version']
        # Several self-play workers share a box and a config, and each one
        # empties its log directory before every session, so they must not
        # share one. MORTAL_WORKER, set by the launcher, keeps them apart.
        worker = os.environ.get('MORTAL_WORKER', '')
        self.log_dir = path.abspath(cfg['log_dir'] + (f'_{worker}' if worker else ''))
        self.train_key = secrets.randbits(64)
        self.train_seed = 10000

        self.seed_count = cfg['games'] // 4
        self.boltzmann_epsilon = cfg['boltzmann_epsilon']
        self.boltzmann_temp = cfg['boltzmann_temp']
        self.top_p = cfg['top_p']

        self.repeats = cfg['repeats']
        self.repeat_counter = 0
        # How many arenas this worker runs side by side. One arena uses a
        # fraction of a box: it advances every game in flight on a single
        # thread, then encodes what must act in parallel, then runs one
        # forward, so the cores are idle through the first and last of those.
        # An evaluation measured 14 of 56 cores busy. Arenas in threads fall
        # into each other's gaps -- libriichi releases the GIL while it plays
        # -- and share the one copy of the weights on the GPU, which a second
        # worker process would not.
        self.arenas = cfg.get('arenas', 1)

    def build_baseline(self):
        """The three opponent seats, under the same rule the trainee plays by."""
        return MortalEngine(
            *self.baseline_parts,
            enable_rule_based_agari_guard = self.agari_guard,
            **self.baseline_engine_args,
        )

    def play_slice(self, engine_chal, first, count, quiet):
        """`count` seeds from `first`, into the session's own directory.

        Games are named after their seed, so slices never collide and the
        whole session still reads back as one directory.
        """
        env = OneVsThree(
            disable_progress_bar = quiet,
            log_dir = self.log_dir,
        )
        return env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (first, self.train_key),
            seed_count = count,
        )

    def play_arenas(self, engine_chal):
        """Every arena at once, and the rankings they came to, added up."""
        # Never more arenas than there are seeds to give them: an arena handed
        # an empty slice fails in Rust, which wants at least one game, and
        # takes the whole session down with it.
        arenas = max(1, min(self.arenas, self.seed_count))
        if arenas <= 1:
            return np.array(self.play_slice(engine_chal, self.train_seed,
                                            self.seed_count, False))

        results, failures = [None] * arenas, []
        # Contiguous slices that tile the range exactly, so the session plays
        # the same seeds however many arenas it is split across.
        per = self.seed_count // arenas
        threads = []
        for i in range(arenas):
            first = self.train_seed + i * per
            count = self.seed_count - i * per if i == arenas - 1 else per

            def run(i=i, first=first, count=count):
                try:
                    results[i] = self.play_slice(engine_chal, first, count, i > 0)
                except BaseException as exc:
                    failures.append(exc)

            threads.append(threading.Thread(target=run))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if failures:
            raise failures[0]
        return np.sum([np.array(r) for r in results], axis=0)

    def train_play(self, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            boltzmann_epsilon = self.boltzmann_epsilon,
            boltzmann_temp = self.boltzmann_temp,
            top_p = self.top_p,
            device = device,
            enable_amp = True,
            enable_rule_based_agari_guard = self.agari_guard,
            name = 'trainee',
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        rankings = self.play_arenas(engine_chal)
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = list(map(lambda p: path.join(self.log_dir, p), os.listdir(self.log_dir)))

        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return rankings, file_list
