import torch
import numpy as np
import os
import gzip
import json
import shutil
import secrets
import logging
from glob import glob
from os import path
from model import Brain, DQN
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
        stable_dqn = DQN(version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(state['current_dqn'])
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
        torch.backends.cudnn.benchmark = False
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
                disable_progress_bar = self.rank != 0,
                log_dir = log_dir,
            )
            env.py_vs_py(
                challenger = engine_chal,
                champion = self.baseline_engine,
                seed_start = (first, 0x2000),
                seed_count = count,
            )
        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']

    def collect(self, track=None):
        # Globs **/*.json.gz, so it picks up every rank's directory at once.
        return Stat.from_dir(self.track_dir(track), 'mortal')

    def paired(self, track):
        """The challenger's rank on `track` minus on the main track, game by game.

        Both tracks deal the same walls from the same seats, so the luck of
        the deal largely cancels and the difference is far tighter than two
        separate averages. Returns (mean difference, its standard error, games).
        """
        def rank_in(file):
            with gzip.open(file, 'rt') as f:
                log = f.read()
            names = json.loads(log.split('\n', 1)[0])['names']
            return Stat.from_log(log, names.index('mortal')).avg_rank

        diffs = []
        for main in glob(path.join(self.log_dir, '**', '*.json.gz'), recursive=True):
            other = path.join(self.track_dir(track), path.relpath(main, self.log_dir))
            if path.exists(other):
                diffs.append(rank_in(other) - rank_in(main))
        if len(diffs) < 2:
            return float('nan'), float('nan'), len(diffs)
        d = np.asarray(diffs)
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)

class TrainPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['train']
        device = torch.device(baseline_cfg['device'])

        state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = DQN(version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(state['current_dqn'])
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

        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        logging.info(f'using profile {profile}')
        cfg = config['train_play'][profile]
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
            enable_rule_based_agari_guard = True,
            name = 'trainee',
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        rankings = env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (self.train_seed, self.train_key),
            seed_count = self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = list(map(lambda p: path.join(self.log_dir, p), os.listdir(self.log_dir)))

        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return rankings, file_list
