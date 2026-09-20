import hashlib
import logging
import os
import queue
import random
import threading
import torch
import numpy as np
from torch.utils.data import IterableDataset
from model import GRP
from reward_calculator import RewardCalculator
from libriichi.dataset import GameplayLoader
from rollout import version_in
from config import config

def digest64(key):
    """A stable signed 64-bit name for a string.

    Not `hash()`: PYTHONHASHSEED is randomised per process and the loader runs
    in several worker processes, so the same game would carry a different
    number depending on which worker happened to decode it. Clusters grouped
    by that would be silently wrong rather than obviously missing, which is
    the worse of the two failures. sha1 is not used here for any security
    property, only because it is the digest already in this file.
    """
    return int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], 'little', signed=True)

class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle = False,
        file_batch_size = 20, # hint: around 660 instances per file
        reserve_ratio = 0,
        parquet = False,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
        final_rank = False,
        skip_rewards = False,
        param_version = False,
        decision_ids = False,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.parquet = parquet
        self.readers = {}
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        # The rank the seat finished the hanchan in, 0 for first, appended to
        # every one of its instances. What a critic of the whole game has to
        # predict, where `player_ranks` is only the standing at the end of the
        # kyoku the instance is in.
        self.final_rank = final_rank
        # Distillation and the critic learn from actions and results, not from
        # the per-kyoku return, and the return costs a GRP forward per game.
        self.skip_rewards = skip_rewards
        # The parameter version that played the game, read off its file name,
        # on every instance it produced. A policy gradient divides by the
        # probability the behaviour policy gave the action, and that policy is
        # whatever was published when the game was played -- not what the
        # trainer holds by the time the batch arrives.
        self.param_version = param_version
        # Which trajectory, which kyoku, and where in the trajectory each
        # decision came from. Without these a batch is a bag of decisions and
        # several things cannot be done at all: the ~30 decisions of a kyoku
        # share one GRP delta, so a standard error that treats them as
        # independent is wrong and there is nothing to resample by; a held-out
        # split that scatters one hanchan across train and validation has
        # already leaked; and any return that walks forward -- TD(lambda),
        # GAE -- needs the order the shuffle threw away.
        self.decision_ids = decision_ids
        self.iterator = None

    def build_iter(self):
        # do not put it in __init__, it won't work on Windows
        if not self.skip_rewards:
            self.grp = GRP(**config['grp']['network'])
            grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
            self.grp.load_state_dict(grp_state['model'])
            self.reward_calc = RewardCalculator(self.grp, self.pts)

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        # shuffle the file list for each epoch
        random.shuffle(self.file_list)

        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
        )
        self.buffer = []

        for entries in self.decoded_ahead(self.iter_batches()):
            old_buffer_size = len(self.buffer)
            self.buffer.extend(entries)
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def iter_batches(self):
        """Games to hand the loader, `file_batch_size` at a time.

        A gz entry is one path; a parquet entry is one (shard, row group) pair
        holding thousands of games, so it is sliced down to the same batch size
        rather than being read whole.
        """
        if not self.parquet:
            for start in range(0, len(self.file_list), self.file_batch_size):
                yield self.file_list[start:start + self.file_batch_size]
            return

        import pyarrow.parquet as pq
        for shard, row_group in self.file_list:
            reader = self.readers.get(shard)
            if reader is None:
                # A handful of shards, so holding every footer open is cheaper
                # than reopening one per group in a shuffled list.
                reader = self.readers[shard] = pq.ParquetFile(shard)
            for chunk in reader.iter_batches(
                batch_size = self.file_batch_size,
                row_groups = [row_group],
                columns = ['events'],
            ):
                yield chunk.column('events').to_pylist()

    def decoded_ahead(self, batches):
        """`load_entries` of each batch, with the next one decoding meanwhile.

        Decoding a v4 batch takes seconds. Done in line, it runs only once a
        worker has been handed a task and found its buffer empty, and tasks
        arrive only as the trainer consumes batches, so the workers drift into
        taking turns to decode instead of decoding side by side. A thread one
        batch ahead keeps every worker decoding; Rust releases the GIL while it
        does. At most one decoded batch waits, which bounds the memory.
        """
        ready = queue.Queue()
        ahead = threading.Semaphore(1)
        stop = threading.Event()
        done = object()

        def decode():
            try:
                for batch in batches:
                    ahead.acquire()
                    if stop.is_set():
                        return
                    ready.put(self.load_entries(batch))
                ready.put(done)
            except BaseException as exc:
                ready.put(exc)

        thread = threading.Thread(target=decode, daemon=True)
        thread.start()
        try:
            while (entries := ready.get()) is not done:
                if isinstance(entries, BaseException):
                    raise entries
                ahead.release()
                yield entries
        finally:
            # A consumer that stops early -- a finished run, an exception, a
            # closed generator -- leaves this thread waiting on a semaphore
            # nothing will release, or part way through a decode. A daemon
            # thread killed in either state at interpreter shutdown aborts the
            # process ("FATAL: exception not rethrown"), which is a crash where
            # there was none. Told to stop it finishes at most the batch it
            # holds, so waiting for it is bounded by one decode.
            stop.set()
            ahead.release()
            thread.join(timeout=300)

    def where(self, source):
        """How a game names itself in a log line: its path, or a digest of it."""
        if not isinstance(source, str):
            # A null cell in a parquet shard, say. Whatever it is, naming it
            # must not be the thing that takes the run down.
            return repr(source)
        if not self.parquet:
            return source
        return 'game ' + hashlib.sha1(source.encode()).hexdigest()[:12]

    def decode_batch(self, batch):
        """`batch` decoded and paired with its sources, minus unreadable games.

        One call for the whole batch: encoding a v4 observation costs enough
        that fanning out over games in Rust rather than looping here is the
        difference between 1.8k and 4k instances/s. But a game libriichi
        refuses to replay (a kakan of a tile already seen, say) fails that one
        call for the whole batch, and the corpus holds a few, so halve the
        batch until the offender is alone, name it, and drop only it. Batches
        without one, which is nearly all of them, decode in a single call.
        """
        load = self.loader.load_logs if self.parquet else self.loader.load_gz_log_files
        try:
            return list(zip(batch, load(batch)))
        except BaseException as exc:
            # Not `Exception`: a Rust panic arrives as pyo3's PanicException,
            # which inherits from BaseException and would sail straight past
            # that and end the run. A real interrupt still has to.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if len(batch) > 1:
                half = len(batch) // 2
                return self.decode_batch(batch[:half]) + self.decode_batch(batch[half:])
            logging.warning(f'skipping an unreadable game: {self.where(batch[0])}: {exc}')
            return []

    def load_entries(self, batch):
        entries = []
        for source, file in self.decode_batch(batch):
            version = version_in(source) if self.param_version else None
            if version is None:
                version = -1
            game_key = self.where(source) if self.decision_ids else None
            for game in file:
                # per move
                obs = game.take_obs()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()

                # per game
                grp = game.take_grp()
                player_id = game.take_player_id()
                # One file holds one log and up to four seats of it, so the
                # trajectory is the pair, not the file.
                game_id = digest64(f'{game_key}#{player_id}') if self.decision_ids else 0

                game_size = len(obs)
                if game_size == 0:
                    # This player never acted, so there is nothing to learn
                    # from and no `at_kyoku` to index. Upstream issue #103.
                    continue

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                if self.skip_rewards:
                    kyoku_rewards = np.zeros(at_kyoku[-1] + 1, dtype=np.float64)
                else:
                    kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1 # usually they are equal, unless there is no action in the last kyoku

                final_scores = grp.take_final_scores()
                scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                for i in range(game_size):
                    action = actions[i]
                    if not (0 <= action < len(masks[i]) and masks[i][action]):
                        # A logged move libriichi does not consider legal there.
                        # The trainer asserts on this, so drop it here, where
                        # the game it came from is still known.
                        logging.warning(
                            f'skipping an illegal logged move: {self.where(source)}, seat {player_id}, '
                            f'move {i} in kyoku #{at_kyoku[i]}, action {action}, '
                            f'legal {np.flatnonzero(masks[i]).tolist()}')
                        continue
                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        steps_to_done[i],
                        kyoku_rewards[at_kyoku[i]],
                        player_ranks[at_kyoku[i] + 1],
                    ]
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    if self.final_rank:
                        # Appended, so every field train.py unpacks keeps its
                        # place and only what asks for this sees it.
                        entry.append(player_ranks[-1])
                    if self.param_version:
                        # -1 for a game whose name does not carry one: an older
                        # worker, or a log put there by hand. The trainer drops
                        # those rather than pricing them with the wrong weights.
                        entry.append(version)
                    if self.decision_ids:
                        # Last, for the reason the two above are appended.
                        entry += [game_id, int(at_kyoku[i]), i]
                    entries.append(entry)
        return entries

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    threads = os.environ.get('MORTAL_LOADER_RAYON_THREADS')
    if threads:
        # The rank's own rayon pool is sized for test play (Dist.setup); this
        # loader's pool, made on its first decode, gets the loader's share.
        os.environ['RAYON_NUM_THREADS'] = threads
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
