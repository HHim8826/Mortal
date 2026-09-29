"""Online v4 training on a TPU, in one process: self-play, training and the gate together.

What `server.py`, `client.py` and `train.py` do across processes when `online = true`,
for a TPU, which belongs to one process:

- self-play: `[tpu_online] arenas` threads, each a libriichi arena where the trainee --
  the weights being trained, published to it every `submit_every` steps -- plays one
  seat against three of the frozen opponent, sampling as client.py's workers do.
  Finished games wait in a buffer of `capacity` logs; self-play pauses while it is full.
- training: each round takes every log in the buffer, decodes the trainee's decisions
  and trains train.py's online step on them (`tpu.train`: no CQL, `freeze_bn`, the last
  `trainable_blocks` blocks), then publishes the weights again.
- the gate, every `test_every` steps: the EMA and the champion play the next block of
  walls never played before, against the v3 baseline, and are paired by wall. The EMA
  takes the title if its rank is ahead by `gate_margin` se; `gate_patience` evaluations
  in a row without a new champion stop the run with exit status 3. train.py's gate.

    MORTAL_CFG=config.online.tpu.toml python -m tpu.online --out /dev/shm/online \\
        --init tpu60.npz --opponent tpu60.npz --baseline baseline.npz --hours 8

The loader's workers come from a forkserver started before anything else. Self-play
starts libriichi's rayon pool in this process, and a worker forked from it afterwards
inherits the pool without its threads and hangs on its first decode -- measured: no
progress in 180 s, where the forkserver's workers decoded each round in seconds. A
forkserver started first forks them from a clean process, without the TPU's device
files either. For the same reason this module must stay safe to import: the
forkserver's children import it again.

In --out: state.msgpack (to resume), weights.npz and weights_ema.npz, champion.npz (the
gate's champion, the model to use), gate.json (the gate's own state) and gate.jsonl (one
line an evaluation).
"""
import argparse
import functools
import glob
import json
import logging
import os
import secrets
import shutil
import sys
import tempfile
import threading
import time
from collections import deque
from os import path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s')

PTS = np.array([90, 45, 0, -135])


def gate_says_replace(diff, se, margin):
    """train.py's rule: the candidate's rank minus the champion's, over the same walls, below
    -`margin` se. A measurement that did not happen keeps the champion."""
    if se is None or not se > 0:
        return False
    # A Python bool: numpy's cannot be written to gate.jsonl.
    return bool(diff < -margin * se)


def full_batches(batches, batch_size):
    """Every full batch, then the ragged ends -- one a worker -- re-cut, as train.py does."""
    import torch
    remaining = []
    for batch in batches:
        if batch[0].shape[0] == batch_size:
            yield batch
        else:
            remaining.append(batch)
    if remaining:
        columns = [torch.cat(col) for col in zip(*remaining)]
        for start in range(0, columns[0].shape[0] - batch_size + 1, batch_size):
            yield [c[start:start + batch_size] for c in columns]


def discard(files):
    """Remove trained-on logs, and the arena directories they leave empty."""
    dirs = set()
    for f in files:
        dirs.add(path.dirname(f))
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    for d in dirs:
        try:
            os.rmdir(d)
        except OSError:
            pass


class SelfPlay:
    """Arenas in threads, the trainee in one seat against three of the frozen opponent.

    client.py's workers without the network: each arena plays `walls` walls at a time --
    four games a wall, the trainee in every seat -- on its own pair of engines, and hands
    the logs to the buffer. `publish` gives every trainee the weights being trained.
    """

    def __init__(self, variables, shape, opponent, *, arenas, walls, play_cfg, capacity, root,
                 devices, history):
        from tpu import convert
        from tpu.engine import JaxEngine
        channels, blocks = shape
        self.walls, self.capacity, self.root = walls, capacity, root
        self.cond = threading.Condition()
        self.files = []
        self.next_seed = 0
        # A key of this session's own, as client.py's workers draw one: walls never dealt before.
        self.key = secrets.randbits(64)
        self.history = deque(maxlen=history)
        self.games = 0
        self.stopping = threading.Event()
        self.failure = None
        guard = play_cfg.get('enable_rule_based_agari_guard', True)
        opp, opp_meta = convert.load_npz(opponent)
        self.trainees, self.opponents = [], []
        for i in range(arenas):
            device = devices[i % len(devices)]
            self.trainees.append(JaxEngine(
                variables, version=4, conv_channels=channels, num_blocks=blocks, device=device,
                name='trainee', enable_rule_based_agari_guard=guard,
                boltzmann_epsilon=play_cfg['boltzmann_epsilon'], boltzmann_temp=play_cfg['boltzmann_temp'],
                top_p=play_cfg['top_p']))
            self.opponents.append(JaxEngine(
                opp, version=opp_meta['version'], conv_channels=opp_meta['conv_channels'],
                num_blocks=opp_meta['num_blocks'], device=device, name='baseline',
                enable_rule_based_agari_guard=guard))
        self.threads = [threading.Thread(target=self._arena, args=(i,), name=f'selfplay{i}')
                        for i in range(arenas)]

    def start(self):
        for t in self.threads:
            t.start()

    def publish(self, variables):
        for engine in self.trainees:
            engine.set_variables(variables)

    def waiting(self):
        with self.cond:
            return len(self.files)

    def take(self, timeout):
        """Every log waiting, once there is one or `timeout` passes; raises if self-play failed."""
        with self.cond:
            self.cond.wait_for(lambda: self.files or self.stopping.is_set(), timeout)
            if self.failure is not None:
                raise RuntimeError('self-play failed') from self.failure
            files, self.files = self.files, []
            self.cond.notify_all()
            return files

    def stop(self):
        """Stop taking new walls; each arena finishes the ones it has, then its thread ends.
        Joined, never left as daemons: one killed inside Rust at exit aborts the process."""
        self.stopping.set()
        with self.cond:
            self.cond.notify_all()
        for t in self.threads:
            t.join()

    def summary(self):
        """(sessions, avg rank, avg pt) of the trainee against the opponent, over the recent sessions."""
        with self.cond:
            if not self.history:
                return 0, float('nan'), float('nan')
            total = np.sum(self.history, axis=0)
            n = len(self.history)
        return n, float(total @ np.arange(1, 5) / total.sum()), float(total @ PTS / total.sum())

    def _arena(self, i):
        from libriichi.arena import OneVsThree
        try:
            while not self.stopping.is_set():
                with self.cond:
                    self.cond.wait_for(lambda: len(self.files) < self.capacity or self.stopping.is_set())
                    if self.stopping.is_set():
                        return
                    first = self.next_seed
                    self.next_seed += self.walls
                logs = path.join(self.root, f'{i:02d}-{first:010d}')
                rankings = OneVsThree(disable_progress_bar=True, log_dir=logs).py_vs_py(
                    challenger=self.trainees[i], champion=self.opponents[i],
                    seed_start=(first, self.key), seed_count=self.walls)
                new = sorted(glob.glob(path.join(logs, '*.json.gz')))
                with self.cond:
                    self.files.extend(new)
                    self.history.append(np.array(rankings))
                    self.games += len(new)
                    self.cond.notify_all()
        except BaseException as exc:
            logging.exception(f'self-play arena {i} failed')
            self.failure = exc
            self.stopping.set()
            with self.cond:
                self.cond.notify_all()


class Gate:
    """train.py's gate, for this run's EMA.

    Each evaluation plays the candidate and the champion on the next `walls` walls at `key`
    -- walls neither has played -- against the v3 baseline, four games a wall, and pairs
    them by wall. The candidate takes the title only if its rank is ahead by `margin` se,
    and `patience` evaluations in a row without a new champion stop the run. The first
    champion is the net the run starts from (`initial`, step 0), as the vast.ai freeze4
    run's was the 560k it was seeded from: the first gate asks whether online has beaten
    its own starting point.
    """

    def __init__(self, baseline, out, shape, initial, *, walls, key, margin, patience, arenas, device,
                 scratch):
        from tpu import convert
        from tpu.engine import JaxEngine
        self.out, self.shape = out, shape
        self.walls, self.key, self.margin, self.patience = walls, key, margin, patience
        self.arenas, self.device, self.scratch = arenas, device, scratch
        self.baseline = JaxEngine.from_npz(baseline, device=device, name='baseline')
        self.state_file = path.join(out, 'gate.json')
        self.champion_file = path.join(out, 'champion.npz')
        if path.exists(self.state_file):
            with open(self.state_file, encoding='utf-8') as f:
                self.state = json.load(f)
            self.champion = convert.load_npz(self.champion_file)[0]
        else:
            self.state = {'evaluations': 0, 'fails': 0, 'champion': 0}
            self.champion = initial
            self._keep_champion(initial, 0)
            self._save_state()

    def _keep_champion(self, variables, steps):
        from tpu import convert
        tmp = self.champion_file + '.tmp.npz'
        convert.save_npz(tmp, variables, {'conv_channels': self.shape[0], 'num_blocks': self.shape[1],
                                          'steps': steps})
        os.replace(tmp, self.champion_file)

    def _save_state(self):
        tmp = self.state_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.state, f)
        os.replace(tmp, self.state_file)

    @property
    def stop(self):
        return bool(self.patience) and self.state['fails'] >= self.patience

    def _play(self, variables, first, label):
        """{'<seed>_<split>': rank} and libriichi's Stat of `variables` as 'mortal' on the gate's walls."""
        from evaluation.evaluate import summarize_logs
        from libriichi.arena import OneVsThree
        from libriichi.stat import Stat
        from tpu.engine import JaxEngine
        engine = JaxEngine(variables, version=4, conv_channels=self.shape[0], num_blocks=self.shape[1],
                           device=self.device, name='mortal')
        logs = tempfile.mkdtemp(prefix=f'gate_{label}_', dir=self.scratch)
        try:
            # Contiguous slices in threads: one arena leaves most of the CPUs idle. Games are
            # named by their wall, so slices never collide in the one directory.
            per = -(-self.walls // self.arenas)
            slices = [(s, min(per, first + self.walls - s)) for s in range(first, first + self.walls, per)]
            failures = []

            def run(start, count):
                try:
                    OneVsThree(disable_progress_bar=True, log_dir=logs).py_vs_py(
                        challenger=engine, champion=self.baseline, seed_start=(start, self.key), seed_count=count)
                except BaseException as exc:
                    failures.append(exc)

            threads = [threading.Thread(target=run, args=s) for s in slices]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if failures:
                raise failures[0]
            return summarize_logs(logs, 'mortal'), Stat.from_dir(logs, 'mortal')
        finally:
            shutil.rmtree(logs, ignore_errors=True)

    def evaluate(self, steps, ema):
        """Play the gate for the EMA `ema` (host arrays) at `steps`; True if it took the title."""
        from evaluation.evaluate import paired, summarize, walls_of
        from tpu import convert
        first = self.state['evaluations'] * self.walls
        logging.info(f'gate at step {steps:,}: walls [{first:,}, {first + self.walls:,}) at key {self.key:#x}')
        started = time.time()
        sides = {'candidate': ema, 'champion': self.champion}
        results, failures = {}, []

        def side(label, variables):
            try:
                results[label] = self._play(variables, first, label)
            except BaseException as exc:
                failures.append(exc)

        threads = [threading.Thread(target=side, args=item) for item in sides.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if failures:
            raise failures[0]

        as_games = lambda r: {(int(k.split('_')[0]), k.split('_')[1]): v for k, v in r.items()}
        record = {'steps': steps, 'walls': [first, first + self.walls], 'key': self.key,
                  'champion_steps': self.state['champion']}
        walls = {}
        for label, (ranks, stat) in results.items():
            walls[label] = walls_of(as_games(ranks))
            seeds = sorted(walls[label])
            record[label] = {**{m: summarize(walls[label], seeds, m) for m in ('pt', 'rank', 'fourth')},
                             'fuuro': stat.fuuro_rate, 'riichi': stat.riichi_rate, 'agari': stat.agari_rate,
                             'houjuu': stat.houjuu_rate, 'rank_1': stat.rank_1_rate, 'rank_4': stat.rank_4_rate}
        cand, champ = record['candidate'], record['champion']
        common = sorted(set(walls['candidate']) & set(walls['champion']))
        gain = paired(walls['candidate'], walls['champion'], common, 'rank', reps=0)
        record['paired'] = {'walls': len(common), 'rank': gain,
                            'pt': paired(walls['candidate'], walls['champion'], common, 'pt', reps=0)}
        replace = gate_says_replace(gain['diff'], gain['se'], self.margin)
        pt = record['paired']['pt']
        line = (f'gate at step {steps:,}: candidate {cand["pt"]["mean"]:+.2f} ± {cand["pt"]["se"]:.2f} pt, '
                f'champion (step {self.state["champion"]:,}) {champ["pt"]["mean"]:+.2f} pt; paired over '
                f'{len(common):,} walls: rank {gain["diff"]:+.4f} ± {gain["se"]:.4f}, pt {pt["diff"]:+.2f} ± '
                f'{pt["se"]:.2f}; {"replacing" if replace else "keeping"} the champion (margin {self.margin} se); '
                f'calls {cand["fuuro"]:.1%} / {champ["fuuro"]:.1%}, riichi {cand["riichi"]:.1%} / '
                f'{champ["riichi"]:.1%}, deal-in {cand["houjuu"]:.1%} / {champ["houjuu"]:.1%}, '
                f'firsts {cand["rank_1"]:.1%} / {champ["rank_1"]:.1%}, fourths {cand["rank_4"]:.1%} / '
                f'{champ["rank_4"]:.1%}')
        self.state['evaluations'] += 1
        self.state['fails'] = 0 if replace else self.state['fails'] + 1
        record.update(replaced=replace, fails=self.state['fails'], seconds=round(time.time() - started, 1))
        if replace:
            self.champion = ema
            self.state['champion'] = steps
            self._keep_champion(ema, steps)
        with open(path.join(self.out, 'gate.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        self._save_state()
        logging.info(f'{line} ({time.time() - started:.0f} s)')
        return replace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', required=True, help='the v4 net to start from: an .npz from tpu.convert export')
    ap.add_argument('--opponent', required=True, help='the frozen net self-play is played against, an .npz')
    ap.add_argument('--baseline', required=True, help='the v3 baseline the gate scores against, an .npz')
    ap.add_argument('--steps', type=int, default=0, help='stop after this many; 0 runs until the gate or --hours')
    ap.add_argument('--hours', type=float, default=0, help='save and stop after this long')
    ap.add_argument('--log-every', type=int, default=500)
    ap.add_argument('--remat', action='store_true')
    args = ap.parse_args()

    # First of all, before JAX and before any arena: see the docstring.
    from multiprocessing import forkserver
    forkserver.ensure_running()

    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from torch.utils.data import DataLoader
    from config import config
    from dataloader import FileDatasetsIter
    from tpu import convert
    from tpu.model import Mortal
    from tpu.run import as_arrays, collate, init_worker
    from tpu.train import loss_fn, make_optimizer, trainable

    tpu_cfg = config['tpu_online']
    # This process's rayon pool, which self-play and the gate share; set after the forkserver
    # started, so the loader's workers keep MORTAL_LOADER_RAYON_THREADS instead.
    os.environ.setdefault('RAYON_NUM_THREADS', str(tpu_cfg['rayon_threads']))
    ctl, ds = config['control'], config['dataset']
    batch_size, save_every = ctl['batch_size'], ctl['save_every']
    test_every, submit_every, ema_decay = ctl['test_every'], ctl['submit_every'], ctl['ema_decay']
    trainable_blocks, freeze_bn = config['freeze']['trainable_blocks'], config['freeze_bn']['mortal']
    if trainable_blocks and not freeze_bn:
        raise SystemExit('trainable_blocks needs freeze_bn here: without it the frozen blocks\' '
                         'BatchNorm statistics would still move, which train.py\'s do not')
    os.makedirs(args.out, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix='online_', dir='/dev/shm' if path.isdir('/dev/shm') else None)

    devices = jax.devices()
    if batch_size % len(devices):
        raise SystemExit(f'batch_size {batch_size} does not split over {len(devices)} devices')
    mesh = Mesh(np.array(devices), ('data',))
    split, whole = NamedSharding(mesh, P('data')), NamedSharding(mesh, P())

    variables, meta = convert.load_npz(args.init)
    if meta['version'] != 4:
        raise SystemExit(f'{args.init} is version {meta["version"]}; only v4 trains here')
    channels, blocks = meta['conv_channels'], meta['num_blocks']
    model = Mortal(channels, blocks, dtype=jnp.bfloat16, remat=args.remat)
    mask = trainable(variables['params'], trainable_blocks)
    tx = make_optimizer(config['optim'], mask)
    params, stats = variables['params'], variables['batch_stats']
    state = {'params': params, 'batch_stats': stats, 'opt': tx.init(params),
             'ema': {'params': params, 'batch_stats': stats}, 'steps': 0}
    ckpt = path.join(args.out, 'state.msgpack')
    if path.exists(ckpt):
        with open(ckpt, 'rb') as f:
            state = serialization.from_state_dict(state, serialization.msgpack_restore(f.read()))
        logging.info(f'resumed {ckpt} at step {int(state["steps"]):,}; --init is only a starting point, ignored')
    else:
        logging.info(f'init: {args.init}, {channels}x{blocks}, step {meta.get("steps", 0):,}')
    steps = int(state['steps'])
    state = jax.device_put(state, whole)
    held = sum(int((np.asarray(m) == 0).sum() * p.size // max(np.asarray(m).size, 1))
               for m, p in zip(jax.tree_util.tree_leaves(mask), jax.tree_util.tree_leaves(params)))
    frozen = max(blocks - trainable_blocks, 0) if trainable_blocks else 0
    logging.info(f'{len(devices)} x {devices[0].device_kind}, batch {batch_size:,}; {frozen} of {blocks} '
                 f'blocks frozen ({held:,} parameters), BatchNorm {"frozen" if freeze_bn else "training"}')

    loss_kw = dict(model=model, gamma=config['env']['gamma'], min_q_weight=config['cql']['min_q_weight'],
                   next_rank_weight=config['aux']['next_rank_weight'], online=True, freeze_bn=freeze_bn)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def step(state, batch):
        (_, (stats, losses)), grads = grad_fn(state['params'], state['batch_stats'], batch, **loss_kw)
        updates, opt = tx.update(grads, state['opt'], state['params'])
        params = optax.apply_updates(state['params'], updates)
        live = {'params': params, 'batch_stats': stats}
        ema = jax.tree_util.tree_map(lambda e, x: ema_decay * e + (1 - ema_decay) * x, state['ema'], live)
        return {'params': params, 'batch_stats': stats, 'opt': opt, 'ema': ema,
                'steps': state['steps'] + 1}, losses

    live = lambda: jax.device_get({'params': state['params'], 'batch_stats': state['batch_stats']})
    saved = [steps]

    def save():
        # Once a step: a step that is a save and a gate, or the last one, would write it twice.
        if saved[0] == steps and path.exists(ckpt):
            return
        saved[0] = steps
        host = jax.device_get(state)
        with open(ckpt + '.tmp', 'wb') as f:
            f.write(serialization.msgpack_serialize(serialization.to_state_dict(host)))
        os.replace(ckpt + '.tmp', ckpt)
        meta_out = {'conv_channels': channels, 'num_blocks': blocks, 'steps': steps}
        for name, v in (('weights.npz', {'params': host['params'], 'batch_stats': host['batch_stats']}),
                        ('weights_ema.npz', host['ema'])):
            convert.save_npz(path.join(args.out, name + '.tmp.npz'), v, meta_out)
            os.replace(path.join(args.out, name + '.tmp.npz'), path.join(args.out, name))
        logging.info(f'saved at step {steps:,}')

    gate = Gate(args.baseline, args.out, (channels, blocks), variables,
                walls=config['test_play']['games'] // 4,
                key=tpu_cfg['gate_key'], margin=config['test_play']['gate_margin'],
                patience=config['test_play']['gate_patience'], arenas=tpu_cfg['gate_arenas'],
                device=devices[-1], scratch=scratch)
    if gate.stop:
        logging.info(f'the gate stopped this run already ({gate.state["fails"]} misses); the champion is '
                     f'step {gate.state["champion"]:,}, in champion.npz')
        return 3
    if args.steps and steps >= args.steps:
        logging.info(f'at step {steps:,} already, --steps {args.steps:,}; nothing to do')
        return 0
    selfplay = SelfPlay(live(), (channels, blocks), args.opponent, arenas=tpu_cfg['arenas'],
                        walls=tpu_cfg['walls'], play_cfg=config['train_play'],
                        capacity=config['online']['server']['capacity'], root=path.join(scratch, 'selfplay'),
                        devices=devices, history=config['online']['history_window'])
    selfplay.start()
    logging.info(f'self-play: {tpu_cfg["arenas"]} arenas of {tpu_cfg["walls"]} walls; '
                 f'gate every {test_every:,} steps on {config["test_play"]["games"] // 4:,} walls')

    started = time.perf_counter()
    window, waited, t_log, games_log = [], 0., time.perf_counter(), selfplay.games
    stop = None
    over = lambda: args.hours and time.perf_counter() - started > args.hours * 3600
    try:
        while stop is None:
            t = time.perf_counter()
            files = selfplay.take(timeout=60)
            waited += time.perf_counter() - t
            if not files:
                stop = 'hours' if over() else None
                continue
            data = FileDatasetsIter(
                version=4, file_list=files, pts=config['env']['pts'], file_batch_size=ds['file_batch_size'],
                reserve_ratio=ds['reserve_ratio'], parquet=False, player_names=['trainee'],
                num_epochs=1, enable_augmentation=False, augmented_first=False)
            workers = min(ds['num_workers'], len(files))
            batches = full_batches(DataLoader(
                data, batch_size=batch_size, drop_last=False, num_workers=workers, worker_init_fn=init_worker,
                collate_fn=functools.partial(collate, slots=None), prefetch_factor=ds['prefetch_factor'],
                multiprocessing_context='forkserver'), batch_size)
            t = time.perf_counter()
            for batch in batches:
                waited += time.perf_counter() - t
                state, losses = step(state, jax.device_put(as_arrays(batch), split))
                steps += 1
                window.append(losses)
                if steps % submit_every == 0:
                    selfplay.publish(live())
                if steps % args.log_every == 0:
                    got = jax.device_get(window)
                    dt = time.perf_counter() - t_log
                    sessions, rank, pt = selfplay.summary()
                    logging.info(
                        f'step {steps:,}: ' + ', '.join(f'{k} {np.mean([w[k] for w in got]):.4f}'
                                                         for k in ('dqn_loss', 'next_rank_loss'))
                        + f'; {len(window) * batch_size / dt:,.0f} samples/s, self-play '
                        f'{(selfplay.games - games_log) / dt:.1f} games/s, {selfplay.waiting():,} waiting, '
                        f'training waited {waited / dt:.0%}; trainee over the last {sessions} sessions: '
                        f'rank {rank:.4f}, {pt:+.2f} pt')
                    window, waited, t_log, games_log = [], 0., time.perf_counter(), selfplay.games
                if steps % save_every == 0:
                    save()
                if steps % test_every == 0:
                    save()
                    gate.evaluate(steps, jax.device_get(state['ema']))
                    if gate.stop:
                        logging.info(f'gate: {gate.state["fails"]} evaluations without a new champion, '
                                     f'stopping; the champion is step {gate.state["champion"]:,}')
                        stop = 'gate'
                        break
                if args.steps and steps >= args.steps:
                    stop = 'steps'
                    break
                if over():
                    logging.info(f'{args.hours} h are up')
                    stop = 'hours'
                    break
                t = time.perf_counter()
            del batches
            # train.py publishes at the end of every round as well.
            selfplay.publish(live())
            discard(files)
    finally:
        selfplay.stop()
        shutil.rmtree(scratch, ignore_errors=True)
    save()
    logging.info(f'done at step {steps:,} ({stop})')
    return 3 if stop == 'gate' else 0


if __name__ == '__main__':
    sys.exit(main())
