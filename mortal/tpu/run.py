"""Offline v4 training in JAX, fed by train.py's own loader.

    MORTAL_CFG=config.tpu.toml python -m tpu.run --out /kaggle/working/run \\
        [--init best_ema.npz] [--grow-to 60] [--steps 200000]

Data parallel over every device JAX finds (the eight chips of a v5e-8, or one
GPU): parameters replicated, each batch split across devices, the gradient
all-reduce left to XLA. What it keeps from train.py's offline run:

- the loader (`FileDatasetsIter` over parquet row groups, in DataLoader
  workers), the loss, AdamW with decay on kernels only, the warm-up cosine;
- an EMA of the parameters and BatchNorm statistics at `ema_decay`, which is
  what gets evaluated: at 800k the average's held-out fit beat the raw
  weights' by more than 280k steps of training had.

`--grow-to` deepens a checkpoint before training: new residual blocks go after
the old ones with their second convolution zeroed, which makes each one exactly
the identity, so step 0 plays exactly as the checkpoint did.

Every `save_every` steps it writes `state.msgpack` (everything, to resume) and
`weights.npz` / `weights_ema.npz` (for `python -m tpu.convert import` on a GPU
box, where the evaluation tools are).
"""
import argparse
import logging
import os
import random
import time
from glob import glob
from os import path

import numpy as np

# prelude's format, without prelude: it pulls in torch's tensorboard, which a TPU host
# running only this has no use for.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s')


def build_file_list(dataset_cfg, seed):
    import pyarrow.parquet as pq
    shards = sorted(s for pat in dataset_cfg.get('parquet_globs', []) for s in glob(pat, recursive=True))
    if shards:
        files = [(s, rg) for s in shards for rg in range(pq.ParquetFile(s).num_row_groups)]
        logging.info(f'{len(shards):,} shards, {len(files):,} row groups')
    else:
        files = sorted(f for pat in dataset_cfg['globs'] for f in glob(pat, recursive=True))
        logging.info(f'{len(files):,} gz logs')
    random.Random(seed).shuffle(files)
    return files


class Slots:
    """Shared memory the loader's workers stack observations into, reused batch after batch.

    Sent the usual way, every batch lands in shared memory allocated for it, and
    the first read of it in this process faults in every page. On the Kaggle host
    that, not decoding, was the limit: torch's DataLoader moved ~46,000 samples/s
    of float32 observations with workers that decoded nothing at all, and the
    device_put that first reads a batch halved what the real loader delivered.
    These slots are allocated and touched here once, before the workers are
    forked, so their pages stay mapped: a worker takes a free slot, stacks a
    batch into it, and sends only the slot's number, which comes back once the
    batch is on the devices.
    """

    def __init__(self, count, batch_size, shape=(1012, 34)):
        import multiprocessing
        import torch
        self.obs = torch.zeros((count, batch_size, *shape), dtype=torch.bfloat16).share_memory_()
        self.free = multiprocessing.get_context('fork').SimpleQueue()
        for i in range(count):
            self.free.put(i)


def init_worker(*args):
    """dataloader.worker_init_fn, in a worker that dies with the process it serves.

    Left alone, a worker outlives a trainer that is killed -- by the OOM killer,
    say: it waits for a slot that nobody will free, and it holds the TPU open,
    because it was forked from a process that had it, so the next run cannot
    start. On Kaggle that took 27 orphans and a kill -9 each.
    """
    import ctypes
    import signal
    from dataloader import worker_init_fn
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)       # PR_SET_PDEATHSIG
    if os.getppid() == 1:                            # it died before that took hold
        os._exit(1)
    worker_init_fn(*args)


def loader(config, files, batch_size, seed, slots=None):
    import functools
    import torch
    from torch.utils.data import DataLoader
    from dataloader import FileDatasetsIter
    ds = config['dataset']
    data = FileDatasetsIter(
        version=4, file_list=files, pts=config['env']['pts'],
        file_batch_size=ds['file_batch_size'], reserve_ratio=ds['reserve_ratio'],
        parquet=bool(files) and not isinstance(files[0], str), player_names=[],
        num_epochs=ds['num_epochs'], enable_augmentation=ds['enable_augmentation'],
        augmented_first=ds['augmented_first'])
    # The slots reach the workers by fork, which is what the DataLoader uses on Linux.
    kw = {'prefetch_factor': ds.get('prefetch_factor', 2), 'in_order': ds.get('in_order', True),
          'multiprocessing_context': 'fork'} if ds['num_workers'] > 0 else {}
    torch.manual_seed(seed)
    return DataLoader(data, batch_size=batch_size, drop_last=True, num_workers=ds['num_workers'],
                      worker_init_fn=init_worker, collate_fn=functools.partial(collate, slots=slots), **kw)


def collate(samples, slots=None):
    """default_collate, with the observations in bfloat16, in a slot if there are slots.

    The step casts them to bfloat16 first thing, and torch rounds float32 to
    bfloat16 as XLA does, to nearest with ties to even, so the net sees the same
    input from half the bytes.
    """
    import torch
    from torch.utils.data import default_collate
    rest = default_collate([s[1:] for s in samples])
    frames = [torch.from_numpy(s[0]) for s in samples]
    if slots is not None:
        slot = slots.free.get()
        torch.stack(frames, out=slots.obs[slot])
        return [slot, *rest]
    obs = torch.empty((len(samples), *samples[0][0].shape), dtype=torch.bfloat16)
    if torch.utils.data.get_worker_info() is not None:
        obs.share_memory_()           # what default_collate does, so it is not copied again
    torch.stack(frames, out=obs)
    return [obs, *rest]


def as_arrays(batch):
    import ml_dtypes
    import torch
    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
    # numpy has no bfloat16 of its own; ml_dtypes' is the one JAX takes.
    return (obs.view(torch.int16).numpy().view(ml_dtypes.bfloat16), actions.numpy().astype(np.int32),
            masks.numpy(), steps_to_done.numpy().astype(np.int32),
            kyoku_rewards.numpy().astype(np.float32), player_ranks.numpy().astype(np.int32))


def on_devices(batches, sharding, slots, threads=2, ready=8):
    """The loader's batches as arrays on the devices, in order.

    A feeder thread takes each batch from the loader and hands it to a pool that
    puts it on the devices; the loop dispatching steps only takes them off a queue
    of at most `ready`. So a batch already on the devices never waits behind the
    loader's next one -- in lockstep, one slow decode held up every finished batch
    behind it, which cost a quarter of the speed at 72,000 samples/s (#47) -- and
    the loader's own next() is off the dispatching thread. Each slot goes back to
    the workers once its batch has arrived, whether or not it has been taken yet.
    """
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import jax
    # A CPU backend may alias the host memory it is given, which the slot's next
    # batch would then overwrite.
    aliasing = any(d.platform == 'cpu' for d in sharding.device_set)

    def put(batch):
        slot, *rest = batch
        obs = slots.obs[slot].clone() if aliasing else slots.obs[slot]
        arrays = jax.block_until_ready(jax.device_put(as_arrays([obs, *rest]), sharding))
        slots.free.put(slot)
        return arrays

    # Started here rather than in the feeder: iter() forks the workers, and their
    # PR_SET_PDEATHSIG fires when the thread that forked them ends, not the process.
    loaded = iter(batches)
    out = queue.Queue(ready)
    stop = threading.Event()
    end = object()

    def offer(item):
        while not stop.is_set():
            try:
                out.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def feed():
        try:
            for batch in loaded:
                if not offer(pool.submit(put, batch)):
                    return
        except BaseException as exc:
            offer(exc)
        else:
            offer(end)

    pool = ThreadPoolExecutor(threads)
    feeder = threading.Thread(target=feed, name='on_devices', daemon=True)
    feeder.start()
    try:
        while (item := out.get()) is not end:
            if isinstance(item, BaseException):
                raise item
            yield item.result()
    finally:
        # Stopped early, or failed: the feeder notices once the loader's next() returns,
        # which is at most one batch away, and the puts under way finish, so their
        # slots are back before the workers are shut down.
        stop.set()
        feeder.join(timeout=120)
        pool.shutdown(wait=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', help='an .npz from `tpu.convert export`; from scratch without it')
    ap.add_argument('--grow-to', type=int, help='deepen the --init net to this many blocks')
    ap.add_argument('--steps', type=int, default=0, help='stop after this many; 0 runs the data out')
    ap.add_argument('--hours', type=float, default=0,
                    help='save and stop after this long; a Kaggle session ends at 9 h, uploads or not')
    ap.add_argument('--remat', action='store_true',
                    help='recompute block activations in the backward pass: less memory, and on a '
                         'v5e-8 faster as well (7.1 ms a step at batch 1,024 against 9.0)')
    ap.add_argument('--log-every', type=int, default=100)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization
    from config import config
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from tpu import convert
    from tpu.model import Mortal, deepen
    from tpu.train import loss_fn, make_optimizer

    os.makedirs(args.out, exist_ok=True)
    ctl, res = config['control'], config['resnet']
    batch_size, save_every = ctl['batch_size'], ctl['save_every']
    ema_decay = ctl.get('ema_decay', 0)
    devices = jax.devices()
    if batch_size % len(devices):
        raise SystemExit(f'batch_size {batch_size} does not split over {len(devices)} devices')
    mesh = Mesh(np.array(devices), ('data',))
    split, whole = NamedSharding(mesh, P('data')), NamedSharding(mesh, P())
    logging.info(f'{len(devices)} x {devices[0].device_kind}, global batch {batch_size:,}')

    tx = make_optimizer(config['optim'])
    rng = jax.random.PRNGKey(ctl.get('seed', 0))
    ckpt = path.join(args.out, 'state.msgpack')
    resume = None
    if path.exists(ckpt):
        # A saved run decides its own shape. It may have been deepened by an earlier
        # session's --grow-to, which this one need not repeat -- and must not
        # contradict, which is caught here, before any data is read.
        with open(ckpt, 'rb') as f:
            resume = serialization.msgpack_restore(f.read())
        shape = resume.pop('shape', None) or {
            'conv_channels': int(resume['params']['brain']['stem']['kernel'].shape[-1]),
            'num_blocks': int(resume['params']['brain']['blocks']['conv1']['kernel'].shape[0])}
        channels, blocks = shape['conv_channels'], shape['num_blocks']
        if args.grow_to and args.grow_to != blocks:
            raise SystemExit(f'{ckpt} holds {blocks} blocks; --grow-to {args.grow_to} contradicts it. '
                             'Resume without --grow-to, or train into a new --out')
        if args.init:
            logging.info(f'resuming {ckpt} ({channels}x{blocks}); --init is only a starting point, ignored')
        variables = Mortal(channels, blocks).init(rng, jnp.zeros((2, 34, 1012)), jnp.ones((2, 46), bool))
    elif args.init:
        variables, meta = convert.load_npz(args.init)
        channels, blocks = meta['conv_channels'], meta['num_blocks']
        logging.info(f'init: {args.init}, {channels}x{blocks}, step {meta.get("steps", 0):,}')
        if args.grow_to and args.grow_to > blocks:
            variables = deepen(variables, channels, blocks, args.grow_to, rng)
            logging.info(f'grown: {blocks} -> {args.grow_to} blocks, the new ones the identity')
            blocks = args.grow_to
    else:
        channels, blocks = res['conv_channels'], res['num_blocks']
        variables = Mortal(channels, blocks).init(rng, jnp.zeros((2, 34, 1012)), jnp.ones((2, 46), bool))
    model = Mortal(channels, blocks, dtype=jnp.bfloat16, remat=args.remat)

    params, stats = variables['params'], variables['batch_stats']
    state = {'params': params, 'batch_stats': stats, 'opt': tx.init(params),
             'ema': {'params': params, 'batch_stats': stats}, 'steps': 0}
    if resume is not None:
        state = serialization.from_state_dict(state, resume)
        logging.info(f'resumed at step {int(state["steps"]):,}')
    steps = int(state['steps'])
    state = jax.device_put(state, whole)
    logging.info(f'parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state["params"])):,}')

    loss_kw = dict(model=model, gamma=config['env']['gamma'], min_q_weight=config['cql']['min_q_weight'],
                   next_rank_weight=config['aux']['next_rank_weight'])
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def step(state, batch):
        (_, (stats, losses)), grads = grad_fn(state['params'], state['batch_stats'], batch, **loss_kw)
        updates, opt = tx.update(grads, state['opt'], state['params'])
        params = optax.apply_updates(state['params'], updates)
        live = {'params': params, 'batch_stats': stats}
        ema = jax.tree_util.tree_map(lambda e, x: ema_decay * e + (1 - ema_decay) * x, state['ema'], live) \
            if ema_decay > 0 else live
        return {'params': params, 'batch_stats': stats, 'opt': opt, 'ema': ema,
                'steps': state['steps'] + 1}, losses

    def save():
        host = jax.device_get(state)
        with open(ckpt + '.tmp', 'wb') as f:
            f.write(serialization.msgpack_serialize(
                {**serialization.to_state_dict(host),
                 'shape': {'conv_channels': channels, 'num_blocks': blocks}}))
        os.replace(ckpt + '.tmp', ckpt)
        meta = {'conv_channels': channels, 'num_blocks': blocks, 'steps': steps}
        # Each through a temporary name, like the state: a copy taken for a backup
        # while a save is under way is then the old file or the new one, never half.
        for name, variables in (('weights.npz', {'params': host['params'], 'batch_stats': host['batch_stats']}),
                                ('weights_ema.npz', host['ema'])):
            convert.save_npz(path.join(args.out, name + '.tmp.npz'), variables, meta)
            os.replace(path.join(args.out, name + '.tmp.npz'), path.join(args.out, name))
        logging.info(f'saved at step {steps:,}')

    started = time.perf_counter()
    files = build_file_list(config['dataset'], seed=steps)
    ds = config['dataset']
    # One for every batch the DataLoader keeps in flight, and for the ones on the way
    # to the devices -- `ready` queued, one being queued, one in the feeder's hand;
    # with fewer the workers wait for slots, which costs speed only.
    ready = 8
    slots = Slots(ds['num_workers'] * ds.get('prefetch_factor', 2) + ready + 2, batch_size)
    window, waited, t_log = [], 0., time.perf_counter()
    # The step is dispatched, not waited for, so the device runs while the next batch
    # is fetched; time spent in the fetch itself is the loader not keeping up.
    t_back = time.perf_counter()
    for arrays in on_devices(loader(config, files, batch_size, steps, slots), split, slots, ready=ready):
        waited += time.perf_counter() - t_back
        state, losses = step(state, arrays)
        steps += 1
        # Kept on the device until the log: adding them up as they come is three more
        # dispatches a step, on the thread the step is waiting for.
        window.append(losses)
        if steps % args.log_every == 0:
            # Reading the losses back waits for the device to finish the window's
            # steps, so the window ends after it: dispatch alone is not training.
            got = jax.device_get(window)
            means = {k: float(np.mean([w[k] for w in got])) for k in got[0]}
            dt = time.perf_counter() - t_log
            logging.info(f'step {steps:,}: ' + ', '.join(f'{k} {v:.4f}' for k, v in means.items())
                         + f'; {len(window) * batch_size / dt:,.0f} samples/s, '
                         f'waiting for data {waited / dt:.0%}')
            window, waited, t_log = [], 0., time.perf_counter()
        if steps % save_every == 0:
            save()
        if args.steps and steps >= args.steps:
            break
        if args.hours and time.perf_counter() - started > args.hours * 3600:
            logging.info(f'{args.hours} h are up')
            break
        t_back = time.perf_counter()
    save()
    logging.info(f'done at step {steps:,}')


if __name__ == '__main__':
    main()
