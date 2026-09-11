"""Multi-GPU offline training with DistributedDataParallel.

    torchrun --standalone --nproc_per_node=2 train.py

Each rank trains on its own shard of the corpus and DDP averages the gradients,
so two ranks at `batch_size = 512` take the same optimizer steps a single GPU
would at 1024. Rank 0 alone writes logs and checkpoints; test play is split
across every rank by seed range, so the games played do not depend on how many
GPUs there are.

Launched as plain `python train.py`, nothing here does anything: one process,
one GPU, the code path it always was.
"""
import logging
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist

# Test play runs long past NCCL's default ten minutes on a slow card, and a
# rank that finishes its share first waits at a barrier for the rest.
TIMEOUT = timedelta(hours=3)


def available_cpus():
    """CPUs this process may really use.

    On a rented slice of a shared host, os.cpu_count() reports the whole
    machine - all 256 threads of a box you were given 64 of - and pools sized
    from it thrash. The cgroup quota is the real limit when there is one, and
    the CPUs the process may run on when they are fewer still (56 against a
    quota of 61 on another box).
    """
    affinity = len(os.sched_getaffinity(0))
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            quota, period = f.read().split()
        if quota != 'max':
            return max(1, min(affinity, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:
            quota = int(f.read())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
            period = int(f.read())
        if quota > 0:
            return max(1, min(affinity, quota // period))
    except (OSError, ValueError):
        pass
    return affinity


class NullWriter:
    """Stands in for SummaryWriter on ranks that do not log."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class Dist:
    def __init__(self):
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.enabled = self.world_size > 1
        self.control = None
        # Before CUDA or NCCL has had a chance to narrow it; see keep_cpus.
        self.cpus = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
        self.window_start = time.perf_counter()
        self.waited = {'data': 0., 'ranks': 0.}

    @property
    def is_main(self):
        return self.rank == 0

    def setup(self, device, loaders_per_rank=1):
        """Join the process group; return the device this rank trains on."""
        if not self.enabled:
            return device

        if os.environ.get('MORTAL_DDP_SHARE_GPU') == '1':
            # For checking the plumbing on a one-GPU machine. NCCL refuses two
            # ranks on the same device; gloo does not, only slower.
            backend = 'gloo'
            device = torch.device('cuda', 0)
        else:
            backend = 'nccl'
            count = torch.cuda.device_count()
            if count < self.world_size:
                raise RuntimeError(
                    f'{self.world_size} ranks but {count} GPU(s). Set '
                    'MORTAL_DDP_SHARE_GPU=1 to put them all on one for testing.')
            device = torch.device('cuda', self.local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend, timeout=TIMEOUT)
        # Barriers and the end-of-data vote go over gloo on the CPU, so they
        # never queue behind GPU work the way an NCCL call on the stream would.
        self.control = dist.new_group(backend='gloo', timeout=TIMEOUT)
        self.keep_cpus()

        cpus = available_cpus()
        per_rank = max(1, cpus // self.world_size)
        # Every loader process fans out over its own rayon pool, which sizes
        # itself to the whole machine. Left alone, ranks x workers pools of
        # that size share the same cores. Set before any loader starts.
        per_loader = max(1, per_rank // max(1, loaders_per_rank))
        os.environ.setdefault('RAYON_NUM_THREADS', str(per_loader))
        torch.set_num_threads(per_rank)

        if not self.is_main:
            # One set of logs is enough; the others speak up only on trouble.
            logging.getLogger().setLevel(logging.WARNING)
        logging.info(
            f'DDP: {self.world_size} ranks over {backend}, {cpus} CPUs, '
            f'RAYON_NUM_THREADS={os.environ["RAYON_NUM_THREADS"]} per loader, '
            f'{loaders_per_rank} loader(s) per rank')
        return device

    def wrap(self, module, device):
        if not self.enabled:
            return module
        from torch.nn.parallel import DistributedDataParallel
        # BatchNorm stays per rank: each sees a full per-rank batch, and
        # SyncBatchNorm would add an all-reduce for every one of the ~80 BN
        # layers on every step. Running stats are broadcast from rank 0.
        net = DistributedDataParallel(module, device_ids=[device.index])
        # Its first broadcast is where NCCL sets up, if setup did not.
        self.keep_cpus()
        return net

    def keep_cpus(self):
        """Put this thread back on every CPU the process started with.

        NCCL's setup can leave the calling thread on just the CPUs next to
        its GPU, and every thread and loader worker started from it
        afterwards inherits that. On a rented slice those can be a handful of the CPUs
        the process may use: one box gave rank 1 eight of its 56, and its
        three loaders' decoding crowded onto them while rank 0 waited in the
        all-reduce. NCCL's own threads keep their placement.
        """
        if self.cpus and os.sched_getaffinity(0) != self.cpus:
            logging.warning(
                f'rank {self.rank}: something narrowed this thread to '
                f'{len(os.sched_getaffinity(0))} of {len(self.cpus)} CPUs; restoring')
            os.sched_setaffinity(0, self.cpus)

    def no_sync(self, net, sync):
        """Skip the gradient all-reduce on accumulation steps that do not step."""
        if self.enabled and not sync:
            return net.no_sync()
        return nullcontext()

    def shard(self, items, seed):
        """This rank's share of `items`; the same split on every rank.

        The shuffle is seeded identically everywhere before slicing, so the
        shards are disjoint and together cover the whole list.
        """
        if not self.enabled:
            return items
        order = list(items)
        random.Random(seed).shuffle(order)
        return order[self.rank::self.world_size]

    def in_lockstep(self, batches):
        """Yield batches for as long as every rank still has one.

        Ranks hold different amounts of data. The first to run out would leave
        the rest waiting forever on a gradient all-reduce it never joins, so
        they vote on every batch, and one empty rank stops them all. Each vote
        is cast a step ahead and runs while the step before it trains: waited
        on at once, it tied every rank's host to the slowest one's, and a
        host that cannot queue the next step leaves its GPU idle.

        Also times the waits, for `pace`: for the next batch, and for the
        other ranks' votes.
        """
        it = iter(batches)
        self.window_start = time.perf_counter()
        self.waited = {'data': 0., 'ranks': 0.}

        def fetch():
            t = time.perf_counter()
            batch = next(it, None)
            self.waited['data'] += time.perf_counter() - t
            if not self.enabled:
                return batch, None
            flag = torch.tensor([0 if batch is None else 1])
            work = dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.control, async_op=True)
            return batch, (flag, work)

        batch, vote = fetch()
        while True:
            if vote is None:
                stop = batch is None
            else:
                t = time.perf_counter()
                flag, work = vote
                work.wait()
                self.waited['ranks'] += time.perf_counter() - t
                stop = not flag.item()
            if stop:
                return
            current = batch
            batch, vote = fetch()
            yield current

    def pace(self):
        """[wall, data, ranks] seconds since the last call, one list per rank.

        What is not waiting on data or on the other ranks is the training step
        itself. Every rank has to call this.
        """
        now = time.perf_counter()
        mine = [now - self.window_start, self.waited['data'], self.waited['ranks']]
        self.window_start = now
        self.waited = {'data': 0., 'ranks': 0.}
        if not self.enabled:
            return [mine]
        t = torch.tensor(mine, dtype=torch.float64)
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t, group=self.control)
        return [g.tolist() for g in gathered]

    def barrier(self):
        if self.enabled:
            dist.barrier(group=self.control)

    def mean(self, value, device):
        """A scalar averaged over ranks, for logging."""
        if not self.enabled:
            return value
        t = torch.as_tensor(value, dtype=torch.float64, device=device).detach().clone()
        dist.all_reduce(t)
        return t / self.world_size

    def check_in_sync(self, modules, device):
        """Raise if the ranks are no longer training the same model.

        DDP averages gradients and nothing else, so anything that lets one rank
        drift - a parameter that never receives a gradient, a stray in-place
        update - stays silent until the ranks are optimising different models.
        Buffers are left out: BN running stats are broadcast from rank 0 at
        each forward and then updated locally, so they differ by design.
        """
        if not self.enabled:
            return
        params = [p.detach().double() for m in modules for p in m.parameters()]
        sig = torch.stack([p.sum() for p in params] + [p.norm() for p in params])
        gathered = [torch.empty_like(sig) for _ in range(self.world_size)]
        dist.all_gather(gathered, sig)
        scale = gathered[0].abs().max().item() + 1.0
        worst = max((g - gathered[0]).abs().max().item() for g in gathered)
        if worst > 1e-9 * scale:
            raise RuntimeError(
                f'ranks have diverged: parameter signatures differ by {worst:.3e}')

    def close(self):
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()
