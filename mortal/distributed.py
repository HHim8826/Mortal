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
    from it thrash. The cgroup quota is the real limit when there is one.
    """
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            quota, period = f.read().split()
        if quota != 'max':
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:
            quota = int(f.read())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
            period = int(f.read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


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
        return DistributedDataParallel(module, device_ids=[device.index])

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
        before each step they vote, and one empty rank stops them all.
        """
        if not self.enabled:
            yield from batches
            return
        it = iter(batches)
        while True:
            batch = next(it, None)
            flag = torch.tensor([0 if batch is None else 1])
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.control)
            if not flag.item():
                return
            yield batch

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
