"""What the box is really doing while an evaluation runs.

Test play looks like it uses a fraction of the box, and the arena's shape says
why it might: it advances every game in flight on one thread, encodes the
states that need a decision in parallel, then runs one GPU forward for all of
them, over and over. If that is what is happening, the CPU and the GPU are
taking turns rather than being short of work -- and no average over seconds can
show it, because a turn lasts milliseconds.

Three measurements, each chosen for what the container makes honest:

  - Cores busy, from the cgroup's own cpu.stat. /proc/stat covers all 128 CPUs
    of the host, of which this container may use 56, so it would answer a
    question about the neighbours. The cgroup counter is this container's CPU
    time and nothing else.
  - How many of the training processes' threads are on, or waiting for, a CPU,
    sampled every 20 ms in bursts. /proc/loadavg counts the host's runnable
    threads, the neighbours' included, so this walks our own /proc/<pid>/task
    instead. Fast enough to see the turns: alternation shows as a split
    distribution, one mode with many threads awake (encoding) and one with
    almost none (waiting for the GPU), where simply being short of work would
    show one low mode throughout.
  - GPU occupancy, from `nvidia-smi dmon`, once a second per GPU. This is the
    share of time a kernel was resident, not how much of the GPU it filled, so
    it answers "was the GPU idle" and nothing else -- which is the question.

    nohup /root/venv/bin/python /root/eval_perf.py &
"""
import os
import subprocess
import time
from collections import Counter

LOG = '/root/Mortal/mortal/logs/v4/train.log'
OUT = '/root/eval_perf.log'
GPU_LOG = '/root/eval_gpu.log'
MINUTES = 12
CPUS = len(os.sched_getaffinity(0))


def say(*args):
    with open(OUT, 'a') as f:
        print(time.strftime('%H:%M:%S', time.gmtime()), *args, file=f, flush=True)


def evals_started():
    """How many test plays this log has announced. The arena prints its seed
    range as it starts, the only line that says one has begun."""
    with open(LOG, 'rb') as f:
        return f.read().replace(b'\r', b'\n').count(b'seed: [10000')


def cpu_usage():
    """The container's CPU time so far, in seconds."""
    with open('/sys/fs/cgroup/cpu.stat') as f:
        for line in f:
            if line.startswith('usage_usec'):
                return int(line.split()[1]) / 1e6
    raise RuntimeError('no usage_usec in cpu.stat')


def rank_pids():
    """The processes that run the arena: the DDP ranks, not their children.

    Each rank's loader workers add hundreds of threads that sit idle through
    test play, and reading all of their stat files would stretch a sample from
    15 ms to 45 ms -- too slow to see what this is looking for. The ranks are
    the training processes whose parent is the launcher, and the launcher is
    the one whose own parent is outside the set.
    """
    pids = subprocess.run(['pgrep', '-f', '[t]rain.py'],
                          capture_output=True, text=True).stdout.split()
    parent = {}
    for pid in pids:
        try:
            with open(f'/proc/{pid}/stat') as f:
                parent[pid] = f.read().rsplit(') ', 1)[1].split()[1]
        except OSError:
            pass
    launchers = {pid for pid, ppid in parent.items() if ppid not in parent}
    return [pid for pid, ppid in parent.items() if ppid in launchers] or list(parent)


def thread_stats():
    """Every thread of those, as a path to its stat file."""
    pids = rank_pids()
    paths = []
    for pid in pids:
        try:
            paths += [f'/proc/{pid}/task/{tid}/stat' for tid in os.listdir(f'/proc/{pid}/task')]
        except OSError:
            pass  # it exited between the listing and now
    return paths


def runnable(paths):
    """Those threads that are on a CPU or queued for one."""
    n = 0
    for p in paths:
        try:
            with open(p) as f:
                line = f.read()
        except OSError:
            continue
        # "pid (comm) S ...", and comm may itself hold spaces and brackets.
        if line.rsplit(') ', 1)[1][0] == 'R':
            n += 1
    return n


def burst(seconds=2, every=0.02):
    paths = thread_stats()
    counts = Counter()
    end = time.time() + seconds
    while time.time() < end:
        counts[runnable(paths)] += 1
        time.sleep(every)
    return counts


def main():
    before = evals_started()
    while evals_started() <= before:
        time.sleep(5)

    say(f'evaluation started; {CPUS} cpus in this container')
    dmon = subprocess.Popen(
        ['nvidia-smi', 'dmon', '-s', 'u', '-d', '1', '-c', str(MINUTES * 60)],
        stdout=open(GPU_LOG, 'w'), stderr=subprocess.DEVNULL)

    for minute in range(MINUTES):
        used, clock = cpu_usage(), time.time()
        time.sleep(55)
        cores = (cpu_usage() - used) / (time.time() - clock)

        counts = burst()
        total = sum(counts.values())
        # The head of the distribution is everything waiting on the GPU; the
        # tail is the encoding that fills the cores.
        quiet = sum(n for r, n in counts.items() if r <= 4) / total
        wide = sum(n for r, n in counts.items() if r >= 16) / total
        common = ', '.join(f'{r} threads {n * 100 // total}%' for r, n in counts.most_common(3))
        say(f'minute {minute + 1}: {cores:5.1f} of {CPUS} cores busy; runnable '
            f'{quiet:.0%} at most 4, {wide:.0%} at least 16; commonest {common}')

    dmon.wait()
    rows = [l.split() for l in open(GPU_LOG) if not l.lstrip().startswith('#')]
    for idx in ('0', '1'):
        sm = sorted(int(r[1]) for r in rows if len(r) > 1 and r[0] == idx and r[1].isdigit())
        if sm:
            say(f'gpu {idx}: a kernel was running {sm[len(sm) // 2]}% of the median '
                f'second, {sm[len(sm) * 9 // 10]}% at p90; '
                f'{sum(1 for x in sm if x < 10) / len(sm):.0%} of seconds under 10%')
    say('done')


if __name__ == '__main__':
    main()
