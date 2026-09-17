"""Evaluate checkpoints on fixed, named sets of walls, and compare them paired.

    # play two checkpoints against the v3 bot on the first 250 walls of `dev`
    python evaluate.py play --set dev --limit-seeds 250 \\
        --model offline520=logs/v4/best_ema.pth --model online560=logs/v4o/best_ema.pth

    # compare what has been played; every model is set against the first
    python evaluate.py report --set dev \\
        --model offline520=logs/v4/best_ema.pth --model online560=logs/v4o/best_ema.pth

    python evaluate.py sets      # what the wall sets are

Why this and not test play inside train.py:

  - The walls are named and never change, so any two checkpoints ever played on
    a set can be compared game by game, however far apart they were played.
  - Games are cached by the checkpoint's sha256, in chunks that are either
    complete or absent. An interrupted run resumes; a checkpoint already played
    is not played again; adding a third model costs one model's games.
  - `dev` is for looking at often, `holdout` for confirming a choice made
    without it. The `legacy` set is the 4,000 games every evaluation of the v4
    runs used. Checkpoints were chosen on those walls, so a number on them is
    biased upward for whichever checkpoint won; they are here to check that
    this tool reproduces the old numbers, not to make new claims.
  - The error is taken over walls, the unit that is actually independent: one
    wall is dealt four times, once with the challenger in each seat.

A checkpoint is `path` for its trained weights or `path#ema` for the weight
average train.py keeps inside it. `best_ema.pth` already holds the average
under the ordinary names, so it is given without `#ema`.
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from os import path

import numpy as np

# Houou-table placement points, the unit every earlier evaluation reported.
PTS = (90, 45, 0, -135)
SPLITS = 'abcd'
# Not *.json: Stat.from_dir reads every *.json under a model's directory as a
# game log, and these sit beside the logs.
SUMMARY = 'summary.jsonl'
META = 'meta.jsonl'
Z_TWO_SIDED_5 = 1.959964
Z_POWER_80 = 0.841621


@dataclass(frozen=True)
class WallSet:
    name: str
    key: int
    first_seed: int
    seeds: int
    note: str

    @property
    def games(self):
        return self.seeds * 4


WALL_SETS = {
    'legacy': WallSet(
        'legacy', 0x2000, 10000, 1000,
        'the walls test play used throughout the v4 runs; checkpoints were '
        'selected on them, so use them to reproduce old numbers only'),
    # Keys spell "dev" and "hold". Self-play draws a random 64-bit key per
    # worker, so neither set is ever dealt in training.
    'dev': WallSet(
        'dev', 0x646576, 0, 5000,
        'day-to-day comparisons; look at it as often as needed'),
    'holdout': WallSet(
        'holdout', 0x686F6C64, 0, 25000,
        'confirming a candidate chosen beforehand; every play is written to a ledger'),
}


# ----------------------------------------------------------------- statistics
#
# A model's games are {(seed, split): rank}, rank 1..4. Everything below works
# on that and nothing else, so it is testable without playing a game.

def walls_of(games):
    """{seed: [rank for split a, b, c, d]} for the walls with all four games."""
    by_seed = {}
    for (seed, split), rank in games.items():
        by_seed.setdefault(seed, {})[split] = rank
    return {seed: [s[x] for x in SPLITS]
            for seed, s in by_seed.items() if len(s) == 4}


def metric(ranks, name):
    ranks = np.asarray(ranks)
    if name == 'rank':
        return ranks.astype(np.float64)
    if name == 'pt':
        return np.asarray(PTS, dtype=np.float64)[ranks - 1]
    if name == 'fourth':
        return (ranks == 4).astype(np.float64)
    raise ValueError(name)


def bootstrap_ci(per_wall, reps=10000, seed=0, batch=500):
    """95% percentile interval of the mean, resampling walls."""
    per_wall = np.asarray(per_wall, dtype=np.float64)
    n = len(per_wall)
    if n < 2 or reps <= 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, reps, batch):
        size = min(batch, reps - start)
        means.append(per_wall[rng.integers(0, n, size=(size, n))].mean(1))
    lo, hi = np.percentile(np.concatenate(means), [2.5, 97.5])
    return float(lo), float(hi)


def summarize(walls, seeds, name, reps=0):
    """A model's own average of `name` over `seeds`, with its error over walls."""
    per_wall = np.array([metric(walls[s], name).mean() for s in seeds])
    out = {'mean': float(per_wall.mean()) if len(per_wall) else float('nan'),
           'se': float(per_wall.std(ddof=1) / np.sqrt(len(per_wall))) if len(per_wall) > 1 else float('nan')}
    if reps:
        out['ci95'] = bootstrap_ci(per_wall, reps)
    return out


def paired(walls_a, walls_b, seeds, name, reps=10000):
    """a minus b on `name`, wall by wall, over `seeds` both have complete."""
    d = np.array([(metric(walls_a[s], name) - metric(walls_b[s], name)).mean() for s in seeds])
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float('nan')
    return {
        'diff': float(d.mean()) if n else float('nan'),
        'se': sd / np.sqrt(n) if n > 1 else float('nan'),
        'sd_per_wall': sd,
        'walls': n,
        'ci95': bootstrap_ci(d, reps),
    }


def in_se(result):
    """The difference in standard errors, as text; identical games have none."""
    if not result['se'] > 0:
        return 'n/a'
    return f'{result["diff"] / result["se"]:+.1f}'


def games_to_detect(sd_per_wall, effect):
    """Games (4 per wall) for a two-sided 5% test to catch `effect` 80% of the time."""
    if not np.isfinite(sd_per_wall) or effect <= 0:
        return float('nan')
    walls = ((Z_TWO_SIDED_5 + Z_POWER_80) * sd_per_wall / effect) ** 2
    return 4 * int(np.ceil(walls))


# -------------------------------------------------------------------- storage

def sha256_of(file, _cache={}):
    st = os.stat(file)
    key = (path.abspath(file), st.st_size, st.st_mtime_ns)
    if key not in _cache:
        h = hashlib.sha256()
        with open(file, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        _cache[key] = h.hexdigest()
    return _cache[key]


@dataclass
class Spec:
    label: str
    file: str
    part: str       # '' for the trained weights, 'ema' for the average inside

    @classmethod
    def parse(cls, text):
        label, sep, rest = text.partition('=')
        if not sep:
            label, rest = '', text
        file, _, part = rest.partition('#')
        if part not in ('', 'ema'):
            raise SystemExit(f'{text}: only #ema can follow a checkpoint')
        if not path.exists(file):
            raise SystemExit(f'{file}: no such checkpoint')
        return cls(label or path.splitext(path.basename(file))[0] + (f'#{part}' if part else ''),
                   file, part)

    @property
    def ident(self):
        return sha256_of(self.file)[:16] + (f'-{self.part}' if self.part else '')


def set_dir(root, wall_set, champion):
    return path.join(root, wall_set.name, f'vs-{champion.ident}')


def chunk_name(first, count):
    return f'seeds-{first:08d}-{count:05d}'


def chunks(wall_set, per_chunk, limit=None):
    """[(first seed, count)] covering the set, or its first `limit` seeds."""
    end = wall_set.first_seed + min(wall_set.seeds, limit or wall_set.seeds)
    return [(first, min(per_chunk, end - first))
            for first in range(wall_set.first_seed, end, per_chunk)]


def read_games(model_dir):
    """Every finished chunk's games for a model, as {(seed, split): rank}."""
    games = {}
    if not path.isdir(model_dir):
        return games
    for name in sorted(os.listdir(model_dir)):
        summary = path.join(model_dir, name, SUMMARY)
        if name.startswith('seeds-') and path.exists(summary):
            with open(summary, encoding='utf-8') as f:
                for key, rank in json.load(f)['ranks'].items():
                    seed, split = key.split('_')
                    games[(int(seed), split)] = rank
    return games


def summarize_logs(log_dir, player_name):
    """{'<seed>_<split>': rank} from the logs one arena run wrote."""
    from libriichi.stat import Stat
    ranks = {}
    for name in os.listdir(log_dir):
        if not name.endswith('.json.gz'):
            continue
        with gzip.open(path.join(log_dir, name), 'rt', encoding='utf-8') as f:
            log = f.read()
        seat = json.loads(log.split('\n', 1)[0])['names'].index(player_name)
        stat = Stat.from_log(log, seat)
        # libriichi names them <seed>_<key>_<split>.json.gz.
        seed, _, split = name[:-len('.json.gz')].split('_')
        ranks[f'{seed}_{split}'] = int(round(stat.avg_rank))
    return ranks


# ---------------------------------------------------------------------- play

def load_model(spec, device):
    import torch
    from model import Brain, DQN
    state = torch.load(spec.file, weights_only=True, map_location='cpu')
    weights = state['ema'] if spec.part == 'ema' else state
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    brain = Brain(version=version, conv_channels=cfg['resnet']['conv_channels'],
                  num_blocks=cfg['resnet']['num_blocks']).eval()
    dqn = DQN(version=version).eval()
    brain.load_state_dict(weights['mortal'])
    dqn.load_state_dict(weights['current_dqn'])
    return brain.to(device), dqn.to(device), version, state.get('steps')


def engine_for(spec, device, name):
    from engine import MortalEngine
    brain, dqn, version, steps = load_model(spec, device)
    # The same settings test play in player.py uses, so numbers carry over.
    return MortalEngine(
        brain, dqn,
        is_oracle = False,
        version = version,
        device = device,
        enable_amp = True,
        enable_rule_based_agari_guard = True,
        name = name,
    ), steps


def write_meta(model_dir, spec, steps, role):
    os.makedirs(model_dir, exist_ok=True)
    meta_file = path.join(model_dir, META)
    meta = {}
    if path.exists(meta_file):
        with open(meta_file, encoding='utf-8') as f:
            meta = json.load(f)
    labels = set(meta.get('labels', [])) | {spec.label}
    meta.update({
        'role': role,
        'sha256': sha256_of(spec.file),
        'part': spec.part or 'trained',
        'steps': steps,
        'labels': sorted(labels),
        'files': sorted(set(meta.get('files', [])) | {path.abspath(spec.file)}),
    })
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f)


def play_chunk(engine, champion, wall_set, first, count, model_dir, quiet):
    """One chunk, all or nothing: written aside, checked, then renamed in."""
    from libriichi.arena import OneVsThree
    final = path.join(model_dir, chunk_name(first, count))
    partial = final + '.partial'
    shutil.rmtree(partial, ignore_errors=True)
    rankings = OneVsThree(disable_progress_bar=quiet, log_dir=partial).py_vs_py(
        challenger = engine,
        champion = champion,
        seed_start = (first, wall_set.key),
        seed_count = count,
    )
    ranks = summarize_logs(partial, engine.name)
    counted = [sum(1 for r in ranks.values() if r == k) for k in (1, 2, 3, 4)]
    # The arena's own tally and the one read back from the logs must agree,
    # or the logs are not the games that were played.
    if counted != list(rankings) or len(ranks) != count * 4:
        raise RuntimeError(f'{partial}: arena counted {list(rankings)}, logs give {counted} '
                           f'over {len(ranks)} games for {count * 4}')
    with open(path.join(partial, SUMMARY), 'w', encoding='utf-8') as f:
        json.dump({'first_seed': first, 'seeds': count, 'key': wall_set.key,
                   'rankings': list(rankings), 'ranks': ranks,
                   'finished': datetime.now(timezone.utc).isoformat()}, f)
    os.replace(partial, final)
    return rankings


def cmd_play(args):
    import torch
    import prelude  # noqa: F401  (logging format, warnings)

    wall_set = WALL_SETS[args.set]
    if wall_set.name == 'holdout' and not args.confirm_holdout:
        raise SystemExit('holdout is for confirming a choice made without it; '
                         'pass --confirm-holdout "<what is being confirmed>"')
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False

    champion_spec = Spec.parse(args.champion)
    models = [Spec.parse(m) for m in args.model]
    base = set_dir(args.root, wall_set, champion_spec)
    todo = chunks(wall_set, args.chunk_seeds, args.limit_seeds)

    pending = {}
    for spec in models:
        model_dir = path.join(base, spec.ident)
        # By the walls already complete, not by chunk directory names, so a
        # run with a different --chunk-seeds does not replay what exists.
        have = walls_of(read_games(model_dir))
        missing = [c for c in todo if any(s not in have for s in range(c[0], c[0] + c[1]))]
        print(f'{spec.label}: {spec.ident}, {len(todo) - len(missing)} of {len(todo)} chunks already played')
        if missing:
            pending[spec.ident] = (spec, model_dir, missing)
    if not pending:
        print('nothing to play')
        return

    if wall_set.name == 'holdout':
        os.makedirs(path.join(args.root, 'holdout'), exist_ok=True)
        with open(path.join(args.root, 'holdout', 'ledger.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'when': datetime.now(timezone.utc).isoformat(),
                                'why': args.confirm_holdout,
                                'champion': champion_spec.ident,
                                'models': {s.label: s.ident for s, _, _ in pending.values()}}) + '\n')

    champion, champion_steps = engine_for(champion_spec, device, 'baseline')
    write_meta(base, champion_spec, champion_steps, 'champion')
    engines = {}
    for ident, (spec, model_dir, _) in pending.items():
        engines[ident], steps = engine_for(spec, device, 'mortal')
        write_meta(model_dir, spec, steps, 'challenger')

    # Chunk by chunk, every model that still needs a chunk plays it at once.
    # libriichi releases the GIL while it plays, and one arena leaves most of
    # the cores idle, so a second arena in a thread is nearly free. They share
    # the champion, which keeps nothing between calls.
    start = time.time()
    played = 0
    for first, count in todo:
        jobs = [(ident, model_dir) for ident, (_, model_dir, missing) in pending.items()
                if (first, count) in missing]
        if not jobs:
            continue
        failures = []

        def run(ident, model_dir, quiet):
            try:
                play_chunk(engines[ident], champion, wall_set, first, count, model_dir, quiet)
            except BaseException as exc:
                failures.append(exc)

        threads = [threading.Thread(target=run, args=(ident, d, i > 0 or args.quiet))
                   for i, (ident, d) in enumerate(jobs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if failures:
            raise failures[0]
        played += count * 4 * len(jobs)
        rate = played / (time.time() - start)
        print(f'seeds [{first}, {first + count}) done for {len(jobs)} model(s); '
              f'{played:,} games, {rate:.2f} games/s', flush=True)


# -------------------------------------------------------------------- report

def cmd_report(args):
    wall_set = WALL_SETS[args.set]
    champion_spec = Spec.parse(args.champion)
    models = [Spec.parse(m) for m in args.model]
    base = set_dir(args.root, wall_set, champion_spec)

    walls = {}
    for spec in models:
        walls[spec.ident] = walls_of(read_games(path.join(base, spec.ident)))
        if not walls[spec.ident]:
            raise SystemExit(f'{spec.label} ({spec.ident}) has no complete walls on '
                             f'{wall_set.name} vs {champion_spec.label}; play it first')
    common = sorted(set.intersection(*(set(w) for w in walls.values())))
    if args.limit_seeds:
        end = wall_set.first_seed + args.limit_seeds
        common = [s for s in common if s < end]
    if len(common) < 2:
        raise SystemExit('fewer than two walls in common; nothing to compare')
    stale = [spec.label for spec in models
             if any(n.endswith('.partial') for n in os.listdir(path.join(base, spec.ident)))]
    if stale and not args.no_style:
        print(f'  note: interrupted chunks under {", ".join(stale)} are counted in the style '
              f'rates until the next play clears them')

    report = {
        'set': wall_set.name, 'key': wall_set.key, 'champion': champion_spec.label,
        'champion_ident': champion_spec.ident, 'walls': len(common),
        'games_per_model': len(common) * 4, 'pts': PTS, 'models': {}, 'paired': {},
    }
    print(f'{wall_set.name} (key {wall_set.key:#x}) vs {champion_spec.label}: '
          f'{len(common):,} walls in common, {len(common) * 4:,} games each; '
          f'error over walls, 2.5 rank / 0 pt = level with the champion')
    if wall_set.name == 'legacy':
        print('  legacy walls: checkpoints were selected on these, so treat gains as optimistic')

    from libriichi.stat import Stat
    header = f'  {"model":<22} {"avg pt":>15} {"avg rank":>17} {"1st":>6} {"2nd":>6} {"3rd":>6} {"4th":>6}'
    print(header)
    for spec in models:
        w = walls[spec.ident]
        ranks = np.array([w[s] for s in common]).ravel()
        dist = [float((ranks == k).mean()) for k in (1, 2, 3, 4)]
        pt = summarize(w, common, 'pt', args.bootstrap)
        rank = summarize(w, common, 'rank')
        entry = {'ident': spec.ident, 'file': spec.file, 'part': spec.part or 'trained',
                 'avg_pt': pt, 'avg_rank': rank, 'rank_rates': dist}
        # Style, over every game this model has on the set: how it gets there.
        if not args.no_style:
            stat = Stat.from_dir(path.join(base, spec.ident), 'mortal', True)
            entry['style'] = {k: getattr(stat, k) for k in (
                'agari_rate', 'houjuu_rate', 'riichi_rate', 'fuuro_rate', 'ryukyoku_rate',
                'avg_point_per_agari', 'avg_point_per_houjuu')}
            entry['style']['games'] = stat.game
        report['models'][spec.label] = entry
        print(f'  {spec.label:<22} {pt["mean"]:+7.3f} ±{pt["se"]:5.3f} '
              f'{rank["mean"]:8.5f} ±{rank["se"]:6.4f} '
              + ' '.join(f'{x:6.3f}' for x in dist))

    if not args.no_style:
        print(f'  {"style (all games)":<22} {"agari":>7} {"houjuu":>7} {"riichi":>7} {"fuuro":>7} {"games":>8}')
        for spec in models:
            s = report['models'][spec.label]['style']
            print(f'  {spec.label:<22} {s["agari_rate"]:7.4f} {s["houjuu_rate"]:7.4f} '
                  f'{s["riichi_rate"]:7.4f} {s["fuuro_rate"]:7.4f} {s["games"]:8,}')

    ref = next((m for m in models if m.label == args.reference), models[0]) if args.reference else models[0]
    others = [m for m in models if m.ident != ref.ident]
    if others:
        print(f'paired against {ref.label}, over the same walls '
              f'(± se; [95% bootstrap interval]; games needed to detect 1 pt / 0.01 rank at 80% power):')
    for spec in others:
        wa, wb = walls[spec.ident], walls[ref.ident]
        res = {name: paired(wa, wb, common, name, args.bootstrap) for name in ('pt', 'rank', 'fourth')}
        res['games_for_1pt'] = games_to_detect(res['pt']['sd_per_wall'], 1.0)
        res['games_for_0.01rank'] = games_to_detect(res['rank']['sd_per_wall'], 0.01)
        report['paired'][f'{spec.label} - {ref.label}'] = res
        pt, rank, fourth = res['pt'], res['rank'], res['fourth']
        print(f'  {spec.label} - {ref.label}:')
        print(f'    pt   {pt["diff"]:+7.3f} ± {pt["se"]:.3f} [{pt["ci95"][0]:+.3f}, {pt["ci95"][1]:+.3f}]'
              f'   ({in_se(pt)} se)')
        print(f'    rank {rank["diff"]:+.5f} ± {rank["se"]:.5f} [{rank["ci95"][0]:+.5f}, {rank["ci95"][1]:+.5f}]'
              f'   ({in_se(rank)} se)')
        print(f'    4th  {fourth["diff"]:+.4f} ± {fourth["se"]:.4f}')
        print(f'    to detect: {res["games_for_1pt"]:,} games for 1 pt, '
              f'{res["games_for_0.01rank"]:,} for 0.01 rank')
    if len(others) > 1:
        print(f'  {len(others)} comparisons: one of them crossing 2 se by chance is not unusual')

    if args.json:
        os.makedirs(path.dirname(path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f'written to {args.json}')


def cmd_sets(_args):
    for s in WALL_SETS.values():
        print(f'{s.name:<8} key {s.key:#012x}  seeds [{s.first_seed}, {s.first_seed + s.seeds})  '
              f'{s.games:>7,} games  {s.note}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--set', choices=WALL_SETS, default='dev')
        p.add_argument('--model', action='append', required=True,
                       help='[label=]checkpoint[#ema]; repeat for more')
        p.add_argument('--champion', default='baseline=logs/baseline.pth',
                       help='[label=]checkpoint[#ema] for the three other seats')
        p.add_argument('--root', default='challenger/eval')
        p.add_argument('--limit-seeds', type=int, default=None,
                       help='only the first N seeds of the set')

    p = sub.add_parser('play', help='play what is missing')
    common(p)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--chunk-seeds', type=int, default=250,
                   help='seeds per resumable chunk (4 games each)')
    p.add_argument('--confirm-holdout', default='')
    p.add_argument('--quiet', action='store_true')
    p.set_defaults(func=cmd_play)

    p = sub.add_parser('report', help='compare what has been played')
    common(p)
    p.add_argument('--reference', default=None, help='label to compare against; the first model by default')
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--no-style', action='store_true')
    p.add_argument('--json', default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser('sets', help='list the wall sets')
    p.set_defaults(func=cmd_sets)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
