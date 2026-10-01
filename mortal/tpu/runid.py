"""What an online run is, kept in run.json beside its output, so that a session resuming it
trains the same run (#61).

An online run spans sessions, and each session brings its own config (a cell's CONFIG_SET)
and fetches its own opponent. Resumed with another gate metric, gate interval or learning
rate, the run would change algorithm or ruler half way and leave no trace of it: the gate
would count pt misses and rank misses towards one patience, and a gate cut short at the old
interval would no longer be owed. So the first session records the config it ran and which
init, opponent and baseline it had, and every later one is held to them:

- `resolve`, before the session starts (ops/kaggle/train_online_tpu.sh): the config to run
  is the run's own, with this session's values only for PLACEMENT -- how the work is spread
  over the box, not what is computed. Each setting of this session's that is not used is
  printed. A new run's config is this session's.
- `settle`, at the trainer's start (tpu.online): a new run's identity is written; a resumed
  run's is checked, and a setting outside PLACEMENT or an opponent other than its own stops
  the trainer, whoever started it. A new baseline is noted (the gate pairs each evaluation
  on one baseline). Each session is appended to `sessions`.

`accept` takes this session's settings and opponent as the run's from here on, and records
what changed in its session's entry.

    python -m tpu.runid --out OUT --config CONFIG --write RESOLVED [--accept-changes]
"""
import argparse
import datetime
import hashlib
import json
import os
from os import path

# How the work is spread, which a session may set for itself; everything else is the run's.
PLACEMENT = frozenset({
    'tpu_online.arenas', 'tpu_online.walls', 'tpu_online.rayon_threads', 'tpu_online.gate_arenas',
    'dataset.num_workers', 'dataset.prefetch_factor', 'online.server.capacity', 'online.history_window',
    'control.save_every', 'control.device',
})


def flatten(tree, prefix=''):
    """{'a.b': value} from nested dicts; lists are values."""
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out.update(flatten(v, f'{prefix}{k}.'))
        else:
            out[f'{prefix}{k}'] = v
    return out


def unflatten(flat):
    out = {}
    for key, v in flat.items():
        *sections, name = key.split('.')
        d = out
        for s in sections:
            d = d.setdefault(s, {})
        d[name] = v
    return out


def sha16(file):
    h = hashlib.sha256()
    with open(file, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()[:16]


def load(out):
    try:
        with open(path.join(out, 'run.json'), encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _save(out, run):
    tmp = path.join(out, 'run.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(run, f, indent=1)
    os.replace(tmp, path.join(out, 'run.json'))


def differences(own, now):
    """{key: (the run's, this session's)} outside PLACEMENT, for keys both have."""
    return {k: (own[k], now[k]) for k in own if k in now and k not in PLACEMENT and own[k] != now[k]}


def resolve(out, config, accept=False):
    """(the config to run this session, lines to print): see the module docstring."""
    run = load(out)
    if run is None:
        return config, ['a new run, or one from before run.json: its settings are this session\'s']
    own, now = flatten(run['config']), flatten(config)
    changed = differences(own, now)
    if accept:
        return config, [f'ACCEPT_CHANGES: {k} is {b!r} from here on (it was {a!r})' for k, (a, b) in changed.items()]
    merged = dict(own)
    lines = []
    for k, v in now.items():
        if k in PLACEMENT:
            merged[k] = v
        elif k not in own:
            merged[k] = v
            lines.append(f'{k} = {v!r}: not in the run\'s settings, taken as this session has it')
    lines += [f'{k}: the run\'s {a!r}, not this session\'s {b!r}' for k, (a, b) in changed.items()]
    return unflatten(merged), ['resuming with the run\'s own settings'] + lines


def settle(out, config, *, steps, init, opponent, baseline, accept=False, log=print):
    """At the trainer's start: run.json written for a new run, checked for a resumed one."""
    os.makedirs(out, exist_ok=True)
    now = flatten(config)
    files = {'init': init, 'opponent': opponent, 'baseline': baseline}
    ids = {k: {'file': path.basename(f), 'sha': sha16(f)} for k, f in files.items()}
    session = {'steps': int(steps), 'time': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               'placement': {k: now[k] for k in sorted(PLACEMENT) if k in now}, 'baseline': ids['baseline']['sha']}
    run = load(out)
    if run is None:
        run = {'config': config, **ids, 'sessions': []}
        if steps:
            log(f'run.json: none at step {steps:,}; this session\'s settings are recorded as the run\'s, '
                'and the earlier sessions\' are not known')
    else:
        own = flatten(run['config'])
        changed = differences(own, now)
        if run['opponent']['sha'] != ids['opponent']['sha']:
            changed['opponent'] = (run['opponent'], ids['opponent'])
        if changed and not accept:
            raise SystemExit(
                'this run was started with other settings than this session\'s, which would make it another '
                'run half way (#61):\n' + '\n'.join(f'  {k}: the run\'s {a!r}, now {b!r}' for k, (a, b) in changed.items())
                + '\nResume it with its own settings (ops/kaggle/train_online_tpu.sh does), start a new RUN_PATH, '
                'or take these as the run\'s from here on with --accept-changes (ACCEPT_CHANGES=1).')
        if changed:
            session['changed'] = {k: [a, b] for k, (a, b) in changed.items()}
            log('run.json: from here on ' + ', '.join(f'{k} {b!r} (was {a!r})' for k, (a, b) in changed.items()))
        added = {k: v for k, v in now.items() if k not in own and k not in PLACEMENT}
        if added:
            session['added'] = added
            log('run.json: settings the run did not have, as this session has them: '
                + ', '.join(f'{k} = {v!r}' for k, v in added.items()))
        if changed or added:
            run['config'] = unflatten({**own, **{k: v for k, v in now.items() if k not in PLACEMENT}})
            run['opponent'] = ids['opponent']
        if run['baseline']['sha'] != ids['baseline']['sha']:
            log(f'run.json: the baseline is now {ids["baseline"]["sha"]} (was {run["baseline"]["sha"]}); '
                'each gate pairs its two sides on one')
            run['baseline'] = ids['baseline']
    run['sessions'].append(session)
    _save(out, run)
    return run


def main():
    import toml
    ap = argparse.ArgumentParser(description='The config a session of an online run trains with: '
                                             'its own, from run.json, but for PLACEMENT.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', required=True, help='this session\'s config')
    ap.add_argument('--write', required=True, help='where to write the config to run')
    ap.add_argument('--accept-changes', action='store_true', help='this session\'s settings, from here on')
    args = ap.parse_args()
    config, lines = resolve(args.out, toml.load(args.config), args.accept_changes)
    for line in lines:
        print(f'  {line}')
    with open(args.write, 'w', encoding='utf-8') as f:
        toml.dump(config, f)


if __name__ == '__main__':
    main()
