"""Everything about the online switch that can be checked without starting it.

Run on the box while the offline run is going: it reads files and imports
modules, and touches no GPU.
"""
import importlib
import os
import socket
import sys
import tomllib
from pathlib import Path

os.chdir('/root/Mortal/mortal')
sys.path.insert(0, '/root/Mortal/mortal')

ok = fail = 0


def check(what, good, detail=''):
    global ok, fail
    if good:
        ok += 1
        print(f'  ok   {what:<46} {detail}')
    else:
        fail += 1
        print(f'  FAIL {what:<46} {detail}')


cfg = tomllib.load(open('config.online.toml', 'rb'))
off = tomllib.load(open('config.vast.toml', 'rb'))

print('the config:')
check('online is on', cfg['control']['online'] is True)
check('version', cfg['control']['version'] == 4, cfg['control']['version'])
check('same net as offline',
      cfg['resnet'] == off['resnet'], str(cfg['resnet']))
check('the trainer has one gpu, workers the other',
      cfg['control']['device'] == 'cuda:0'
      and cfg['baseline']['train']['device'] == 'cuda:1')
check('test play is the same ruler as offline',
      cfg['test_play']['games'] == off['test_play']['games']
      and cfg['baseline']['test']['state_file'] == off['baseline']['test']['state_file'],
      f"{cfg['test_play']['games']} games vs {cfg['baseline']['test']['state_file']}")
check('the schedule is flat at the floor',
      cfg['optim']['scheduler']['peak'] == cfg['optim']['scheduler']['final'],
      f"{cfg['optim']['scheduler']['peak']:g}")
check('batchnorm is frozen', cfg['freeze_bn']['mortal'] is True)
check('no corpus globs', cfg['dataset']['parquet_globs'] == []
      and cfg['dataset']['globs'] == [])
check('a train_play profile exists', 'default' in cfg['train_play'],
      f"games={cfg['train_play']['default']['games']}")

print('the files:')
seed = Path('logs/v4/best_ema.pth')
check('the seed is there', seed.exists(), str(seed))
check('the v3 ruler is there', Path(cfg['baseline']['test']['state_file']).exists())
check('the grp is there', Path(cfg['grp']['state_file']).exists())
check('the online run directory is free',
      not Path(cfg['control']['state_file']).exists(),
      'a leftover logs/v4o/mortal.pth would be resumed instead of the seed')

import torch  # noqa: E402  (after the cheap checks, it is slow to import)

state = torch.load(seed, weights_only=True, map_location='cpu')
print(f"the seed: step {state['steps']:,}, best {state['best_perf']}")
check('it is an offline checkpoint',
      state['config']['control']['online'] is False,
      'so train.py will build a fresh optimizer and schedule for online')
check('its net matches the online config',
      state['config']['resnet'] == cfg['resnet'])
check('it carries aux_net', 'aux_net' in state)
check('it carries an ema to continue from', 'ema' in state)
same = all(torch.equal(state['mortal'][k], state['ema']['mortal'][k])
           for k in state['mortal'])
check('its weights are the averaged ones', same,
      'best_ema.pth saves the average under the ordinary names')

print('the code:')
for mod in ('server', 'client', 'train', 'player', 'engine', 'model', 'common'):
    try:
        importlib.import_module(mod)
        check(f'{mod}.py imports', True)
    except Exception as exc:
        check(f'{mod}.py imports', False, repr(exc))

print('the box:')
port = cfg['online']['remote']['port']
with socket.socket() as s:
    check(f'port {port} is free', s.connect_ex(('127.0.0.1', port)) != 0)
free = os.statvfs('/root')
free_gb = free.f_bavail * free.f_frsize / 2**30
check('disk', free_gb > 6, f'{free_gb:.0f}G free')
check('both gpus visible', torch.cuda.device_count() == 2,
      f'{torch.cuda.device_count()} cuda devices')

print(f'\n{ok} passed, {fail} failed')
sys.exit(1 if fail else 0)
