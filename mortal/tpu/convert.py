"""Checkpoints between PyTorch (`model.py`) and Flax (`tpu.model`), both ways.

The two sides never share a process: the TPU host has JAX and a CPU-only torch,
and the GPU boxes that evaluate have torch and no JAX. So the interchange is an
.npz of plain arrays under their PyTorch names, `mortal/...`, `dqn/...` and
`aux/...`, with the shape of the net in `meta/...`, and the mapping itself is
numpy on both ends.

    python -m tpu.convert export logs/best_ema.pth best_ema.npz          # torch side
    python -m tpu.convert import trained.npz logs/best_ema.pth out.pth   # torch side

`export` takes the weights `train.py` plays with: a best_ema.pth holds its average
at the top level already, and `--ema` takes the `ema` entry of a plain
checkpoint instead of its raw weights.
"""
import argparse

import numpy as np

# PyTorch Conv1d weights are (out, in, k), Flax's (k, in, out); Linear is (out, in)
# against Dense's (in, out).
conv_in, conv_out = (lambda w: w.transpose(2, 1, 0)), (lambda w: w.transpose(2, 1, 0))
dense_in, dense_out = (lambda w: w.T), (lambda w: w.T)

BN = (('weight', 'scale'), ('bias', 'bias'))
BN_STATS = (('running_mean', 'mean'), ('running_var', 'var'))


def layout(num_blocks):
    """Where each part sits in `ResNet.net`: the stem, the blocks, then the tail."""
    return {'stem': 0, 'bn_final': num_blocks + 1, 'conv_out': num_blocks + 3, 'fc': num_blocks + 6}


def block_parts(i):
    """PyTorch prefix of each Flax submodule of residual block i (0-based)."""
    p = f'encoder.net.{i + 1}.'
    return {'bn1': p + 'res_unit.0.', 'conv1': p + 'res_unit.2.', 'bn2': p + 'res_unit.3.',
            'conv2': p + 'res_unit.5.', 'fc1': p + 'ca.shared_mlp.0.', 'fc2': p + 'ca.shared_mlp.2.'}


def from_torch(mortal, dqn, aux, num_blocks):
    """Three PyTorch state dicts, as numpy arrays, to Flax {'params', 'batch_stats'}."""
    at = layout(num_blocks)
    net = lambda k: f'encoder.net.{k}.'
    params = {'stem': {'kernel': conv_in(mortal[net(at['stem']) + 'weight'])},
              'bn_final': {f: mortal[net(at['bn_final']) + t] for t, f in BN},
              'conv_out': {'kernel': conv_in(mortal[net(at['conv_out']) + 'weight']),
                           'bias': mortal[net(at['conv_out']) + 'bias']},
              'fc': {'kernel': dense_in(mortal[net(at['fc']) + 'weight']),
                     'bias': mortal[net(at['fc']) + 'bias']}}
    stats = {'bn_final': {f: mortal[net(at['bn_final']) + t] for t, f in BN_STATS}}

    per_block = [block_parts(i) for i in range(num_blocks)]
    stack = lambda key, fn=lambda w: w: np.stack([fn(mortal[b[key[0]] + key[1]]) for b in per_block])
    params['blocks'] = {
        'bn1': {f: stack(('bn1', t)) for t, f in BN},
        'conv1': {'kernel': stack(('conv1', 'weight'), conv_in)},
        'bn2': {f: stack(('bn2', t)) for t, f in BN},
        'conv2': {'kernel': stack(('conv2', 'weight'), conv_in)},
        'ca': {'fc1': {'kernel': stack(('fc1', 'weight'), dense_in), 'bias': stack(('fc1', 'bias'))},
               'fc2': {'kernel': stack(('fc2', 'weight'), dense_in), 'bias': stack(('fc2', 'bias'))}},
    }
    stats['blocks'] = {'bn1': {f: stack(('bn1', t)) for t, f in BN_STATS},
                       'bn2': {f: stack(('bn2', t)) for t, f in BN_STATS}}
    return {'params': {'brain': params,
                       'dqn': {'net': {'kernel': dense_in(dqn['net.weight']), 'bias': dqn['net.bias']}},
                       'aux': {'kernel': dense_in(aux['net.weight'])}},
            'batch_stats': {'brain': stats}}


def to_torch(variables, num_blocks):
    """Flax {'params', 'batch_stats'} back to the three PyTorch state dicts, as numpy."""
    p, s = variables['params']['brain'], variables['batch_stats']['brain']
    at = layout(num_blocks)
    net = lambda k: f'encoder.net.{k}.'
    a = lambda x: np.asarray(x, dtype=np.float32)
    mortal = {net(at['stem']) + 'weight': conv_out(a(p['stem']['kernel'])),
              net(at['conv_out']) + 'weight': conv_out(a(p['conv_out']['kernel'])),
              net(at['conv_out']) + 'bias': a(p['conv_out']['bias']),
              net(at['fc']) + 'weight': dense_out(a(p['fc']['kernel'])),
              net(at['fc']) + 'bias': a(p['fc']['bias'])}
    for t, f in BN:
        mortal[net(at['bn_final']) + t] = a(p['bn_final'][f])
    for t, f in BN_STATS:
        mortal[net(at['bn_final']) + t] = a(s['bn_final'][f])
    mortal[net(at['bn_final']) + 'num_batches_tracked'] = np.array(0, dtype=np.int64)

    b, bs = p['blocks'], s['blocks']
    for i, parts in enumerate(block_parts(k) for k in range(num_blocks)):
        for bn in ('bn1', 'bn2'):
            for t, f in BN:
                mortal[parts[bn] + t] = a(b[bn][f][i])
            for t, f in BN_STATS:
                mortal[parts[bn] + t] = a(bs[bn][f][i])
            mortal[parts[bn] + 'num_batches_tracked'] = np.array(0, dtype=np.int64)
        for c in ('conv1', 'conv2'):
            mortal[parts[c] + 'weight'] = conv_out(a(b[c]['kernel'][i]))
        for fc in ('fc1', 'fc2'):
            mortal[parts[fc] + 'weight'] = dense_out(a(b['ca'][fc]['kernel'][i]))
            mortal[parts[fc] + 'bias'] = a(b['ca'][fc]['bias'][i])

    q = variables['params']['dqn']['net']
    dqn = {'net.weight': dense_out(a(q['kernel'])), 'net.bias': a(q['bias'])}
    aux = {'net.weight': dense_out(a(variables['params']['aux']['kernel']))}
    return mortal, dqn, aux


def load_npz(path):
    """An exported .npz as (variables, meta)."""
    z = np.load(path)
    part = lambda prefix: {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}
    meta = {k: int(v) for k, v in part('meta/').items()}
    return from_torch(part('mortal/'), part('dqn/'), part('aux/'), meta['num_blocks']), meta


def save_npz(path, variables, meta):
    mortal, dqn, aux = to_torch(variables, meta['num_blocks'])
    out = {f'meta/{k}': np.array(v) for k, v in meta.items()}
    for prefix, sd in (('mortal/', mortal), ('dqn/', dqn), ('aux/', aux)):
        out.update({prefix + k: v for k, v in sd.items()})
    np.savez(path, **out)


def cmd_export(args):
    import torch
    state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    if version != 4:
        raise SystemExit(f'{args.checkpoint} is version {version}; only v4 is ported')
    weights = state['ema'] if args.ema else state
    out = {'meta/conv_channels': np.array(cfg['resnet']['conv_channels']),
           'meta/num_blocks': np.array(cfg['resnet']['num_blocks']),
           'meta/steps': np.array(state.get('steps', 0))}
    for prefix, key, sd in (('mortal/', 'mortal', weights['mortal']),
                            ('dqn/', 'current_dqn', weights['current_dqn']),
                            ('aux/', 'aux_net', state['aux_net'])):
        out.update({prefix + k: v.float().numpy() if v.is_floating_point() else v.numpy()
                    for k, v in sd.items()})
    np.savez(args.out, **out)
    print(f'{args.checkpoint} ({"ema" if args.ema else "as saved"}, step {out["meta/steps"]}) -> {args.out}')


def cmd_import(args):
    """Trained Flax weights into a PyTorch checkpoint the GPU tools can play.

    The config and everything else come from `base`, except the net's shape, which
    comes from the .npz: a deepened net has more blocks than the checkpoint it grew
    from. The weights are loaded strictly into a model of that shape before saving,
    so a checkpoint that would not load is never written.
    """
    import torch
    from model import AuxNet, Brain, DQN
    z = np.load(args.npz)
    state = torch.load(args.base, weights_only=True, map_location='cpu')
    shape = {'conv_channels': int(z['meta/conv_channels']), 'num_blocks': int(z['meta/num_blocks'])}
    to_t = lambda prefix: {k[len(prefix):]: torch.from_numpy(np.asarray(z[k])) for k in z.files if k.startswith(prefix)}
    modules = {'mortal': Brain(version=4, **shape), 'current_dqn': DQN(version=4), 'aux_net': AuxNet((4,))}
    for key, prefix in (('mortal', 'mortal/'), ('current_dqn', 'dqn/'), ('aux_net', 'aux/')):
        modules[key].load_state_dict(to_t(prefix))            # strict: every key, every shape
        state[key] = to_t(prefix)
    state['config']['resnet'].update(shape)
    state.pop('ema', None)
    state['steps'] = int(z['meta/steps']) if 'meta/steps' in z.files else state.get('steps', 0)
    torch.save(state, args.out)
    print(f'{args.npz} ({shape["conv_channels"]}x{shape["num_blocks"]}) -> {args.out}, '
          f'config and the rest from {args.base}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    e = sub.add_parser('export')
    e.add_argument('checkpoint')
    e.add_argument('out')
    e.add_argument('--ema', action='store_true')
    e.set_defaults(fn=cmd_export)
    i = sub.add_parser('import')
    i.add_argument('npz')
    i.add_argument('base')
    i.add_argument('out')
    i.set_defaults(fn=cmd_import)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
