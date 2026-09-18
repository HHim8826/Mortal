"""Fold a play temperature into the policy head, so that what is trained is what is played.

    python sharpen_policy.py --from logs/policy/policy.pth --temperature 0.05

The head out of `train_policy.py` carries the teacher's own scale, and that
scale is not a policy: sampling it deviates from its own best action on two
decisions in three and takes 88% fourth places. Measured on 1,000 dev walls,
the usable range is far colder -- 0.05 deviates 9.1% of the time, about six
decisions a game, for no cost the evaluation can resolve.

A temperature could be passed to the sampler instead, but then the
distribution being optimized is not the one being sampled, and every
importance ratio in a policy gradient would be taken against the wrong
denominator. Multiplying the last linear map by 1/T moves the temperature
into the weights: softmax(W'phi + b') is exactly softmax((W phi + b) / T), the
argmax is untouched, and from then on the policy is a plain softmax of its own
logits -- which is what the ratio, the entropy term and the KL all assume.
"""
import argparse
from os import path

import torch

from model import PolicyHead


def sharpen(state, temperature):
    """The policy head's parameters, scaled so its softmax is the tempered one."""
    scaled = {k: v / temperature for k, v in state.items()}
    return scaled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='src', required=True, help='a train_policy.py checkpoint')
    ap.add_argument('--temperature', type=float, required=True,
                    help='the play temperature to fold in; below 1 sharpens')
    ap.add_argument('--out', default=None, help='default: <from>-t<temperature>.pth')
    args = ap.parse_args()
    if args.temperature <= 0:
        raise SystemExit('a temperature must be positive')

    state = torch.load(args.src, weights_only=True, map_location='cpu')
    if 'policy' not in state:
        raise SystemExit(f'{args.src} has no policy head')
    if state.get('play_temperature'):
        raise SystemExit(f'{args.src} already has {state["play_temperature"]} folded in')

    sharp = sharpen(state['policy'], args.temperature)

    # The claim is exact, so check it rather than trusting it: on random
    # features the two distributions must agree to float precision, and the
    # argmax must be the same tile.
    before, after = PolicyHead(version=4), PolicyHead(version=4)
    before.load_state_dict(state['policy'])
    after.load_state_dict(sharp)
    phi = torch.randn(256, 1024)
    mask = torch.rand(256, 46) < 0.3
    mask[:, 0] = True
    with torch.inference_mode():
        old = before(phi, mask)
        new = after(phi, mask)
        gap = (new.softmax(-1) - (old / args.temperature).softmax(-1)).abs().max()
        moved = (new.argmax(-1) != old.argmax(-1)).sum()
    if gap > 1e-5 or moved:
        raise SystemExit(f'sharpening changed the policy: max prob gap {gap:g}, {moved} argmax moved')

    out = args.out or f'{path.splitext(args.src)[0]}-t{args.temperature:g}.pth'
    torch.save({**state, 'policy': sharp, 'play_temperature': args.temperature,
                'sharpened_from': path.abspath(args.src)}, out)
    print(f'{args.src} -> {out}')
    print(f'temperature {args.temperature:g} folded in; same argmax, '
          f'max probability gap {gap:.2e}')
    print('play it with epsilon 1 and temperature 1: the head is the policy now')


if __name__ == '__main__':
    main()
