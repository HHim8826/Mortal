"""Improve the distilled policy with a clipped policy gradient.

    python train_ppo.py --from logs/policy/policy-t0.05.pth --out logs/ppo/kl
    python train_ppo.py --from ... --out logs/ppo/kl-refresh --ref-refresh 10
    python train_ppo.py --from ... --out logs/ppo/plain      --kl-coef 0

The v4 online run regressed only the Q of the action that was played, left
every other action to drift through the shared trunk, and explored so little
-- a non-argmax draw once in two thousand decisions -- that there was almost
no signal to learn from. It got worse for 600,000 steps and nothing in the
objective could have stopped it. This trains the policy instead:

    the ratio     pi(a|s) / mu(a|s), where mu is the policy that actually
                  played the game. The server stamps every replay with the
                  version that produced it and the trainer keeps those weights,
                  so the denominator is the real one rather than "the current
                  model", which is hundreds of updates newer by then.
    the clip      an update that would move a decision's probability by more
                  than `--clip` stops paying. This is the brake the MC recipe
                  had no way to express.
    the baseline  the critic's expected placement utility, subtracted from what
                  the hanchan actually paid, so the gradient sees what a
                  decision changed rather than how the game happened to end.
    the anchor    an exact KL to a reference policy, over the legal actions
                  rather than sampled. `--ref-refresh` decides whether it stays
                  the distilled start (a fixed anchor) or follows the policy
                  every K rounds (a trust region that moves).

The trunk is frozen by default. That is not only cheap -- one forward serves
the policy, the critic and the behaviour policy -- it is the point: with the
representation fixed, whatever this changes is the decision rule, and the
drift that the shared trunk spread through every action in the old run cannot
happen. `--train-trunk` lifts it once the method itself is shown to work.
"""
import argparse
import logging
import os
import time
from collections import OrderedDict
from copy import deepcopy
from os import path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import prelude                                          # noqa: F401
from common import drain, submit_param
from config import config
from dataloader import FileDatasetsIter, worker_init_fn
from model import Brain, PolicyHead, RankCritic


def load_start(file, device):
    """The distilled checkpoint this run departs from, and everything in it."""
    state = torch.load(file, weights_only=True, map_location='cpu')
    for key in ('policy', 'critic', 'mortal'):
        if key not in state:
            raise SystemExit(f'{file} has no {key}; train one with train_policy.py')
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    if version != 4:
        raise SystemExit(f'{file} is a v{version} model; this wants v4')
    if not state.get('play_temperature'):
        logging.warning(
            f'{file} has no play temperature folded in. Its softmax is the teacher own '
            'scale, which samples 65% non-argmax moves and plays at 88% fourths -- run '
            'sharpen_policy.py first unless this is deliberate.')
    brain = Brain(version=4, conv_channels=cfg['resnet']['conv_channels'],
                  num_blocks=cfg['resnet']['num_blocks']).eval()
    brain.load_state_dict(state['mortal'])
    policy = PolicyHead(version=4)
    policy.load_state_dict(state['policy'])
    critic = RankCritic(pts=tuple(state.get('pts', config['env']['pts'])))
    critic.load_state_dict(state['critic'])
    return brain.to(device), policy.to(device), critic.to(device), state, cfg


class Behaviour:
    """The policies that played the games, kept so their probabilities can be recomputed.

    A frozen trunk makes this nearly free: the features the current policy
    reads are the ones the old one read, so a published version is 46x1024
    numbers and pricing a batch under it is one matrix multiply. With
    `--train-trunk` the whole network has to come back, which is why the
    snapshots are written to disk as well and loaded on demand.
    """

    def __init__(self, out_dir, frozen_trunk, keep, device):
        self.dir = path.join(out_dir, 'params')
        os.makedirs(self.dir, exist_ok=True)
        self.frozen_trunk = frozen_trunk
        self.keep = keep
        self.device = device
        self.heads = OrderedDict()
        self.nets = OrderedDict()

    def add(self, version, brain, policy):
        self.heads[version] = deepcopy(policy).eval().requires_grad_(False)
        torch.save({'policy': policy.state_dict(), 'mortal': brain.state_dict()},
                   path.join(self.dir, f'v{version}.pth'))
        while len(self.heads) > self.keep:
            old, _ = self.heads.popitem(last=False)
            self.nets.pop(old, None)
            stale = path.join(self.dir, f'v{old}.pth')
            if path.exists(stale):
                os.remove(stale)

    def known(self, version):
        return version in self.heads

    def logits(self, version, phi, obs, masks):
        """The logits the behaviour policy gave these decisions."""
        if self.frozen_trunk:
            return self.heads[version](phi, masks)
        net = self.nets.get(version)
        if net is None:
            state = torch.load(path.join(self.dir, f'v{version}.pth'),
                               weights_only=True, map_location='cpu')
            res = config['resnet']
            brain = Brain(version=4, conv_channels=res['conv_channels'],
                          num_blocks=res['num_blocks']).eval()
            brain.load_state_dict(state['mortal'])
            head = PolicyHead(version=4).eval()
            head.load_state_dict(state['policy'])
            net = self.nets[version] = (brain.to(self.device).requires_grad_(False),
                                        head.to(self.device).requires_grad_(False))
            while len(self.nets) > 2:
                self.nets.popitem(last=False)
        brain, head = net
        return head(brain(obs), masks)


def log_prob_of(logits, actions):
    return logits.log_softmax(-1).gather(-1, actions[:, None]).squeeze(-1)


def entropy_and_kl(logits, ref_logits):
    """The policy's entropy, and its exact KL to the reference over legal actions.

    Exact rather than the usual one-sample estimate: there are at most 46
    actions and the mask has already made the illegal ones -inf, so the sum is
    cheap and carries none of the variance a sampled KL would add to the very
    term that is meant to keep the update small.
    """
    logp = logits.log_softmax(-1)
    p = logp.exp()
    finite = torch.isfinite(logp)
    entropy = -torch.where(finite, p * logp, torch.zeros_like(logp)).sum(-1)
    if ref_logits is None:
        return entropy, torch.zeros_like(entropy)
    ref_logp = ref_logits.log_softmax(-1)
    kl = torch.where(finite, p * (logp - ref_logp), torch.zeros_like(logp)).sum(-1)
    return entropy, kl


class Meter:
    def __init__(self):
        self.sums = {}
        self.n = 0

    def add(self, **kw):
        for key, val in kw.items():
            # Several of these are still attached to the graph; reading one as
            # a number is a measurement, not a use of it.
            self.sums[key] = self.sums.get(key, 0.) + float(val.detach())
        self.n += 1

    def take(self):
        means = {k: v / max(self.n, 1) for k, v in self.sums.items()}
        self.sums.clear()
        self.n = 0
        return means


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start', required=True,
                    help='the sharpened policy checkpoint this run departs from')
    ap.add_argument('--out', default='logs/ppo/run')
    ap.add_argument('--rounds', type=int, default=0, help='0 runs until stopped')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--trunk-lr', type=float, default=1e-6, help='only with --train-trunk')
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--kl-coef', type=float, default=0.1, help='0 turns the anchor off')
    ap.add_argument('--ref-refresh', type=int, default=0,
                    help='rounds between reference updates; 0 keeps the start policy')
    ap.add_argument('--ent-coef', type=float, default=1e-3)
    ap.add_argument('--v-coef', type=float, default=0.5)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--train-trunk', action='store_true')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--file-batch-size', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--save-every-rounds', type=int, default=5)
    return ap.parse_args(argv)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device or config['control']['device'])
    brain, policy, critic, state, cfg = load_start(args.start, device)
    frozen_trunk = not args.train_trunk
    brain.requires_grad_(not frozen_trunk)
    logging.info(f'start {args.start}: distilled at step {state.get("steps")}, '
                 f'play temperature {state.get("play_temperature")}, '
                 f'trunk {"frozen" if frozen_trunk else "training"}')

    reference = deepcopy(policy).eval().requires_grad_(False) if args.kl_coef > 0 else None
    groups = [{'params': list(policy.parameters()) + list(critic.parameters()), 'lr': args.lr}]
    if not frozen_trunk:
        groups.append({'params': list(brain.parameters()), 'lr': args.trunk_lr})
    optimizer = optim.AdamW(groups, weight_decay=0.)
    scaler = torch.amp.GradScaler(device.type)
    cross_entropy = nn.CrossEntropyLoss()
    trained = [p for g in groups for p in g['params']]

    behaviour = Behaviour(args.out, frozen_trunk,
                          config['online'].get('param_history', 8), device)
    # The workers play the policy head, so it goes in the slot the Q head
    # usually takes; client.py builds whichever the config names.
    version = submit_param(brain, policy, is_idle=True)
    behaviour.add(version, brain, policy)
    logging.info(f'published v{version}, waiting for games')

    workers = args.workers if args.workers is not None else config['dataset']['num_workers']
    file_batch_size = (args.file_batch_size if args.file_batch_size is not None
                       else config['dataset'].get('file_batch_size', 20))

    def save(round_no, steps):
        out = path.join(args.out, 'policy.pth')
        torch.save({
            'policy': policy.state_dict(),
            'critic': critic.state_dict(),
            'mortal': brain.state_dict(),
            'current_dqn': state['current_dqn'],
            'config': cfg,
            'pts': critic.pts.tolist(),
            'play_temperature': state.get('play_temperature'),
            'steps': steps,
            'rounds': round_no,
            'started_from': path.abspath(args.start),
            'args': vars(args),
        }, out)
        return out

    steps = 0
    meter = Meter()
    started = time.time()
    round_no = 0
    while not args.rounds or round_no < args.rounds:
        round_no += 1
        dirname = drain()
        file_list = [path.join(dirname, p) for p in sorted(os.listdir(dirname))
                     if p.endswith('.json.gz') and path.isfile(path.join(dirname, p))]
        logging.info(f'round {round_no}: {len(file_list):,} games from {dirname}')

        data = FileDatasetsIter(
            version = 4,
            file_list = file_list,
            pts = critic.pts.tolist(),
            player_names = ['trainee'],
            file_batch_size = file_batch_size,
            num_epochs = 1,
            final_rank = True,
            param_version = True,
            # The advantage is the hanchan's own result against the critic's
            # estimate of it, so the per-kyoku GRP return -- and the GRP
            # forward each game would cost -- is not needed.
            skip_rewards = True,
            # Known gap: the rule-based agari guard can replace what the policy
            # sampled, and those decisions are then priced at an action the
            # behaviour policy did not draw. Both sides use the same action, so
            # the ratio stays consistent and the check above still holds; what
            # breaks is the importance-sampling identity, for the handful of
            # decisions a hanchan where a win was on offer.
        )
        loader_kwargs = {}
        if workers > 0:
            loader_kwargs['prefetch_factor'] = config['dataset'].get('prefetch_factor', 2)
            loader_kwargs['in_order'] = config['dataset'].get('in_order', True)
        batches = DataLoader(
            dataset = data, batch_size = args.batch_size, drop_last = True,
            num_workers = workers, pin_memory = True,
            worker_init_fn = worker_init_fn, **loader_kwargs,
        )

        # Nothing has moved since the last publish, so every decision stamped
        # with it must price at a ratio of exactly one. If the versions, the
        # snapshots or the sampling rule were mismatched, this is where it
        # shows -- loudly, on the first batch, rather than as a policy that
        # quietly learns from the wrong denominator.
        unchecked = True
        dropped = kept = 0
        for batch in batches:
            (obs, actions, masks, _steps_to_done, _kyoku_rewards,
             _player_ranks, final_rank, versions) = batch
            obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
            actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
            masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
            final_rank = final_rank.to(dtype=torch.int64, device=device, non_blocking=True)
            versions = versions.to(dtype=torch.int64)

            usable = torch.tensor([behaviour.known(int(v)) for v in versions])
            dropped += int((~usable).sum())
            kept += int(usable.sum())
            if not bool(usable.any()):
                continue
            keep = usable.to(device)

            with torch.autocast(device.type):
                phi = brain(obs)
                if frozen_trunk:
                    phi = phi.detach()
                logits = policy(phi, masks)
                rank_logits = critic(phi)

                with torch.no_grad():
                    mu_logp = torch.zeros_like(actions, dtype=torch.float32)
                    for v in versions[usable].unique().tolist():
                        rows = (versions == v).to(device) & keep
                        mu_logits = behaviour.logits(v, phi[rows], obs[rows], masks[rows])
                        mu_logp[rows] = log_prob_of(mu_logits.float(), actions[rows])
                    ref_logits = reference(phi, masks) if reference is not None else None

                logp = log_prob_of(logits.float(), actions)
                ratio = (logp - mu_logp).exp()
                value = critic.value(rank_logits.float())
                paid = critic.pts[final_rank]
                advantage = paid - value.detach()
                advantage = (advantage - advantage[keep].mean()) / (advantage[keep].std() + 1e-8)

                clipped = ratio.clamp(1 - args.clip, 1 + args.clip)
                policy_loss = -torch.min(ratio * advantage, clipped * advantage)
                entropy, kl = entropy_and_kl(logits.float(), ref_logits)
                per_decision = policy_loss - args.ent_coef * entropy + args.kl_coef * kl
                critic_loss = cross_entropy(rank_logits[keep], final_rank[keep])
                loss = per_decision[keep].mean() + args.v_coef * critic_loss

            if unchecked:
                unchecked = False
                current = (versions == version).to(device) & keep
                if bool(current.any()):
                    off = (ratio[current] - 1).abs().max().detach()
                    logging.info(f'{int(current.sum())} decisions from v{version} price at '
                                 f'ratio 1 within {float(off):.2e}')
                    if off > 1e-2:
                        raise SystemExit(
                            f'v{version} was published from these weights, yet its own '
                            f'decisions price up to {float(off):.4f} away from a ratio of 1. '
                            'The behaviour policy is not the one that played: check the '
                            'version stamping and what the workers sample.')
                else:
                    logging.warning(f'no decisions from the current v{version} in the first '
                                    'batch; the ratio check did not run')

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trained, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            steps += 1

            with torch.inference_mode():
                meter.add(
                    policy_loss = policy_loss[keep].mean(),
                    critic_loss = critic_loss,
                    ratio = ratio[keep].mean(),
                    clipped = ((ratio[keep] - 1).abs() > args.clip).float().mean(),
                    approx_kl = (mu_logp - logp)[keep].mean(),
                    ref_kl = kl[keep].mean(),
                    entropy = entropy[keep].mean(),
                    value = value[keep].mean(),
                    paid = paid[keep].float().mean(),
                )
            if steps % args.log_every == 0:
                m = meter.take()
                logging.info(
                    f'round {round_no} step {steps:,} ({steps / (time.time() - started):.1f}/s) '
                    f'policy {m["policy_loss"]:+.4f} critic {m["critic_loss"]:.4f} | '
                    f'ratio {m["ratio"]:.4f}, clipped {m["clipped"]:.1%}, '
                    f'kl to mu {m["approx_kl"]:+.5f}, to ref {m["ref_kl"]:.5f} | '
                    f'entropy {m["entropy"]:.3f}, value {m["value"]:+.3f} '
                    f'vs paid {m["paid"]:+.3f}')

        del batches
        if data.iterator is not None:
            data.iterator.close()
        if dropped:
            logging.warning(f'round {round_no}: dropped {dropped:,} of {dropped + kept:,} '
                            'decisions whose parameters are no longer held')

        if args.ref_refresh and reference is not None and round_no % args.ref_refresh == 0:
            reference = deepcopy(policy).eval().requires_grad_(False)
            logging.info(f'reference refreshed to the policy after round {round_no}')

        version = submit_param(brain, policy, is_idle=False)
        behaviour.add(version, brain, policy)
        logging.info(f'round {round_no} done, published v{version} after {steps:,} steps')
        if round_no % args.save_every_rounds == 0:
            logging.info(f'saved {save(round_no, steps)}')
    save(round_no, steps)


if __name__ == '__main__':
    main()
