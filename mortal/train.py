def train():
    import prelude

    import logging
    import sys
    import os
    import gc
    import copy
    import gzip
    import json
    import shutil
    import random
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from distributed import Dist, NullWriter
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet
    from libriichi.consts import obs_shape
    from config import config

    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    min_q_weight = config['cql']['min_q_weight']
    next_rank_weight = config['aux']['next_rank_weight']
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    # One process per GPU under torchrun; otherwise the single-GPU path.
    ddp = Dist()
    device = ddp.setup(torch.device(config['control']['device']),
                       loaders_per_rank=config['dataset']['num_workers'])
    if ddp.enabled and online:
        raise RuntimeError('DDP is for offline training: online mode drains one '
                           'shared replay buffer and has no notion of shards')
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    gamma = config['env']['gamma']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    all_models = (mortal, dqn, aux_net)

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])

    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    # Fused keeps the step free of host syncs: with GradScaler it skips an
    # overflowed step on the device instead of reading found_inf back, and it
    # has no per-parameter `step.item()` for the bias correction, which the
    # foreach version pays hundreds of times a step once a resumed state has
    # put the step counts on the GPU. In DDP any such wait on one rank leaves
    # the other idling in the gradient all-reduce.
    fused = device.type == 'cuda'
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps, fused=fused)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer(
        device = device if ddp.enabled else None,
        rank = ddp.rank,
        world_size = ddp.world_size,
    )
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])
            if fused:
                # load_state_dict takes every group setting from the checkpoint,
                # `fused` included, so one saved by the foreach version would
                # quietly switch it back off; and fused wants the step counts
                # as float32 on the parameters' device.
                for group in optimizer.param_groups:
                    group['fused'] = True
                    group['foreach'] = None
                for param_state in optimizer.state.values():
                    if 'step' in param_state:
                        param_state['step'] = param_state['step'].to(device=device, dtype=torch.float32)
            scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        steps = state['steps']

    class TrainNet(nn.Module):
        """The three trained modules as one, so DDP syncs them with one reducer.

        Checkpoints still save each module on its own, so nothing that loads
        them needs to know this exists.
        """
        def __init__(self):
            super().__init__()
            self.mortal = mortal
            self.dqn = dqn
            self.aux_net = aux_net

        def forward(self, obs, masks):
            phi = self.mortal(obs)
            return self.dqn(phi, masks), self.aux_net(phi)

    # Wrapped after loading, so every rank starts from the same weights.
    net = ddp.wrap(TrainNet(), device)

    # A running average of the weights, test-played beside the trained ones on
    # the same walls. It is kept and compared, and saved as its own best, but
    # only becomes the model to use once test play shows it is the stronger.
    #
    # After the wrap, never before it: starting from scratch the ranks hold
    # different random weights until DDP broadcasts rank 0's, and an average
    # copied before that would carry weights no rank is training, for the
    # ~1/(1-decay) steps it takes to forget them. Each rank test-plays its own
    # average, so they would not even be measuring one model.
    ema_decay = config['control'].get('ema_decay', 0)
    ema_models = ()
    best_perf_ema = {'avg_rank': 4., 'avg_pt': -135.}
    if ema_decay > 0:
        mortal_ema = copy.deepcopy(mortal).requires_grad_(False)
        dqn_ema = copy.deepcopy(dqn).requires_grad_(False)
        ema_models = (mortal_ema, dqn_ema)
        if path.exists(state_file) and 'ema' in state:
            mortal_ema.load_state_dict(state['ema']['mortal'])
            dqn_ema.load_state_dict(state['ema']['current_dqn'])
            best_perf_ema = state['best_perf_ema']
        else:
            logging.info('ema: starting from the current weights')
        # Parameters and the BN running stats, in matching order.
        ema_tensors = [t for m in ema_models for t in chain(m.parameters(), m.buffers()) if t.is_floating_point()]
        live_tensors = [t for m in (mortal, dqn) for t in chain(m.parameters(), m.buffers()) if t.is_floating_point()]
        best_ema_file = path.splitext(best_state_file)[0] + '_ema.pth'
        # The average as it stood at the previous evaluation, played beside the
        # current one on the same walls. Two absolute numbers 40,000 steps apart
        # carry about 0.02 of noise each when compared, which is more than any
        # progress they are being asked to show; the difference between the two
        # models measured over the same walls is the thing worth reporting, and
        # this is the only way to get it.
        prev_ema_file = path.splitext(state_file)[0] + '_ema_prev.pth'
        mortal_prev = copy.deepcopy(mortal).requires_grad_(False)
        dqn_prev = copy.deepcopy(dqn).requires_grad_(False)
        logging.info(f'ema: decay {ema_decay}, over ~{1 / (1 - ema_decay):,.0f} steps')

    if enable_compile:
        # Fuses the ResNet's thousands of small kernels a step, whose launches
        # otherwise leave the GPU waiting on the host: 240 -> 138 ms a step on
        # a laptop 3060, for a couple of minutes of compiling at the start.
        # The whole wrapped net rather than each module in place, as this used
        # to: DDP keeps overlapping the gradient all-reduce with the backward,
        # and test play, which runs the modules themselves with a batch size
        # of its own, stays uncompiled instead of recompiling for it.
        compile_mode = config['control'].get('compile_mode')
        # Every mode that captures CUDA graphs, which is more than the obvious
        # one: max-autotune captures them too, and says so only by not being
        # the -no-cudagraphs spelling of itself.
        cuda_graphs = compile_mode in ('reduce-overhead', 'max-autotune')
        if cuda_graphs and opt_step_every > 1:
            # A CUDA graph's replay overwrites the gradients it produced last
            # time, which accumulation still needs to add to.
            raise ValueError(f'compile_mode = {compile_mode!r} (CUDA graphs) cannot be used '
                             'with opt_step_every > 1')
        net = torch.compile(net, mode=compile_mode)

    def optimizer_step():
        """Apply what has accumulated, and move the weight average with it."""
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
            clip_grad_norm_(params, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema_models:
            with torch.no_grad():
                torch._foreach_lerp_(ema_tensors, live_tensors, 1 - ema_decay)

    def save_state():
        """Write the checkpoint and return it.

        Everything needed to carry on from here, which is more than the
        weights: resuming without the optimizer's moments and the schedule
        restarts the run in all but name.
        """
        state = {
            'mortal': mortal.state_dict(),
            'current_dqn': dqn.state_dict(),
            'aux_net': aux_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': steps,
            'timestamp': datetime.now().timestamp(),
            'best_perf': best_perf,
            'config': config,
        }
        if ema_models:
            state['ema'] = {'mortal': mortal_ema.state_dict(), 'current_dqn': dqn_ema.state_dict()}
            state['best_perf_ema'] = best_perf_ema
        if ddp.is_main:
            torch.save(state, state_file)
        return state

    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(mortal, dqn, is_idle=True)
        logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir']) if ddp.is_main else NullWriter()
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
        'next_rank_loss': 0,
    }
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    def train_epoch():
        nonlocal steps
        nonlocal idx

        player_names = []
        if online:
            player_names = ['trainee']
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            parquet_globs = config['dataset'].get('parquet_globs') or []
            # Rank 0 builds a missing index while the others wait, so two
            # processes never race to write one file; then all read it back.
            build = ddp.is_main and not path.exists(file_index)
            if build and parquet_globs:
                logging.info('building parquet row group index...')
                shards = []
                for pat in parquet_globs:
                    shards.extend(glob(pat, recursive=True))
                import pyarrow.parquet as pq
                file_list = []
                for shard in tqdm(sorted(shards), unit='shard'):
                    num_row_groups = pq.ParquetFile(shard).num_row_groups
                    file_list.extend((shard, rg) for rg in range(num_row_groups))
                logging.info(f'{len(shards):,} shards, {len(file_list):,} row groups')
                torch.save({'file_list': file_list}, file_index)
            elif build:
                logging.info('building file index...')
                file_list = []
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            if not set(start['names']).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                torch.save({'file_list': file_list}, file_index)
            ddp.barrier()
            file_list = torch.load(file_index, weights_only=True)['file_list']
        # A disjoint slice per rank. `steps` is the same on every rank here, so
        # they agree on the split without having to ask each other.
        file_list = ddp.shard(file_list, seed=steps)
        logging.info(f'file list size: {len(file_list):,}'
                     + (f' (rank 0 of {ddp.world_size})' if ddp.enabled else ''))

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        if num_workers > 1:
            random.shuffle(file_list)
        # A gz entry is a path and a parquet entry is a (shard, row group)
        # pair, which is also what a reloaded index holds.
        use_parquet = bool(file_list) and not isinstance(file_list[0], str)
        file_data = FileDatasetsIter(
            version = version,
            file_list = file_list,
            pts = pts,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            parquet = use_parquet,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
        )
        # A worker stops yielding while it decodes its next read, and in-order
        # delivery waits for that worker even with another's batches ready.
        # Out of order, with enough queued ahead, the refills stay hidden.
        # Both default to upstream's behaviour when not configured.
        loader_kwargs = {}
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = config['dataset'].get('prefetch_factor', 2)
            loader_kwargs['in_order'] = config['dataset'].get('in_order', True)
        data_loader = iter(DataLoader(
            dataset = file_data,
            batch_size = batch_size,
            drop_last = False,
            num_workers = num_workers,
            pin_memory = True,
            worker_init_fn = worker_init_fn,
            **loader_kwargs,
        ))

        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every,
                  disable=not ddp.is_main)
        batch_idx = torch.arange(batch_size, device=device)

        def train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks):
            nonlocal steps
            nonlocal idx
            nonlocal pb

            # From pinned memory these copies need not wait for the GPU, and
            # the mask check runs on the device, so nothing here makes the
            # host wait: it can queue the next step while this one runs.
            obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
            actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
            masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device, non_blocking=True)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device, non_blocking=True)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device, non_blocking=True)
            torch._assert_async(masks[batch_idx, actions].all())

            q_target_mc = gamma ** steps_to_done * kyoku_rewards
            q_target_mc = q_target_mc.to(torch.float32)

            # Only the step that updates the weights needs the all-reduce.
            # Accumulation boundaries are counted in `steps`, which nothing
            # resets, rather than in `idx`, which the save window zeroes. Keyed
            # to `idx` a save landing mid-accumulation dropped the boundary,
            # and the batches on either side of it were summed into one update
            # that was still scaled as though there had been opt_step_every of
            # them. Counted this way a resume also lands where it left off.
            with ddp.no_sync(net, (steps + 1) % opt_step_every == 0):
                with torch.autocast(device.type, enabled=enable_amp):
                    q_out, (next_rank_logits,) = net(obs, masks)
                    q = q_out[batch_idx, actions]
                    dqn_loss = 0.5 * mse(q, q_target_mc)
                    cql_loss = 0
                    if not online:
                        cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                    next_rank_loss = ce(next_rank_logits, player_ranks)

                    loss = sum((
                        dqn_loss,
                        cql_loss * min_q_weight,
                        next_rank_loss * next_rank_weight,
                    ))
                scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                stats['dqn_loss'] += dqn_loss
                if not online:
                    stats['cql_loss'] += cql_loss
                stats['next_rank_loss'] += next_rank_loss
                all_q[idx] = q
                all_q_target[idx] = q_target_mc

            steps += 1
            idx += 1
            if steps % opt_step_every == 0:
                optimizer_step()
            scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                # Every rank has to take part in these before rank 0 goes on
                # alone to write things down.
                for k in stats:
                    stats[k] = ddp.mean(stats[k], device)
                ddp.check_in_sync(all_models, device)
                # Where the time went. A loader that falls behind shows as data
                # waits; a slow rank as the others waiting on it.
                pace = ddp.pace()
                wall = pace[0][0]
                # `idx`, not save_every: a window cut short by a resume holds
                # fewer batches than the save interval names, and every average
                # over it has to be divided by what it actually holds.
                writer.add_scalar('perf/steps_per_sec', idx / wall, steps)
                writer.add_scalar('perf/data_wait', max(data for _, data, _ in pace) / wall, steps)
                logging.info(
                    f'{idx} steps in {wall:.0f} s; waiting for data / other ranks: '
                    + ', '.join(f'rank {r} {data / wall:.0%} / {ranks / wall:.0%}'
                                for r, (_, data, ranks) in enumerate(pace)))

                # downsample to reduce tensorboard event size. Only the rows
                # this window actually filled: a window cut short by a resume
                # leaves the rest holding the last run's numbers.
                all_q_1d = all_q[:idx].cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target[:idx].cpu().numpy().flatten()[::128]

                writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / idx, steps)
                if not online:
                    writer.add_scalar('loss/cql_loss', stats['cql_loss'] / idx, steps)
                writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / idx, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                writer.add_histogram('q_predicted', all_q_1d, steps)
                writer.add_histogram('q_target', all_q_target_1d, steps)
                writer.flush()

                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

                state = save_state()

                if online and steps % submit_every != 0:
                    submit_param(mortal, dqn, is_idle=False)
                    logging.info('param has been submitted')

                if steps % test_every == 0:
                    # Each rank plays its own slice of the same seeds on its own
                    # GPU. Rank 0 clears the last round's games first, and all
                    # read the finished set back once the slowest rank is done.
                    prev_steps = None
                    if ema_models and path.exists(prev_ema_file):
                        was = torch.load(prev_ema_file, weights_only=True, map_location=device)
                        mortal_prev.load_state_dict(was['mortal'])
                        dqn_prev.load_state_dict(was['current_dqn'])
                        prev_steps = was['steps']
                        del was
                    ddp.barrier()
                    if ddp.is_main:
                        test_player.clear()
                        if ema_models:
                            test_player.clear('ema')
                            test_player.clear('prev')
                    ddp.barrier()
                    ddp.sync_buffers(all_models)
                    jobs = [(mortal, dqn, None)]
                    if ema_models:
                        ddp.sync_buffers(ema_models)
                        jobs.append((mortal_ema, dqn_ema, 'ema'))
                        if prev_steps is not None:
                            jobs.append((mortal_prev, dqn_prev, 'prev'))
                    # All of them at once: one arena alone leaves most of the
                    # box idle, and they are independent games. See play_all.
                    test_player.play_all(test_games // 4, jobs, device)
                    ddp.barrier()
                    stat = test_player.collect()
                    mortal.train()
                    dqn.train()

                    avg_pt = stat.avg_pt([90, 45, 0, -135]) # for display only, never used in training
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    better_ema = False
                    if ema_models:
                        stat_ema = test_player.collect('ema')
                        avg_pt_ema = stat_ema.avg_pt([90, 45, 0, -135])
                        better_ema = avg_pt_ema >= best_perf_ema['avg_pt'] and stat_ema.avg_rank <= best_perf_ema['avg_rank']
                        if better_ema:
                            best_perf_ema['avg_pt'] = avg_pt_ema
                            best_perf_ema['avg_rank'] = stat_ema.avg_rank
                        writer.add_scalar('test_play_ema/avg_ranking', stat_ema.avg_rank, steps)
                        writer.add_scalar('test_play_ema/avg_pt', avg_pt_ema, steps)
                        writer.add_scalars('test_play_ema/ranking', {
                            '1st': stat_ema.rank_1_rate,
                            '2nd': stat_ema.rank_2_rate,
                            '3rd': stat_ema.rank_3_rate,
                            '4th': stat_ema.rank_4_rate,
                        }, steps)
                        if ddp.is_main:
                            diff, se, games, seeds = test_player.paired('ema')
                            writer.add_scalar('test_play_ema/rank_minus_trained', diff, steps)
                            logging.info(f'ema avg rank: {stat_ema.avg_rank:.6}, avg pt: {avg_pt_ema:.6}; '
                                         f'ema minus trained over the same {games:,} games '
                                         f'({seeds:,} walls): rank {diff:+.4f} +- {se:.4f}')
                            if prev_steps is not None:
                                # Negative is better, as everywhere else here.
                                gain, gain_se, _, walls = test_player.paired('ema', against='prev')
                                writer.add_scalar('test_play_ema/rank_since_last', gain, steps)
                                logging.info(
                                    f'progress since step {prev_steps:,}, over the same '
                                    f'{walls:,} walls: rank {gain:+.4f} +- {gain_se:.4f}')
                            torch.save({
                                'mortal': mortal_ema.state_dict(),
                                'current_dqn': dqn_ema.state_dict(),
                                'steps': steps,
                            }, prev_ema_file)
                        writer.flush()

                    if (better or better_ema) and ddp.is_main:
                        torch.save(state, state_file)  # with the new best_perf in it
                    if better and ddp.is_main:
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                    if better_ema and ddp.is_main:
                        # A whole checkpoint, with the averaged weights under
                        # the usual names: it loads anywhere mortal.pth does,
                        # including as the state an online run starts from,
                        # which wants aux_net and the optimizer as well.
                        torch.save({
                            **state,
                            'mortal': mortal_ema.state_dict(),
                            'current_dqn': dqn_ema.state_dict(),
                            'best_perf': best_perf_ema,
                        }, best_ema_file)
                        logging.info(f'a new ema record: rank {best_perf_ema["avg_rank"]:.4}, '
                                     f'pt {best_perf_ema["avg_pt"]:.4}, saving to {best_ema_file}')
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN', disable=not ddp.is_main)

        def full_batches():
            """Every full batch of the epoch, with the ragged ends re-cut last."""
            remaining = []
            remaining_bs = 0
            for batch in data_loader:
                bs = batch[0].shape[0]
                if bs == batch_size:
                    yield batch
                    continue
                remaining.append(batch)
                remaining_bs += bs
            if remaining_bs >= batch_size:
                columns = [torch.cat(col, dim=0) for col in zip(*remaining)]
                for start in range(0, remaining_bs - batch_size + 1, batch_size):
                    yield [col[start:start + batch_size] for col in columns]

        # One place for both the full batches and the re-cut ends, so DDP can
        # stop every rank together when the first of them runs dry.
        for batch in ddp.in_lockstep(full_batches()):
            train_batch(*batch)
        pb.close()

        if online:
            submit_param(mortal, dqn, is_idle=True)
            logging.info('param has been submitted')

    while True:
        train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # only run one epoch for offline for easier control
            break
    # Whatever is still accumulated belongs in the weights before they are
    # written down; the checkpoint holds no gradients, so anything left here is
    # simply lost.
    if steps % opt_step_every != 0:
        optimizer_step()
    # The corpus ran out mid-window, so the last steps are in no checkpoint
    # yet, and the line below tells a supervisor to stop watching for a run to
    # come back. They would be lost between the two.
    if steps % save_every != 0:
        save_state()
    ddp.close()
    if ddp.is_main:
        # The last line of a run that ended because it was finished. A
        # supervisor watching the log tells that from a crash by this.
        logging.info(f'training is complete after {steps:,} steps')

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
