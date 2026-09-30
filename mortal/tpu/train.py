"""The offline (supervised) v4 training step in JAX: train.py's loss, optimizer and schedule.

What `train.py` does per batch when `online = false`, for `tpu.model.Mortal`:

    q_target = gamma ** steps_to_done * kyoku_reward
    loss = 0.5 * mse(Q(a), q_target)                          # dqn
         + min_q_weight * (logsumexp(Q) - Q(a)).mean()        # cql
         + next_rank_weight * ce(next_rank_logits, rank)      # aux

AdamW with decoupled weight decay on the weights of linear and convolutional
layers only (`optimizer_groups`), under `LinearWarmUpCosineAnnealingLR` with a
base learning rate of 1. The only deliberate difference is precision: bf16
compute where train.py runs fp16 autocast with a GradScaler, which bf16's range
makes unnecessary.

Online (`online = true`) it is train.py's online branch: no CQL term; with
`freeze_bn`, BatchNorm normalises by its running statistics and never updates
them, as a BatchNorm held in eval does, while its scale and bias still train; and
with `trainable_blocks`, `split` keeps the stem and every block but the last few
out of the gradient and the optimizer, as `Brain.freeze_trunk` does.
"""
import math

import jax
import jax.numpy as jnp
import optax


def lr_schedule(*, peak, final, warm_up_steps, max_steps, init=1e-8, offset=0, epoch_size=0):
    """`lr_scheduler.LinearWarmUpCosineAnnealingLR`, as an optax schedule."""
    def schedule(step):
        step = step + offset
        if epoch_size > 0:
            step = step % epoch_size
        warm = init + (peak - init) / max(warm_up_steps, 1) * step
        progress = (step - warm_up_steps) / max(max_steps - warm_up_steps, 1)
        cos = final + 0.5 * (peak - final) * (1 + jnp.cos(jnp.clip(progress, 0., 1.) * math.pi))
        return jnp.where((warm_up_steps > 0) & (step < warm_up_steps), warm,
                         jnp.where(step < max_steps, cos, final))
    return schedule


def decay_mask(params):
    """Weight decay on kernels -- the weights of Dense and Conv -- and nothing else."""
    return jax.tree_util.tree_map_with_path(lambda path, _: path[-1].key == 'kernel', params)


def split(tree, frozen_blocks):
    """(trained, frozen): a params or batch_stats tree cut where `Brain.freeze_trunk` cuts.

    The frozen part is the stem and the first `frozen_blocks` residual blocks; the rest of
    the Brain, the heads and the aux net are the trained part. The blocks are one stacked
    array, cut along its first axis. `join` puts them back together.
    """
    brain = tree['brain']
    cut = lambda sub, s: jax.tree_util.tree_map(lambda a: a[s], sub)
    frozen = {'blocks': cut(brain['blocks'], slice(None, frozen_blocks))}
    if 'stem' in brain:
        frozen['stem'] = brain['stem']
    trained = {k: v for k, v in brain.items() if k not in ('stem', 'blocks')}
    trained['blocks'] = cut(brain['blocks'], slice(frozen_blocks, None))
    return {**{k: v for k, v in tree.items() if k != 'brain'}, 'brain': trained}, {'brain': frozen}


def join(trained, frozen):
    """The tree `split` cut into `trained` and `frozen`."""
    brain = {**trained['brain'], **{k: v for k, v in frozen['brain'].items() if k != 'blocks'}}
    brain['blocks'] = jax.tree_util.tree_map(lambda f, t: jnp.concatenate([f, t]),
                                             frozen['brain']['blocks'], trained['brain']['blocks'])
    return {**trained, 'brain': brain}


def before_split(opt):
    """Whether `opt`, an online state's optimizer state as msgpack_restore gives it, is from
    before `split` (#59): the whole net's AdamW between two masks, chain(hold, AdamW, hold),
    whose EmptyStates restore as empty dicts."""
    return isinstance(opt, dict) and set(opt) == {'0', '1', '2'} and opt['0'] == {} and opt['2'] == {}


def from_before_split(opt, frozen_blocks):
    """Such a state as the trained part's AdamW state now: the moments, zero where the net
    was held, cut to the trained part; the step counts as they were."""
    if not frozen_blocks:
        return opt['1']

    def walk(d):
        if isinstance(d, dict) and isinstance(d.get('brain'), dict) and 'stem' in d['brain']:
            return split(d, frozen_blocks)[0]
        return {k: walk(v) for k, v in d.items()} if isinstance(d, dict) else d
    return walk(opt['1'])


def make_optimizer(optim_cfg):
    """AdamW as train.py builds it. A frozen parameter stays out of it, as in train.py:
    online, it is given `split`'s trained part only."""
    tx = optax.adamw(lr_schedule(**optim_cfg['scheduler']), b1=optim_cfg['betas'][0],
                     b2=optim_cfg['betas'][1], eps=optim_cfg['eps'],
                     weight_decay=optim_cfg['weight_decay'], mask=decay_mask)
    if optim_cfg.get('max_grad_norm', 0) > 0:
        tx = optax.chain(optax.clip_by_global_norm(optim_cfg['max_grad_norm']), tx)
    return tx


def loss_fn(params, batch_stats, batch, *, model, gamma, min_q_weight, next_rank_weight,
            dtype=jnp.bfloat16, online=False, freeze_bn=False, frozen=None):
    """train.py's loss on one batch.

    Offline, BatchNorm normalises by the batch's own statistics and updates its running
    ones. `online` drops the CQL term; `freeze_bn` normalises by the running statistics
    and leaves them as they are.

    With `frozen`, the (params, batch_stats) of `split`'s frozen part, `params` and
    `batch_stats` are the trained part: the frozen blocks run forward only, into the
    trained ones, so the gradient -- `params`' alone -- goes back through those and no
    further (#59). It needs `freeze_bn`, which holds the frozen blocks' BatchNorm in eval.

    `batch` is train.py's tuple as arrays: obs (n, 1012, 34), actions, masks, steps_to_done,
    kyoku_rewards, player_ranks.
    """
    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
    x = obs.transpose(0, 2, 1).astype(dtype)                  # channels last
    variables = {'params': params, 'batch_stats': batch_stats}
    if frozen is not None:
        if not freeze_bn:
            raise ValueError('a frozen trunk needs freeze_bn')
        cut = jax.tree_util.tree_leaves(frozen[0]['brain']['blocks'])[0].shape[0]
        trunk = model.clone(part='trunk', num_blocks=cut, remat=False)
        x = jax.lax.stop_gradient(trunk.apply({'params': frozen[0], 'batch_stats': frozen[1]}, x))
        q_out, logits = model.clone(part='tail', num_blocks=model.num_blocks - cut).apply(
            variables, x, masks, train=False)
    elif freeze_bn:
        q_out, logits = model.apply(variables, x, masks, train=False)
    else:
        (q_out, logits), updates = model.apply(variables, x, masks, train=True, mutable=['batch_stats'])
        batch_stats = updates['batch_stats']
    q_out, logits = q_out.astype(jnp.float32), logits.astype(jnp.float32)
    q = jnp.take_along_axis(q_out, actions[:, None], 1)[:, 0]
    q_target = (gamma ** steps_to_done.astype(jnp.float32)) * kyoku_rewards.astype(jnp.float32)
    dqn_loss = 0.5 * jnp.mean((q - q_target) ** 2)
    cql_loss = jnp.zeros(()) if online else jnp.mean(jax.nn.logsumexp(q_out, axis=-1)) - jnp.mean(q)
    rank_loss = optax.softmax_cross_entropy_with_integer_labels(logits, player_ranks).mean()
    loss = dqn_loss + min_q_weight * cql_loss + next_rank_weight * rank_loss
    return loss, (batch_stats, {'dqn_loss': dqn_loss, 'cql_loss': cql_loss, 'next_rank_loss': rank_loss})


def make_train_step(tx, **loss_kw):
    """One jitted update: (params, batch_stats, opt_state, batch) -> the same, and the losses."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(params, batch_stats, opt_state, batch):
        (_, (batch_stats, stats)), grads = grad_fn(params, batch_stats, batch, **loss_kw)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), batch_stats, opt_state, stats
    return train_step
