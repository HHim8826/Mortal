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


def make_optimizer(optim_cfg):
    tx = optax.adamw(lr_schedule(**optim_cfg['scheduler']), b1=optim_cfg['betas'][0],
                     b2=optim_cfg['betas'][1], eps=optim_cfg['eps'],
                     weight_decay=optim_cfg['weight_decay'], mask=decay_mask)
    if optim_cfg.get('max_grad_norm', 0) > 0:
        tx = optax.chain(optax.clip_by_global_norm(optim_cfg['max_grad_norm']), tx)
    return tx


def loss_fn(params, batch_stats, batch, *, model, gamma, min_q_weight, next_rank_weight,
            dtype=jnp.bfloat16):
    """The offline loss on one batch, with BatchNorm on the batch's own statistics.

    `batch` is train.py's tuple as arrays: obs (n, 1012, 34), actions, masks, steps_to_done,
    kyoku_rewards, player_ranks.
    """
    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
    x = obs.transpose(0, 2, 1).astype(dtype)                  # channels last
    (q_out, logits), updates = model.apply({'params': params, 'batch_stats': batch_stats}, x,
                                           masks, train=True, mutable=['batch_stats'])
    q_out, logits = q_out.astype(jnp.float32), logits.astype(jnp.float32)
    q = jnp.take_along_axis(q_out, actions[:, None], 1)[:, 0]
    q_target = (gamma ** steps_to_done.astype(jnp.float32)) * kyoku_rewards.astype(jnp.float32)
    dqn_loss = 0.5 * jnp.mean((q - q_target) ** 2)
    cql_loss = jnp.mean(jax.nn.logsumexp(q_out, axis=-1)) - jnp.mean(q)
    rank_loss = optax.softmax_cross_entropy_with_integer_labels(logits, player_ranks).mean()
    loss = dqn_loss + min_q_weight * cql_loss + next_rank_weight * rank_loss
    return loss, (updates['batch_stats'], {'dqn_loss': dqn_loss, 'cql_loss': cql_loss,
                                           'next_rank_loss': rank_loss})


def make_train_step(tx, **loss_kw):
    """One jitted update: (params, batch_stats, opt_state, batch) -> the same, and the losses."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(params, batch_stats, opt_state, batch):
        (_, (batch_stats, stats)), grads = grad_fn(params, batch_stats, batch, **loss_kw)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), batch_stats, opt_state, stats
    return train_step
