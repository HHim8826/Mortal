"""The v4 Brain, DQN and AuxNet of `model.py`, in Flax, for training on a TPU.

Layer for layer the same network, so a checkpoint moves between the two with
`tpu.convert` and plays the same moves. Two things differ in form only:

- Activations are channels-last, (batch, 34, channels), which is what Flax's
  convolutions take. The one place the order matters is the flatten before the
  1024-wide linear: PyTorch flattens (32, 34) channel-major, so the port
  transposes back before flattening and the weights carry over unchanged.
- The residual blocks are one `nn.scan` over stacked weights rather than forty
  modules. Unrolled, the forty-block graph took over 60 GB of host memory to
  compile and had not finished after fifteen minutes on the Kaggle TPU host.

Only version 4 is ported: its BatchNorm eps is 1e-3 and momentum 0.01 in
PyTorch's convention, 0.99 in Flax's.
"""
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

OBS_CHANNELS, WIDTH, ACTIONS = 1012, 34, 46


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def batch_norm(train, name, dtype):
    return nn.BatchNorm(use_running_average=not train, momentum=0.99, epsilon=1e-3, dtype=dtype, name=name)


class ChannelAttention(nn.Module):
    channels: int
    dtype: Any = jnp.float32
    ratio: int = 16

    @nn.compact
    def __call__(self, x):
        fc1 = nn.Dense(self.channels // self.ratio, dtype=self.dtype, name='fc1')
        fc2 = nn.Dense(self.channels, dtype=self.dtype, name='fc2')
        mlp = lambda v: fc2(mish(fc1(v)))
        weight = jax.nn.sigmoid(mlp(x.mean(1)) + mlp(x.max(1)))
        return x * weight[:, None, :]


class ResBlock(nn.Module):
    """Pre-activation: BN, Mish, conv, BN, Mish, conv, channel attention, plus the input."""
    channels: int
    train: bool
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, _):
        conv = lambda name: nn.Conv(self.channels, (3,), padding='SAME', use_bias=False,
                                    dtype=self.dtype, name=name)
        out = conv('conv1')(mish(batch_norm(self.train, 'bn1', self.dtype)(x)))
        out = conv('conv2')(mish(batch_norm(self.train, 'bn2', self.dtype)(out)))
        return ChannelAttention(self.channels, self.dtype, name='ca')(out) + x, None


def blocks(num_blocks, remat=False):
    """The residual stack as one scan; `remat` recomputes each block's activations in the
    backward pass instead of keeping all forty blocks' worth of them."""
    return nn.scan(nn.remat(ResBlock) if remat else ResBlock,
                   variable_axes={'params': 0, 'batch_stats': 0},
                   split_rngs={'params': True}, length=num_blocks)


class Brain(nn.Module):
    """obs (batch, 34, 1012) -> phi (batch, 1024).

    `dtype` is what the layers compute in; parameters stay float32 either way.
    Left at float32 a layer computes in float32 whatever its input is, because
    Flax promotes a bfloat16 input against float32 weights: training has to ask
    for bfloat16, and conversion and the equivalence checks keep float32.
    """
    conv_channels: int = 192
    num_blocks: int = 40
    dtype: Any = jnp.float32
    remat: bool = False

    @nn.compact
    def __call__(self, obs, train=False):
        d = self.dtype
        x = nn.Conv(self.conv_channels, (3,), padding='SAME', use_bias=False, dtype=d, name='stem')(obs)
        x, _ = blocks(self.num_blocks, self.remat)(self.conv_channels, train, d, name='blocks')(x, None)
        x = mish(batch_norm(train, 'bn_final', d)(x))
        x = mish(nn.Conv(32, (3,), padding='SAME', dtype=d, name='conv_out')(x))
        x = x.transpose(0, 2, 1).reshape(x.shape[0], -1)       # PyTorch's (32, 34) order
        return mish(nn.Dense(1024, dtype=d, name='fc')(x))


class DQN(nn.Module):
    """The v4 dueling head: one linear to V and the 46 advantages."""
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, phi, mask):
        va = nn.Dense(1 + ACTIONS, dtype=self.dtype, name='net')(phi)
        v, a = va[:, :1], va[:, 1:]
        a_mean = jnp.where(mask, a, 0.).sum(-1, keepdims=True) / mask.sum(-1, keepdims=True)
        return jnp.where(mask, v + a - a_mean, -jnp.inf)


class Mortal(nn.Module):
    """Brain, DQN and the next-rank AuxNet together, as `train.py` runs them."""
    conv_channels: int = 192
    num_blocks: int = 40
    dtype: Any = jnp.float32
    remat: bool = False

    @nn.compact
    def __call__(self, obs, mask, train=False):
        phi = Brain(self.conv_channels, self.num_blocks, self.dtype, self.remat, name='brain')(obs, train)
        q = DQN(self.dtype, name='dqn')(phi, mask)
        next_rank_logits = nn.Dense(4, use_bias=False, dtype=self.dtype, name='aux')(phi)
        return q, next_rank_logits


def deepen(variables, channels, old_blocks, new_blocks, rng):
    """`variables` with residual blocks appended up to `new_blocks`, each exactly the identity.

    A pre-activation block adds its residual branch to its input, and the branch ends in
    the second convolution (then channel attention, which scales it). Zero that one
    convolution and the branch is zero whatever the rest holds, so the new blocks start
    as fresh weights around a closed gate: the deeper net plays exactly as the old one,
    and training opens them.
    """
    fresh = Mortal(channels, new_blocks).init(rng, jnp.zeros((2, WIDTH, OBS_CHANNELS)),
                                              jnp.ones((2, ACTIONS), bool))
    out = {}
    for col in ('params', 'batch_stats'):
        out[col] = dict(variables[col])
        brain = dict(variables[col]['brain'])
        brain['blocks'] = jax.tree_util.tree_map(lambda o, f: jnp.concatenate([o, f[old_blocks:]]),
                                                 variables[col]['brain']['blocks'],
                                                 fresh[col]['brain']['blocks'])
        out[col]['brain'] = brain
    blocks = dict(out['params']['brain']['blocks'])
    blocks['conv2'] = {'kernel': blocks['conv2']['kernel'].at[old_blocks:].set(0.)}
    out['params']['brain']['blocks'] = blocks
    return out
