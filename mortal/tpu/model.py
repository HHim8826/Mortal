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
import jax
import jax.numpy as jnp
import flax.linen as nn

OBS_CHANNELS, WIDTH, ACTIONS = 1012, 34, 46


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def batch_norm(train, name):
    return nn.BatchNorm(use_running_average=not train, momentum=0.99, epsilon=1e-3, name=name)


class ChannelAttention(nn.Module):
    channels: int
    ratio: int = 16

    @nn.compact
    def __call__(self, x):
        fc1 = nn.Dense(self.channels // self.ratio, name='fc1')
        fc2 = nn.Dense(self.channels, name='fc2')
        mlp = lambda v: fc2(mish(fc1(v)))
        weight = jax.nn.sigmoid(mlp(x.mean(1)) + mlp(x.max(1)))
        return x * weight[:, None, :]


class ResBlock(nn.Module):
    """Pre-activation: BN, Mish, conv, BN, Mish, conv, channel attention, plus the input."""
    channels: int
    train: bool

    @nn.compact
    def __call__(self, x, _):
        conv = lambda name: nn.Conv(self.channels, (3,), padding='SAME', use_bias=False, name=name)
        out = conv('conv1')(mish(batch_norm(self.train, 'bn1')(x)))
        out = conv('conv2')(mish(batch_norm(self.train, 'bn2')(out)))
        return ChannelAttention(self.channels, name='ca')(out) + x, None


def blocks(num_blocks):
    return nn.scan(ResBlock, variable_axes={'params': 0, 'batch_stats': 0},
                   split_rngs={'params': True}, length=num_blocks)


class Brain(nn.Module):
    """obs (batch, 34, 1012) -> phi (batch, 1024)."""
    conv_channels: int = 192
    num_blocks: int = 40

    @nn.compact
    def __call__(self, obs, train=False):
        x = nn.Conv(self.conv_channels, (3,), padding='SAME', use_bias=False, name='stem')(obs)
        x, _ = blocks(self.num_blocks)(self.conv_channels, train, name='blocks')(x, None)
        x = mish(batch_norm(train, 'bn_final')(x))
        x = mish(nn.Conv(32, (3,), padding='SAME', name='conv_out')(x))
        x = x.transpose(0, 2, 1).reshape(x.shape[0], -1)       # PyTorch's (32, 34) order
        return mish(nn.Dense(1024, name='fc')(x))


class DQN(nn.Module):
    """The v4 dueling head: one linear to V and the 46 advantages."""

    @nn.compact
    def __call__(self, phi, mask):
        va = nn.Dense(1 + ACTIONS, name='net')(phi)
        v, a = va[:, :1], va[:, 1:]
        a_mean = jnp.where(mask, a, 0.).sum(-1, keepdims=True) / mask.sum(-1, keepdims=True)
        return jnp.where(mask, v + a - a_mean, -jnp.inf)


class Mortal(nn.Module):
    """Brain, DQN and the next-rank AuxNet together, as `train.py` runs them."""
    conv_channels: int = 192
    num_blocks: int = 40

    @nn.compact
    def __call__(self, obs, mask, train=False):
        phi = Brain(self.conv_channels, self.num_blocks, name='brain')(obs, train)
        q = DQN(name='dqn')(phi, mask)
        next_rank_logits = nn.Dense(4, use_bias=False, name='aux')(phi)
        return q, next_rank_logits
