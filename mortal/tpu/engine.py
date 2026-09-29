"""A libriichi engine that plays a Flax net: `MortalEngine`'s interface, on a JAX device.

libriichi's arena reads six attributes of an engine (`engine_type`, `name`,
`is_oracle`, `version`, `enable_quick_eval`, `enable_rule_based_agari_guard`) and
calls `react_batch` with a list of observations; this is that, for the nets of
`tpu.model`, v4 and the v3 baseline alike.

What makes it more than a forward pass is the batch. The arena hands over however
many seats are waiting, a different number nearly every call, and XLA compiles a
new program for every new shape. So a batch is padded up to the next of a few fixed
sizes and cut back after, and each size compiles once, the first time it is seen.
The padding rows are all-legal, so their mean advantage stays finite.

Self-play samples as `MortalEngine` does: each move is the argmax with probability
1 - `boltzmann_epsilon`, and otherwise drawn from softmax(Q / `boltzmann_temp`)
over the legal actions, cut to the top `top_p` of the mass. The draw is numpy on
the host, from Q the device already sent back.

    engine = JaxEngine.from_npz('best_ema.npz', name='mortal')
"""
import traceback

import numpy as np

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
_FORWARD = {}


def forward_for(conv_channels, num_blocks, version):
    """One jitted forward per architecture, shared by every engine that plays it: online runs
    an engine per arena, and each would otherwise compile every batch size again."""
    key = (conv_channels, num_blocks, version)
    if key not in _FORWARD:
        import jax
        from tpu.model import Player
        net = Player(conv_channels, num_blocks, version)
        _FORWARD[key] = jax.jit(lambda v, obs, mask: net.apply(v, obs, mask))
    return _FORWARD[key]


def sample_top_p(logits, p, rng):
    """One action a row from softmax(logits) cut to the top `p` of the mass: `engine.sample_top_p`.

    Illegal actions carry -inf and get no mass. As there, an action stays while the
    mass before it is at most `p`, so the most likely one always does.
    """
    if p <= 0:
        return logits.argmax(-1)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    if p < 1:
        order = np.argsort(-probs, -1, kind='stable')
        ranked = np.take_along_axis(probs, order, -1)
        ranked = ranked / ranked.sum(-1, keepdims=True)
        ranked[np.cumsum(ranked, -1) - ranked > p] = 0.
        probs = np.zeros_like(probs)
        np.put_along_axis(probs, order, ranked, -1)
    cdf = np.cumsum(probs, -1)
    u = rng.random((len(cdf), 1)) * cdf[:, -1:]
    return np.minimum((cdf <= u).sum(-1), cdf.shape[-1] - 1)


class JaxEngine:
    engine_type = 'mortal'
    is_oracle = False

    def __init__(self, variables, *, version, conv_channels, num_blocks, device=None, name='NoName',
                 enable_quick_eval=True, enable_rule_based_agari_guard=True,
                 boltzmann_epsilon=0., boltzmann_temp=1., top_p=1., seed=None):
        import jax
        self.name = name
        self.version = version
        # As evaluation and test play set them on `MortalEngine`.
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p
        # Generator draws take its lock, so arenas in threads can share one engine.
        self.rng = np.random.default_rng(seed)
        self.device = device or jax.devices()[0]
        self._q = forward_for(conv_channels, num_blocks, version)
        self.set_variables(variables)

    @classmethod
    def from_npz(cls, path, **kw):
        from tpu import convert
        variables, meta = convert.load_npz(path)
        return cls(variables, version=meta['version'], conv_channels=meta['conv_channels'],
                   num_blocks=meta['num_blocks'], **kw)

    def set_variables(self, variables):
        """Play these weights from the next batch on: a newer EMA, say, mid-run."""
        import jax
        keep = {'params': {k: variables['params'][k] for k in ('brain', 'dqn')},
                'batch_stats': variables['batch_stats']}
        self._variables = jax.device_put(keep, self.device)

    def react_batch(self, obs, masks, invisible_obs):
        try:
            step = BUCKETS[-1]
            q = np.concatenate([self._forward(obs[i:i + step], masks[i:i + step])
                                for i in range(0, len(obs), step)])
            legal = np.stack(masks)
            actions = q.argmax(-1)
            greedy = np.ones(len(q), bool)
            if self.boltzmann_epsilon > 0:
                greedy = self.rng.random(len(q)) >= self.boltzmann_epsilon
                logits = np.where(legal, q / self.boltzmann_temp, -np.inf)
                actions = np.where(greedy, actions, sample_top_p(logits, self.top_p, self.rng))
            return actions.tolist(), q.tolist(), legal.tolist(), greedy.tolist()
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _forward(self, obs, masks):
        import jax
        n = len(obs)
        size = next(b for b in BUCKETS if b >= n)
        # libriichi's observations are (channels, 34); tpu.model takes channels last.
        x = np.zeros((size, obs[0].shape[1], obs[0].shape[0]), np.float32)
        x[:n] = np.stack(obs).transpose(0, 2, 1)
        m = np.ones((size, len(masks[0])), bool)
        m[:n] = np.stack(masks)
        q = self._q(self._variables, jax.device_put(x, self.device), jax.device_put(m, self.device))
        return np.asarray(q)[:n]
