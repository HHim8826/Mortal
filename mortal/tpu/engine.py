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

    engine = JaxEngine.from_npz('best_ema.npz', name='mortal')
"""
import traceback

import numpy as np

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


class JaxEngine:
    engine_type = 'mortal'
    is_oracle = False

    def __init__(self, variables, *, version, conv_channels, num_blocks, device=None, name='NoName',
                 enable_quick_eval=True, enable_rule_based_agari_guard=True):
        import jax
        from tpu.model import Player
        self.name = name
        self.version = version
        # As evaluation and test play set them on `MortalEngine`.
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.device = device or jax.devices()[0]
        net = Player(conv_channels, num_blocks, version)
        self._q = jax.jit(lambda v, obs, mask: net.apply(v, obs, mask))
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
            # Argmax only: every move is greedy, as `MortalEngine` with epsilon 0.
            return q.argmax(-1).tolist(), q.tolist(), np.stack(masks).tolist(), [True] * len(obs)
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
