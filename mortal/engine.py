import json
import threading
import traceback
import torch
import numpy as np
from torch.distributions import Normal, Categorical
from typing import *

class PinnedPool:
    """Pinned host blocks that engines stack their batches in, lent one batch at a time.

    A block is out only while a batch is stacked into it and copied to the GPU,
    and the copy blocks, so a few blocks serve any number of arenas and
    engines. What the pool keeps is capped whatever the number of threads: a
    batch that finds no free block that fits, and no room under the cap for a
    new one, is stacked the old way, into an array of its own.
    """
    def __init__(self, cap):
        self.cap = cap
        self.kept = 0
        self.free = []
        self.lock = threading.Lock()

    def take(self, nbytes):
        with self.lock:
            fits = [i for i, b in enumerate(self.free) if b.nbytes >= nbytes]
            if fits:
                return self.free.pop(min(fits, key=lambda i: self.free[i].nbytes))
            # The pinned allocator hands out power-of-two blocks, so ask for a
            # whole one; a block it has cached is never given back to the system.
            size = max(1 << 24, 1 << (nbytes - 1).bit_length())
            if self.kept + size > self.cap:
                return None
            self.kept += size
        try:
            return torch.empty(size // 4, dtype=torch.float32, pin_memory=True)
        except RuntimeError:
            # No pinned memory to be had: the batch goes the old way instead.
            with self.lock:
                self.kept -= size
            return None

    def give(self, block):
        with self.lock:
            self.free.append(block)

# One pool for the process. An evaluation's batches reach ~100 MB, so this holds
# two arenas' largest at once and the smaller ones beside them.
STAGING = PinnedPool(512 << 20)

class MortalEngine:
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        device = None,
        stochastic_latent = False,
        enable_amp = False,
        enable_quick_eval = True,
        enable_rule_based_agari_guard = False,
        name = 'NoName',
        boltzmann_epsilon = 0,
        boltzmann_temp = 1,
        top_p = 1,
    ):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

        # Only a batch bound for the GPU is stacked in the pool; see `_stage`.
        self._pin = self.device.type == 'cuda'

    def _stage(self, obs):
        """The batch `obs` as one tensor on the device, stacked in a pinned block from STAGING.

        A v4 observation is 1012 x 34 floats, 137 KB, and the champion's seats in
        an evaluation act a few hundred at a time: a fresh `np.stack` asked the
        kernel for up to 100 MB every step, about 85 GB per arena for 250 walls,
        and gave it back on the way out. On the one-card box, where memory is too
        fragmented for huge pages and nearly every attempt to compact it fails,
        each of those arrays was faulted in 4 KB at a time, and the evaluation's
        arena threads spent 57-85% of their time in the kernel with the GPU idle.
        On the CPU the stacked array is the model's input itself, one row a move
        for the bot, and there is nothing to keep.
        """
        need = len(obs) * obs[0].size * 4
        block = STAGING.take(need) if self._pin else None
        if block is None:
            return torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        try:
            view = block[:need // 4].view(len(obs), *obs[0].shape)
            np.stack(obs, axis=0, out=view.numpy())
            # Not non_blocking: the batch is on the device when this returns,
            # so the block can go straight back for the next one.
            return view.to(self.device)
        finally:
            STAGING.give(block)

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react_batch(self, obs, masks, invisible_obs):
        obs = self._stage(obs)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        if invisible_obs is not None:
            invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)
        batch_size = obs.shape[0]

        match self.version:
            case 1:
                mu, logsig = self.brain(obs, invisible_obs)
                if self.stochastic_latent:
                    latent = Normal(mu, logsig.exp() + 1e-6).sample()
                else:
                    latent = mu
                q_out = self.dqn(latent, masks)
            case 2 | 3 | 4:
                phi = self.brain(obs)
                q_out = self.dqn(phi, masks)

        if self.boltzmann_epsilon > 0:
            is_greedy = torch.full((batch_size,), 1-self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            logits = (q_out / self.boltzmann_temp).masked_fill(~masks, -torch.inf)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()

def sample_top_p(logits, p):
    if p >= 1:
        return Categorical(logits=logits).sample()
    if p <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled

class ExampleMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = 'mjai-log'
        self.name = name
        self.player_ids = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states):
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]['type'] == 'start_kyoku'

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(json.dumps({
                    'type': 'dahai',
                    'actor': player_id,
                    'pai': tile,
                    'tsumogiri': True,
                }))
            else:
                res.append('{"type":"none"}')
        return res

    # They will be executed at specific events. They can be no-op but must be
    # defined.
    def start_game(self, game_idx: int):
        pass
    def end_kyoku(self, game_idx: int):
        pass
    def end_game(self, game_idx: int, scores: List[int]):
        pass
