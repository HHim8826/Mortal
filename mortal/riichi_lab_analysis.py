"""Exact own-hand facts and optional opponent estimates, isolated from play.

The tenpai model always uses v3 observations (934 x 34), even when the playing
model uses v4. Wait probabilities are NOT ron/deal-in probabilities.
"""
import asyncio
import logging
import time

import numpy as np
import torch

from tenpai_net import TenpaiNet


TILES = tuple(f'{n}{s}' for s in 'mps' for n in range(1, 10)) + tuple('ESWNPFC')
OUTPUTS = ('tenpai', 'waits', 'any_wait', 'furiten')
SHAPES = ((1, 3), (1, 3, 34), (1, 34), (1, 3))


class TenpaiPredictor:
    def __init__(self, model):
        self.model = model.cpu().eval()

    @classmethod
    def load(cls, filename):
        saved = torch.load(filename, map_location='cpu', weights_only=True)
        if isinstance(saved, dict) and 'model' in saved:
            weights = saved['model']
            channels, inputs, _ = weights['stem.0.weight'].shape
            if inputs != 934:
                raise ValueError(f'tenpai model needs v3 / 934 inputs, got {inputs}')
            blocks = len({k.split('.')[1] for k in weights if k.startswith('trunk.')})
            model = TenpaiNet(934, channels, blocks)
            model.load_state_dict(weights)
            result = cls(model)
        else:
            raise ValueError('unsupported tenpai checkpoint format')
        result.predict(np.zeros((934, 34), dtype=np.float32))
        return result

    def predict(self, obs):
        with torch.inference_mode():
            out = self.model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            if not isinstance(out, dict) or set(out) != set(OUTPUTS):
                raise ValueError('tenpai output must contain tenpai, waits, any_wait and furiten')
            result = {}
            for name, shape in zip(OUTPUTS, SHAPES):
                tensor = out[name].sigmoid()
                if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                    raise ValueError(f'invalid tenpai output: {name}')
                if torch.any((tensor < 0) | (tensor > 1)):
                    raise ValueError(f'tenpai probabilities outside [0, 1]: {name}')
                result[name] = tensor[0].tolist()
            return result


class TableAnalysis:
    def __init__(self, table):
        self.table = table
        self.state = None
        self.active = False
        self.generation = 0
        self.revision = 0
        self.predictor = None

    def pause(self):
        self.active = False
        self.generation += 1
        self.table.prediction = None

    def update(self, line, event):
        from libriichi.state import PlayerState

        kind = event['type']
        if kind in ('start_game', 'start_kyoku', 'end_kyoku', 'end_game'):
            self.generation += 1
            self.table.prediction = None
            self.active = kind == 'start_kyoku'
        if kind == 'start_game':
            self.state = PlayerState(event.get('id', 0))
        elif kind == 'start_kyoku' and self.state is None:
            self.state = PlayerState(self.table.seat)
        if self.state is None:
            return
        try:
            self.state.update(line)
            self.revision += 1
            if not self.active:
                return
            shanten = self.state.shanten
            settled = sum(self.state.tehai) % 3 == 1
            if settled and shanten == 0:
                waits = [tile for tile, waiting in zip(TILES, self.state.waits) if waiting]
                status = 'TENPAI: ' + (' '.join(waits) or 'no live waits')
                if self.state.at_furiten:
                    status += ' | FURITEN'
            elif not settled:
                status = ('COMPLETE' if shanten < 0 else
                          f'{shanten} shanten after best discard')
            else:
                status = f'{shanten} shanten'
            self.table.hand_status = status
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            # pyo3 panics inherit BaseException. Analysis must never take the
            # playing bot down or present an old hand as a current fact.
            self.state = None
            self.active = False
            self.table.hand_status = 'unavailable'
            self.table.prediction = None
            logging.warning('hand analysis unavailable until next round: %s', exc)

    async def sample(self):
        """One in-flight prediction, at most 1 Hz. No queue of stale frames."""
        seen = -1
        while True:
            await asyncio.sleep(1)
            if not self.active or self.state is None or self.revision == seen:
                continue
            generation, seen = self.generation, self.revision
            seat = self.table.seat
            captured = time.monotonic()
            try:
                obs, _ = self.state.encode_obs(3, False)
                result = await asyncio.to_thread(self.predictor.predict, obs)
                if generation != self.generation or not self.active:
                    continue
                # Model output k maps to (observer + 1 + k) % 4, never to
                # absolute seat k. No hidden hands enter this observation.
                result['by_seat'] = {
                    (seat + k + 1) % 4: {
                        'tenpai': result['tenpai'][k], 'furiten': result['furiten'][k],
                        'waits': result['waits'][k]}
                    for k in range(3)}
                result['captured'] = captured
                self.table.prediction = result
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                self.table.prediction = None
                self.table.prediction_status = 'Predictions unavailable'
                logging.warning('opponent predictions disabled: %s', exc)
                return

    async def run(self, awaitable):
        worker = asyncio.create_task(self.sample()) if self.predictor else None
        try:
            return await awaitable
        finally:
            if worker:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass
