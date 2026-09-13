"""Training data for reading the other three hands, out of the same corpus.

Every label here is a fact, not a judgement. Four PlayerStates fed one log know
each seat's hand exactly, so who was tenpai and on what needs no annotation and
no model -- and the input is the same log seen from one seat, which is exactly
what that seat could know at the time. The hidden hands make the labels and
never enter the features.

The v3 observation is the input rather than v4's. It carries what reading a
hand is made of -- each opponent's discards in order, every one of them flagged
tedashi or tsumogiri, weighted by how recently it was let go, their melds,
their riichi and its turn -- and the 78 channels v4 adds to that are an
expected-value solve over *our own* hand, which is both irrelevant here and the
most expensive thing in the whole pipeline. Measured on the box: replaying a
hanchan costs 7 ms, and a v3 observation 0.01 ms on top.
"""
import json
import random

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from libriichi.consts import obs_shape
from libriichi.state import PlayerState

TILE_KINDS = 34
# Relative seats, always in this order: the model learns "the player to my
# right", which means the same thing every hand, rather than "seat 2".
OTHERS = (1, 2, 3)
# riichi, turn, has melds, then what they discarded and the suji of it.
RIICHI, TURN, FUURO, DISCARDED, SUJI = 0, 1, 2, 3, 3 + TILE_KINDS
CONTEXT_WIDTH = 3 + 2 * TILE_KINDS


def _suji(discards):
    """Tiles that are suji of `discards`: three away in the same number suit.

    The oldest read there is. It says nothing about honours, and nothing at all
    about a hand that has not declared riichi, which is the point: it is the
    yardstick the model has to beat, not a feature.
    """
    out = np.zeros(TILE_KINDS, dtype=np.float32)
    for tile in np.flatnonzero(discards):
        if tile >= 27:
            continue
        num = tile % 9
        for step in (-3, 3):
            if 0 <= num + step < 9:
                out[tile + step] = 1.
    return out


class TenpaiDataset(IterableDataset):
    """(what one seat saw, what the other three actually held), sample by sample.

    Sampling is by decision: every point at which a seat may discard, which is
    where the question "how dangerous is this tile" is asked. Consecutive
    decisions in one hand are nearly the same position, so `keep_prob` thins
    them -- the replay is cheap and the encoding cheaper, so it costs less to
    play many games and keep a few moments each than to keep every moment of a
    few.
    """

    def __init__(self, groups, keep_prob=0.25, version=3, seed=0):
        super().__init__()
        self.groups = groups            # [(shard path, row group index), ...]
        self.keep_prob = keep_prob
        self.version = version
        self.seed = seed
        self.channels = obs_shape(version)[0]

    def _my_groups(self):
        """This worker's share, so no two workers read the same games."""
        info = get_worker_info()
        if info is None:
            return list(self.groups)
        return [g for n, g in enumerate(self.groups) if n % info.num_workers == info.id]

    def __iter__(self):
        info = get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0))
        groups = self._my_groups()
        rng.shuffle(groups)

        import pyarrow.parquet as pq
        readers = {}
        for shard, row_group in groups:
            reader = readers.get(shard)
            if reader is None:
                reader = readers[shard] = pq.ParquetFile(shard)
            for chunk in reader.iter_batches(batch_size=64, row_groups=[row_group],
                                             columns=['events']):
                for log in chunk.column('events').to_pylist():
                    if not isinstance(log, str):
                        continue
                    try:
                        yield from self.samples(log, rng)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except BaseException:
                        # One unreadable game is not worth an epoch. Not
                        # `Exception`: a malformed log makes the Rust side
                        # panic, and pyo3 raises that as a PanicException,
                        # which inherits from BaseException and so walked
                        # straight out of an `except Exception` and killed the
                        # run. The main loader learned this in 2ac97f9; this
                        # pipeline was written afterwards and repeated it.
                        continue

    def samples(self, log, rng):
        states = [PlayerState(i) for i in range(4)]
        # Their own discards, which every seat can see and the suji baseline
        # needs; PlayerState keeps no list of them for other seats.
        discarded = np.zeros((4, TILE_KINDS), dtype=np.float32)
        riichi = np.zeros(4, dtype=np.float32)
        turn = 0

        for line in log.splitlines():
            if not line:
                continue
            cans = [state.update(line) for state in states]
            event = json.loads(line)
            kind = event.get('type')
            if kind == 'start_kyoku':
                discarded[:] = 0.
                riichi[:] = 0.
                turn = 0
            elif kind == 'reach':
                riichi[event['actor']] = 1.

            for seat, can in enumerate(cans):
                if not can.can_discard or rng.random() >= self.keep_prob:
                    continue
                yield self.one(states, seat, discarded, riichi, turn)

            if kind == 'dahai':
                actor = event['actor']
                discarded[actor, _kind_of(event['pai'])] = 1.
                if actor == 3:
                    turn += 1

    def one(self, states, seat, discarded, riichi, turn):
        obs, _ = states[seat].encode_obs(self.version, False)
        others = [(seat + k) % 4 for k in OTHERS]

        tenpai = np.zeros(3, dtype=np.float32)
        furiten = np.zeros(3, dtype=np.float32)
        waits = np.zeros((3, TILE_KINDS), dtype=np.float32)
        # Carried for the yardstick, not for the model: riichi, turn, what each
        # opponent has discarded and the suji of it. All of it is in `obs`
        # already, so handing it over separately would prove nothing -- it is
        # here so validation can score the old rules on the same samples.
        context = np.zeros((3, CONTEXT_WIDTH), dtype=np.float32)
        for k, other in enumerate(others):
            state = states[other]
            if state.shanten == 0:
                tenpai[k] = 1.
                waits[k] = np.asarray(state.waits, dtype=np.float32)
            if state.at_furiten:
                furiten[k] = 1.
            context[k, 0] = riichi[other]
            context[k, 1] = turn / 18.
            context[k, 2] = 1. if (state.chis or state.pons or state.minkans) else 0.
            context[k, 3:3 + TILE_KINDS] = discarded[other]
            context[k, 3 + TILE_KINDS:] = _suji(discarded[other])

        # What matters for a discard: is anyone waiting on this tile at all.
        # Trained as its own output rather than combined from the three, which
        # would have to assume they are independent, and they are not: they see
        # the same discards and draw from the same wall.
        any_wait = waits.max(axis=0)
        return obs, tenpai, waits, any_wait, furiten, context


_SUITS = {'m': 0, 'p': 9, 's': 18}
_HONOURS = {'E': 27, 'S': 28, 'W': 29, 'N': 30, 'P': 31, 'F': 32, 'C': 33}


def _kind_of(pai):
    """mjai's tile name as its kind index; a red five counts as a five."""
    if pai in _HONOURS:
        return _HONOURS[pai]
    return _SUITS[pai[1]] + int(pai[0]) - 1
