"""What the policy actually played, and with what probability.

A policy gradient divides by the probability the behaviour policy gave the
action it took. Getting that from the run rather than recomputing it later is
the difference between an importance ratio and a guess: by the time a batch is
trained on, the weights have moved, the sampling rule may have changed, and a
rule-based guard may have replaced the action that was sampled.

Nothing extra is written to do this. Every decision the engine evaluates
already carries its own logits in the game log's `meta` (`q_values`, compact
over the legal actions, with `mask_bits` saying which those were), and the
sampling rule is a property of the run. This module pairs those with the
instances libriichi decodes from the same log and says, for each one, what the
behaviour probability was -- or why it cannot be known.

Three kinds of decision have no logits in the log:

    forced      `enable_quick_eval` answers a turn whose only legal action is
                one discard without calling the model at all. The probability
                is 1 whatever the policy is.
    declined    a call the model was offered and turned down. mjai has no
                event for passing, so a decision the model really made -- and
                one of the decisions that matter most -- leaves nothing behind
                but the instance itself.
    ended       taking the win, or the abortive draw. The board writes its own
                `hora` or `ryukyoku` event, with the scores it worked out, and
                the reaction's `meta` does not survive that.
    kan_select  a kan with more than one candidate tile is a second query to
                the model, and the dataset makes a second instance of it, but
                the log carries one event and one `meta` for the pair. The
                choice was sampled and its logits are not written down.
    unknown     anything else: a decision whose mask does not line up with the
                next unused `meta`. That is a silent mismatch between the log
                and the decode, and it is exactly what must not pass quietly.

`guard` marks a fourth kind, which does have logits: the rule-based agari
guard replaces action 43 with the best of the rest, so two sampled actions
lead to the one that was played and its probability is the sum of theirs.
Whether the guard was in force is not in the log, so the probability here is
the one the sampling rule gives on its own, a lower bound, and the decision is
flagged rather than quietly trusted.
"""
import gzip
import json
from dataclasses import dataclass
from os import path

import numpy as np

AGARI = 43          # the action the rule-based guard overrides
KAN = 42            # the action a kan-select instance follows
NONE = 45           # declining a call, which leaves no event in the log
ENDS_KYOKU = (43, 44)   # agari and the abortive draw, rewritten by the board


@dataclass(frozen=True)
class Sampler:
    """The rule a worker picks actions with, from its `train_play` profile.

    `MortalEngine` takes the best action with probability `1 - epsilon`, and
    otherwise samples a temperature-`temperature` softmax of the same values,
    optionally cut down to the smallest set of actions holding `top_p` of the
    mass. The two branches together are the behaviour policy; a ratio wants
    that mixture, not the branch that happened to fire.
    """

    epsilon: float = 0.
    temperature: float = 1.
    top_p: float = 1.

    def probs(self, logits):
        """The probability of each of the legal actions, in the order given."""
        logits = np.asarray(logits, dtype=np.float64)
        out = np.zeros(len(logits))
        out[logits.argmax()] = 1. - self.epsilon
        if self.epsilon <= 0:
            return out

        scaled = logits / self.temperature
        soft = np.exp(scaled - scaled.max())
        soft /= soft.sum()
        if self.top_p < 1:
            # The same nucleus as engine.sample_top_p: sort by probability and
            # drop every action whose predecessors already hold `top_p`.
            order = np.argsort(-soft, kind='stable')
            ranked = soft[order]
            keep = ranked.cumsum() - ranked <= self.top_p
            soft = np.zeros_like(soft)
            soft[order[keep]] = ranked[keep]
            soft /= soft.sum()
        return out + self.epsilon * soft


def metas_of(log, seat):
    """Every `meta` this seat's own events carry, in the order they were made.

    `log` is the text of an mjai log, one JSON event per line, as the arena
    writes it.
    """
    out = []
    for line in log.splitlines():
        if not line:
            continue
        event = json.loads(line)
        meta = event.get('meta')
        if meta and event.get('actor') == seat and meta.get('mask_bits') is not None:
            out.append(meta)
    return out


def read_log(source):
    """The text of a log, given either the text itself or a path to a .json.gz."""
    if isinstance(source, str) and source.endswith('.json.gz'):
        with gzip.open(source, 'rt', encoding='utf-8') as f:
            return f.read()
    return source


def mask_bits_of(mask):
    """A boolean legal-action mask as the `mask_bits` integer the log stores.

    Bit i is action i, which is how `gen_meta` builds it in libriichi.
    """
    bits = 0
    for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
        bits |= 1 << int(i)
    return bits


def behaviour_of(actions, masks, log, seat, sampler):
    """(probabilities, kinds) for one game's instances, in their order.

    `actions` and `masks` are what `GameplayLoader` decoded for `seat` from the
    same log the `meta` comes from. A probability is nan where `kind` is not
    'sampled' or 'forced'.
    """
    metas = metas_of(read_log(log), seat)
    probs = np.full(len(actions), np.nan)
    kinds = []
    used = 0
    for i, (action, mask) in enumerate(zip(actions, masks)):
        bits = mask_bits_of(mask)
        legal = int(np.count_nonzero(mask))
        meta = metas[used] if used < len(metas) else None
        # A decline and a kyoku-ending action never have a `meta` of their
        # own, so they are named before one is tried. Tried first, a call
        # declined and then taken on the very next discard -- the same offer,
        # the same mask -- gave the decline the taken call's `meta`, and the
        # call that was actually made came out 'unknown'.
        if action == NONE:
            kinds.append('declined')
        elif action in ENDS_KYOKU:
            kinds.append('ended')
        elif meta is not None and meta['mask_bits'] == bits:
            used += 1
            values = meta['q_values']
            if len(values) != legal:
                kinds.append('unknown')
                continue
            # `q_values` is compact over the legal actions, in action order,
            # so the action's place in it is how many legal actions precede it.
            where = int(np.count_nonzero(mask[:action]))
            if not mask[action]:
                kinds.append('unknown')
                continue
            probs[i] = sampler.probs(values)[where]
            kinds.append('guard' if mask[AGARI] and action != AGARI else 'sampled')
        elif legal == 1 and mask[action]:
            probs[i] = 1.
            kinds.append('forced')
        elif i > 0 and actions[i - 1] == KAN:
            kinds.append('kan_select')
        else:
            kinds.append('unknown')
    return probs, np.array(kinds)


def aligned_metas(actions, masks, log, seat):
    """The `meta` belonging to each instance, or None where there is none.

    `behaviour_of` says what the probability of the action taken was;
    this says what the whole distribution over that decision's legal actions
    was, which is what picking a counterfactual to force needs. Same cursor,
    same rule -- the alignment between a log's events and the instances
    libriichi decodes from it is subtle enough that it should be written down
    once, and `test_rollout` checks the two agree on which instances have one.
    """
    metas = metas_of(read_log(log), seat)
    out = []
    used = 0
    for action, mask in zip(actions, masks):
        meta = metas[used] if used < len(metas) else None
        if action == NONE or action in ENDS_KYOKU:
            # Never one of their own; see behaviour_of.
            out.append(None)
        elif meta is not None and meta['mask_bits'] == mask_bits_of(mask):
            used += 1
            legal = int(np.count_nonzero(mask))
            # A length that does not match the mask means the pairing has
            # slipped, and a distribution read off it would be wrong rather
            # than missing.
            out.append(meta if len(meta['q_values']) == legal else None)
        else:
            out.append(None)
    return out


def version_in(file):
    """The parameter version a replay file's name carries, or None.

    server.py names what a worker submits `<submission>_v<version>_<game>`, so
    a batch can be paired with the weights that played it and the behaviour
    logits recomputed from them -- including for the decisions the log cannot
    carry, such as declining a call. `check_against_meta` is how that pairing
    is proven right rather than assumed.
    """
    name = path.basename(str(file))
    for part in name.split('_'):
        if part.startswith('v') and part[1:].isdigit():
            return int(part[1:])
    return None


def check_against_meta(logits, actions, masks, log, seat):
    """Compare recomputed logits with the ones the log recorded.

    `logits` is the full 46-wide output for each instance, as a rerun of the
    behaviour weights gives it. Returns (compared, largest difference); a
    mismatch means the weights are not the ones that played, and a ratio built
    on them would be wrong without ever looking wrong.
    """
    metas = metas_of(read_log(log), seat)
    _, kinds = behaviour_of(actions, masks, log, seat, Sampler())
    compared = 0
    worst = 0.
    used = 0
    for i, kind in enumerate(kinds):
        if kind not in ('sampled', 'guard'):
            continue
        meta = metas[used]
        used += 1
        mask = np.asarray(masks[i], dtype=bool)
        mine = np.asarray(logits[i], dtype=np.float64)[mask]
        worst = max(worst, float(np.abs(mine - np.asarray(meta['q_values'])).max()))
        compared += 1
    return compared, worst


def summarize(kinds):
    """How many decisions of each kind, for a log line that says what was kept."""
    names, counts = np.unique(np.asarray(kinds), return_counts=True)
    return dict(zip(names.tolist(), counts.tolist()))
