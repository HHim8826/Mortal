"""A small net for reading the other three hands.

The observation's 34 positions are the 34 tile kinds, so the answers that are
per tile -- who is waiting on what -- come out of a convolution over those
positions rather than a layer that has to learn the geometry back. The answers
that are per seat come from pooling across them.

Deliberately not weighted against the class imbalance. A tile is a wait about
one time in fifty, and a pos_weight would push the outputs up to match; but
what this model is for is printing a number on a screen, and a number is only
worth printing if 20% means twenty times in a hundred. Plain cross-entropy is
a proper scoring rule, so the calibration comes out of the training rather than
having to be repaired after it.
"""
import torch
from torch import nn

TILE_KINDS = 34


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.out = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.out(x + self.net(x))


class TenpaiNet(nn.Module):
    """Four outputs over one trunk, all of them logits.

    waits    (B, 3, 34)  is seat k waiting on this tile
    any_wait (B, 34)     is any of the three, which is what a discard asks
    tenpai   (B, 3)      is seat k tenpai at all
    furiten  (B, 3)      is seat k furiten, which decides whether tenpai bites
    """

    def __init__(self, in_channels, channels=128, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*(ResBlock(channels) for _ in range(blocks)))
        self.waits = nn.Conv1d(channels, 3, 1)
        self.any_wait = nn.Conv1d(channels, 1, 1)
        # Mean and max over the tiles: how much is going on, and where the
        # strongest single signal is.
        self.seat = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 6),
        )

    def forward(self, obs):
        phi = self.trunk(self.stem(obs))
        pooled = torch.cat((phi.mean(-1), phi.amax(-1)), dim=-1)
        tenpai, furiten = self.seat(pooled).split(3, dim=-1)
        return {
            'waits': self.waits(phi),
            'any_wait': self.any_wait(phi).squeeze(1),
            'tenpai': tenpai,
            'furiten': furiten,
        }


def losses(out, batch):
    """One binary cross-entropy per head, and their sum.

    The waits head is weighted down: it carries 102 numbers to the seat heads'
    three, and left alone it would decide the trunk on its own.
    """
    bce = nn.functional.binary_cross_entropy_with_logits
    parts = {
        'waits': bce(out['waits'], batch['waits']),
        'any_wait': bce(out['any_wait'], batch['any_wait']),
        'tenpai': bce(out['tenpai'], batch['tenpai']),
        'furiten': bce(out['furiten'], batch['furiten']),
    }
    total = (0.3 * parts['waits'] + parts['any_wait']
             + parts['tenpai'] + 0.3 * parts['furiten'])
    return total, parts
