import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from typing import *
from functools import partial
from itertools import permutations
from libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE, GRP_SIZE

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16, actv_builder=nn.ReLU, bias=True):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=bias),
            actv_builder(),
            nn.Linear(channels // ratio, channels, bias=bias),
        )
        if bias:
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.constant_(mod.bias, 0)

    def forward(self, x: Tensor):
        avg_out = self.shared_mlp(x.mean(-1))
        max_out = self.shared_mlp(x.amax(-1))
        weight = (avg_out + max_out).sigmoid()
        x = weight.unsqueeze(-1) * x
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        *,
        norm_builder = nn.Identity,
        actv_builder = nn.ReLU,
        pre_actv = False,
    ):
        super().__init__()
        self.pre_actv = pre_actv

        if pre_actv:
            self.res_unit = nn.Sequential(
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            )
        else:
            self.res_unit = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
            )
            self.actv = actv_builder()
        self.ca = ChannelAttention(channels, actv_builder=actv_builder, bias=True)

    def forward(self, x):
        out = self.res_unit(x)
        out = self.ca(out)
        out = out + x
        if not self.pre_actv:
            out = self.actv(out)
        return out

class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels,
        num_blocks,
        *,
        norm_builder = nn.Identity,
        actv_builder = nn.ReLU,
        pre_actv = False,
    ):
        super().__init__()

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(
                conv_channels,
                norm_builder = norm_builder,
                actv_builder = actv_builder,
                pre_actv = pre_actv,
            ))

        layers = [nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False)]
        if pre_actv:
            layers += [*blocks, norm_builder(), actv_builder()]
        else:
            layers += [norm_builder(), actv_builder(), *blocks]
        layers += [
            nn.Conv1d(conv_channels, 32, kernel_size=3, padding=1),
            actv_builder(),
            nn.Flatten(),
            nn.Linear(32 * 34, 1024),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Brain(nn.Module):
    def __init__(self, *, conv_channels, num_blocks, is_oracle=False, version=1):
        super().__init__()
        self.is_oracle = is_oracle
        self.version = version

        in_channels = obs_shape(version)[0]
        if is_oracle:
            in_channels += oracle_obs_shape(version)[0]

        norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01)
        actv_builder = partial(nn.Mish, inplace=True)
        pre_actv = True

        match version:
            case 1:
                actv_builder = partial(nn.ReLU, inplace=True)
                pre_actv = False
                self.latent_net = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                )
                self.mu_head = nn.Linear(512, 512)
                self.logsig_head = nn.Linear(512, 512)
            case 2:
                pass
            case 3 | 4:
                norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01, eps=1e-3)
            case _:
                raise ValueError(f'Unexpected version {self.version}')

        self.encoder = ResNet(
            in_channels = in_channels,
            conv_channels = conv_channels,
            num_blocks = num_blocks,
            norm_builder = norm_builder,
            actv_builder = actv_builder,
            pre_actv = pre_actv,
        )
        self.actv = actv_builder()

        # always use EMA or CMA when True
        self._freeze_bn = False
        # Modules held still by `freeze_trunk`. Kept so `train()` can put their
        # BatchNorm back into eval every time the model is switched to train,
        # which is the only way a frozen block stays frozen: `requires_grad_`
        # stops the weights moving, and running statistics are not weights.
        self._frozen_mods = []

    def forward(self, obs: Tensor, invisible_obs: Optional[Tensor] = None) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.is_oracle:
            assert invisible_obs is not None
            obs = torch.cat((obs, invisible_obs), dim=1)
        phi = self.encoder(obs)

        match self.version:
            case 1:
                latent_out = self.latent_net(phi)
                mu = self.mu_head(latent_out)
                logsig = self.logsig_head(latent_out)
                return mu, logsig
            case 2 | 3 | 4:
                return self.actv(phi)
            case _:
                raise ValueError(f'Unexpected version {self.version}')

    def train(self, mode=True):
        super().train(mode)
        if self._freeze_bn:
            for mod in self.modules():
                if isinstance(mod, nn.BatchNorm1d):
                    mod.eval()
                    # I don't think this benefits
                    # module.requires_grad_(False)
        for mod in self._frozen_mods:
            mod.eval()
        return self

    def freeze_trunk(self, trainable_blocks):
        """Hold every residual block but the last `trainable_blocks` still.

        The online phase trains the whole trunk on a Monte-Carlo regression of
        Q onto the kyoku it just played. Nothing in that objective maintains a
        value for an action the policy has stopped taking, so what it can do is
        contract onto what it already does -- which is what it did, folding
        further along one axis over 80,000 steps. Holding the early trunk still
        leaves the features that were learned from 1.6M human games where they
        are, and lets self-play move only the part that reads them.

        0, or anything at or above the block count, trains everything and is
        the behaviour this had before. Returns (blocks frozen, parameters
        frozen) so a run can log what it actually did rather than what it asked
        for.
        """
        self._frozen_mods = self.trunk_frozen_by(trainable_blocks)
        # Everything back first, so going from 4 trainable blocks to 6 thaws
        # the two between rather than leaving them held from last time.
        self.encoder.requires_grad_(True)
        for mod in self._frozen_mods:
            mod.requires_grad_(False)
        self.train(self.training)
        blocks = sum(isinstance(m, ResBlock) for m in self._frozen_mods)
        return blocks, sum(p.numel() for m in self._frozen_mods for p in m.parameters())

    def trunk_frozen_by(self, trainable_blocks):
        """The encoder modules `freeze_trunk(trainable_blocks)` holds, without holding them.

        Its own method for resuming: a checkpoint saved under one setting and
        read under another has to know which parameters the saved optimizer
        was carrying moments for, and that follows from the setting alone.
        """
        mods = list(self.encoder.net)
        at = [i for i, m in enumerate(mods) if isinstance(m, ResBlock)]
        if not trainable_blocks or trainable_blocks >= len(at):
            return []
        # Everything up to and including the last block being frozen: the stem
        # convolution, every earlier block, and for a post-activation layout
        # the norm and activation that sit between them.
        return mods[:at[len(at) - trainable_blocks - 1] + 1]

    def reset_running_stats(self):
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.reset_running_stats()

    def freeze_bn(self, value: bool):
        self._freeze_bn = value
        return self.train(self.training)

class AuxNet(nn.Module):
    def __init__(self, dims=None):
        super().__init__()
        self.dims = dims
        self.net = nn.Linear(1024, sum(dims), bias=False)

    def forward(self, x):
        return self.net(x).split(self.dims, dim=-1)

class DQN(nn.Module):
    def __init__(self, *, version=1):
        super().__init__()
        self.version = version
        match version:
            case 1:
                self.v_head = nn.Linear(512, 1)
                self.a_head = nn.Linear(512, ACTION_SPACE)
            case 2 | 3:
                hidden_size = 512 if version == 2 else 256
                self.v_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, 1),
                )
                self.a_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, ACTION_SPACE),
                )
            case 4:
                self.net = nn.Linear(1024, 1 + ACTION_SPACE)
                nn.init.constant_(self.net.bias, 0)

    def forward(self, phi, mask):
        if self.version == 4:
            v, a = self.net(phi).split((1, ACTION_SPACE), dim=-1)
        else:
            v = self.v_head(phi)
            a = self.a_head(phi)
        a_sum = a.masked_fill(~mask, 0.).sum(-1, keepdim=True)
        mask_sum = mask.sum(-1, keepdim=True)
        a_mean = a_sum / mask_sum
        q = (v + a - a_mean).masked_fill(~mask, -torch.inf)
        return q

class PolicyHead(nn.Module):
    """What to play, as logits over the legal actions.

    Kept apart from `DQN` on purpose. Mortal's Q carries three jobs at once:
    it is the estimate of the return, the thing the CQL term shapes into
    human-action logits, and, divided by a temperature nobody calibrated, the
    scale exploration is drawn on. A policy gradient needs a distribution that
    is free to move without dragging a value estimate with it, so this owns the
    distribution and `RankCritic` owns the value.

    The same shape as v4's Q head, a single linear map of the 1024 trunk
    features, and for a reason: masked softmax of a v4 Q is softmax of its
    advantage -- the state value and the mean cancel -- so a linear policy can
    reproduce the teacher it is distilled from exactly, and any gap that
    remains is optimization, not capacity.
    """

    def __init__(self, *, version=4):
        super().__init__()
        if version != 4:
            raise ValueError(f'Unexpected version {version}')
        self.version = version
        self.net = nn.Linear(1024, ACTION_SPACE)
        nn.init.constant_(self.net.bias, 0)

    def forward(self, phi: Tensor, mask: Tensor) -> Tensor:
        return self.net(phi).masked_fill(~mask, -torch.inf)

def head_for(kind, *, version=4):
    """The head a run plays with: the dueling Q, or the policy distilled from it.

    Both answer the same call -- logits over the legal actions -- so
    `MortalEngine` plays either without knowing which it holds, and a config
    key is enough to decide. The Q head is what every run before the policy
    used, so it stays the default.
    """
    match kind:
        case 'dqn':
            return DQN(version=version)
        case 'policy':
            return PolicyHead(version=version)
    raise ValueError(f'unknown head {kind!r}; expected dqn or policy')

def head_state(kind, state):
    """Where that head's weights sit in a checkpoint."""
    match kind:
        case 'dqn':
            return state['current_dqn']
        case 'policy':
            if 'policy' not in state:
                raise KeyError('this checkpoint has no policy head; '
                               'train one with train_policy.py')
            return state['policy']
    raise ValueError(f'unknown head {kind!r}; expected dqn or policy')

class RankCritic(nn.Module):
    """How the hanchan ends, as the probability of each final placement.

    The whole game, not the kyoku: the thing being played for is the placement
    at the end, and a distribution over the four of them carries what a single
    number cannot -- that a hand which avoids fourth is worth more than its
    average says. `value` turns it back into the expected placement utility
    when a baseline needs one number.
    """

    def __init__(self, *, pts=(3., 1.5, 0., -4.5)):
        super().__init__()
        self.net = nn.Linear(1024, 4)
        nn.init.constant_(self.net.bias, 0)
        self.register_buffer('pts', torch.tensor(pts, dtype=torch.float32))

    def forward(self, phi: Tensor) -> Tensor:
        return self.net(phi)

    def value(self, logits: Tensor) -> Tensor:
        return logits.float().softmax(-1) @ self.pts

class KyokuValue(nn.Module):
    """What this kyoku is still worth, in expected placement utility.

    The critic a policy gradient needs is one that predicts the same return the
    advantage is taken against. `RankCritic` predicts how the hanchan ends,
    which makes every decision in it share one number: 248 decisions, one
    outcome, and a gradient that measured out as pure noise -- cosine between
    consecutive batches 0.003, a policy that walked 0.0068 of KL in 106,000
    steps and played exactly as well at the end.

    This predicts the per-kyoku return instead, the GRP's change in expected
    placement utility over the kyoku the decision is in. About thirty decisions
    share one of those rather than 248.

    It starts as the teacher's own value: a v4 Q head is a dueling
    `Linear(1024, 1 + ACTION_SPACE)` whose first row is exactly this, V(s), and
    it was trained on exactly this return. So the baseline is useful from the
    first step rather than after a warm-up.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1024, 1)
        nn.init.constant_(self.net.bias, 0)

    @classmethod
    def from_dueling(cls, dqn):
        """The value stream of a v4 Q head, which is already this function."""
        if getattr(dqn, 'version', None) != 4:
            raise ValueError('only a v4 Q head carries a linear value stream')
        head = cls()
        with torch.no_grad():
            head.net.weight.copy_(dqn.net.weight[:1])
            head.net.bias.copy_(dqn.net.bias[:1])
        return head

    def forward(self, phi: Tensor) -> Tensor:
        return self.net(phi).squeeze(-1)

class GRP(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(input_size=GRP_SIZE, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * num_layers, hidden_size * num_layers),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * num_layers, 24),
        )
        for mod in self.modules():
            mod.to(torch.float64)

        # perms are the permutations of all possible rank-by-player result
        perms = torch.tensor(list(permutations(range(4))))
        perms_t = perms.transpose(0, 1)
        self.register_buffer('perms', perms)     # (24, 4)
        self.register_buffer('perms_t', perms_t) # (4, 24)

    # input: [grand_kyoku, honba, kyotaku, s[0], s[1], s[2], s[3]]
    # grand_kyoku: E1 = 0, S4 = 7, W4 = 11
    # s is 2.5 at E1
    # s[0] is score of player id 0
    def forward(self, inputs: List[Tensor]):
        lengths = torch.tensor([t.shape[0] for t in inputs], dtype=torch.int64)
        inputs = pad_sequence(inputs, batch_first=True)
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        return self.forward_packed(packed_inputs)

    def forward_packed(self, packed_inputs):
        _, state = self.rnn(packed_inputs)
        state = state.transpose(0, 1).flatten(1)
        logits = self.fc(state)
        return logits

    # (N, 24) -> (N, player, rank_prob)
    def calc_matrix(self, logits: Tensor):
        batch_size = logits.shape[0]
        probs = logits.softmax(-1)
        matrix = torch.zeros(batch_size, 4, 4, dtype=probs.dtype)
        for player in range(4):
            for rank in range(4):
                cond = self.perms_t[player] == rank
                matrix[:, player, rank] = probs[:, cond].sum(-1)
        return matrix

    # (N, 4) -> (N)
    def get_label(self, rank_by_player: Tensor):
        batch_size = rank_by_player.shape[0]
        perms = self.perms.expand(batch_size, -1, -1).transpose(0, 1)
        mappings = (perms == rank_by_player).all(-1).nonzero()

        labels = torch.zeros(batch_size, dtype=torch.int64, device=mappings.device)
        labels[mappings[:, 1]] = mappings[:, 0]
        return labels
