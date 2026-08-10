"""Network primitives and the numerical tricks the DreamerV3 line depends on.

Three of these are load-bearing and worth stating explicitly, because without them a
world model on this task diverges rather than merely underperforming:

* **symlog / symexp** (Hafner et al. 2023 §"robust predictions"). Rewards here span
  ~0.002 (a step cost) to 64 (a gym badge). Regressing that range directly makes the
  gradient from the badge event swamp everything else for thousands of updates. symlog
  compresses the range without the fixed clipping that would erase the badge signal.
* **two-hot / HL-Gauss return head**. Predicting returns by classification over a fixed
  bin grid rather than by MSE regression. The critic then represents a *distribution*,
  which is what makes it stable under the extremely heavy-tailed returns produced by
  sparse milestone rewards (Imani & White 2018; Farebrother et al. 2024; adopted by
  Simulus 2025).
* **unimix categoricals**. Mixing 1% uniform into every categorical latent keeps the KL
  finite and stops the posterior from collapsing to a one-hot that the prior can never
  match.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# --------------------------------------------------------------------------------------
# Scalar transforms
# --------------------------------------------------------------------------------------


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHot:
    """Two-hot encoding / decoding over a fixed grid of symlog-spaced bins.

    `encode` places unit mass on the two bins bracketing the target, split by linear
    interpolation, so the expectation of the decoded distribution equals the target
    exactly. `decode` takes logits and returns the mean in the original (symexp) space.
    """

    def __init__(self, bins: int, low: float, high: float, device=None) -> None:
        assert bins >= 2
        self.bins = bins
        self.values = torch.linspace(low, high, bins, device=device)

    def to(self, device) -> "TwoHot":
        self.values = self.values.to(device)
        return self

    def encode(self, x: Tensor) -> Tensor:
        """x: (...,) in *symlog* space -> (..., bins) probabilities."""
        x = x.unsqueeze(-1)
        values = self.values.to(x.device)
        below = (values <= x).sum(dim=-1) - 1
        above = self.bins - (values > x).sum(dim=-1)
        below = below.clamp(0, self.bins - 1)
        above = above.clamp(0, self.bins - 1)
        equal = below == above
        # Guard the degenerate case where the target lands exactly on a bin edge.
        d_below = torch.where(equal, torch.ones_like(x[..., 0]), x[..., 0] - values[below])
        d_above = torch.where(equal, torch.ones_like(x[..., 0]), values[above] - x[..., 0])
        total = d_below + d_above
        w_below = d_above / total
        w_above = d_below / total
        return (
            F.one_hot(below, self.bins) * w_below.unsqueeze(-1)
            + F.one_hot(above, self.bins) * w_above.unsqueeze(-1)
        ).float()

    def decode(self, logits: Tensor) -> Tensor:
        """logits: (..., bins) -> scalar in the *original* space."""
        probs = torch.softmax(logits, dim=-1)
        return symexp((probs * self.values.to(logits.device)).sum(dim=-1))

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """Cross-entropy against the two-hot encoding of symlog(target)."""
        with torch.no_grad():
            tgt = self.encode(symlog(target).clamp(self.values[0], self.values[-1]))
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(tgt * log_probs).sum(dim=-1)


# --------------------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------------------


def unimix_logits(logits: Tensor, unimix: float = 0.01) -> Tensor:
    if unimix <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / probs.shape[-1]
    probs = (1.0 - unimix) * probs + unimix * uniform
    return torch.log(probs + 1e-12)


class OneHotCategoricalST(torch.distributions.OneHotCategorical):
    """OneHot categorical with a straight-through gradient estimator."""

    def rsample(self, sample_shape=torch.Size()) -> Tensor:  # type: ignore[override]
        sample = self.sample(sample_shape)
        probs = self.probs
        return sample + (probs - probs.detach())


# --------------------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------------------


def _init_(module: nn.Module) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.trunc_normal_(module.weight, std=0.02, a=-2 * 0.02, b=2 * 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        layers: int,
        *,
        norm: bool = True,
        act=nn.SiLU,
        out_scale: float = 1.0,
    ) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods.append(_init_(nn.Linear(d, hidden, bias=not norm)))
            if norm:
                mods.append(nn.LayerNorm(hidden))
            mods.append(act())
            d = hidden
        head = _init_(nn.Linear(d, out_dim))
        if out_scale != 1.0:
            with torch.no_grad():
                head.weight.mul_(out_scale)
                if head.bias is not None:
                    head.bias.mul_(out_scale)
        mods.append(head)
        self.net = nn.Sequential(*mods)
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# Three stride-2 stages. The Game Boy screen is 144x160, which halves to 72x80 -- and 72
# is divisible by 8 but not by 16, so a fourth stage would not invert cleanly in the
# decoder. Three stages take 72x80 -> 9x10, which the transposed convs reconstruct
# exactly.
CONV_STAGES = 3
CONV_DIVISOR = 2**CONV_STAGES


def check_frame_shape(h: int, w: int) -> None:
    if h % CONV_DIVISOR or w % CONV_DIVISOR:
        raise ValueError(
            f"frame {h}x{w} must be divisible by {CONV_DIVISOR} for the conv "
            "encoder/decoder to be exactly invertible"
        )


class ConvEncoder(nn.Module):
    """(C, H, W) -> flat feature vector. GroupNorm(1, C) == LayerNorm over (C,H,W)."""

    def __init__(self, in_channels: int, depth: int, h: int = 72, w: int = 80) -> None:
        super().__init__()
        check_frame_shape(h, w)
        chans = [in_channels] + [depth * (2**i) for i in range(CONV_STAGES)]
        blocks: list[nn.Module] = []
        for i in range(CONV_STAGES):
            blocks += [
                _init_(nn.Conv2d(chans[i], chans[i + 1], 4, stride=2, padding=1)),
                nn.GroupNorm(1, chans[i + 1]),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*blocks)
        self.out_hw = (h // CONV_DIVISOR, w // CONV_DIVISOR)
        self.out_channels = chans[-1]
        self.out_dim = self.out_channels * self.out_hw[0] * self.out_hw[1]

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) uint8 or float in [0,255]
        x = x.float() / 255.0 - 0.5
        return self.net(x).flatten(1)


class ConvDecoder(nn.Module):
    def __init__(
        self, in_dim: int, out_channels: int, depth: int, h: int = 72, w: int = 80
    ) -> None:
        super().__init__()
        check_frame_shape(h, w)
        self.hw = (h // CONV_DIVISOR, w // CONV_DIVISOR)
        chans = [depth * (2**i) for i in reversed(range(CONV_STAGES))] + [out_channels]
        self.chans = chans[0]
        self.fc = _init_(nn.Linear(in_dim, self.chans * self.hw[0] * self.hw[1]))
        blocks: list[nn.Module] = []
        for i in range(CONV_STAGES):
            blocks.append(
                _init_(nn.ConvTranspose2d(chans[i], chans[i + 1], 4, stride=2, padding=1))
            )
            if i < CONV_STAGES - 1:
                blocks += [nn.GroupNorm(1, chans[i + 1]), nn.SiLU()]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc(x).view(-1, self.chans, *self.hw)
        return self.net(h)  # (B, C, H, W), predicts (pixels/255 - 0.5)


class SymlogMLPHead(nn.Module):
    """Regression head trained with an MSE loss in symlog space."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.net = MLP(in_dim, out_dim, hidden, layers, out_scale=0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def loss(self, x: Tensor, target: Tensor) -> Tensor:
        pred = self.net(x)
        return 0.5 * (pred - symlog(target)).pow(2).sum(dim=-1)

    def predict(self, x: Tensor) -> Tensor:
        return symexp(self.net(x))


class BernoulliHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.net = MLP(in_dim, 1, hidden, layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)

    def loss(self, x: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            self.net(x).squeeze(-1), target.float(), reduction="none"
        )

    def prob(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x).squeeze(-1))


class ReturnNormalizer:
    """Exponentially-decayed percentile scale used to normalise imagined returns.

    DreamerV3 normalises advantages by the running 5th-95th percentile spread rather
    than by standard deviation, because sparse milestone rewards give returns with
    occasional huge outliers that would otherwise crush the effective learning rate.
    """

    def __init__(self, low: float, high: float, decay: float, limit: float) -> None:
        self.low_q, self.high_q = low, high
        self.decay = decay
        self.limit = limit
        self.low = torch.tensor(0.0)
        self.high = torch.tensor(0.0)
        self._initialized = False

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        flat = x.detach().flatten().float()
        lo = torch.quantile(flat, self.low_q / 100.0)
        hi = torch.quantile(flat, self.high_q / 100.0)
        self.low = self.low.to(lo.device)
        self.high = self.high.to(hi.device)
        if not self._initialized:
            self.low, self.high = lo, hi
            self._initialized = True
        else:
            self.low = self.decay * self.low + (1 - self.decay) * lo
            self.high = self.decay * self.high + (1 - self.decay) * hi

    @property
    def scale(self) -> Tensor:
        return torch.clamp(self.high - self.low, min=self.limit)

    def state_dict(self) -> dict:
        return {
            "low": float(self.low),
            "high": float(self.high),
            "init": self._initialized,
        }

    def load_state_dict(self, d: dict) -> None:
        self.low = torch.tensor(float(d.get("low", 0.0)))
        self.high = torch.tensor(float(d.get("high", 0.0)))
        self._initialized = bool(d.get("init", False))


def lambda_return(
    reward: Tensor, value: Tensor, cont: Tensor, gamma: float, lam: float
) -> Tensor:
    """TD(lambda) returns over an imagined rollout.

    Shapes are (T, B). `value[t]` is V(s_t); `reward[t]` is received on the transition
    *into* s_t, matching the DreamerV3 convention. Returns targets for t = 0..T-2.
    """
    assert reward.shape == value.shape == cont.shape
    next_values = value[1:]
    disc = gamma * cont[1:]
    inputs = reward[1:] + disc * next_values * (1 - lam)
    returns: list[Tensor] = []
    acc = value[-1]
    for t in reversed(range(inputs.shape[0])):
        acc = inputs[t] + disc[t] * lam * acc
        returns.append(acc)
    return torch.stack(list(reversed(returns)), dim=0)


def kl_divergence_categorical(
    post_logits: Tensor, prior_logits: Tensor, free: float
) -> Tensor:
    """Per-timestep KL between two factorised categoricals, with free bits.

    Shapes: (..., stoch, classes). Returns (...,) summed over the stoch axis, with the
    free-nats floor applied to the *mean over variables* as in the reference
    implementation.
    """
    post = torch.softmax(post_logits, dim=-1)
    log_post = torch.log_softmax(post_logits, dim=-1)
    log_prior = torch.log_softmax(prior_logits, dim=-1)
    kl = (post * (log_post - log_prior)).sum(dim=-1)  # (..., stoch)
    kl = kl.sum(dim=-1)  # (...)
    return torch.clamp(kl, min=free)


def cosine_embedding(index: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding for the integer subgoal id.

    A learned nn.Embedding would work too, but a fixed basis means a subgoal the LLM has
    never proposed before still lands somewhere sensible in the conditioning space
    instead of at a randomly-initialised vector.
    """
    device = index.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device).float() / max(half, 1)
    )
    ang = index.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb
