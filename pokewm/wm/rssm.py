"""Recurrent State-Space Model with discrete latents.

The state is a pair (h, z):

* h_t = f(h_{t-1}, z_{t-1}, a_{t-1})  -- deterministic GRU path, carries long-range
  context (which town you are in, whether you already beat Brock).
* z_t ~ q(z_t | h_t, x_t)             -- stochastic posterior, 32 categoricals of 32
  classes, absorbs the parts of the frame the deterministic path cannot predict.

Imagination uses the prior p(z_t | h_t) instead of the posterior, which is precisely
what lets the actor-critic be trained on rollouts that never touch the emulator.

Why discrete latents for a Game Boy game specifically: the observable state is genuinely
discrete (tile grids, menu indices, HP integers), and categorical posteriors do not
suffer the posterior collapse that Gaussian latents show on sharply multi-modal
transitions like a screen transition or a battle intro. This is the same argument
Hafner et al. make for Atari, and it applies more strongly here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .nets import MLP, OneHotCategoricalST, kl_divergence_categorical, unimix_logits


@dataclass
class RSSMState:
    deter: Tensor  # (B, deter)
    logits: Tensor  # (B, stoch, classes)
    stoch: Tensor  # (B, stoch, classes) one-hot (straight-through)

    def feature(self) -> Tensor:
        return torch.cat([self.deter, self.stoch.flatten(1)], dim=-1)

    def detach(self) -> "RSSMState":
        return RSSMState(self.deter.detach(), self.logits.detach(), self.stoch.detach())

    def __getitem__(self, idx) -> "RSSMState":
        return RSSMState(self.deter[idx], self.logits[idx], self.stoch[idx])


class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter: int = 1024,
        stoch: int = 32,
        classes: int = 32,
        hidden: int = 512,
        layers: int = 1,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.deter_size = deter
        self.stoch_size = stoch
        self.classes = classes
        self.unimix = unimix
        self.action_dim = action_dim
        self.feat_dim = deter + stoch * classes

        # (z_{t-1}, a_{t-1}) -> GRU input
        self.pre_gru = MLP(stoch * classes + action_dim, hidden, hidden, layers)
        self.gru = nn.GRUCell(hidden, deter)
        # h_t -> prior logits
        self.prior_net = MLP(deter, stoch * classes, hidden, layers)
        # (h_t, embed_t) -> posterior logits
        self.post_net = MLP(deter + embed_dim, stoch * classes, hidden, layers)

    # ------------------------------------------------------------------ helpers

    def initial(self, batch: int, device) -> RSSMState:
        deter = torch.zeros(batch, self.deter_size, device=device)
        logits = torch.zeros(batch, self.stoch_size, self.classes, device=device)
        stoch = torch.zeros(batch, self.stoch_size, self.classes, device=device)
        return RSSMState(deter, logits, stoch)

    def _sample(self, logits: Tensor, sample: bool = True) -> Tensor:
        logits = unimix_logits(logits, self.unimix)
        dist = OneHotCategoricalST(logits=logits)
        if sample:
            return dist.rsample()
        # Deterministic readout: argmax with a straight-through gradient.
        probs = dist.probs
        idx = probs.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
        return hard + probs - probs.detach()

    # ------------------------------------------------------------------ dynamics

    def img_step(self, prev: RSSMState, action: Tensor, sample: bool = True) -> RSSMState:
        """One step of the *prior*: no observation used. This is imagination."""
        x = torch.cat([prev.stoch.flatten(1), action], dim=-1)
        deter = self.gru(self.pre_gru(x), prev.deter)
        logits = self.prior_net(deter).view(-1, self.stoch_size, self.classes)
        return RSSMState(deter, logits, self._sample(logits, sample))

    def obs_step(
        self, prev: RSSMState, action: Tensor, embed: Tensor, sample: bool = True
    ) -> tuple[RSSMState, RSSMState]:
        """One filtering step. Returns (posterior, prior)."""
        prior = self.img_step(prev, action, sample)
        logits = self.post_net(torch.cat([prior.deter, embed], dim=-1)).view(
            -1, self.stoch_size, self.classes
        )
        post = RSSMState(prior.deter, logits, self._sample(logits, sample))
        return post, prior

    # ------------------------------------------------------------------ sequences

    def observe(
        self,
        embed: Tensor,  # (B, T, embed)
        action: Tensor,  # (B, T, action_dim)
        is_first: Tensor,  # (B, T) float/bool
        state: RSSMState | None = None,
    ) -> tuple[RSSMState, RSSMState]:
        """Filter a batch of sequences. Returns stacked (post, prior), each (B, T, ...).

        `is_first` resets the recurrent state *inside* the sequence, so a replay window
        may straddle an episode boundary without leaking state across it.
        """
        b, t = embed.shape[:2]
        device = embed.device
        state = state or self.initial(b, device)
        posts, priors = [], []
        for i in range(t):
            mask = (1.0 - is_first[:, i].float()).view(b, 1)
            state = RSSMState(
                state.deter * mask,
                state.logits * mask.unsqueeze(-1),
                state.stoch * mask.unsqueeze(-1),
            )
            act = action[:, i] * mask
            post, prior = self.obs_step(state, act, embed[:, i])
            posts.append(post)
            priors.append(prior)
            state = post
        return _stack(posts), _stack(priors)

    def imagine(
        self, state: RSSMState, policy, horizon: int
    ) -> tuple[RSSMState, Tensor, Tensor]:
        """Roll the prior forward under `policy`.

        `policy(feature) -> (action_onehot, action_logprob)`. Returns states stacked as
        (H+1, B, ...) including the starting state, plus actions and log-probs (H, B).
        """
        states = [state]
        actions, logps = [], []
        s = state
        for _ in range(horizon):
            feat = s.feature()
            act, logp = policy(feat)
            s = self.img_step(s, act)
            states.append(s)
            actions.append(act)
            logps.append(logp)
        return (
            _stack(states, dim=0),
            torch.stack(actions, dim=0),
            torch.stack(logps, dim=0),
        )

    # ------------------------------------------------------------------ losses

    def kl_loss(
        self,
        post: RSSMState,
        prior: RSSMState,
        free: float,
        dyn_scale: float,
        rep_scale: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """KL balancing (DreamerV3 eq. 5).

        Two separately-scaled terms: the *dynamics* loss pulls the prior toward the
        (stopped) posterior, teaching the model to predict; the *representation* loss
        pulls the posterior toward the (stopped) prior, keeping the encoder from
        inventing information the dynamics can never reproduce. Weighting dynamics
        higher (0.5 vs 0.1) is what makes imagination rollouts stay on-manifold.
        """
        dyn = kl_divergence_categorical(post.logits.detach(), prior.logits, free)
        rep = kl_divergence_categorical(post.logits, prior.logits.detach(), free)
        return dyn_scale * dyn + rep_scale * rep, dyn, rep


def _stack(states: list[RSSMState], dim: int = 1) -> RSSMState:
    return RSSMState(
        torch.stack([s.deter for s in states], dim=dim),
        torch.stack([s.logits for s in states], dim=dim),
        torch.stack([s.stoch for s in states], dim=dim),
    )


def flatten_state(state: RSSMState) -> RSSMState:
    """(B, T, ...) -> (B*T, ...) for feeding heads."""
    b, t = state.deter.shape[:2]
    return RSSMState(
        state.deter.reshape(b * t, -1),
        state.logits.reshape(b * t, state.logits.shape[-2], state.logits.shape[-1]),
        state.stoch.reshape(b * t, state.stoch.shape[-2], state.stoch.shape[-1]),
    )
