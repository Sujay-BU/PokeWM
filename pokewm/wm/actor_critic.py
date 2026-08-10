"""Actor and critic, trained entirely on latent imagination.

No gradient in this file ever touches the emulator. The actor maximises the
lambda-return of trajectories rolled out through the RSSM prior, which is what buys the
sample efficiency: one replay batch of 16x64 real steps yields 16*64 imagination starts
x 15 imagined steps = ~15k policy-gradient samples per update.

Discrete actions here use REINFORCE with a learned baseline rather than the reparam
("dynamics backprop") path. Straight-through gradients through 15 chained categorical
samples are extremely high-variance, and DreamerV3 reports the same choice for discrete
control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ActorCriticConfig, WorldModelConfig
from .nets import MLP, ReturnNormalizer, TwoHot, lambda_return, symlog, unimix_logits


@dataclass
class ACLosses:
    actor: Tensor
    critic: Tensor
    alpha_loss: Tensor
    alpha: Tensor
    entropy: Tensor
    value_mean: Tensor
    return_mean: Tensor
    adv_std: Tensor
    imag_reward: Tensor


class Actor(nn.Module):
    def __init__(self, feat_dim: int, num_actions: int, cfg: ActorCriticConfig) -> None:
        super().__init__()
        self.net = MLP(feat_dim, num_actions, cfg.hidden, cfg.layers, out_scale=0.01)
        self.num_actions = num_actions

    def logits(self, feat: Tensor) -> Tensor:
        return unimix_logits(self.net(feat), 0.01)

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        """Sample an action. Returns (one-hot action, log-prob)."""
        logits = self.logits(feat)
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        return F.one_hot(idx, self.num_actions).float(), dist.log_prob(idx)

    def act(self, feat: Tensor, greedy: bool = False) -> tuple[Tensor, Tensor]:
        logits = self.logits(feat)
        dist = torch.distributions.Categorical(logits=logits)
        idx = logits.argmax(dim=-1) if greedy else dist.sample()
        return idx, dist.log_prob(idx)

    def entropy(self, feat: Tensor) -> Tensor:
        return torch.distributions.Categorical(logits=self.logits(feat)).entropy()


class Critic(nn.Module):
    """Distributional value head over a two-hot bin grid, plus an EMA slow copy.

    The slow copy provides the bootstrap target. Without it, the critic chases its own
    predictions over a 15-step imagination horizon and the value estimate drifts upward
    without bound -- the classic deadly triad, which sparse milestone rewards make much
    easier to trigger.
    """

    def __init__(
        self, feat_dim: int, ac_cfg: ActorCriticConfig, wm_cfg: WorldModelConfig
    ) -> None:
        super().__init__()
        self.net = MLP(feat_dim, wm_cfg.bins, ac_cfg.hidden, ac_cfg.layers, out_scale=0.0)
        self.slow = MLP(feat_dim, wm_cfg.bins, ac_cfg.hidden, ac_cfg.layers, out_scale=0.0)
        self.slow.load_state_dict(self.net.state_dict())
        for p in self.slow.parameters():
            p.requires_grad_(False)
        self.twohot = TwoHot(wm_cfg.bins, wm_cfg.bin_low, wm_cfg.bin_high)
        self._updates = 0
        self.cfg = ac_cfg

    def forward(self, feat: Tensor) -> Tensor:
        return self.net(feat)

    def value(self, feat: Tensor) -> Tensor:
        return self.twohot.to(feat.device).decode(self.net(feat))

    @torch.no_grad()
    def slow_value(self, feat: Tensor) -> Tensor:
        return self.twohot.to(feat.device).decode(self.slow(feat))

    @torch.no_grad()
    def update_slow(self) -> None:
        self._updates += 1
        if self._updates % max(self.cfg.slow_critic_update, 1) != 0:
            return
        tau = self.cfg.slow_critic_fraction
        for p_slow, p in zip(self.slow.parameters(), self.net.parameters()):
            p_slow.data.lerp_(p.data, tau)


class ImaginationActorCritic(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_actions: int,
        ac_cfg: ActorCriticConfig,
        wm_cfg: WorldModelConfig,
    ) -> None:
        super().__init__()
        self.actor = Actor(feat_dim, num_actions, ac_cfg)
        self.critic = Critic(feat_dim, ac_cfg, wm_cfg)
        self.cfg = ac_cfg
        self.wm_cfg = wm_cfg
        self.ret_norm = ReturnNormalizer(
            ac_cfg.return_norm_low,
            ac_cfg.return_norm_high,
            ac_cfg.return_norm_decay,
            ac_cfg.return_norm_limit,
        )
        # Entropy coefficient as a *dual variable* rather than a constant (SAC-style
        # automatic temperature, Haarnoja et al. 2018).
        #
        # The right constant depends on the advantage scale, which moves by two orders
        # of magnitude across a run here (adv_std 0.006 -> 0.3). Four hand-picked values
        # produced two policy collapses and one policy pinned at exactly ln|A|; both
        # failure modes stall the game, because a deterministic policy never stumbles
        # into a scripted button press and a uniform one never commits to a route.
        # Tracking a target entropy removes the choice.
        self.target_entropy = float(
            ac_cfg.entropy_target_frac * np.log(num_actions)
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(float(np.log(max(ac_cfg.entropy, 1e-8))))
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy coefficient, clamped to a sane band."""
        if not self.cfg.entropy_adaptive:
            return torch.tensor(self.cfg.entropy, device=self.log_alpha.device)
        return self.log_alpha.exp().clamp(self.cfg.entropy_min, self.cfg.entropy_max)

    @torch.no_grad()
    def clamp_log_alpha(self) -> None:
        """Clamp the *dual variable itself*, not just its exponential.

        `alpha` clamps the exponential, but the dual loss is linear in `log_alpha`, so
        its gradient is the entropy gap regardless of whether alpha is saturated. Left
        alone, a long stretch below target winds `log_alpha` far past the ceiling and
        the coefficient then takes just as long to come back down -- integral windup,
        with a policy stuck at uniform for the duration. Clamping the parameter after
        each step bounds the recovery time to a few hundred updates.
        """
        if not self.cfg.entropy_adaptive:
            return
        self.log_alpha.data.clamp_(
            float(np.log(self.cfg.entropy_min)), float(np.log(self.cfg.entropy_max))
        )

    def losses(
        self,
        feat: Tensor,  # (H+1, B, F)
        logps: Tensor,  # (H,   B)
        reward: Tensor,  # (H+1, B)
        cont: Tensor,  # (H+1, B)
        epistemic: Tensor | None = None,  # (H, B)
        epistemic_scale: float = 0.0,
    ) -> ACLosses:
        cfg = self.cfg
        if epistemic is not None and epistemic_scale:
            reward = reward.clone()
            reward[1:] = reward[1:] + epistemic_scale * epistemic

        value = self.critic.value(feat.reshape(-1, feat.shape[-1])).view(feat.shape[:2])
        with torch.no_grad():
            slow = self.critic.slow_value(
                feat.reshape(-1, feat.shape[-1])
            ).view(feat.shape[:2])
            target = lambda_return(reward, slow, cont, cfg.gamma, cfg.lam)  # (H, B)

        # -- actor ------------------------------------------------------------------
        self.ret_norm.update(target)
        scale = self.ret_norm.scale.to(target.device)
        adv = (target - value[:-1].detach()) / scale
        entropy = self.actor.entropy(feat[:-1].reshape(-1, feat.shape[-1])).view(
            logps.shape
        )
        # Discount the policy gradient along the imagination horizon so early steps,
        # which are on-distribution, dominate late ones, which are model-hallucinated.
        with torch.no_grad():
            weight = torch.cumprod(
                torch.cat([torch.ones_like(cont[:1]), cfg.gamma * cont[:-1]], dim=0),
                dim=0,
            )[:-1]
        alpha = self.alpha
        actor_loss = -(
            weight * (logps * adv.detach() + alpha.detach() * entropy)
        ).mean()
        # Dual update: push alpha up while realised entropy sits below target, down when
        # above. Detached from the actor so the coefficient tracks entropy rather than
        # the actor learning to game its own bonus.
        if cfg.entropy_adaptive:
            # gap > 0 means the policy is *less* random than we want, so the bonus must
            # grow. d(alpha_loss)/d(log_alpha) = -gap, and gradient descent then moves
            # log_alpha along +gap. The sign matters more than it looks: with it flipped
            # the dual variable actively drives the policy toward whichever failure mode
            # it is already heading for.
            gap = (self.target_entropy - entropy.detach().mean())
            alpha_loss = -self.log_alpha * gap
        else:
            alpha_loss = torch.zeros((), device=entropy.device)

        # -- critic -----------------------------------------------------------------
        logits = self.critic(feat[:-1].reshape(-1, feat.shape[-1]))
        twohot = self.critic.twohot.to(logits.device)
        critic_loss = twohot.loss(logits, target.detach().reshape(-1)).view(target.shape)
        # Regularise toward the slow critic; keeps the distributional head from
        # collapsing onto a single bin early in training.
        slow_logits = self.critic.slow(feat[:-1].reshape(-1, feat.shape[-1])).detach()
        reg = -(torch.softmax(slow_logits, -1) * torch.log_softmax(logits, -1)).sum(-1)
        critic_loss = (weight * (critic_loss + 1.0 * reg.view(target.shape))).mean()

        return ACLosses(
            actor=actor_loss,
            critic=critic_loss,
            alpha_loss=alpha_loss,
            alpha=alpha.detach(),
            entropy=entropy.mean().detach(),
            value_mean=value.mean().detach(),
            return_mean=target.mean().detach(),
            adv_std=adv.std().detach(),
            imag_reward=reward.mean().detach(),
        )

    @torch.no_grad()
    def policy(self, feat: Tensor, greedy: bool = False) -> Tensor:
        idx, _ = self.actor.act(feat, greedy=greedy)
        return idx

    def imagination_policy(self):
        """Callable handed to `RSSM.imagine`."""

        def _policy(feat: Tensor):
            return self.actor(feat)

        return _policy


__all__ = ["Actor", "Critic", "ImaginationActorCritic", "ACLosses", "symlog"]
