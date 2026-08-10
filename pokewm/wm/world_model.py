"""The world model: encoder, RSSM, prediction heads, and an epistemic ensemble.

Loss (per timestep, summed over a replay window):

    L = L_frame + L_symbolic + L_reward + L_cont + beta * L_KL + L_ensemble

`L_ensemble` trains N independent heads to predict the *next* categorical latent from
(state, action). Their Jensen-Shannon divergence is a calibrated estimate of epistemic
uncertainty -- high where the model has not seen enough data to know what happens next,
and, crucially, *low* on stochastic-but-familiar transitions like wild-encounter RNG.
That distinction is what makes it usable as an exploration bonus where a prediction-error
bonus would instead chase the noisy-TV (Pathak et al. 2017 -> Plan2Explore, Sekar et al.
2020 -> Simulus, 2025).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import WorldModelConfig
from .nets import (
    MLP,
    BernoulliHead,
    ConvDecoder,
    ConvEncoder,
    SymlogMLPHead,
    TwoHot,
    cosine_embedding,
    symlog,
)
from .rssm import RSSM, RSSMState


@dataclass
class WorldModelLosses:
    total: Tensor
    frame: Tensor
    symbolic: Tensor
    reward: Tensor
    cont: Tensor
    kl: Tensor
    dyn: Tensor
    rep: Tensor
    ensemble: Tensor
    # Per-sequence model loss, used to prioritise replay (Simulus §3.3).
    per_sequence: Tensor


class Encoder(nn.Module):
    """Multi-modal encoder: pixels + symbolic RAM + the active subgoal."""

    def __init__(
        self,
        frame_channels: int,
        frame_hw: tuple[int, int],
        symbolic_dim: int,
        num_subgoals: int,
        cfg: WorldModelConfig,
    ) -> None:
        super().__init__()
        self.conv = ConvEncoder(frame_channels, cfg.cnn_depth, *frame_hw)
        self.sym = MLP(symbolic_dim, cfg.hidden, cfg.hidden, cfg.layers)
        self.subgoal = MLP(cfg.subgoal_dim, cfg.subgoal_dim, cfg.hidden, 1)
        self.num_subgoals = num_subgoals
        self.subgoal_dim = cfg.subgoal_dim
        self.out_dim = self.conv.out_dim + cfg.hidden + cfg.subgoal_dim

    def forward(
        self, frame: Tensor, symbolic: Tensor, subgoal: Tensor
    ) -> Tensor:
        # subgoal arrives one-hot; convert to an index then to a fixed sinusoidal basis
        # so the conditioning space has structure even for unseen ids.
        idx = subgoal.argmax(dim=-1)
        sg = self.subgoal(cosine_embedding(idx, self.subgoal_dim))
        return torch.cat([self.conv(frame), self.sym(symlog(symbolic)), sg], dim=-1)


class EnsembleHead(nn.Module):
    """N independent one-step latent predictors sharing no parameters."""

    def __init__(self, in_dim: int, stoch: int, classes: int, cfg: WorldModelConfig):
        super().__init__()
        self.n = cfg.ensemble_size
        self.stoch, self.classes = stoch, classes
        self.heads = nn.ModuleList(
            MLP(in_dim, stoch * classes, cfg.hidden, cfg.layers) for _ in range(self.n)
        )

    def forward(self, x: Tensor) -> Tensor:
        """-> (N, B, stoch, classes) logits"""
        return torch.stack(
            [h(x).view(-1, self.stoch, self.classes) for h in self.heads], dim=0
        )

    @staticmethod
    def jsd(logits: Tensor) -> Tensor:
        """Jensen-Shannon divergence across the ensemble axis.

        logits: (N, B, stoch, classes) -> (B,) in nats, averaged over latent variables.
        JSD = H(mean_i p_i) - mean_i H(p_i), which is 0 iff all members agree and is
        bounded by log(N) -- so the resulting intrinsic reward cannot explode.
        """
        probs = torch.softmax(logits, dim=-1)
        mean = probs.mean(dim=0)
        h_mean = -(mean * torch.log(mean + 1e-12)).sum(dim=-1)
        h_each = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean(dim=0)
        return (h_mean - h_each).mean(dim=-1)


class WorldModel(nn.Module):
    def __init__(
        self,
        frame_channels: int,
        frame_hw: tuple[int, int],
        symbolic_dim: int,
        num_actions: int,
        num_subgoals: int,
        cfg: WorldModelConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_actions = num_actions
        self.frame_channels = frame_channels
        self.frame_hw = frame_hw

        self.encoder = Encoder(frame_channels, frame_hw, symbolic_dim, num_subgoals, cfg)
        self.rssm = RSSM(
            embed_dim=self.encoder.out_dim,
            action_dim=num_actions,
            deter=cfg.deter,
            stoch=cfg.stoch,
            classes=cfg.classes,
            hidden=cfg.hidden,
            layers=1,
        )
        feat = self.rssm.feat_dim

        self.frame_decoder = ConvDecoder(feat, frame_channels, cfg.cnn_depth, *frame_hw)
        self.symbolic_decoder = SymlogMLPHead(feat, symbolic_dim, cfg.hidden, cfg.layers)
        self.reward_head = MLP(feat, cfg.bins, cfg.hidden, cfg.layers, out_scale=0.0)
        self.cont_head = BernoulliHead(feat, cfg.hidden, cfg.layers)
        self.ensemble = EnsembleHead(feat + num_actions, cfg.stoch, cfg.classes, cfg)

        self.twohot = TwoHot(cfg.bins, cfg.bin_low, cfg.bin_high)

    # ------------------------------------------------------------------ inference

    def embed(self, obs: dict[str, Tensor]) -> Tensor:
        b, t = obs["symbolic"].shape[:2]
        flat = self.encoder(
            obs["frame"].reshape(b * t, *obs["frame"].shape[2:]),
            obs["symbolic"].reshape(b * t, -1),
            obs["subgoal"].reshape(b * t, -1),
        )
        return flat.view(b, t, -1)

    @torch.no_grad()
    def observe_step(
        self, state: RSSMState, obs: dict[str, Tensor], action: Tensor, is_first: Tensor
    ) -> RSSMState:
        """Single-step filtering used by the online collector."""
        embed = self.encoder(obs["frame"], obs["symbolic"], obs["subgoal"])
        mask = (1.0 - is_first.float()).view(-1, 1)
        state = RSSMState(
            state.deter * mask,
            state.logits * mask.unsqueeze(-1),
            state.stoch * mask.unsqueeze(-1),
        )
        post, _ = self.rssm.obs_step(state, action * mask, embed)
        return post

    @torch.no_grad()
    def epistemic_bonus(self, state: RSSMState, action: Tensor) -> Tensor:
        logits = self.ensemble(torch.cat([state.feature(), action], dim=-1))
        return EnsembleHead.jsd(logits)

    # ------------------------------------------------------------------ training

    def loss(self, batch: dict[str, Tensor]) -> tuple[WorldModelLosses, RSSMState]:
        """`batch` holds (B, T, ...) tensors: frame, symbolic, subgoal, action (one-hot),
        reward, cont, is_first."""
        cfg = self.cfg
        b, t = batch["reward"].shape
        embed = self.embed(batch)
        post, prior = self.rssm.observe(embed, batch["action"], batch["is_first"])

        feat = torch.cat([post.deter, post.stoch.flatten(2)], dim=-1)  # (B, T, feat)
        flat_feat = feat.reshape(b * t, -1)

        # -- reconstruction ---------------------------------------------------------
        target_frame = batch["frame"].reshape(b * t, *batch["frame"].shape[2:]).float()
        target_frame = target_frame / 255.0 - 0.5
        pred_frame = self.frame_decoder(flat_feat)
        frame_loss = 0.5 * (pred_frame - target_frame).pow(2).flatten(1).sum(-1)

        sym_loss = self.symbolic_decoder.loss(
            flat_feat, batch["symbolic"].reshape(b * t, -1)
        )

        # -- reward / continuation ---------------------------------------------------
        reward_logits = self.reward_head(flat_feat)
        reward_loss = self.twohot.to(flat_feat.device).loss(
            reward_logits, batch["reward"].reshape(b * t)
        )
        cont_loss = self.cont_head.loss(flat_feat, batch["cont"].reshape(b * t))

        # -- KL ----------------------------------------------------------------------
        kl, dyn, rep = self.rssm.kl_loss(
            post, prior, cfg.kl_free, cfg.kl_dyn_scale, cfg.kl_rep_scale
        )
        kl_flat = kl.reshape(b * t)

        # -- ensemble ----------------------------------------------------------------
        # Predict z_{t+1} from (feat_t, a_t). Targets are the posterior one-hots,
        # detached: the ensemble must not be able to move the representation.
        ens_in = torch.cat([feat[:, :-1], batch["action"][:, 1:]], dim=-1)
        ens_logits = self.ensemble(ens_in.reshape(b * (t - 1), -1))
        ens_target = post.stoch[:, 1:].detach().reshape(b * (t - 1), cfg.stoch, cfg.classes)
        ens_loss = -(
            ens_target.unsqueeze(0) * torch.log_softmax(ens_logits, dim=-1)
        ).sum(dim=(-1, -2)).mean(dim=0)
        ens_loss = F.pad(ens_loss.view(b, t - 1), (1, 0)).reshape(b * t)

        per_step = frame_loss + sym_loss + reward_loss + cont_loss + kl_flat + ens_loss
        per_sequence = per_step.view(b, t).mean(dim=1)
        total = per_step.mean()

        losses = WorldModelLosses(
            total=total,
            frame=frame_loss.mean().detach(),
            symbolic=sym_loss.mean().detach(),
            reward=reward_loss.mean().detach(),
            cont=cont_loss.mean().detach(),
            kl=kl_flat.mean().detach(),
            dyn=dyn.mean().detach(),
            rep=rep.mean().detach(),
            ensemble=ens_loss.mean().detach(),
            per_sequence=per_sequence.detach(),
        )
        return losses, post

    # ------------------------------------------------------------------ imagination

    def imagine(
        self, start: RSSMState, policy, horizon: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Roll out in latent space.

        Returns (features, actions, log_probs, rewards, continues), each leading with the
        imagination axis of length H+1 (H for actions/log-probs).
        """
        states, actions, logps = self.rssm.imagine(start, policy, horizon)
        feat = torch.cat([states.deter, states.stoch.flatten(2)], dim=-1)
        h1, bsz = feat.shape[0], feat.shape[1]
        flat = feat.reshape(h1 * bsz, -1)
        reward = self.twohot.to(flat.device).decode(self.reward_head(flat)).view(h1, bsz)
        cont = self.cont_head.prob(flat).view(h1, bsz)
        return feat, actions, logps, reward, cont

    @torch.no_grad()
    def imagine_epistemic(self, feat: Tensor, actions: Tensor) -> Tensor:
        """Epistemic bonus along an imagined rollout. feat (H+1,B,F), actions (H,B,A)."""
        h, bsz = actions.shape[0], actions.shape[1]
        x = torch.cat([feat[:-1], actions], dim=-1).reshape(h * bsz, -1)
        return EnsembleHead.jsd(self.ensemble(x)).view(h, bsz)
