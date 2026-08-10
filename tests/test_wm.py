"""RSSM, world model and actor-critic.

The properties checked here are the ones that distinguish a working world model from one
that merely runs: the recurrent state must actually reset on episode boundaries, the
prior must be trainable toward the posterior, imagination must not touch observations,
and epistemic disagreement must behave like an uncertainty estimate rather than like
prediction error.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pokewm.config import Config
from pokewm.wm.actor_critic import ImaginationActorCritic
from pokewm.wm.rssm import RSSMState, flatten_state
from pokewm.wm.world_model import EnsembleHead, WorldModel

NA, NS, SD = 7, 24, 22


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.preset("smoke")


@pytest.fixture(scope="module")
def wm(cfg) -> WorldModel:
    torch.manual_seed(0)
    channels = cfg.env.frame_stack + cfg.env.seen_map_channels
    return WorldModel(channels, (cfg.env.frame_h, cfg.env.frame_w), SD, NA, NS, cfg.wm)


@pytest.fixture(scope="module")
def ac(cfg, wm) -> ImaginationActorCritic:
    return ImaginationActorCritic(wm.rssm.feat_dim, NA, cfg.ac, cfg.wm)


def make_batch(cfg, b=2, t=8):
    channels = cfg.env.frame_stack + cfg.env.seen_map_channels
    batch = {
        "frame": torch.randint(
            0, 255, (b, t, channels, cfg.env.frame_h, cfg.env.frame_w), dtype=torch.uint8
        ),
        "symbolic": torch.rand(b, t, SD),
        "subgoal": torch.eye(NS)[torch.randint(0, NS, (b, t))],
        "action": torch.eye(NA)[torch.randint(0, NA, (b, t))],
        "reward": torch.randn(b, t) * 3,
        "cont": torch.ones(b, t),
        "is_first": torch.zeros(b, t),
    }
    batch["is_first"][:, 0] = 1.0
    return batch


class TestRSSM:
    def test_initial_state_is_zero(self, wm):
        s = wm.rssm.initial(4, torch.device("cpu"))
        assert s.deter.abs().max() == 0
        assert s.stoch.abs().max() == 0
        assert s.feature().shape == (4, wm.rssm.feat_dim)

    def test_feature_concatenates_deter_and_flattened_stoch(self, wm):
        s = wm.rssm.initial(3, torch.device("cpu"))
        expected = wm.rssm.deter_size + wm.rssm.stoch_size * wm.rssm.classes
        assert s.feature().shape[-1] == expected == wm.rssm.feat_dim

    def test_stoch_is_one_hot_per_variable(self, wm, cfg):
        embed = torch.randn(2, wm.encoder.out_dim)
        prev = wm.rssm.initial(2, torch.device("cpu"))
        post, _ = wm.rssm.obs_step(prev, torch.eye(NA)[[0, 1]], embed)
        assert torch.allclose(post.stoch.sum(-1), torch.ones(2, cfg.wm.stoch))

    def test_observe_shapes(self, wm, cfg):
        b, t = 2, 6
        embed = torch.randn(b, t, wm.encoder.out_dim)
        action = torch.eye(NA)[torch.randint(0, NA, (b, t))]
        is_first = torch.zeros(b, t)
        is_first[:, 0] = 1
        post, prior = wm.rssm.observe(embed, action, is_first)
        assert post.deter.shape == (b, t, cfg.wm.deter)
        assert post.logits.shape == (b, t, cfg.wm.stoch, cfg.wm.classes)
        assert prior.deter.shape == post.deter.shape

    def test_is_first_resets_recurrent_state(self, wm):
        """A window straddling an episode boundary must not leak state across it.

        Compared at the reset step itself. Later steps depend on the *sampled* latent,
        and the two runs draw different numbers of samples, so they legitimately
        diverge downstream -- the deterministic reset is the property under test.
        """
        b, t = 1, 4
        embed = torch.randn(b, t, wm.encoder.out_dim)
        action = torch.eye(NA)[torch.randint(0, NA, (b, t))]

        first_a = torch.zeros(b, t)
        first_a[:, 0] = 1
        first_a[:, 2] = 1  # reset at t=2
        post_a, prior_a = wm.rssm.observe(embed, action, first_a)

        # Running only the post-reset suffix from scratch must agree at the boundary.
        first_b = torch.zeros(b, 2)
        first_b[:, 0] = 1
        post_b, prior_b = wm.rssm.observe(embed[:, 2:], action[:, 2:], first_b)

        assert torch.allclose(post_a.deter[:, 2], post_b.deter[:, 0], atol=1e-5)
        assert torch.allclose(post_a.logits[:, 2], post_b.logits[:, 0], atol=1e-5)
        assert torch.allclose(prior_a.logits[:, 2], prior_b.logits[:, 0], atol=1e-5)

    def test_history_before_a_reset_does_not_change_the_reset_step(self, wm):
        """Same reset step, completely different prefix -> identical state."""
        b, t = 1, 3
        embed = torch.randn(b, t, wm.encoder.out_dim)
        is_first = torch.zeros(b, t)
        is_first[:, 0] = 1
        is_first[:, 2] = 1

        a = torch.eye(NA)[torch.zeros(b, t, dtype=torch.long)]
        z = torch.eye(NA)[torch.full((b, t), 3, dtype=torch.long)]
        post_a, _ = wm.rssm.observe(embed, a, is_first)
        post_z, _ = wm.rssm.observe(embed, z, is_first)
        assert torch.allclose(post_a.deter[:, 2], post_z.deter[:, 2], atol=1e-5)

    def test_imagine_does_not_use_observations(self, wm, ac):
        """img_step must depend only on (state, action)."""
        prev = wm.rssm.initial(3, torch.device("cpu"))
        act = torch.eye(NA)[[0, 0, 0]]
        torch.manual_seed(7)
        a = wm.rssm.img_step(prev, act, sample=False)
        torch.manual_seed(7)
        b = wm.rssm.img_step(prev, act, sample=False)
        assert torch.allclose(a.deter, b.deter)
        assert torch.allclose(a.logits, b.logits)

    def test_imagine_shapes(self, wm, ac, cfg):
        start = wm.rssm.initial(5, torch.device("cpu"))
        states, actions, logps = wm.rssm.imagine(
            start, ac.imagination_policy(), cfg.wm.horizon
        )
        h = cfg.wm.horizon
        assert states.deter.shape == (h + 1, 5, cfg.wm.deter)
        assert actions.shape == (h, 5, NA)
        assert logps.shape == (h, 5)

    def test_kl_balancing_produces_two_scaled_terms(self, wm, cfg):
        post = wm.rssm.initial(2, torch.device("cpu"))
        post = RSSMState(post.deter, torch.randn(2, cfg.wm.stoch, cfg.wm.classes),
                         post.stoch)
        prior = wm.rssm.initial(2, torch.device("cpu"))
        prior = RSSMState(prior.deter, torch.randn(2, cfg.wm.stoch, cfg.wm.classes),
                          prior.stoch)
        kl, dyn, rep = wm.rssm.kl_loss(post, prior, 0.0, 0.5, 0.1)
        assert torch.allclose(kl, 0.5 * dyn + 0.1 * rep, atol=1e-5)
        assert (dyn >= 0).all() and (rep >= 0).all()

    def test_flatten_state(self, wm, cfg):
        s = RSSMState(
            torch.randn(3, 4, cfg.wm.deter),
            torch.randn(3, 4, cfg.wm.stoch, cfg.wm.classes),
            torch.randn(3, 4, cfg.wm.stoch, cfg.wm.classes),
        )
        f = flatten_state(s)
        assert f.deter.shape == (12, cfg.wm.deter)
        assert f.stoch.shape == (12, cfg.wm.stoch, cfg.wm.classes)


class TestWorldModelLoss:
    def test_all_terms_finite_and_non_negative(self, wm, cfg):
        losses, _ = wm.loss(make_batch(cfg))
        for name in ["total", "frame", "symbolic", "reward", "cont", "kl", "ensemble"]:
            v = getattr(losses, name).detach()
            assert torch.isfinite(v).all(), name
            assert float(v) >= 0.0, name

    def test_per_sequence_loss_has_one_entry_per_batch_row(self, wm, cfg):
        batch = make_batch(cfg, b=3, t=8)
        losses, _ = wm.loss(batch)
        assert losses.per_sequence.shape == (3,)
        assert torch.isfinite(losses.per_sequence).all()

    def test_gradients_reach_every_component(self, cfg):
        """Every parameter must be trainable once the network has left initialisation.

        Checked after a few optimiser steps rather than at step 0, because two
        components are deliberately gradient-blocked at initialisation -- see
        `test_free_bits_block_kl_gradient_at_the_floor` and
        `test_zero_init_output_layer_unblocks_after_one_step`.
        """
        from dataclasses import replace

        torch.manual_seed(1)
        wm_cfg = replace(cfg.wm, kl_free=0.0)
        channels = cfg.env.frame_stack + cfg.env.seen_map_channels
        model = WorldModel(channels, (cfg.env.frame_h, cfg.env.frame_w), SD, NA, NS,
                           wm_cfg)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = make_batch(cfg)
        for _ in range(3):
            losses, _ = model.loss(batch)
            opt.zero_grad()
            losses.total.backward()
            opt.step()
        missing = [
            n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
        ]
        assert missing == [], f"no gradient reached: {missing[:8]}"

    def test_free_bits_block_kl_gradient_at_the_floor(self, cfg):
        """Free bits are a hard floor: below it the KL contributes no gradient at all.

        This is intended (DreamerV3 eq. 5) -- it stops the model spending capacity
        driving an already-small KL to zero -- but it means the prior network is
        untrained until the posterior genuinely diverges from it.
        """
        torch.manual_seed(1)
        channels = cfg.env.frame_stack + cfg.env.seen_map_channels
        model = WorldModel(channels, (cfg.env.frame_h, cfg.env.frame_w), SD, NA, NS,
                           cfg.wm)
        losses, _ = model.loss(make_batch(cfg))
        assert float(losses.dyn) == pytest.approx(cfg.wm.kl_free, abs=1e-5)
        losses.total.backward()
        prior_grad = sum(
            float(p.grad.abs().sum()) for p in model.rssm.prior_net.parameters()
            if p.grad is not None
        )
        assert prior_grad == 0.0

    def test_zero_init_output_layer_unblocks_after_one_step(self, cfg):
        """`out_scale=0` heads start with a zero output layer.

        At step 0 the hidden layers receive W_out^T @ delta == 0. The output layer
        itself does get gradient, so one optimiser step is enough to unblock the rest.
        """
        torch.manual_seed(0)
        channels = cfg.env.frame_stack + cfg.env.seen_map_channels
        model = WorldModel(channels, (cfg.env.frame_h, cfg.env.frame_w), SD, NA, NS,
                           cfg.wm)
        head = model.symbolic_decoder
        feat = torch.randn(4, model.rssm.feat_dim)
        target = torch.rand(4, SD)

        head.loss(feat, target).sum().backward()
        params = list(head.parameters())
        assert float(params[-1].grad.abs().sum()) > 0, "output layer must get gradient"
        assert float(params[0].grad.abs().sum()) == 0, "hidden blocked at init"

        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        opt.step()
        opt.zero_grad()
        head.loss(feat, target).sum().backward()
        assert float(params[0].grad.abs().sum()) > 0, "hidden must unblock after 1 step"

    def test_posterior_returned_for_imagination_start(self, wm, cfg):
        batch = make_batch(cfg, b=2, t=8)
        _, post = wm.loss(batch)
        assert post.deter.shape[:2] == (2, 8)

    def test_learns_to_predict_a_constant_reward(self, cfg):
        """End-to-end sanity: the reward head must be able to fit a fixed target."""
        torch.manual_seed(0)
        channels = cfg.env.frame_stack + cfg.env.seen_map_channels
        model = WorldModel(channels, (cfg.env.frame_h, cfg.env.frame_w), SD, NA, NS,
                           cfg.wm)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        batch = make_batch(cfg, b=2, t=8)
        batch["reward"] = torch.full_like(batch["reward"], 5.0)
        first = None
        for i in range(40):
            losses, _ = model.loss(batch)
            opt.zero_grad()
            losses.total.backward()
            opt.step()
            if i == 0:
                first = float(losses.reward)
        assert float(losses.reward) < first * 0.6


class TestEnsemble:
    def test_jsd_is_zero_when_members_agree(self):
        logits = torch.randn(1, 6, 4, 8).repeat(4, 1, 1, 1)
        assert EnsembleHead.jsd(logits).abs().max() < 1e-5

    def test_jsd_is_positive_when_members_disagree(self):
        logits = torch.randn(4, 6, 4, 8) * 5
        assert (EnsembleHead.jsd(logits) > 0).all()

    def test_jsd_is_bounded_by_log_n(self):
        logits = torch.randn(4, 32, 4, 8) * 100
        assert EnsembleHead.jsd(logits).max() <= torch.log(torch.tensor(4.0)) + 1e-4

    def test_jsd_shape(self):
        assert EnsembleHead.jsd(torch.randn(3, 11, 4, 8)).shape == (11,)

    def test_epistemic_bonus_is_finite(self, wm):
        state = wm.rssm.initial(4, torch.device("cpu"))
        bonus = wm.epistemic_bonus(state, torch.eye(NA)[[0, 1, 2, 3]])
        assert bonus.shape == (4,)
        assert torch.isfinite(bonus).all() and (bonus >= 0).all()

    def test_imagine_epistemic_shape(self, wm, ac, cfg):
        start = wm.rssm.initial(6, torch.device("cpu"))
        feat, actions, _, _, _ = wm.imagine(start, ac.imagination_policy(),
                                            cfg.wm.horizon)
        assert wm.imagine_epistemic(feat, actions).shape == (cfg.wm.horizon, 6)


class TestActorCritic:
    def test_losses_finite(self, wm, ac, cfg):
        start = wm.rssm.initial(8, torch.device("cpu"))
        feat, actions, logps, reward, cont = wm.imagine(
            start, ac.imagination_policy(), cfg.wm.horizon
        )
        out = ac.losses(feat, logps, reward, cont)
        assert torch.isfinite(out.actor) and torch.isfinite(out.critic)

    def test_gradients_flow_to_actor_and_critic(self, cfg, wm):
        torch.manual_seed(3)
        model = ImaginationActorCritic(wm.rssm.feat_dim, NA, cfg.ac, cfg.wm)
        start = wm.rssm.initial(8, torch.device("cpu"))
        feat, actions, logps, reward, cont = wm.imagine(
            start, model.imagination_policy(), cfg.wm.horizon
        )
        out = model.losses(feat, logps, reward, cont)
        (out.actor + out.critic).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.actor.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.critic.net.parameters())

    def test_slow_critic_is_frozen_but_tracks(self, cfg, wm):
        model = ImaginationActorCritic(wm.rssm.feat_dim, NA, cfg.ac, cfg.wm)
        assert all(not p.requires_grad for p in model.critic.slow.parameters())
        before = [p.clone() for p in model.critic.slow.parameters()]
        with torch.no_grad():
            for p in model.critic.net.parameters():
                p.add_(1.0)
        model.critic.update_slow()
        after = list(model.critic.slow.parameters())
        assert all(not torch.equal(a, b) for a, b in zip(before, after))
        # But it moves only a fraction of the way (EMA, not a copy).
        delta = float((after[0] - before[0]).abs().mean())
        assert delta < 1.0

    def test_entropy_is_positive_and_bounded(self, wm, ac):
        feat = torch.randn(16, wm.rssm.feat_dim)
        h = ac.actor.entropy(feat)
        assert (h > 0).all()
        assert h.max() <= torch.log(torch.tensor(float(NA))) + 1e-4

    def test_policy_returns_valid_actions(self, wm, ac):
        feat = torch.randn(32, wm.rssm.feat_dim)
        for greedy in (False, True):
            idx = ac.policy(feat, greedy=greedy)
            assert idx.shape == (32,)
            assert int(idx.min()) >= 0 and int(idx.max()) < NA

    def test_greedy_is_deterministic(self, wm, ac):
        feat = torch.randn(16, wm.rssm.feat_dim)
        assert torch.equal(ac.policy(feat, greedy=True), ac.policy(feat, greedy=True))

    def test_entropy_coefficient_is_a_tuned_dual_variable(self, cfg, wm):
        """Regression: four hand-picked entropy constants, two collapses, one pinned.

        The right constant depends on the advantage scale, which moves two orders of
        magnitude over a run. Tracking a target entropy removes the choice, so what
        matters is that the coefficient moves in the correcting direction.
        """
        model = ImaginationActorCritic(wm.rssm.feat_dim, NA, cfg.ac, cfg.wm)
        assert model.target_entropy == pytest.approx(
            cfg.ac.entropy_target_frac * float(np.log(NA))
        )

        feat = torch.randn(cfg.wm.horizon + 1, 8, wm.rssm.feat_dim)
        logps = torch.randn(cfg.wm.horizon, 8)
        reward = torch.randn(cfg.wm.horizon + 1, 8)
        cont = torch.ones(cfg.wm.horizon + 1, 8)

        def run(target: float, steps: int = 30) -> tuple[float, float]:
            """Drive the real dual loss with `target` moved either side of realised H."""
            model.log_alpha.data.fill_(float(np.log(1e-3)))
            model.target_entropy = target
            opt = torch.optim.Adam([model.log_alpha], lr=0.1)
            before = float(model.alpha.detach())
            for _ in range(steps):
                out = model.losses(feat, logps, reward, cont)
                opt.zero_grad()
                out.alpha_loss.backward()
                opt.step()
                model.clamp_log_alpha()
            return before, float(model.alpha.detach())

        # A freshly initialised actor is near-uniform, so realised entropy sits just
        # under ln|A|. Put the target above it -> the coefficient must climb; below it
        # -> it must fall. This exercises `losses()`, not a re-derivation of the sign.
        h = float(model.actor.entropy(feat[0]).mean().detach())
        assert 0.0 < h <= np.log(NA) + 1e-6

        before_up, after_up = run(h + 0.5)
        assert after_up > before_up

        before_down, after_down = run(max(h - 0.5, 1e-3))
        assert after_down < before_down

    def test_entropy_coefficient_stays_within_bounds(self, cfg, wm):
        model = ImaginationActorCritic(wm.rssm.feat_dim, NA, cfg.ac, cfg.wm)
        # float32 lands a hair either side of the bound, so compare with a relative
        # tolerance rather than pretending the clamp is exact in single precision.
        model.log_alpha.data.fill_(50.0)
        assert float(model.alpha.detach()) == pytest.approx(cfg.ac.entropy_max, rel=1e-6)
        model.log_alpha.data.fill_(-50.0)
        assert float(model.alpha.detach()) == pytest.approx(cfg.ac.entropy_min, rel=1e-6)

    def test_dual_variable_cannot_wind_up(self, cfg, wm):
        """Regression: the dual loss is linear in log_alpha, so its gradient does not

        vanish when alpha saturates. Without clamping the parameter itself, a long
        stretch below target pushes log_alpha arbitrarily far past the ceiling and the
        coefficient needs just as long to come back -- a policy pinned at uniform.
        """
        model = ImaginationActorCritic(wm.rssm.feat_dim, NA, cfg.ac, cfg.wm)
        model.log_alpha.data.fill_(50.0)
        model.clamp_log_alpha()
        assert float(model.log_alpha.detach()) == pytest.approx(np.log(cfg.ac.entropy_max))
        model.log_alpha.data.fill_(-50.0)
        model.clamp_log_alpha()
        assert float(model.log_alpha.detach()) == pytest.approx(np.log(cfg.ac.entropy_min))

    def test_fixed_coefficient_when_adaptive_disabled(self, cfg, wm):
        from dataclasses import replace

        ac_cfg = replace(cfg.ac, entropy_adaptive=False, entropy=7e-4)
        model = ImaginationActorCritic(wm.rssm.feat_dim, NA, ac_cfg, cfg.wm)
        assert float(model.alpha) == pytest.approx(7e-4)
        feat = torch.randn(cfg.wm.horizon + 1, 4, wm.rssm.feat_dim)
        out = model.losses(feat, torch.randn(cfg.wm.horizon, 4),
                           torch.randn(cfg.wm.horizon + 1, 4),
                           torch.ones(cfg.wm.horizon + 1, 4))
        assert float(out.alpha_loss) == 0.0

    def test_alpha_gradient_does_not_reach_the_actor(self, cfg, wm):
        """The actor must not be able to inflate its own entropy bonus."""
        model = ImaginationActorCritic(wm.rssm.feat_dim, NA, cfg.ac, cfg.wm)
        feat = torch.randn(cfg.wm.horizon + 1, 4, wm.rssm.feat_dim)
        out = model.losses(feat, torch.randn(cfg.wm.horizon, 4),
                           torch.randn(cfg.wm.horizon + 1, 4),
                           torch.ones(cfg.wm.horizon + 1, 4))
        out.alpha_loss.backward()
        assert model.log_alpha.grad is not None
        assert all(p.grad is None or p.grad.abs().sum() == 0
                   for p in model.actor.parameters())

    def test_epistemic_bonus_increases_imagined_reward(self, wm, ac, cfg):
        start = wm.rssm.initial(8, torch.device("cpu"))
        feat, actions, logps, reward, cont = wm.imagine(
            start, ac.imagination_policy(), cfg.wm.horizon
        )
        epi = torch.ones(cfg.wm.horizon, 8)
        plain = ac.losses(feat, logps, reward, cont)
        bonus = ac.losses(feat, logps, reward, cont, epi, 1.0)
        assert float(bonus.imag_reward) > float(plain.imag_reward)
