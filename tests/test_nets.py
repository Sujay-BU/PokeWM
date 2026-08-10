"""Numerical primitives.

These are the pieces where a subtle sign or indexing error produces a model that trains
without erroring and simply never learns, so each is checked against its mathematical
definition rather than against a golden value.
"""

from __future__ import annotations


import pytest
import torch

from pokewm.wm.nets import (
    MLP,
    ConvDecoder,
    ConvEncoder,
    OneHotCategoricalST,
    ReturnNormalizer,
    TwoHot,
    check_frame_shape,
    cosine_embedding,
    kl_divergence_categorical,
    lambda_return,
    symexp,
    symlog,
    unimix_logits,
)


class TestSymlog:
    def test_is_inverse_of_symexp(self):
        x = torch.tensor([-1e4, -100.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 100.0, 1e4])
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-3, rtol=1e-4)

    def test_odd_symmetry(self):
        x = torch.randn(64) * 50
        assert torch.allclose(symlog(-x), -symlog(x), atol=1e-6)

    def test_fixes_zero(self):
        assert symlog(torch.zeros(4)).abs().max() == 0.0

    def test_compresses_large_values(self):
        # The point of symlog: a badge (+64) must not dominate a step cost (-0.002) by
        # the raw 32000x factor. It measures ~2089x here, a 15x compression, which is
        # what keeps the badge gradient from erasing everything else for thousands of
        # updates while still leaving it clearly the largest signal.
        ratio_raw = 64.0 / 0.002
        ratio_log = float(symlog(torch.tensor(64.0)) / symlog(torch.tensor(0.002)))
        assert 1.0 < abs(ratio_log) < ratio_raw / 10

    def test_monotone(self):
        x = torch.linspace(-100, 100, 500)
        assert torch.all(torch.diff(symlog(x)) > 0)


class TestTwoHot:
    @pytest.fixture
    def th(self):
        return TwoHot(bins=51, low=-5.0, high=5.0)

    def test_encoding_sums_to_one(self, th):
        x = torch.linspace(-5, 5, 37)
        assert torch.allclose(th.encode(x).sum(-1), torch.ones(37), atol=1e-5)

    def test_encoding_has_at_most_two_nonzeros(self, th):
        enc = th.encode(torch.tensor([0.37, -2.2, 4.9]))
        assert ((enc > 1e-6).sum(-1) <= 2).all()

    def test_expectation_recovers_the_target(self, th):
        """The defining property: sum_i p_i * v_i == x."""
        x = torch.linspace(-4.9, 4.9, 51)
        enc = th.encode(x)
        assert torch.allclose((enc * th.values).sum(-1), x, atol=1e-4)

    def test_exact_bin_edges_do_not_produce_nan(self, th):
        enc = th.encode(th.values.clone())
        assert torch.isfinite(enc).all()
        assert torch.allclose(enc.sum(-1), torch.ones(th.bins), atol=1e-5)

    def test_decode_inverts_encode_through_symexp(self, th):
        target = torch.tensor([0.0, 1.0, -3.0, 20.0, -20.0])
        logits = torch.log(th.encode(symlog(target).clamp(-5, 5)) + 1e-12)
        assert torch.allclose(th.decode(logits), target, atol=1e-2)

    def test_loss_is_minimised_at_the_target(self, th):
        target = torch.tensor([2.0])
        good = torch.log(th.encode(symlog(target)) + 1e-12)
        bad = torch.zeros(1, th.bins)  # uniform
        assert float(th.loss(good, target)) < float(th.loss(bad, target))

    def test_loss_is_non_negative(self, th):
        logits = torch.randn(32, th.bins)
        assert (th.loss(logits, torch.randn(32) * 10) >= 0).all()


class TestLambdaReturn:
    def test_lam_zero_is_one_step_td(self):
        t, b = 6, 3
        reward = torch.randn(t, b)
        value = torch.randn(t, b)
        cont = torch.ones(t, b)
        gamma = 0.9
        got = lambda_return(reward, value, cont, gamma, 0.0)
        want = reward[1:] + gamma * value[1:]
        assert torch.allclose(got, want, atol=1e-5)

    def test_lam_one_is_monte_carlo(self):
        t, b = 5, 2
        reward = torch.randn(t, b)
        value = torch.randn(t, b)
        cont = torch.ones(t, b)
        gamma = 0.95
        got = lambda_return(reward, value, cont, gamma, 1.0)
        # Discounted sum of future rewards, bootstrapping on the final value.
        want = torch.zeros(t - 1, b)
        for i in range(t - 1):
            acc = value[-1]
            for k in reversed(range(i, t - 1)):
                acc = reward[k + 1] + gamma * acc
            want[i] = acc
        assert torch.allclose(got, want, atol=1e-4)

    def test_termination_stops_bootstrapping(self):
        t, b = 4, 1
        reward = torch.ones(t, b)
        value = torch.full((t, b), 100.0)
        cont = torch.ones(t, b)
        cont[1:] = 0.0  # everything after the first step is terminal
        got = lambda_return(reward, value, cont, 0.99, 0.95)
        assert torch.allclose(got[0], torch.ones(b), atol=1e-5)

    def test_output_shape(self):
        out = lambda_return(torch.randn(9, 5), torch.randn(9, 5), torch.ones(9, 5),
                            0.99, 0.95)
        assert out.shape == (8, 5)

    def test_zero_reward_zero_value_gives_zero(self):
        z = torch.zeros(7, 4)
        assert lambda_return(z, z, torch.ones(7, 4), 0.99, 0.95).abs().max() == 0.0


class TestKL:
    def test_zero_for_identical_distributions_up_to_free_bits(self):
        logits = torch.randn(8, 32, 32)
        kl = kl_divergence_categorical(logits, logits.clone(), free=0.0)
        assert kl.abs().max() < 1e-4

    def test_free_bits_floor_is_applied(self):
        logits = torch.randn(8, 32, 32)
        kl = kl_divergence_categorical(logits, logits.clone(), free=1.5)
        assert torch.allclose(kl, torch.full_like(kl, 1.5))

    def test_positive_for_different_distributions(self):
        a = torch.randn(4, 16, 8) * 3
        b = torch.randn(4, 16, 8) * 3
        assert (kl_divergence_categorical(a, b, free=0.0) > 0).all()

    def test_shape_reduces_over_stoch_and_classes(self):
        kl = kl_divergence_categorical(torch.randn(5, 7, 16, 8),
                                       torch.randn(5, 7, 16, 8), free=0.0)
        assert kl.shape == (5, 7)


class TestUnimix:
    def test_bounds_probabilities_away_from_zero(self):
        logits = torch.tensor([[100.0, -100.0, -100.0]])
        probs = torch.softmax(unimix_logits(logits, 0.01), -1)
        assert probs.min() > 1e-3
        assert torch.allclose(probs.sum(-1), torch.ones(1), atol=1e-5)

    def test_disabled_when_zero(self):
        logits = torch.randn(4, 9)
        assert torch.equal(unimix_logits(logits, 0.0), logits)


class TestStraightThrough:
    def test_sample_is_one_hot(self):
        d = OneHotCategoricalST(logits=torch.randn(16, 10))
        s = d.rsample()
        assert torch.allclose(s.sum(-1), torch.ones(16))
        assert ((s == 0) | (s == 1)).all()

    def test_gradient_flows_to_logits(self):
        """Reduce with weights, not a plain sum.

        Straight-through gives `sample = onehot + probs - probs.detach()`, so
        `sample.sum()` is identically 1 per row and its gradient to the logits is
        analytically *zero*. Asserting that gradient is non-zero therefore tested
        floating-point noise, and failed whenever the noise happened to cancel. A
        non-constant reduction has a genuine gradient.
        """
        torch.manual_seed(0)
        logits = torch.randn(8, 5, requires_grad=True)
        weights = torch.randn(8, 5)
        (OneHotCategoricalST(logits=logits).rsample() * weights).sum().backward()
        assert logits.grad is not None and logits.grad.abs().sum() > 0

    def test_a_constant_reduction_has_no_gradient(self):
        """Pins the reason the test above needs weights."""
        torch.manual_seed(0)
        logits = torch.randn(8, 5, requires_grad=True)
        OneHotCategoricalST(logits=logits).rsample().sum().backward()
        assert logits.grad.abs().max() < 1e-5


class TestConvShapes:
    @pytest.mark.parametrize("hw", [(72, 80), (64, 64), (144, 160)])
    def test_encoder_decoder_roundtrip_shape(self, hw):
        h, w = hw
        enc = ConvEncoder(5, 8, h, w)
        dec = ConvDecoder(enc.out_dim, 5, 8, h, w)
        x = torch.randint(0, 255, (2, 5, h, w), dtype=torch.uint8)
        feat = enc(x)
        assert feat.shape == (2, enc.out_dim)
        assert dec(feat).shape == (2, 5, h, w)

    def test_rejects_indivisible_frame_size(self):
        with pytest.raises(ValueError, match="divisible"):
            check_frame_shape(72, 75)

    def test_encoder_normalises_input_range(self):
        enc = ConvEncoder(1, 4, 32, 32)
        out = enc(torch.full((1, 1, 32, 32), 255, dtype=torch.uint8))
        assert torch.isfinite(out).all()


class TestReturnNormalizer:
    def test_scale_is_positive_and_respects_limit(self):
        rn = ReturnNormalizer(5.0, 95.0, 0.99, 1.0)
        rn.update(torch.zeros(1000))
        assert float(rn.scale) >= 1.0

    def test_tracks_spread(self):
        rn = ReturnNormalizer(5.0, 95.0, 0.0, 1e-8)  # no decay -> immediate
        rn.update(torch.linspace(0, 100, 1001))
        assert 70 < float(rn.scale) < 100

    def test_outliers_do_not_dominate(self):
        rn = ReturnNormalizer(5.0, 95.0, 0.0, 1e-8)
        x = torch.cat([torch.zeros(1000), torch.tensor([1e6])])
        rn.update(x)
        assert float(rn.scale) < 10.0

    def test_state_dict_roundtrip(self):
        rn = ReturnNormalizer(5.0, 95.0, 0.99, 1.0)
        rn.update(torch.randn(500) * 7)
        other = ReturnNormalizer(5.0, 95.0, 0.99, 1.0)
        other.load_state_dict(rn.state_dict())
        assert float(other.scale) == pytest.approx(float(rn.scale), abs=1e-5)


class TestMisc:
    def test_cosine_embedding_shape_and_range(self):
        emb = cosine_embedding(torch.arange(24), 32)
        assert emb.shape == (24, 32)
        assert emb.abs().max() <= 1.0 + 1e-6

    def test_cosine_embedding_distinguishes_ids(self):
        emb = cosine_embedding(torch.arange(24), 32)
        sims = emb @ emb.t()
        diag = sims.diagonal().clone()
        off_diag_max = sims.fill_diagonal_(float("-inf")).max(dim=-1).values
        assert (diag > off_diag_max).all()

    def test_cosine_embedding_handles_odd_dim(self):
        assert cosine_embedding(torch.arange(3), 7).shape == (3, 7)

    def test_mlp_out_scale_zero_initialises_near_zero(self):
        mlp = MLP(8, 4, 16, 2, out_scale=0.0)
        assert mlp(torch.randn(5, 8)).abs().max() < 1e-5

    def test_mlp_shapes(self):
        assert MLP(11, 3, 7, 2)(torch.randn(6, 11)).shape == (6, 3)
