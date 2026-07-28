"""Unit tests for the length-adaptive GAE lambda."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vime.backends.megatron_utils.loss import _gae_lambda


def _args(**kwargs):
    return SimpleNamespace(**{"lambd": 0.95, **kwargs})


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_disabled_returns_the_configured_constant(alpha):
    assert _gae_lambda(_args(gae_lambda_alpha=alpha), [10, 100], device="cpu") == 0.95


@pytest.mark.unit
def test_an_absent_setting_returns_the_configured_constant():
    assert _gae_lambda(_args(), [10, 100], device="cpu") == 0.95


@pytest.mark.unit
def test_one_value_per_sample():
    lambdas = _gae_lambda(_args(gae_lambda_alpha=1.5), [10, 100, 1000], device="cpu")
    assert torch.is_tensor(lambdas)
    assert lambdas.shape == (3,)


@pytest.mark.unit
def test_longer_responses_get_a_larger_lambda():
    lambdas = _gae_lambda(_args(gae_lambda_alpha=1.5), [1, 10, 100, 4096], device="cpu")
    assert torch.all(lambdas[1:] > lambdas[:-1])
    assert lambdas[-1] < 1.0


@pytest.mark.unit
def test_matches_the_formula():
    alpha, lengths = 1.5, [7, 250]
    lambdas = _gae_lambda(_args(gae_lambda_alpha=alpha), lengths, device="cpu")
    expected = [1.0 - 1.0 / (alpha * length) for length in lengths]
    assert lambdas.tolist() == pytest.approx(expected, abs=1e-6)


@pytest.mark.unit
def test_never_goes_negative():
    # A small alpha would otherwise drive short samples below zero.
    lambdas = _gae_lambda(_args(gae_lambda_alpha=0.1), [1, 2, 3], device="cpu")
    assert torch.all(lambdas >= 0)


@pytest.mark.unit
def test_a_zero_length_does_not_divide_by_zero():
    lambdas = _gae_lambda(_args(gae_lambda_alpha=1.5), [0], device="cpu")
    assert torch.isfinite(lambdas).all()


@pytest.mark.unit
def test_the_result_feeds_the_gae_recursion():
    from vime.utils.ppo_utils import vanilla_gae

    lambdas = _gae_lambda(_args(gae_lambda_alpha=1.5), [4, 4], device="cpu")
    rewards, values = torch.randn(2, 4), torch.randn(2, 4)
    advantages, _ = vanilla_gae(rewards=rewards, values=values, gamma=1.0, lambd=lambdas)
    assert advantages.shape == (2, 4)
