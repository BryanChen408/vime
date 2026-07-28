"""Unit tests for ``distributed_explained_variance``.

Single-process, so the all-reduce is a no-op and the statistics are whatever this rank holds.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from vime.utils.distributed_utils import distributed_explained_variance


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, group=None: None)


def _reference(returns, values, mask):
    kept_returns = returns[mask > 0].double()
    kept_values = values[mask > 0].double()
    error = kept_returns - kept_values
    return 1.0 - (error.var(unbiased=False) / (kept_returns.var(unbiased=False) + 1e-8)).item()


@pytest.mark.unit
def test_perfect_predictions_explain_everything(single_rank):
    torch.manual_seed(0)
    returns = torch.randn(200)
    ones = torch.ones_like(returns)
    assert distributed_explained_variance(returns, returns.clone(), ones) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.unit
def test_predicting_the_mean_explains_nothing(single_rank):
    torch.manual_seed(0)
    returns = torch.randn(200) * 2
    values = torch.full_like(returns, returns.mean().item())
    assert distributed_explained_variance(returns, values, torch.ones_like(returns)) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.unit
def test_worse_than_the_mean_goes_negative(single_rank):
    torch.manual_seed(0)
    returns = torch.randn(200)
    values = -returns * 3
    assert distributed_explained_variance(returns, values, torch.ones_like(returns)) < 0


@pytest.mark.unit
def test_matches_the_reference_formula(single_rank):
    torch.manual_seed(0)
    returns = torch.randn(500) * 2
    values = returns + torch.randn(500) * 0.5
    mask = torch.ones_like(returns)
    got = distributed_explained_variance(returns, values, mask)
    assert got == pytest.approx(_reference(returns, values, mask), abs=1e-5)


@pytest.mark.unit
def test_masked_positions_are_ignored(single_rank):
    torch.manual_seed(0)
    returns = torch.randn(100) * 3
    values = returns.clone()
    values[50:] = 0  # would look terrible if counted
    mask = torch.zeros_like(returns)
    mask[:50] = 1
    assert distributed_explained_variance(returns, values, mask) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.unit
def test_too_few_points_reports_zero(single_rank):
    returns = torch.tensor([1.0])
    assert distributed_explained_variance(returns, returns.clone(), torch.ones(1)) == 0.0


@pytest.mark.unit
def test_result_is_a_plain_float(single_rank):
    # It must not be a tensor: whole-batch ratios bypass the per-sample metric reduction,
    # which would sum them across micro-batches and ranks and then divide by the batch size.
    torch.manual_seed(0)
    returns = torch.randn(50)
    result = distributed_explained_variance(returns, returns.clone(), torch.ones(50))
    assert isinstance(result, float)
