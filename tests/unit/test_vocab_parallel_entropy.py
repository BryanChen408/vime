"""Unit tests for the vocab-parallel entropy in ``vime.utils.ppo_utils``.

Runs single-process, so the all-reduces are no-ops and the maths reduces to the tp=1 case.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from vime.utils.ppo_utils import _VocabParallelEntropy

SHAPES = [(4, 7), (3, 129), (2, 1000)]


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None, group=None: None)


def _reference_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Textbook -sum(p * log p), in double, as an independent check."""
    log_probs = torch.log_softmax(logits.double(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


@pytest.mark.unit
@pytest.mark.parametrize("shape", SHAPES)
def test_entropy_matches_the_textbook_formula(single_rank, shape):
    torch.manual_seed(0)
    logits = torch.randn(*shape) * 3
    entropy = _VocabParallelEntropy.apply(logits.clone(), None)
    assert torch.allclose(entropy.double(), _reference_entropy(logits), atol=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize("shape", SHAPES)
def test_gradient_matches_autograd_on_the_textbook_formula(single_rank, shape):
    torch.manual_seed(0)
    logits = torch.randn(*shape) * 3
    grad_output = torch.randn(shape[0])

    ours = logits.clone().requires_grad_(True)
    _VocabParallelEntropy.apply(ours, None).backward(grad_output)

    reference = logits.clone().double().requires_grad_(True)
    _reference_entropy(reference).backward(grad_output.double())

    scale = reference.grad.abs().max().clamp_min(1e-9)
    assert ((ours.grad.double() - reference.grad).abs().max() / scale) < 1e-4


@pytest.mark.unit
def test_backward_keeps_a_single_saved_tensor(single_rank):
    # The gradient is finished in forward so that only it has to be kept alive; keeping the
    # logits and the softmax to combine later would double what the chunk holds.
    torch.manual_seed(0)
    logits = torch.randn(3, 64, requires_grad=True)
    entropy = _VocabParallelEntropy.apply(logits, None)
    assert len(entropy.grad_fn.saved_tensors) == 1


@pytest.mark.unit
def test_input_is_used_as_scratch_and_only_roughly_restored(single_rank):
    # forward borrows the input tensor while assembling the gradient. The undo is a subtract
    # followed by an add, so it comes back close but not bit-identical — which is why every
    # caller in this module hands over a clone rather than the tensor it still needs.
    torch.manual_seed(0)
    logits = torch.randn(3, 64)
    before = logits.clone()
    _VocabParallelEntropy.apply(logits, None)
    assert not torch.equal(logits, before)
    assert torch.allclose(logits, before, atol=1e-5, rtol=1e-3)


@pytest.mark.unit
def test_every_caller_passes_a_clone():
    # Guards the contract above: compute_entropy_from_logits must never be handed a tensor the
    # caller still needs.
    import inspect

    from vime.utils import ppo_utils

    source = inspect.getsource(ppo_utils)
    calls = [line.strip() for line in source.splitlines() if "compute_entropy_from_logits(" in line]
    calls = [line for line in calls if not line.startswith("def ")]
    assert calls, "expected the helper to have callers"
    for call in calls:
        assert ".clone()" in call or "entropy_input" in call, call
