"""Unit tests for ``vanilla_gae``, including the masked multi-turn variant."""

from __future__ import annotations

import pytest
import torch

from vime.utils.ppo_utils import vanilla_gae

GAMMA_LAMBDA = [(1.0, 1.0), (0.99, 0.95), (0.9, 0.5)]


def _textbook_gae(rewards, values, gamma, lambd):
    """Straightforward reference recursion, one sample at a time."""
    B, T = rewards.shape
    advantages = torch.zeros(B, T)
    for b in range(B):
        running = 0.0
        for t in reversed(range(T)):
            next_value = values[b, t + 1] if t < T - 1 else 0.0
            delta = rewards[b, t] + gamma * next_value - values[b, t]
            running = delta + gamma * lambd * running
            advantages[b, t] = running
    return advantages


@pytest.mark.unit
@pytest.mark.parametrize(("gamma", "lambd"), GAMMA_LAMBDA)
def test_matches_the_textbook_recursion(gamma, lambd):
    torch.manual_seed(0)
    rewards, values = torch.randn(4, 25), torch.randn(4, 25)
    advantages, returns = vanilla_gae(rewards=rewards, values=values, gamma=gamma, lambd=lambd)
    assert torch.allclose(advantages, _textbook_gae(rewards, values, gamma, lambd), atol=1e-5)
    assert torch.allclose(returns, advantages + values)


@pytest.mark.unit
@pytest.mark.parametrize(("gamma", "lambd"), GAMMA_LAMBDA)
def test_an_all_ones_mask_changes_nothing(gamma, lambd):
    torch.manual_seed(0)
    rewards, values = torch.randn(4, 25), torch.randn(4, 25)
    plain, _ = vanilla_gae(rewards=rewards, values=values, gamma=gamma, lambd=lambd)
    masked, _ = vanilla_gae(
        rewards=rewards, values=values, gamma=gamma, lambd=lambd, mask=torch.ones(4, 25)
    )
    assert torch.equal(plain, masked)


@pytest.mark.unit
@pytest.mark.parametrize(("gamma", "lambd"), GAMMA_LAMBDA)
def test_a_uniform_per_sample_lambda_matches_the_scalar(gamma, lambd):
    torch.manual_seed(0)
    rewards, values = torch.randn(4, 25), torch.randn(4, 25)
    scalar, _ = vanilla_gae(rewards=rewards, values=values, gamma=gamma, lambd=lambd)
    per_sample, _ = vanilla_gae(
        rewards=rewards, values=values, gamma=gamma, lambd=torch.full((4,), lambd)
    )
    assert torch.equal(scalar, per_sample)


@pytest.mark.unit
def test_per_sample_lambda_is_applied_per_sample():
    rewards = torch.zeros(2, 4)
    rewards[:, -1] = 1.0
    values = torch.zeros(2, 4)
    lambdas = torch.tensor([1.0, 0.0])
    advantages, _ = vanilla_gae(rewards=rewards, values=values, gamma=1.0, lambd=lambdas)
    # lambda 1 carries the terminal reward all the way back; lambda 0 leaves it where it fell.
    assert torch.allclose(advantages[0], torch.ones(4))
    assert torch.allclose(advantages[1], torch.tensor([0.0, 0.0, 0.0, 1.0]))


@pytest.mark.unit
def test_masked_positions_get_no_advantage():
    torch.manual_seed(0)
    rewards, values = torch.randn(2, 8), torch.randn(2, 8)
    mask = torch.ones(2, 8)
    mask[:, 3:5] = 0
    advantages, _ = vanilla_gae(rewards=rewards, values=values, gamma=0.9, lambd=0.5, mask=mask)
    assert torch.equal(advantages[:, 3:5], torch.zeros(2, 2))
    assert (advantages[:, :3] != 0).any()


@pytest.mark.unit
def test_a_reward_on_a_masked_position_reaches_the_previous_action():
    # One action, then tool output that the episode's reward lands on. Undiscounted, so the
    # whole reward has to arrive at the action.
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    values = torch.zeros(1, 4)
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    advantages, _ = vanilla_gae(rewards=rewards, values=values, gamma=1.0, lambd=1.0, mask=mask)
    assert advantages[0, 0] == pytest.approx(1.0)
    assert torch.equal(advantages[0, 1:], torch.zeros(3))


@pytest.mark.unit
def test_a_carried_reward_is_discounted_once_per_position_it_passes():
    # Pins the discounting, which is only visible below gamma 1: the reward is discounted for
    # each masked position it travels past, not for the one it landed on.
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    values = torch.zeros(1, 4)
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    advantages, _ = vanilla_gae(rewards=rewards, values=values, gamma=0.5, lambd=1.0, mask=mask)
    assert advantages[0, 0] == pytest.approx(0.5**2)


@pytest.mark.unit
def test_masked_values_stay_out_of_the_recursion():
    # A wild prediction on a masked position must not leak into the action's advantage.
    rewards = torch.zeros(1, 3)
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    tame = torch.tensor([[0.0, 0.0, 0.0]])
    wild = torch.tensor([[0.0, 999.0, 0.0]])
    tame_advantages, _ = vanilla_gae(rewards=rewards, values=tame, gamma=0.9, lambd=0.9, mask=mask)
    wild_advantages, _ = vanilla_gae(rewards=rewards, values=wild, gamma=0.9, lambd=0.9, mask=mask)
    assert torch.equal(tame_advantages[:, [0, 2]], wild_advantages[:, [0, 2]])
