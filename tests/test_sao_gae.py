"""[SAO C10/C12b] Card-free 单测:skip-observation GAE + 逐样本 λ 的正确性。
关键断言:mask 全 1 时与标准 GAE 逐位一致(backward-compat);skip-obs 手算例;终止 reward 落 masked 尾仍回流到最后 action。
运行:PYTHONPATH=<worktree> python tests/test_sao_gae.py
"""
import torch

from vime.utils.ppo_utils import vanilla_gae


def ref_gae(rewards, values, gamma, lam):
    B, T = rewards.shape
    adv = torch.zeros(B, T)
    last = torch.zeros(B)
    for t in reversed(range(T)):
        nv = values[:, t + 1] if t < T - 1 else torch.zeros(B)
        delta = rewards[:, t] + gamma * nv - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv


def test_backward_compat():
    torch.manual_seed(0)
    r, v = torch.randn(3, 7), torch.randn(3, 7)
    adv, ret = vanilla_gae(r, v, 0.99, 0.95)
    assert torch.allclose(adv, ref_gae(r, v, 0.99, 0.95), atol=1e-5)
    assert torch.allclose(ret, adv + v, atol=1e-5)
    print("test_backward_compat OK")


def test_mask_all_ones_equals_standard():
    torch.manual_seed(1)
    r, v = torch.randn(2, 5), torch.randn(2, 5)
    adv_m, _ = vanilla_gae(r, v, 1.0, 1.0, mask=torch.ones(2, 5))
    adv_s, _ = vanilla_gae(r, v, 1.0, 1.0)
    assert torch.allclose(adv_m, adv_s, atol=1e-5), (adv_m, adv_s)
    # 也和 gamma/lambda<1 比
    adv_m2, _ = vanilla_gae(r, v, 0.9, 0.8, mask=torch.ones(2, 5))
    assert torch.allclose(adv_m2, ref_gae(r, v, 0.9, 0.8), atol=1e-5)
    print("test_mask_all_ones_equals_standard OK")


def test_per_sample_lambda():
    torch.manual_seed(2)
    r, v = torch.randn(2, 4), torch.randn(2, 4)
    lam = torch.tensor([0.9, 0.5])
    adv, _ = vanilla_gae(r, v, 1.0, lam)
    for i in range(2):
        assert torch.allclose(adv[i : i + 1], ref_gae(r[i : i + 1], v[i : i + 1], 1.0, float(lam[i])), atol=1e-5)
    print("test_per_sample_lambda OK")


def test_skip_observation_small():
    # T=3: action, observation, action(terminal reward=5). gamma=lambda=1.
    r = torch.tensor([[0.0, 0.0, 5.0]])
    v = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    adv, ret = vanilla_gae(r, v, 1.0, 1.0, mask=mask)
    # 手算:t2 action: gae=5-3=2; t1 obs: adv=0,链不变; t0 action: 0+next_av(3)-1+last(2)=4
    assert torch.allclose(adv, torch.tensor([[4.0, 0.0, 2.0]]), atol=1e-5), adv
    assert torch.allclose(ret, adv + v, atol=1e-5)
    print("test_skip_observation_small OK")


def test_terminal_reward_on_masked_tail():
    # 终止 reward=7 落在 masked(observation)尾 token → 必须仍回流到最后一个 action。
    r = torch.tensor([[0.0, 0.0, 7.0]])
    v = torch.tensor([[1.0, 2.0, 0.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    adv, _ = vanilla_gae(r, v, 1.0, 1.0, mask=mask)
    # t2 obs: carry=7,adv=0; t1 action: 7+0-2=5; t0 action: next_av(2)-1+last(5)=6
    assert torch.allclose(adv, torch.tensor([[6.0, 5.0, 0.0]]), atol=1e-5), adv
    print("test_terminal_reward_on_masked_tail OK (终止 reward 穿过 masked 尾回流)")


def test_explained_variance():
    # [SAO Q-15] EV = 1 − Var(R−V)/Var(R)。
    from vime.utils.ppo_utils import explained_variance

    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.allclose(explained_variance(r, r.clone()), torch.tensor(1.0), atol=1e-4)  # 完美预测 → 1
    assert explained_variance(r, torch.full_like(r, r.mean())).abs() < 1e-3  # 只预测均值 → 0
    assert explained_variance(r, -r) < 0  # 比均值还差 → <0
    assert explained_variance(torch.tensor([5.0]), torch.tensor([0.0])) == 0.0  # numel<=1 兜底
    print("test_explained_variance OK")


def test_critic_update_count_logic():
    # [SAO C12a] 镜像 train_async.py 的 K 循环:K → 调 K 次 async_train,末次 ref 传 actor;K=1 逐位不变。
    calls = []

    def fake_async_train(_rollout_id, data):
        calls.append(data)
        return f"ref{len(calls)}"

    def run(K):
        calls.clear()
        value_refs = fake_async_train(0, "d")
        for _ in range(max(1, K) - 1):  # 中间 ray.get 省略,不影响计数/末次 ref
            value_refs = fake_async_train(0, "d")
        return value_refs, len(calls)

    assert run(1) == ("ref1", 1)
    assert run(2) == ("ref2", 2)
    assert run(3) == ("ref3", 3)
    print("test_critic_update_count_logic OK (K→K calls, last ref→actor)")


if __name__ == "__main__":
    test_backward_compat()
    test_mask_all_ones_equals_standard()
    test_per_sample_lambda()
    test_skip_observation_small()
    test_terminal_reward_on_masked_tail()
    test_explained_variance()
    test_critic_update_count_logic()
    print("ALL GAE TESTS PASSED")
