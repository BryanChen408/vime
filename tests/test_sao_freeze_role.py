"""[SAO C8] Card-free 单测:freeze_model_params 的 role 隔离。
关键断言:critic 专用冻结列表只作用 critic、绝不误冻 actor;全局列表对两个 role 都生效(backward-compat)。
运行:PYTHONPATH=<worktree> python tests/test_sao_freeze_role.py
"""
import argparse

import torch.nn as nn

from vime.backends.megatron_utils.model_provider import freeze_model_params


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.Linear(2, 2)  # 注意力(critic 要冻)
        self.mlp = nn.ModuleDict({"experts": nn.Linear(2, 2)})  # MoE experts(critic 要训)
        self.output_layer = nn.Linear(2, 1)  # value head(critic 要训)


def _args(**kw):
    d = dict(
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        critic_only_train_params_name_list=None,
        critic_freeze_params_name_list=None,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def grad_set(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_critic_list_does_not_touch_actor():
    args = _args(critic_only_train_params_name_list=["mlp.experts", "output_layer"])
    actor, critic = Toy(), Toy()
    freeze_model_params(actor, args, role="actor")
    freeze_model_params(critic, args, role="critic")
    # actor:critic 列表不生效 → 全可训
    assert grad_set(actor) == {n for n, _ in actor.named_parameters()}, "actor 被误冻!"
    # critic:只有 mlp.experts + output_layer 可训,self_attention 冻结
    trainable = grad_set(critic)
    assert all("mlp.experts" in n or "output_layer" in n for n in trainable), trainable
    assert not any("self_attention" in n for n in trainable), "critic 的 attention 没冻!"
    assert trainable, "critic 没有可训参数(regex 全 miss)!"
    print("test_critic_list_does_not_touch_actor OK")


def test_global_list_affects_both_roles():
    args = _args(only_train_params_name_list=["output_layer"])
    actor, critic = Toy(), Toy()
    freeze_model_params(actor, args, role="actor")
    freeze_model_params(critic, args, role="critic")
    for m, tag in [(actor, "actor"), (critic, "critic")]:
        assert grad_set(m) == {n for n, _ in m.named_parameters() if "output_layer" in n}, tag
    print("test_global_list_affects_both_roles OK")


def test_critic_overrides_global_for_critic():
    # 全局训 output_layer;critic 专用训 mlp.experts → critic 用专用(覆盖),actor 用全局。
    args = _args(
        only_train_params_name_list=["output_layer"],
        critic_only_train_params_name_list=["mlp.experts"],
    )
    actor, critic = Toy(), Toy()
    freeze_model_params(actor, args, role="actor")
    freeze_model_params(critic, args, role="critic")
    assert grad_set(actor) == {n for n, _ in actor.named_parameters() if "output_layer" in n}
    assert grad_set(critic) == {n for n, _ in critic.named_parameters() if "mlp.experts" in n}
    print("test_critic_overrides_global_for_critic OK")


if __name__ == "__main__":
    test_critic_list_does_not_touch_actor()
    test_global_list_affects_both_roles()
    test_critic_overrides_global_for_critic()
    print("ALL FREEZE-ROLE TESTS PASSED")
