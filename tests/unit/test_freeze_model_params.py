"""Unit tests for ``freeze_model_params``, including the critic-only lists."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vime.backends.megatron_utils.model_provider import freeze_model_params

PARAM_NAMES = [
    "embedding.word_embeddings.weight",
    "layers.0.self_attention.linear_qkv.weight",
    "layers.0.mlp.experts.weight1",
    "layers.1.mlp.experts.weight1",
    "output_layer.weight",
]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._params = {name: nn.Parameter(torch.zeros(1)) for name in PARAM_NAMES}

    def named_parameters(self, *args, **kwargs):
        return iter(self._params.items())


def _trainable(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def _args(**kwargs):
    defaults = {
        "only_train_params_name_list": None,
        "freeze_params_name_list": None,
        "critic_only_train_params_name_list": None,
        "critic_freeze_params_name_list": None,
    }
    return SimpleNamespace(**{**defaults, **kwargs})


@pytest.mark.unit
@pytest.mark.parametrize("role", ["actor", "critic"])
def test_nothing_configured_leaves_everything_trainable(role):
    model = _Model()
    freeze_model_params(model, _args(), role)
    assert _trainable(model) == set(PARAM_NAMES)


@pytest.mark.unit
@pytest.mark.parametrize("role", ["actor", "critic"])
def test_the_global_list_applies_to_every_role(role):
    model = _Model()
    freeze_model_params(model, _args(only_train_params_name_list=["experts"]), role)
    assert _trainable(model) == {"layers.0.mlp.experts.weight1", "layers.1.mlp.experts.weight1"}


@pytest.mark.unit
def test_the_critic_list_never_reaches_the_actor():
    model = _Model()
    freeze_model_params(model, _args(critic_only_train_params_name_list=["experts"]), "actor")
    assert _trainable(model) == set(PARAM_NAMES)


@pytest.mark.unit
def test_the_critic_list_replaces_the_global_one_for_the_critic():
    args = _args(
        only_train_params_name_list=["embedding"],
        critic_only_train_params_name_list=["mlp.experts", "output_layer"],
    )
    critic, actor = _Model(), _Model()
    freeze_model_params(critic, args, "critic")
    freeze_model_params(actor, args, "actor")
    assert _trainable(critic) == {
        "layers.0.mlp.experts.weight1",
        "layers.1.mlp.experts.weight1",
        "output_layer.weight",
    }
    assert _trainable(actor) == {"embedding.word_embeddings.weight"}


@pytest.mark.unit
def test_the_freeze_list_removes_from_what_is_trainable():
    model = _Model()
    freeze_model_params(model, _args(freeze_params_name_list=["embedding", "output_layer"]), "actor")
    assert _trainable(model) == {
        "layers.0.self_attention.linear_qkv.weight",
        "layers.0.mlp.experts.weight1",
        "layers.1.mlp.experts.weight1",
    }


@pytest.mark.unit
def test_the_freeze_list_is_applied_after_the_train_list():
    args = _args(only_train_params_name_list=["experts"], freeze_params_name_list=["layers.1"])
    model = _Model()
    freeze_model_params(model, args, "actor")
    assert _trainable(model) == {"layers.0.mlp.experts.weight1"}


@pytest.mark.unit
def test_patterns_are_regexes():
    model = _Model()
    freeze_model_params(model, _args(only_train_params_name_list=[r"layers\.[01]\.mlp"]), "actor")
    assert _trainable(model) == {"layers.0.mlp.experts.weight1", "layers.1.mlp.experts.weight1"}


@pytest.mark.unit
def test_an_empty_critic_list_falls_back_to_the_global_one():
    args = _args(only_train_params_name_list=["experts"], critic_only_train_params_name_list=[])
    model = _Model()
    freeze_model_params(model, args, "critic")
    assert _trainable(model) == {"layers.0.mlp.experts.weight1", "layers.1.mlp.experts.weight1"}
