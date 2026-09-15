import importlib.util
import sys
import types
from pathlib import Path

import pytest


NUM_GPUS = 0


def load_arguments_module(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    training_mod = types.ModuleType("megatron.training")
    arguments_mod = types.ModuleType("megatron.training.arguments")
    tokenizer_pkg_mod = types.ModuleType("megatron.training.tokenizer")
    tokenizer_mod = types.ModuleType("megatron.training.tokenizer.tokenizer")
    transformers_mod = types.ModuleType("transformers")

    arguments_mod.parse_args = lambda *args, **kwargs: None
    arguments_mod.validate_args = lambda args: args
    tokenizer_mod._vocab_size_with_padding = lambda vocab_size, _args: vocab_size
    transformers_mod.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", arguments_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer", tokenizer_pkg_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer.tokenizer", tokenizer_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    module_path = Path(__file__).resolve().parents[1] / "vime" / "backends" / "megatron_utils" / "arguments.py"
    module_name = "test_megatron_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_qwen3_6_args(**overrides):
    values = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_layers=40,
        ffn_hidden_size=512,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        moe_layer_freq=[1] * 40,
        untie_embeddings_and_output_weights=True,
        norm_epsilon=1e-6,
        layernorm_epsilon=1e-6,
        rotary_base=10000000,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_qwen3_6_hf_config():
    text_config = types.SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        intermediate_size=5632,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000000},
    )
    return types.SimpleNamespace(text_config=text_config)


def make_allgather_cp_args(**overrides):
    values = dict(
        allgather_cp=True,
        context_parallel_size=2,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_yarn_args(**overrides):
    rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "factor": 4.0,
        "original_max_position_embeddings": 262144,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 0.0,
        "truncate": True,
    }
    values = dict(
        position_embedding_type="yarn",
        rotary_base=10000000,
        rotary_percent=0.25,
        rotary_scaling_factor=4.0,
        yarn_original_max_position_embeddings=262144,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=0.0,
        yarn_correction_range_round_to_int=True,
        vllm_hf_overrides={"text_config": {"rope_parameters": rope_parameters}},
        vllm_allow_long_max_model_len=True,
        seq_length=131072,
        max_position_embeddings=131072,
        rollout_max_context_len=131072,
        vllm_max_model_len=131072,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_hf_validate_all_moe_skips_dense_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    module._hf_validate_args(make_qwen3_6_args(), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_moe_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    with pytest.raises(AssertionError, match="moe_intermediate_size"):
        module._hf_validate_args(make_qwen3_6_args(moe_ffn_hidden_size=256), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_dense_intermediate_size_when_moe_has_dense_layers(monkeypatch):
    module = load_arguments_module(monkeypatch)

    args = make_qwen3_6_args(moe_layer_freq=[0] + [1] * 39)

    with pytest.raises(AssertionError, match="intermediate_size"):
        module._hf_validate_args(args, make_qwen3_6_hf_config())


@pytest.mark.unit
def test_allgather_cp_rejects_non_dsa_cp_models(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args()
    hf_config = types.SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3")

    with pytest.raises(ValueError, match="only supported for DSA attention models"):
        module._validate_allgather_cp_supported(args, hf_config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "hf_config",
    [
        types.SimpleNamespace(architectures=["DeepseekV32ForCausalLM"], model_type="deepseek_v3"),
        types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], model_type="glm"),
    ],
)
def test_allgather_cp_allows_dsa_architectures(monkeypatch, hf_config):
    module = load_arguments_module(monkeypatch)

    module._validate_allgather_cp_supported(make_allgather_cp_args(), hf_config)


@pytest.mark.unit
def test_allgather_cp_ignores_cp_size_one(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args(context_parallel_size=1)

    module._validate_allgather_cp_supported(args)


@pytest.mark.unit
def test_yarn_training_and_rollout_fingerprints_match(monkeypatch, caplog):
    module = load_arguments_module(monkeypatch)

    with caplog.at_level("INFO"):
        module._validate_yarn_consistency(make_yarn_args())

    assert "Resolved matched YaRN fingerprint" in caplog.text


@pytest.mark.unit
def test_yarn_training_only_replay_does_not_require_rollout_config(monkeypatch, caplog):
    module = load_arguments_module(monkeypatch)
    args = make_yarn_args(
        debug_train_only=True,
        vllm_hf_overrides=None,
        vllm_allow_long_max_model_len=False,
        rollout_max_context_len=None,
        vllm_max_model_len=None,
    )

    with caplog.at_level("INFO"):
        module._validate_yarn_consistency(args)

    assert "Resolved training-only YaRN fingerprint" in caplog.text


@pytest.mark.unit
def test_yarn_training_only_replay_rejects_training_capacity_mismatch(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_yarn_args(
        debug_train_only=True,
        vllm_hf_overrides=None,
        max_position_embeddings=65536,
    )

    with pytest.raises(ValueError, match="training replay capacity mismatch"):
        module._validate_yarn_consistency(args)


@pytest.mark.unit
def test_yarn_fingerprint_mismatch_is_rejected(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_yarn_args()
    args.vllm_hf_overrides["text_config"]["rope_parameters"]["factor"] = 8.0

    with pytest.raises(ValueError, match=r"fingerprint mismatch: factor: training=4\.0, rollout=8\.0"):
        module._validate_yarn_consistency(args)


@pytest.mark.unit
def test_yarn_requires_both_training_and_rollout(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_yarn_args(position_embedding_type="rope")

    with pytest.raises(ValueError, match="must be enabled on both training and rollout"):
        module._validate_yarn_consistency(args)


@pytest.mark.unit
def test_yarn_requires_explicit_vllm_long_length_gate(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_yarn_args(vllm_allow_long_max_model_len=False)

    with pytest.raises(ValueError, match="--vllm-allow-long-max-model-len"):
        module._validate_yarn_consistency(args)


@pytest.mark.unit
def test_yarn_capacity_mismatch_is_rejected(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_yarn_args(vllm_max_model_len=270000)

    with pytest.raises(ValueError, match="capacity mismatch"):
        module._validate_yarn_consistency(args)


@pytest.mark.unit
def test_default_rope_does_not_require_yarn_rollout_fields(monkeypatch):
    module = load_arguments_module(monkeypatch)

    module._validate_yarn_consistency(types.SimpleNamespace(position_embedding_type="rope"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
