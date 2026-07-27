import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def install_bridge_stubs():
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    models_mod = types.ModuleType("megatron.core.models")
    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    gpt_layer_specs_mod = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    gpt_layer_specs_mod.get_gpt_mtp_block_spec = lambda _config, transformer_layer_spec, **_kwargs: (
        "mtp-spec",
        transformer_layer_spec,
    )

    mbridge_mod = types.ModuleType("mbridge")
    mbridge_core_mod = types.ModuleType("mbridge.core")
    mbridge_models_mod = types.ModuleType("mbridge.models")

    def register_model(_names):
        def decorator(cls):
            return cls

        return decorator

    class Qwen2MoEBridge:
        _MLP_MAPPING = {
            "shared_experts.linear_fc1.weight": [
                "model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
                "model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
            ],
            "pre_mlp_layernorm": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
            "shared_experts.linear_fc2.weight": ["model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"],
            "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
            "shared_experts.gate_weight": ["model.layers.{layer_number}.mlp.shared_expert_gate.weight"],
            "mlp.experts.linear_fc1": [
                "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
                "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
            ],
            "mlp.experts.linear_fc2": ["model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"],
        }

        def _weight_name_mapping_mlp(self, name: str) -> list[str]:
            layer_number = name.split(".")[2]
            convert_names = []
            for keyword, mapping_names in self._MLP_MAPPING.items():
                if keyword in name:
                    if "{expert_id}" in mapping_names[0]:
                        expert_id = name.split("weight")[-1]
                        convert_names.extend(
                            [x.format(layer_number=layer_number, expert_id=expert_id) for x in mapping_names]
                        )
                    else:
                        convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                    break
            if len(convert_names) == 0:
                raise NotImplementedError(f"Unsupported parameter name: {name}")
            return convert_names

        def _weight_name_mapping_attention(self, name: str) -> list[str]:
            raise NotImplementedError(f"Unexpected attention mapping lookup: {name}")

        def _get_transformer_layer_spec(self, vp_stage=None):
            return "REAL_LAYER_SPEC" if vp_stage is None else f"REAL_LAYER_SPEC_VP{vp_stage}"

        def _get_gptmodel_args(self) -> dict:
            return {"base": "ok"}

        def _model_provider(self, callbacks):
            def provider(pre_process, post_process, vp_stage=None):
                transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)
                gptmodel_args = self._get_gptmodel_args()
                return {"transformer_layer_spec": transformer_layer_spec, **gptmodel_args}

            return provider

        def _weight_to_mcore_format(self, _mcore_weights_name, hf_weights):
            assert len(hf_weights) == 1
            return hf_weights[0]

        def _weight_to_hf_format(self, mcore_weights_name, mcore_weights):
            return [mcore_weights_name], [mcore_weights]

        def _build_base_config(self, **kwargs):
            return kwargs

    mbridge_core_mod.register_model = register_model
    mbridge_models_mod.Qwen2MoEBridge = Qwen2MoEBridge

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.models"] = models_mod
    sys.modules["megatron.core.models.gpt"] = gpt_mod
    sys.modules["megatron.core.models.gpt.gpt_layer_specs"] = gpt_layer_specs_mod
    sys.modules["mbridge"] = mbridge_mod
    sys.modules["mbridge.core"] = mbridge_core_mod
    sys.modules["mbridge.models"] = mbridge_models_mod


def load_bridge_module():
    install_bridge_stubs()
    # The bridge imports gdn_param_mapping relatively, so it needs a package to be loaded from.
    package_dir = Path(__file__).resolve().parents[1] / "vime_plugins" / "mbridge"
    package_name = "test_qwen3_5_bridge_pkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    sys.modules[package_name] = package

    module_name = f"{package_name}.qwen3_5"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, package_dir / "qwen3_5.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_raw_export_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "vime" / "backends" / "megatron_utils" / "megatron_to_hf" / "qwen3_5.py"
    )
    module_name = "test_qwen3_5_raw_export_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_mtp_moe_expert_mapping_uses_fused_hf_weights():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)

    fc1_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight42")
    fc2_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight42")

    assert fc1_names == ["mtp.layers.0.mlp.experts.gate_up_proj"]
    assert fc2_names == ["mtp.layers.0.mlp.experts.down_proj"]


@pytest.mark.unit
def test_mtp_dense_mlp_mapping_still_uses_dense_hf_weights():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)

    fc1_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.linear_fc1.weight")
    fc2_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.linear_fc2.weight")

    assert fc1_names == ["mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight"]
    assert fc2_names == ["mtp.layers.0.mlp.down_proj.weight"]


@pytest.mark.unit
def test_mtp_block_spec_uses_current_transformer_layer_spec():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.config = "CONFIG_OBJECT"
    bridge.hf_config = types.SimpleNamespace(text_config=types.SimpleNamespace(mtp_num_hidden_layers=1))

    provider = bridge._model_provider([])
    result = provider(True, True, vp_stage=3)

    assert result["transformer_layer_spec"] == "REAL_LAYER_SPEC_VP3"
    assert result["mtp_block_spec"] == ("mtp-spec", "REAL_LAYER_SPEC_VP3")


@pytest.mark.unit
def test_tied_qwen3_5_uses_language_embedding_for_output_layer():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.hf_config = types.SimpleNamespace(text_config=types.SimpleNamespace(tie_word_embeddings=True))

    bridge._adjust_mapping_for_shared_weights()

    assert bridge._DIRECT_MAPPING["output_layer.weight"] == "model.language_model.embed_tokens.weight"
    assert module.Qwen3_5Bridge._DIRECT_MAPPING["output_layer.weight"] == "lm_head.weight"


@pytest.mark.unit
def test_eh_proj_keeps_column_order_when_loading_to_mcore():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)

    weight = torch.arange(24, dtype=torch.float32).view(3, 8)
    converted = bridge._weight_to_mcore_format("mtp.layers.0.eh_proj.weight", [weight])

    assert torch.equal(converted, weight)


@pytest.mark.unit
def test_build_config_enables_gated_attention_when_transformer_config_supports_it():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.hf_config = types.SimpleNamespace(text_config=types.SimpleNamespace(mtp_num_hidden_layers=1))
    bridge.TransformerConfigClass = types.SimpleNamespace(
        __dataclass_fields__={
            "mtp_num_layers": None,
            "attention_output_gate": None,
            "use_gated_attention": None,
        }
    )

    config = bridge._build_config()

    assert config["mtp_num_layers"] == 1
    assert config["attention_output_gate"] is True
    assert config["use_gated_attention"] is True


@pytest.mark.unit
def test_build_config_skips_gated_attention_when_transformer_config_does_not_support_it():
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.hf_config = types.SimpleNamespace(text_config=types.SimpleNamespace(mtp_num_hidden_layers=1))
    bridge.TransformerConfigClass = types.SimpleNamespace(
        __dataclass_fields__={
            "mtp_num_layers": None,
            "attention_output_gate": None,
        }
    )

    config = bridge._build_config()

    assert config["mtp_num_layers"] == 1
    assert config["attention_output_gate"] is True
    assert "use_gated_attention" not in config


@pytest.mark.unit
def test_raw_qwen3_5_mtp_export_keeps_eh_proj_column_order():
    module = load_raw_export_module()

    weight = torch.arange(24, dtype=torch.float32).view(3, 8)
    converted = module.convert_qwen3_5_to_hf(
        types.SimpleNamespace(), "module.module.mtp.layers.0.eh_proj.weight", weight
    )

    assert converted == [("mtp.fc.weight", weight)]


def load_gdn_param_mapping_module():
    """Load gdn_param_mapping directly; it only needs torch, no megatron stubs."""
    module_path = Path(__file__).resolve().parents[1] / "vime_plugins" / "mbridge" / "gdn_param_mapping.py"
    module_name = "test_gdn_param_mapping_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def gdn_config():
    """Small GDN geometry whose dims stay divisible by the tp sizes under test."""
    return types.SimpleNamespace(
        hidden_size=32,
        linear_key_head_dim=4,
        linear_value_head_dim=8,
        linear_num_key_heads=8,
        linear_num_value_heads=16,
    )


def gdn_dims(config):
    return (
        config.linear_key_head_dim * config.linear_num_key_heads,
        config.linear_value_head_dim * config.linear_num_value_heads,
    )


@pytest.mark.unit
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_gdn_in_proj_merge_split_roundtrip(tp_size):
    module = load_gdn_param_mapping_module()
    config = gdn_config()
    qk_dim, v_dim = gdn_dims(config)
    generator = torch.Generator().manual_seed(0)
    hidden = config.hidden_size
    qkv = torch.randn(qk_dim * 2 + v_dim, hidden, generator=generator)
    z = torch.randn(v_dim, hidden, generator=generator)
    b = torch.randn(config.linear_num_value_heads, hidden, generator=generator)
    a = torch.randn(config.linear_num_value_heads, hidden, generator=generator)

    qkvz, ba = module._fuse_gdn_separate_to_grouped(config, qkv, z, b, a)
    in_proj = module.merge_gdn_linear_weights(config, qkvz, ba, tp_size=tp_size)
    assert in_proj.shape == (qk_dim * 2 + v_dim * 2 + config.linear_num_value_heads * 2, hidden)

    qkvz_back, ba_back = module.split_gdn_linear_weights(config, in_proj, tp_size=tp_size)
    qkv_back, z_back, b_back, a_back = module._split_gdn_grouped_to_separate(config, qkvz_back, ba_back)

    assert torch.equal(qkv_back, qkv)
    assert torch.equal(z_back, z)
    assert torch.equal(b_back, b)
    assert torch.equal(a_back, a)


@pytest.mark.unit
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_gdn_conv1d_interleave_roundtrip(tp_size):
    module = load_gdn_param_mapping_module()
    config = gdn_config()
    qk_dim, v_dim = gdn_dims(config)
    conv = torch.randn(qk_dim * 2 + v_dim, 1, 4, generator=torch.Generator().manual_seed(1))

    interleaved = module.interleave_gdn_conv1d(conv, config, tp_size)
    assert interleaved.shape == conv.shape

    restored = module.deinterleave_gdn_conv1d(interleaved, config, tp_size)
    assert torch.equal(restored, conv)


@pytest.mark.unit
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_gdn_conv1d_interleave_keeps_ranks_segment_aligned(tp_size):
    """Each rank's chunk must hold whole [q|k|v] segments, in that order."""
    module = load_gdn_param_mapping_module()
    config = gdn_config()
    qk_dim, v_dim = gdn_dims(config)
    marker = torch.cat(
        [torch.zeros(qk_dim, 1, 1), torch.ones(qk_dim, 1, 1), torch.full((v_dim, 1, 1), 2.0)]
    )

    chunks = module.interleave_gdn_conv1d(marker, config, tp_size).chunk(tp_size, dim=0)
    expected = torch.cat(
        [
            torch.zeros(qk_dim // tp_size),
            torch.ones(qk_dim // tp_size),
            torch.full((v_dim // tp_size,), 2.0),
        ]
    )
    for chunk in chunks:
        assert torch.equal(chunk.flatten(), expected)


@pytest.mark.unit
def test_gdn_conv1d_interleave_rejects_indivisible_tp_size():
    module = load_gdn_param_mapping_module()
    config = gdn_config()
    qk_dim, v_dim = gdn_dims(config)
    conv = torch.zeros(qk_dim * 2 + v_dim, 1, 4)

    with pytest.raises(AssertionError, match="divisible by tp_size"):
        module.interleave_gdn_conv1d(conv, config, tp_size=5)
