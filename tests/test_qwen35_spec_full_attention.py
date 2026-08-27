"""Regression coverage for the Qwen3.5 MindSpeed full-attention lifecycle."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = REPO_ROOT.parent / "Megatron-LM"
MINDSPEED_ROOT = REPO_ROOT.parent / "MindSpeed"
HAS_LOCAL_MINDSPEED_STACK = MEGATRON_ROOT.is_dir() and MINDSPEED_ROOT.is_dir()


def _named_calls(node: ast.AST, names: set[str]) -> dict[str, list[int]]:
    calls = {name: [] for name in names}
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in names:
            calls[child.func.id].append(child.lineno)
    return calls


@pytest.mark.unit
def test_synchronous_driver_does_not_import_mindspeed():
    """MindSpeed patch state belongs to Ray training workers, not the driver."""
    tree = ast.parse((REPO_ROOT / "train.py").read_text())
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert not [module for module in imported_modules if module == "mindspeed" or module.startswith("mindspeed.")]


@pytest.mark.unit
def test_actor_uses_mindspeed_full_args_before_megatron_model_parallel_init():
    """Lock the verl lifecycle and the canonical MindSpeed argument handoff."""
    actor_path = REPO_ROOT / "vime" / "backends" / "megatron_utils" / "actor.py"
    tree = ast.parse(actor_path.read_text())
    actor_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    init_method = next(
        node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name == "init"
    )
    repatch_imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "mindspeed.megatron_adaptor"
        for alias in node.names
    ]
    full_args_imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "mindspeed.args_utils"
        for alias in node.names
    ]
    super_init_lines = [
        child.lineno
        for child in ast.walk(init_method)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "init"
        and isinstance(child.func.value, ast.Call)
        and isinstance(child.func.value.func, ast.Name)
        and child.func.value.func.id == "super"
    ]

    calls = _named_calls(
        init_method,
        {
            "repatch",
            "get_full_args",
            "init",
            "initialize_context_parallel_group_for_double_ring",
            "initialize_context_parallel_group_for_hybrid_cp",
            "initialize_context_parallel_group_for_send_recv_overlap",
        },
    )
    canonical_args_lines = [
        child.lineno
        for child in ast.walk(init_method)
        if isinstance(child, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "args" for target in child.targets)
        and isinstance(child.value, ast.Call)
        and isinstance(child.value.func, ast.Name)
        and child.value.func.id == "get_full_args"
    ]
    all_self_args_lines = [
        child.lineno
        for child in ast.walk(init_method)
        if isinstance(child, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "args"
            for target in child.targets
        )
        and isinstance(child.value, ast.Name)
        and child.value.id == "args"
    ]
    assert repatch_imports == ["repatch"]
    assert full_args_imports == ["get_full_args"]
    assert len(super_init_lines) == 1
    assert len(calls["repatch"]) == 1
    assert len(calls["get_full_args"]) == 1
    assert len(canonical_args_lines) == 1
    self_args_lines = [line for line in all_self_args_lines if line > calls["get_full_args"][0]]
    assert len(self_args_lines) == 1
    assert len(calls["init"]) == 1
    assert (
        super_init_lines[0]
        < calls["repatch"][0]
        < calls["get_full_args"][0]
        == canonical_args_lines[0]
        < self_args_lines[0]
        < calls["init"][0]
    )
    assert not calls["initialize_context_parallel_group_for_double_ring"]
    assert not calls["initialize_context_parallel_group_for_hybrid_cp"]
    assert not calls["initialize_context_parallel_group_for_send_recv_overlap"]


SPEC_PROBE = r"""
import inspect
import tempfile
import types

import torch

from mindspeed.args_utils import get_full_args
from mindspeed.megatron_adaptor import repatch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import set_args
import vime_plugins.models.qwen3_5 as q35

# Match the production order: the Ray actor initializes torch.distributed, then
# repatch enriches MindSpeed's full Namespace before Megatron installs it as the
# global args object and initializes model-parallel groups.
store_dir = tempfile.TemporaryDirectory()
torch.distributed.init_process_group(
    backend="gloo", init_method=f"file://{store_dir.name}/store", rank=0, world_size=1
)
raw_args = types.SimpleNamespace(context_parallel_size=8, context_parallel_algo="megatron_cp_algo")
assert not hasattr(raw_args, "use_cp_send_recv_overlap")
repatch(raw_args)
full_args = get_full_args()
assert full_args is not raw_args
assert full_args.context_parallel_size == 8
assert full_args.use_cp_send_recv_overlap is False
assert full_args.tp_2d is False
set_args(full_args)

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from mindspeed.core.context_parallel.adaptor import MindSpeedCPDotProductAttention

native_core_attention = TESpecProvider().core_attention()
assert native_core_attention is MindSpeedCPDotProductAttention
assert "packed_seq_params" in inspect.signature(native_core_attention.forward).parameters

from megatron.core import parallel_state
from mindspeed.core.context_parallel import model_parallel_utils as cp_group_utils

# One process is sufficient for spec construction.  The CP=8 choice was already
# applied by repatch above; use CP=1 only for this probe's local process group.
double_ring_calls = []
cp_group_utils.initialize_context_parallel_group_for_double_ring = (
    lambda *call_args: double_ring_calls.append(call_args)
)
parallel_state.initialize_model_parallel(1, 1, context_parallel_size=1)
assert len(double_ring_calls) == 1
assert double_ring_calls[0][:3] == (1, 1, 1)


def make_config():
    config = TransformerConfig(
        num_layers=8,
        hidden_size=2048,
        num_attention_heads=16,
        moe_layer_freq=[1] * 8,
        num_moe_experts=256,
        pipeline_model_parallel_size=1,
        context_parallel_size=8,
    )
    config.context_parallel_algo = "megatron_cp_algo"
    config.qkv_format = "thd"
    return config


args = types.SimpleNamespace(
    num_experts=256,
    hf_checkpoint="unused-by-test",
    qwen_gdn_backend="npu",
    qkv_format="thd",
    context_parallel_size=8,
)
q35._load_hf_config = lambda _: types.SimpleNamespace(num_hidden_layers=8, full_attention_interval=4)

spec = q35.get_qwen3_5_spec(args, make_config(), None)
linear_layers = (0, 1, 2, 4, 5, 6)
full_layers = (3, 7)

for layer_id in linear_layers:
    assert spec.layer_specs[layer_id].submodules.self_attention.module is q35.Attention
for layer_id in full_layers:
    self_attention = spec.layer_specs[layer_id].submodules.self_attention
    assert self_attention.module.__name__ == "SelfAttention"
    assert self_attention.submodules.core_attention is MindSpeedCPDotProductAttention
for linear_layer in linear_layers:
    for full_layer in full_layers:
        assert spec.layer_specs[linear_layer].submodules is not spec.layer_specs[full_layer].submodules


class UncopyableMock:
    def __deepcopy__(self, memo):
        raise TypeError("cannot pickle 'cell' object")


# Reproduce the old poisoned provider state.  GDN construction must no longer
# deepcopy the unusable self-attention object, and full attention must reject
# the broken lifecycle instead of guessing/hard-coding another implementation.
real_builder = q35.get_gpt_decoder_block_spec


def broken_builder(config, **kwargs):
    block_spec = real_builder(config, **kwargs)
    for layer_spec in block_spec.layer_specs:
        layer_spec.submodules.self_attention.submodules.core_attention = UncopyableMock()
    return block_spec


q35.get_gpt_decoder_block_spec = broken_builder
try:
    q35.get_qwen3_5_spec(args, make_config(), None)
except RuntimeError as error:
    assert "repatch(args) must run before" in str(error)
else:
    raise AssertionError("broken MindSpeed patch lifecycle was silently accepted")
"""


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LOCAL_MINDSPEED_STACK, reason="requires local Megatron-LM and MindSpeed checkouts")
def test_cp8_spec_uses_mindspeed_native_mapping_without_shared_template_pollution():
    env = dict(os.environ)
    env["GLOO_SOCKET_IFNAME"] = "lo"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(MEGATRON_ROOT), str(REPO_ROOT), str(MINDSPEED_ROOT), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", SPEC_PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"spec probe failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
