"""qwen3_5 spec 构建回归:全注意力层 core_attention 不得被 GDN 修复污染。

run 20260825-172714:eb10b0bd 的就地替换(Mock→DotProductAttention)经共享模板
泄漏到全注意力层 → 本地 DPA 不支持 packed sequence → train forward 断言。
修复:整体换新对象(dataclasses.replace),不碰共享模板;全注意力层遇 Mock
兜底 MindSpeedTEDotProductAttention(与健康路径一致)。

注:子进程隔离执行 —— pytest 环境会使 MindSpeed adaptor 的动态 dataclass 注入
(arguments_basic.py make_dataclass)误炸,与 spec 逻辑无关。

运行: python3 -m pytest tests/test_qwen35_spec_full_attention.py -q
"""
import os
import subprocess
import sys

PROBE = r"""
import tempfile, types, torch
from unittest.mock import MagicMock
torch.distributed.init_process_group(backend="gloo", init_method=f"file://{tempfile.mktemp()}", rank=0, world_size=1)
from megatron.core import parallel_state
parallel_state.initialize_model_parallel(1, 1)
import mindspeed.megatron_adaptor  # noqa: F401
from megatron.core.transformer.transformer_config import TransformerConfig
import vime_plugins.models.qwen3_5 as q35

def mk_config():
    return TransformerConfig(num_layers=4, hidden_size=2048, num_attention_heads=16,
                             moe_layer_freq=[1]*4, num_moe_experts=256, pipeline_model_parallel_size=1)
args = types.SimpleNamespace(num_experts=256,
    hf_checkpoint="/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16", qwen_gdn_backend="npu")

def show(tag, spec):
    for i, ls in enumerate(spec.layer_specs):
        sa = ls.submodules.self_attention
        ca = getattr(sa.submodules, "core_attention", None) if getattr(sa, "submodules", None) else None
        print(f"RESULT [{tag}] layer{i}: sa={getattr(sa.module,'__name__',sa.module)} ca={getattr(ca,'__name__',type(ca).__name__)}")
    print(f"RESULT [{tag}] no_shared_template:", spec.layer_specs[0].submodules is not spec.layer_specs[3].submodules)

# 场景1:健康(TE stub 正常)
show("healthy", q35.get_qwen3_5_spec(args, mk_config(), None))

# 场景2:模拟 TE 补丁丢失(共享模板 core_attention = MagicMock)
real_builder = q35.get_gpt_decoder_block_spec
def fake_builder(config, **kw):
    s = real_builder(config, **kw)
    for ls in s.layer_specs:
        ls.submodules.self_attention.submodules.core_attention = MagicMock()
    return s
q35.get_gpt_decoder_block_spec = fake_builder
show("te_lost", q35.get_qwen3_5_spec(args, mk_config(), None))
"""


def test_full_attention_not_poisoned_in_either_te_state():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        ["/workspace/Megatron-LM", "/workspace/vime", "/workspace/MindSpeed", env.get("PYTHONPATH", "")]
    )
    r = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"probe 失败:\n{r.stderr[-2000:]}"
    lines = [l.split("RESULT ", 1)[1] for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    assert len(lines) == 10, f"输出行数异常: {lines}"
    for tag in ("healthy", "te_lost"):
        got = [l for l in lines if l.startswith(f"[{tag}]")]
        for i in range(3):  # GDN 层:整体换成 vime Attention
            assert f"layer{i}: sa=Attention" in got[i], got
        # 全注意力层:必须是 packed 支持的 MindSpeed 实现,绝不能是本地 DPA 或 Mock
        assert "layer3: sa=SelfAttention ca=MindSpeedTEDotProductAttention" in got[3], got
        assert "no_shared_template: True" in got[4], got  # 模板未被污染
