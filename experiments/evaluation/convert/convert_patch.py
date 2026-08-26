"""评测专用的转换补丁：不改 vime 源码，运行时覆盖两处框架行为。

Qwen3.6-35B-A3B（Qwen3_5Moe）转 HF 时，框架默认行为不适用，这里在评测入口打补丁：
1. get_expert_param：Qwen3_5Moe 的 HF 用 grouped experts（gate_up_proj/down_proj），
   保持不展开（框架默认把 grouped 展开成逐个 expert，产出错误格式）。
2. convert_qwen3_5_to_hf：
   - 跳过 critic 的 value head（output_layer.weight [1, hidden] / output_layer.bias），
     lm_head 由 save_tensors 从 origin HF 补齐。
   - grouped experts 分支匹配 torch_dist 的真实 key（mlp.experts.experts.linear_fc1.weight）。
"""

import importlib.util
import re
from pathlib import Path

VIME_ROOT = Path(__file__).resolve().parents[3]

# 按路径加载 tools/convert_torch_dist_to_hf.py，绕开 tools 包的命名冲突（无需 __init__.py）。
_spec = importlib.util.spec_from_file_location(
    "eval_convert_base", VIME_ROOT / "tools" / "convert_torch_dist_to_hf.py"
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

import vime.backends.megatron_utils.megatron_to_hf as m2h


# ---- 补丁 1：Qwen3_5Moe 的 experts 保持 grouped，不展开 ----

_orig_get_expert_param = base.get_expert_param


def _get_expert_param(args, name, param):
    if ".experts." not in name:
        yield name, param
        return
    if hasattr(args, "linear_num_key_heads"):
        yield name, param
        return
    yield from _orig_get_expert_param(args, name, param)


base.get_expert_param = _get_expert_param


# ---- 补丁 2：output_layer 跳过 value head + grouped experts 匹配 ----

_orig_convert_qwen3_5 = m2h.convert_qwen3_5_to_hf


def _convert_qwen3_5(args, name, param):
    # critic 的 value head：评测只需要 actor，跳过（lm_head 从 origin HF 补齐）。
    if name == "module.module.output_layer.bias":
        return []
    if name == "module.module.output_layer.weight" and param.shape[0] == 1:
        return []

    # grouped experts：匹配 torch_dist 的真实 key，输出 grouped 的 gate_up_proj/down_proj。
    m = re.match(
        r"module\.module\.decoder\.layers\.(\d+)\.mlp\.experts\.experts\.(linear_fc[12])\.weight",
        name,
    )
    if m:
        layer_idx, fc = m.groups()
        if fc == "linear_fc1":
            return [(f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj", param)]
        return [(f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj", param)]

    return _orig_convert_qwen3_5(args, name, param)


m2h.convert_qwen3_5_to_hf = _convert_qwen3_5
