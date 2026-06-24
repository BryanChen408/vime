"""Lazy bridge imports — defer until accessed to avoid pulling in heavy mbridge deps
(e.g. ``qwen2_5_vl.transformer_config`` has a dataclass with a mutable
``CompilationConfig`` default that breaks when vllm compilation config is set).
"""


def __getattr__(name: str):
    if name == "DeepseekV32Bridge":
        from .deepseek_v32 import DeepseekV32Bridge as _cls
        return _cls
    if name == "GLM4Bridge":
        from .glm4 import GLM4Bridge as _cls
        return _cls
    if name == "GLM4MoEBridge":
        from .glm4moe import GLM4MoEBridge as _cls
        return _cls
    if name == "GLM4MoELiteBridge":
        from .glm4moe_lite import GLM4MoELiteBridge as _cls
        return _cls
    if name == "GptOssBridge":
        from .gpt_oss import GptOssBridge as _cls
        return _cls
    if name == "MimoBridge":
        from .mimo import MimoBridge as _cls
        return _cls
    if name == "MiniMaxM2Bridge":
        from .minimax_m2 import MiniMaxM2Bridge as _cls
        return _cls
    if name == "Qwen3_5Bridge":
        from .qwen3_5 import Qwen3_5Bridge as _cls
        return _cls
    if name == "Qwen3NextBridge":
        from .qwen3_next import Qwen3NextBridge as _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DeepseekV32Bridge",
    "GLM4Bridge",
    "GLM4MoEBridge",
    "GLM4MoELiteBridge",
    "GptOssBridge",
    "MimoBridge",
    "MiniMaxM2Bridge",
    "Qwen3NextBridge",
    "Qwen3_5Bridge",
]
