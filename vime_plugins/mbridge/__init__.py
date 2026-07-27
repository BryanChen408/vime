"""Lazy bridge imports.

Importing the bridges eagerly pulls in heavy mbridge dependencies; one of them
(``qwen2_5_vl.transformer_config``) has a dataclass whose ``CompilationConfig`` default is
mutable and breaks once a vllm compilation config has been set.
"""

from importlib import import_module

_BRIDGES = {
    "DeepseekV32Bridge": ".deepseek_v32",
    "GLM4Bridge": ".glm4",
    "GLM4MoEBridge": ".glm4moe",
    "GLM4MoELiteBridge": ".glm4moe_lite",
    "GptOssBridge": ".gpt_oss",
    "MimoBridge": ".mimo",
    "MiniMaxM2Bridge": ".minimax_m2",
    "Qwen3NextBridge": ".qwen3_next",
    "Qwen3_5Bridge": ".qwen3_5",
}

__all__ = sorted(_BRIDGES)


def __getattr__(name: str):
    if name not in _BRIDGES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_BRIDGES[name], __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_BRIDGES))
