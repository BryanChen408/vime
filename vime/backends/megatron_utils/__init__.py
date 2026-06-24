import logging

import torch
from vime.utils.common import is_npu


def _ensure_npu_adaptor():
    """Lazy-import mindspeed.megatron_adaptor to patch torch for NPU.

    This MUST be called before any megatron code that touches NPU (model
    build, checkpoint load, etc.).  It is deliberately lazy (not at module
    level) so that the **vLLM subprocess** — which only imports
    ``update_weight_from_tensor`` for the colocate worker extension — does
    NOT pull in mindspeed.  mindspeed breaks ``torch.compile``'s
    ``aot_compile`` path, which would otherwise kill cudagraph capture.
    """
    if is_npu():
        import mindspeed.megatron_adaptor  # noqa: F811

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLMoETextRotaryEmbedding,
        Qwen3VLTextRotaryEmbedding,
    )

    def patch_rotary_embedding(cls):
        _original_forward = cls.forward

        def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
            return _original_forward(self, *args, **kwargs)

        cls.forward = _patched_forward

    patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
    patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
except ImportError:
    pass

logging.getLogger("megatron").setLevel(logging.WARNING)

from . import megatron_patch  # noqa: F401, E402
