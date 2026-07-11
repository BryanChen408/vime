"""No-card verification for F-PPO-1: chunk-lm-head patch must NOT bypass the
critic value head (output dim 1). Stubs megatron's GPTModel, applies the REAL
patched closure from the edited module, and checks the gate decision.

Detection: the bypass block sets module-global _LM_HEAD_WEIGHT; the early
fall-through (return _orig_forward) does not. So:
  - value head (weight.shape[0]==1)  -> fall-through -> _LM_HEAD_WEIGHT stays None
  - LM head    (weight.shape[0]==1000) -> bypass      -> _LM_HEAD_WEIGHT set
"""
import os
import sys
import types

import torch

# --- stub megatron.core.models.gpt.gpt_model.GPTModel BEFORE import ---
for pkg in ("megatron", "megatron.core", "megatron.core.models",
            "megatron.core.models.gpt", "megatron.core.models.gpt.gpt_model"):
    if pkg not in sys.modules:
        sys.modules[pkg] = types.ModuleType(pkg)

_ORIG_MARKER = {"ran": False}


class FakeGPTModel:
    # original forward returns hidden; records that it ran (proves fall-through
    # dispatches to the real value forward for the critic).
    def forward(self, *a, **k):
        _ORIG_MARKER["ran"] = True
        return torch.zeros(1, 4, self._out_dim)  # [b, s, out_dim]


sys.modules["megatron.core.models.gpt.gpt_model"].GPTModel = FakeGPTModel

import chunked_lm_head_patch as P  # the EDITED copy in this dir

os.environ["QWEN36_CHUNK_LMHEAD"] = "1"
P.apply_chunked_lm_head_patch()          # patches FakeGPTModel.forward
assert getattr(FakeGPTModel, "_chunked_lm_head_patched", False), "patch not applied"


class _OutputLayer:
    def __init__(self, out_features, hidden=8):
        self.weight = torch.zeros(out_features, hidden)  # [out, in]
        self.sequence_parallel = False
        self.tp_group = None

    def forward(self, *a, **k):
        return (torch.zeros(4, 1, self.weight.shape[0]), None)


def _make(out_features):
    m = FakeGPTModel()
    m.post_process = True
    m.mtp_process = False
    m.output_layer = _OutputLayer(out_features)
    m._out_dim = 8 if out_features == 1 else out_features  # value head returns hidden dim
    return m


def _run(out_features):
    P._LM_HEAD_WEIGHT = None
    _ORIG_MARKER["ran"] = False
    m = _make(out_features)
    m.forward(labels=None)
    return P.get_captured_lm_head_weight(), _ORIG_MARKER["ran"]


# --- value head: dim 1 -> must FALL THROUGH (not bypass) ---
cap, orig_ran = _run(1)
assert cap is None, f"[FAIL] value head was bypassed (_LM_HEAD_WEIGHT set: {cap is not None})"
assert orig_ran, "[FAIL] value head did not dispatch to original value forward"
print("[OK] value head (dim=1): fall-through, original value forward ran, NOT bypassed")

# --- LM head: dim>>1 -> must still BYPASS (chunk path preserved) ---
cap, _ = _run(1000)
assert cap is not None, "[FAIL] LM head no longer bypassed -> chunk path broken (regression!)"
assert tuple(cap.shape) == (1000, 8), f"unexpected captured weight shape {tuple(cap.shape)}"
print("[OK] LM head (dim=1000): still bypassed, _LM_HEAD_WEIGHT captured (chunk path intact)")

# --- gate off: QWEN36_CHUNK_LMHEAD=0 -> everything falls through (retro-compat) ---
os.environ["QWEN36_CHUNK_LMHEAD"] = "0"
cap, orig_ran = _run(1000)
assert cap is None and orig_ran, "[FAIL] gate-off did not fall through"
print("[OK] gate off: retro-compat, original forward ran")

print("\nALL PASS — F-PPO-1 gate correct: value head protected, LM-head chunk path & retro-compat intact")
