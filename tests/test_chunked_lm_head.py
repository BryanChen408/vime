import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def install_fused_cross_entropy_stub():
    """Stand in for megatron's fused vocab-parallel cross entropy.

    The tests below compare chunked against non-chunked results, so only consistency
    between the two matters, not fidelity to the real kernel.
    """
    module_names = [
        "megatron",
        "megatron.core",
        "megatron.core.fusions",
        "megatron.core.fusions.fused_cross_entropy",
    ]
    for name in module_names:
        sys.modules.setdefault(name, types.ModuleType(name))

    def fused_vocab_parallel_cross_entropy(logits, tokens, process_group):
        assert process_group is None, "the stub only covers the single-rank case"
        flat = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tokens.reshape(-1), reduction="none"
        )
        return flat.reshape(tokens.shape)

    sys.modules["megatron.core.fusions.fused_cross_entropy"].fused_vocab_parallel_cross_entropy = (
        fused_vocab_parallel_cross_entropy
    )


def load_module(relative_path: str, module_name: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_patch_module():
    return load_module("vime/backends/megatron_utils/chunked_lm_head_patch.py", "test_chunked_lm_head_patch")


@pytest.mark.unit
def test_captured_weight_is_none_before_any_forward():
    module = load_patch_module()
    assert module.get_captured_lm_head_weight() is None


def install_gpt_model_stub():
    """A minimal GPTModel whose forward mimics the real one: logits = output_layer(hidden)."""
    for name in ["megatron", "megatron.core", "megatron.core.models",
                 "megatron.core.models.gpt", "megatron.core.models.gpt.gpt_model"]:
        sys.modules.setdefault(name, types.ModuleType(name))

    class OutputLayer:
        def __init__(self, rows):
            self.weight = torch.randn(rows, 4)
            self.sequence_parallel = False
            self.tp_group = None

        def forward(self, hidden, weight=None, runtime_gather_output=None):
            return torch.matmul(hidden, self.weight.t()), None

    class GPTModel:
        def __init__(self, rows=16, post_process=True, mtp_process=False):
            self.output_layer = OutputLayer(rows)
            self.post_process = post_process
            self.mtp_process = mtp_process

        def forward(self, hidden, labels=None):
            out, _ = self.output_layer.forward(hidden)
            return out

    sys.modules["megatron.core.models.gpt.gpt_model"].GPTModel = GPTModel
    return GPTModel


@pytest.mark.unit
def test_patch_returns_hidden_and_captures_weight_for_lm_head():
    gpt_model_cls = install_gpt_model_stub()
    module = load_patch_module()
    module.apply_chunked_lm_head_patch()

    model = gpt_model_cls(rows=16)
    hidden = torch.randn(5, 4)

    out = model.forward(hidden)

    assert torch.equal(out, hidden), "the output layer should have been bypassed"
    assert module.get_captured_lm_head_weight() is model.output_layer.weight


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"rows": 1}, "a critic value head emits [T, 1] and must not be bypassed"),
        ({"post_process": False}, "only the last pipeline stage holds an output layer"),
        ({"mtp_process": True}, "MTP computes its loss inside postprocess"),
    ],
)
def test_patch_leaves_other_forwards_alone(kwargs, reason):
    gpt_model_cls = install_gpt_model_stub()
    module = load_patch_module()
    module.apply_chunked_lm_head_patch()

    model = gpt_model_cls(**kwargs)
    hidden = torch.randn(5, 4)

    out = model.forward(hidden)

    assert out.shape == (5, model.output_layer.weight.shape[0]), reason
    assert module.get_captured_lm_head_weight() is None


@pytest.mark.unit
def test_patch_clears_stale_weight_between_forwards():
    """An LM forward followed by a value-head forward must not leave the LM weight behind."""
    gpt_model_cls = install_gpt_model_stub()
    module = load_patch_module()
    module.apply_chunked_lm_head_patch()

    lm_model = gpt_model_cls(rows=16)
    lm_model.forward(torch.randn(5, 4))
    assert module.get_captured_lm_head_weight() is not None

    gpt_model_cls(rows=1).forward(torch.randn(5, 4))
    assert module.get_captured_lm_head_weight() is None


@pytest.mark.unit
def test_patch_is_idempotent():
    gpt_model_cls = install_gpt_model_stub()
    module = load_patch_module()
    module.apply_chunked_lm_head_patch()
    first = gpt_model_cls.forward
    module.apply_chunked_lm_head_patch()

    assert gpt_model_cls.forward is first


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [1, 3, 8, 64])
def test_chunked_logprob_matches_full_logits(chunk_size):
    """Chunking must not change the log-probs compared with materialising all logits."""
    install_fused_cross_entropy_stub()
    ppo_utils = load_module("vime/utils/ppo_utils.py", "test_chunked_ppo_utils")

    torch.manual_seed(0)
    seq, hidden, vocab = 17, 8, 32
    h = torch.randn(seq, hidden)
    weight = torch.randn(vocab, hidden)
    tokens = torch.randint(0, vocab, (seq,))

    chunked, _ = ppo_utils.chunked_logprob_entropy_from_hidden(
        h, weight, tokens, None, chunk_size=chunk_size, with_entropy=False
    )
    reference, _ = ppo_utils.calculate_log_probs_and_entropy(
        torch.matmul(h, weight.t()).float(), tokens, None, with_entropy=False
    )

    torch.testing.assert_close(chunked, reference)


@pytest.mark.unit
def test_chunked_logprob_handles_empty_input():
    install_fused_cross_entropy_stub()
    ppo_utils = load_module("vime/utils/ppo_utils.py", "test_chunked_ppo_utils")
    h = torch.zeros(0, 8)
    log_prob, entropy = ppo_utils.chunked_logprob_entropy_from_hidden(
        h, torch.zeros(32, 8), torch.zeros(0, dtype=torch.long), None, with_entropy=True
    )
    assert log_prob.numel() == 0
    assert entropy.numel() == 0
