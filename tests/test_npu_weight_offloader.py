import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import torch


def load_offloader_module():
    """Load the offloader with a CPU-only clear_memory, so no NPU runtime is needed."""
    memory_utils = types.ModuleType("vime.utils.memory_utils")
    memory_utils.clear_memory = lambda clear_host_memory=False: None
    sys.modules["vime.utils.memory_utils"] = memory_utils
    for name in ["vime", "vime.utils"]:
        sys.modules.setdefault(name, types.ModuleType(name))

    module_path = Path(__file__).resolve().parents[1] / "vime" / "utils" / "npu_weight_offloader.py"
    module_name = "test_npu_weight_offloader_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def storage_size(tensor: torch.Tensor) -> int:
    """Storage size in bytes.

    Assert on this rather than on the tensor: once the storage is resized to zero, any
    attempt to format the tensor — which pytest does when an assertion fails — reads freed
    memory and takes the interpreter down with it.
    """
    return tensor.untyped_storage().size()


class FakeParamAndGradBuffer:
    """Stands in for Megatron's _ParamAndGradBuffer: two flat tensors plus views."""

    def __init__(self, numel=16):
        self.param_data = torch.arange(numel, dtype=torch.float32)
        self.grad_data = torch.arange(numel, dtype=torch.float32) * 10
        # A parameter's data is a view into the flat buffer, which is what makes
        # attribute swapping useless and storage resizing necessary.
        self.param_view = self.param_data[:4]


class FakeDDP:
    def __init__(self, buffers, expert_parallel_buffers=None):
        self.buffers = buffers
        self.expert_parallel_buffers = expert_parallel_buffers or []


@pytest.mark.unit
def test_offload_releases_grad_storage_and_onload_zeroes_it():
    module = load_offloader_module()
    buffer = FakeParamAndGradBuffer()
    model = FakeDDP([buffer])
    offloader = module.NPUWeightOffloader()

    params_before = buffer.param_data.clone()
    released = offloader.offload(model)

    assert released == 16 * 4
    assert storage_size(buffer.grad_data) == 0
    assert storage_size(buffer.param_data) != 0, "grad-only mode keeps the weights"
    assert offloader.is_offloaded

    offloader.onload(model)

    # Bind the sizes before asserting: the assertion below must not mention a tensor whose
    # storage might still be zero, or pytest's repr of it takes the interpreter down.
    grad_size, param_size = storage_size(buffer.grad_data), storage_size(buffer.param_data)
    assert grad_size != 0, "grad buffer must be restored too, not just the params"
    assert param_size != 0

    assert torch.equal(buffer.grad_data, torch.zeros(16)), "restored pages must be zeroed"
    assert torch.equal(buffer.param_data, params_before)
    assert not offloader.is_offloaded


@pytest.mark.unit
def test_release_param_buffer_restores_weights_exactly():
    module = load_offloader_module()
    buffer = FakeParamAndGradBuffer()
    model = FakeDDP([buffer])
    offloader = module.NPUWeightOffloader(release_param_buffer=True)

    params_before = buffer.param_data.clone()
    offloader.offload(model)

    assert storage_size(buffer.param_data) == 0
    assert storage_size(buffer.grad_data) == 0

    offloader.onload(model)

    grad_size, param_size = storage_size(buffer.grad_data), storage_size(buffer.param_data)
    assert grad_size != 0 and param_size != 0

    assert torch.equal(buffer.param_data, params_before)
    assert torch.equal(buffer.grad_data, torch.zeros(16))


@pytest.mark.unit
def test_views_become_valid_again_after_onload():
    """The point of resizing rather than swapping: views survive the round trip."""
    module = load_offloader_module()
    buffer = FakeParamAndGradBuffer()
    model = FakeDDP([buffer])
    offloader = module.NPUWeightOffloader(release_param_buffer=True)

    view = buffer.param_view
    offloader.offload(model)
    offloader.onload(model)

    view_size = storage_size(view)
    assert view_size != 0
    assert torch.equal(view, torch.arange(4, dtype=torch.float32))


@pytest.mark.unit
def test_expert_parallel_buffers_are_included():
    module = load_offloader_module()
    dense, expert = FakeParamAndGradBuffer(), FakeParamAndGradBuffer()
    model = FakeDDP([dense], expert_parallel_buffers=[expert])

    module.NPUWeightOffloader().offload(model)

    assert storage_size(dense.grad_data) == 0
    assert storage_size(expert.grad_data) == 0


@pytest.mark.unit
def test_pipeline_chunks_are_walked_and_deduplicated():
    module = load_offloader_module()
    shared = FakeParamAndGradBuffer()
    own = FakeParamAndGradBuffer()
    model = [FakeDDP([shared]), FakeDDP([shared, own])]

    offloader = module.NPUWeightOffloader()
    offloader.offload(model)

    assert offloader.stats["buffers_offloaded_last_call"] == 2


@pytest.mark.unit
def test_double_offload_is_a_noop():
    module = load_offloader_module()
    buffer = FakeParamAndGradBuffer()
    model = FakeDDP([buffer])
    offloader = module.NPUWeightOffloader()

    first = offloader.offload(model)
    second = offloader.offload(model)

    assert second == first
    assert offloader.stats["buffers_currently_offloaded"] == 1


@pytest.mark.unit
def test_onload_without_offload_is_a_noop():
    module = load_offloader_module()
    model = FakeDDP([FakeParamAndGradBuffer()])

    assert module.NPUWeightOffloader().onload(model) == 0


@pytest.mark.unit
def test_stats_report_counts_not_sentinels():
    module = load_offloader_module()
    model = FakeDDP([FakeParamAndGradBuffer()])
    offloader = module.NPUWeightOffloader()

    offloader.offload(model)
    assert offloader.stats["buffers_currently_offloaded"] == 1

    offloader.onload(model)
    stats = offloader.stats
    assert stats["buffers_currently_offloaded"] == 0
    assert stats["buffers_offloaded_last_call"] == 1
    assert stats["offloaded_mb"] >= 0


def load_backend_resolver():
    """Load just the resolver out of arguments.py, which is too heavy to import whole."""
    import ast

    source = (Path(__file__).resolve().parents[1] / "vime" / "utils" / "arguments.py").read_text()
    tree = ast.parse(source)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_npu_offload_backend"
    )
    namespace = {"os": __import__("os"), "logger": logging.getLogger("test-resolver")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<resolver>", "exec"), namespace)
    return namespace["_resolve_npu_offload_backend"]


@pytest.fixture
def resolve(monkeypatch):
    common = types.ModuleType("vime.utils.common")
    common.is_npu = lambda: True
    monkeypatch.setitem(sys.modules, "vime.utils.common", common)
    return load_backend_resolver(), common


@pytest.mark.unit
@pytest.mark.parametrize("requested, expected", [("auto", "tms"), ("tms", "tms"), ("storage-resize", "storage-resize")])
def test_backend_resolution_on_npu(resolve, monkeypatch, requested, expected):
    resolver, _ = resolve
    monkeypatch.delenv("PYTORCH_NPU_ALLOC_CONF", raising=False)
    assert resolver(types.SimpleNamespace(npu_offload_backend=requested)) == expected


@pytest.mark.unit
def test_backend_is_always_tms_off_npu(resolve, monkeypatch):
    resolver, common = resolve
    common.is_npu = lambda: False
    monkeypatch.delenv("PYTORCH_NPU_ALLOC_CONF", raising=False)
    assert resolver(types.SimpleNamespace(npu_offload_backend="storage-resize")) == "tms"


@pytest.mark.unit
def test_tms_warns_that_it_overrides_expandable_segments(resolve, monkeypatch, caplog):
    """tms turns expandable segments off; say so rather than dropping the request silently."""
    resolver, _ = resolve
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

    with caplog.at_level(logging.WARNING):
        assert resolver(types.SimpleNamespace(npu_offload_backend="tms")) == "tms"

    assert "expandable_segments" in caplog.text


@pytest.mark.unit
def test_storage_resize_does_not_warn_about_expandable_segments(resolve, monkeypatch, caplog):
    resolver, _ = resolve
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

    with caplog.at_level(logging.WARNING):
        resolver(types.SimpleNamespace(npu_offload_backend="storage-resize"))

    assert caplog.text == ""


@pytest.mark.unit
def test_storage_resize_allows_expandable_segments(resolve, monkeypatch):
    resolver, _ = resolve
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    assert resolver(types.SimpleNamespace(npu_offload_backend="storage-resize")) == "storage-resize"
