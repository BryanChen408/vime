"""CPU regressions for Megatron NPU flat-buffer offload lifecycle."""

from __future__ import annotations

import types
from unittest import mock

import pytest
import torch

from vime.utils.npu_weight_offloader import NPUWeightOffloader

NUM_GPUS = 0


class _FlatBuffer:
    def __init__(self) -> None:
        self.param_data = torch.arange(1, 9, dtype=torch.float32)
        self.grad_data = torch.arange(11, 19, dtype=torch.float32)


def _model_with_buffer():
    buffer = _FlatBuffer()
    model = types.SimpleNamespace(buffers=[buffer], expert_parallel_buffers=[])
    return model, buffer


@pytest.fixture(autouse=True)
def _disable_real_npu_cache(monkeypatch):
    npu = getattr(torch, "npu", None)
    if npu is None:
        npu = types.SimpleNamespace()
        monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(npu, "empty_cache", lambda: None, raising=False)


def test_external_actor_backup_releases_param_and_grad_without_private_cpu_copy(monkeypatch):
    """Shared actor path reuses TensorBackuper instead of duplicating host weights."""
    monkeypatch.setenv("VIME_OFFLOAD_PARAM_BUFFER", "0")
    model, buffer = _model_with_buffer()
    expected_param = buffer.param_data.clone()
    param_view = buffer.param_data[:4]
    param_tensor_id = id(buffer.param_data)
    view_tensor_id = id(param_view)
    restore_calls = 0

    def restore_actor() -> None:
        nonlocal restore_calls
        restore_calls += 1
        buffer.param_data.copy_(expected_param)

    offloader = NPUWeightOffloader(
        release_param_buffer=True,
        param_restorer=restore_actor,
    )

    released = offloader.offload(model, verbose=False)

    assert released == expected_param.untyped_storage().nbytes() * 2
    assert buffer.param_data.untyped_storage().size() == 0
    assert buffer.grad_data.untyped_storage().size() == 0
    assert offloader.stats["param_backup_source"] == "external"
    assert all(backup is None for _, backup in offloader._saved.values())
    assert restore_calls == 0

    restored = offloader.onload(model, verbose=False)

    assert restored == released
    assert restore_calls == 1
    assert id(buffer.param_data) == param_tensor_id
    assert id(param_view) == view_tensor_id
    assert torch.equal(buffer.param_data, expected_param)
    assert torch.equal(param_view, expected_param[:4])
    assert torch.count_nonzero(buffer.grad_data) == 0
    assert not offloader.is_offloaded


def test_default_mode_keeps_params_and_only_releases_grad(monkeypatch):
    """Unset legacy switch preserves the asynchronous PD grad-only path."""
    monkeypatch.delenv("VIME_OFFLOAD_PARAM_BUFFER", raising=False)
    model, buffer = _model_with_buffer()
    expected_param = buffer.param_data.clone()
    original_param_storage_size = buffer.param_data.untyped_storage().size()
    restore_calls = 0

    def restore_actor() -> None:
        nonlocal restore_calls
        restore_calls += 1
        buffer.param_data.copy_(expected_param)

    offloader = NPUWeightOffloader(param_restorer=restore_actor)
    offloader.offload(model, verbose=False)

    assert not offloader.release_param_buffer
    assert buffer.param_data.untyped_storage().size() == original_param_storage_size
    assert buffer.grad_data.untyped_storage().size() == 0

    offloader.onload(model, verbose=False)

    assert restore_calls == 1
    assert torch.equal(buffer.param_data, expected_param)
    assert torch.count_nonzero(buffer.grad_data) == 0


def test_explicit_full_release_overrides_legacy_environment(monkeypatch):
    """Hybrid topology policy wins even if an inherited environment says A mode."""
    monkeypatch.setenv("VIME_OFFLOAD_PARAM_BUFFER", "0")
    model, buffer = _model_with_buffer()
    expected_param = buffer.param_data.clone()
    offloader = NPUWeightOffloader(
        release_param_buffer=True,
        param_restorer=lambda: buffer.param_data.copy_(expected_param),
    )

    offloader.offload(model, verbose=False)

    assert offloader.release_param_buffer
    assert buffer.param_data.untyped_storage().size() == 0
    offloader.onload(model, verbose=False)
    assert torch.equal(buffer.param_data, expected_param)


def test_legacy_full_release_keeps_private_backup_for_callers_without_backuper(monkeypatch):
    """Existing VIME_OFFLOAD_PARAM_BUFFER=1 callers remain self-contained."""
    monkeypatch.setenv("VIME_OFFLOAD_PARAM_BUFFER", "1")
    model, buffer = _model_with_buffer()
    expected_param = buffer.param_data.clone()

    real_empty_like = torch.empty_like

    def cpu_empty_like(tensor, *, device=None, pin_memory=False, **kwargs):
        # CPU-only CI may not have a pin-memory allocator. Preserve the copy
        # semantics while recording that production requested pinned memory.
        assert device == "cpu"
        assert pin_memory is True
        return real_empty_like(tensor, device=device, **kwargs)

    with mock.patch.object(torch, "empty_like", side_effect=cpu_empty_like):
        offloader = NPUWeightOffloader()
        offloader.offload(model, verbose=False)

    private_backup = offloader._saved[(id(buffer), "param_data")][1]
    assert private_backup is not None
    assert private_backup.device.type == "cpu"
    assert torch.equal(private_backup, expected_param)

    offloader.onload(model, verbose=False)
    assert torch.equal(buffer.param_data, expected_param)
    assert torch.count_nonzero(buffer.grad_data) == 0


def test_onload_fails_loudly_if_ddp_buffer_identity_changed(monkeypatch):
    monkeypatch.setenv("VIME_OFFLOAD_PARAM_BUFFER", "0")
    model, _ = _model_with_buffer()
    offloader = NPUWeightOffloader(release_param_buffer=True, param_restorer=lambda: None)
    offloader.offload(model, verbose=False)
    model.buffers = [_FlatBuffer()]

    with pytest.raises(RuntimeError, match="model rebuilt"):
        offloader.onload(model, verbose=False)

    assert offloader.is_offloaded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
