from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.analyze_qwen36_yarn_evidence import compare, summarize
from vime.backends.megatron_utils import validation_evidence


def _record(*, cp_rank: int, cp_size: int, positions: list[int], values: list[float]):
    count = len(positions)
    return {
        "schema_version": 1,
        "step": 0,
        "microbatch_index": 0,
        "global_rank": cp_rank,
        "tp_rank": 0,
        "tp_size": 2,
        "cp_rank": cp_rank,
        "cp_size": cp_size,
        "dp_rank": 0,
        "dp_size": 1,
        "sample_indices": [7],
        "rollout_ids": [3],
        "total_lengths": [6],
        "response_lengths": [4],
        "sample_token_sha256": ["abc"],
        "loss_masks": [torch.tensor([1, 0, 1, 1])],
        "local_loss_masks": [torch.ones(count, dtype=torch.int)],
        "local_logit_positions": [torch.tensor(positions)],
        "local_target_token_ids": [torch.tensor([100 + value for value in positions])],
        "train_log_probs": [torch.tensor(values)],
        "current_log_probs": [torch.tensor(values)],
        "rollout_log_probs": [torch.tensor([value - 0.01 for value in values])],
        "packed_seq_params": {},
        "parallel_config": {"context_parallel_size": cp_size},
        "yarn_fingerprint": {"position_embedding_type": "yarn"},
    }


def _write(path: Path, records: list[dict]) -> None:
    path.mkdir()
    for index, record in enumerate(records):
        torch.save(record, path / f"tis_evidence_step0_mb0_rank{index}.pt")


def test_summary_uses_valid_tokens_and_checks_cp_position_coverage(tmp_path: Path) -> None:
    evidence = tmp_path / "cp2"
    _write(
        evidence,
        [
            _record(cp_rank=0, cp_size=2, positions=[1, 3], values=[-1.0, -2.0]),
            _record(cp_rank=1, cp_size=2, positions=[4], values=[-3.0]),
        ],
    )

    result = summarize(evidence)

    assert result["all_finite"] is True
    assert result["packed_position_coverage_ok"] is True
    assert result["train_rollout_abs_diff"]["count"] == 3
    assert result["train_rollout_abs_diff"]["mean"] == pytest.approx(0.01, abs=1e-6)
    assert result["gates"]["train_rollout_mean_abs_diff_le_0_05"] is True


def test_compare_requires_and_compares_identical_token_keys(tmp_path: Path) -> None:
    cp1 = tmp_path / "cp1"
    cp2 = tmp_path / "cp2"
    _write(cp1, [_record(cp_rank=0, cp_size=1, positions=[1, 3, 4], values=[-1.0, -2.0, -3.0])])
    _write(
        cp2,
        [
            _record(cp_rank=0, cp_size=2, positions=[1, 3], values=[-1.0, -2.01]),
            _record(cp_rank=1, cp_size=2, positions=[4], values=[-3.0]),
        ],
    )

    result = compare(cp1, cp2)

    assert result["token_sets_equal"] is True
    assert result["shared_token_count"] == 3
    assert result["train_logprob_abs_diff"]["max"] == pytest.approx(0.01, abs=1e-6)


def test_runtime_dump_is_microbatch_unique_and_self_describing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VIME_SAVE_TIS_LOGPROBS", str(tmp_path))
    monkeypatch.setattr(validation_evidence.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(validation_evidence.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(validation_evidence.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(validation_evidence.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(validation_evidence.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(validation_evidence.mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(validation_evidence.mpu, "get_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(validation_evidence.torch.distributed, "get_rank", lambda: 5)
    args = SimpleNamespace(
        curr_iteration=2,
        qkv_format="thd",
        position_embedding_type="yarn",
        rotary_base=10_000_000,
        rotary_percent=0.25,
        rotary_scaling_factor=4.0,
        yarn_original_max_position_embeddings=262_144,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=0.0,
        yarn_correction_range_round_to_int=True,
        seq_length=262_144,
        max_position_embeddings=262_144,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=8,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([10, 11, 12, 13, 14, 15])],
        "loss_masks": [torch.tensor([1, 0, 1, 1])],
        "rollout_log_probs": [torch.tensor([-1.01, -2.01, -3.01, -4.01])],
        "total_lengths": [6],
        "response_lengths": [4],
        "sample_indices": [9],
        "rollout_ids": [4],
        "packed_seq_params": None,
    }

    first = validation_evidence.save_tis_validation_evidence(
        args,
        batch,
        train_log_probs=[torch.tensor([-1.0, -2.0, -3.0, -4.0])],
        current_log_probs=[torch.tensor([-1.0, -2.0, -3.0, -4.0])],
    )
    second = validation_evidence.save_tis_validation_evidence(
        args,
        batch,
        train_log_probs=[torch.tensor([-1.0, -2.0, -3.0, -4.0])],
        current_log_probs=[torch.tensor([-1.0, -2.0, -3.0, -4.0])],
    )

    assert first is not None and second is not None and first != second
    payload = torch.load(first, weights_only=False)
    assert payload["schema_version"] == 1
    assert payload["sample_indices"] == [9]
    assert payload["local_logit_positions"][0].tolist() == [1, 2, 3, 4]
    assert payload["local_target_token_ids"][0].tolist() == [12, 13, 14, 15]
    assert payload["sample_token_sha256"][0]
