"""Unit tests for external-LB DP size sourcing from the resource layout (#6).

Locks that ``rollout.vllm_dp_size`` (see docs/design/vime_vllm_native_dp_rollout.md §19.2 #6)
is parsed by the loader and that the shipped dual-node layout keeps it consistent with the
engine count — the invariant arguments.py enforces at validate time (external-LB requires
``vllm_dp_size == rollout_num_gpus // rollout_num_gpus_per_engine``). This is the single-source
replacement for the earlier shell ROLLOUT_NUM_GPUS/PER_ENGINE double-source.
"""

from __future__ import annotations

import textwrap

import pytest

from vime.ray.resource_layout import load_resource_layout


def _write(tmp_path, body: str) -> str:
    p = tmp_path / "layout.yaml"
    p.write_text(textwrap.dedent(body))
    return str(p)


@pytest.mark.unit
def test_loader_parses_vllm_dp_size(tmp_path):
    path = _write(
        tmp_path,
        """
        roles:
          actor:
            - {node: 80.48.5.88, devices: "0-7"}
          rollout:
            - {node: 80.48.5.52, devices: "0-15"}
        rollout:
          num_gpus_per_engine: 4
          vllm_dp_size: 4
        """,
    )
    layout = load_resource_layout(path)
    assert layout.vllm_dp_size == 4
    # invariant arguments.py enforces for external-LB: dp_size == engine count
    assert layout.vllm_dp_size == layout.rollout_num_gpus // layout.rollout_num_gpus_per_engine


@pytest.mark.unit
def test_loader_vllm_dp_size_absent_is_none(tmp_path):
    path = _write(
        tmp_path,
        """
        roles:
          actor:
            - {node: 80.48.5.88, devices: "0-7"}
          rollout:
            - {node: 80.48.5.52, devices: "0-15"}
        rollout:
          num_gpus_per_engine: 4
        """,
    )
    layout = load_resource_layout(path)
    assert layout.vllm_dp_size is None


@pytest.mark.unit
def test_shipped_dual_node_layout_dp_size_matches_engine_count():
    # Regression guard for the exact misconfig: bumping num_gpus_per_engine but forgetting to
    # update vllm_dp_size (→ arguments.py would raise at launch). Keeps them in lockstep.
    layout = load_resource_layout("scripts/resource_layout.dual88train52infer.yaml")
    if layout.vllm_dp_size is not None:
        assert layout.vllm_dp_size == layout.rollout_num_gpus // layout.rollout_num_gpus_per_engine
