"""Unit tests for ``vime.ray.rollout`` external-LB DP address/port allocation (#3).

Locks the invariants of ``_allocate_external_lb_addr_and_ports`` (see
docs/design/vime_vllm_native_dp_rollout.md §19.2 #3 + §21.2 T1):

- every slot IS one DP rank and gets its OWN api-server ``port`` / ``nccl_port``;
- co-located slots draw from a per-node cursor so they never collide (the §5 failure class);
- the whole DP group shares one ``(data_parallel_address, data_parallel_rpc_port)``
  rendezvous, allocated once from slot 0;
- ``data_parallel_rank`` == slot index, ``data_parallel_size`` == N, ``dist_init_addr`` None.

These import the real module (no vllm stubs), matching test_vllm_engine.py — so they run on
both the pure-CPU CI (vllm installed) and the NPU dev box.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vime.ray import rollout as R


class _FakeFreePort:
    """Deterministic stand-in for ``engine._get_current_node_ip_and_free_port.remote``.

    Returns ``start_port`` verbatim (not a real OS probe) so the test asserts the *cursor*
    logic — that co-located slots advance past each other — independent of host free ports.
    """

    def __init__(self, ip: str):
        self._ip = ip

    def remote(self, start_port=15000, consecutive=1):
        return (self._ip, start_port)


class _FakeEngine:
    def __init__(self, ip: str):
        self._get_current_node_ip_and_free_port = _FakeFreePort(ip)


@pytest.fixture
def passthrough_ray(monkeypatch):
    # helper calls ray.get(engine....remote(...)); make it a passthrough of the fake return.
    monkeypatch.setattr(R.ray, "get", lambda x: x)


@pytest.mark.unit
def test_external_lb_colocated_slots_do_not_collide(passthrough_ray):
    engines = [(0, _FakeEngine("80.48.5.52")), (1, _FakeEngine("80.48.5.52"))]
    aps, _ = R._allocate_external_lb_addr_and_ports(
        args=SimpleNamespace(), rollout_engines=engines, worker_type="regular"
    )
    # per-rank server + nccl ports are all distinct (no 15000/15000 collision, the §5 bug)
    ports = [aps[0]["port"], aps[0]["nccl_port"], aps[1]["port"], aps[1]["nccl_port"]]
    assert len(set(ports)) == 4, ports
    # DP group shares ONE rendezvous, allocated from slot 0
    assert aps[0]["data_parallel_address"] == aps[1]["data_parallel_address"] == "80.48.5.52"
    assert aps[0]["data_parallel_rpc_port"] == aps[1]["data_parallel_rpc_port"]
    # slot index == DP rank; size == N; dist_init_addr unused on external-LB
    assert (aps[0]["data_parallel_rank"], aps[1]["data_parallel_rank"]) == (0, 1)
    assert aps[0]["data_parallel_size"] == aps[1]["data_parallel_size"] == 2
    assert aps[0]["dist_init_addr"] is None and aps[1]["dist_init_addr"] is None


@pytest.mark.unit
def test_external_lb_master_is_slot0_across_nodes(passthrough_ray):
    # Two nodes: each rank's server lives on its own node, but the DP master addr is slot 0's.
    engines = [(0, _FakeEngine("80.48.5.52")), (1, _FakeEngine("80.48.5.53"))]
    aps, _ = R._allocate_external_lb_addr_and_ports(
        args=SimpleNamespace(), rollout_engines=engines, worker_type="regular"
    )
    assert aps[0]["host"] == "80.48.5.52"
    assert aps[1]["host"] == "80.48.5.53"
    assert aps[0]["data_parallel_address"] == aps[1]["data_parallel_address"] == "80.48.5.52"


@pytest.mark.unit
def test_external_lb_branch_selected_only_when_flag_set(passthrough_ray, monkeypatch):
    # The dispatcher must route to the external-LB helper iff vllm_data_parallel_external_lb;
    # otherwise the byte-identical non-external-LB path runs.
    seen = {}
    monkeypatch.setattr(
        R,
        "_allocate_external_lb_addr_and_ports",
        lambda **kw: (seen.setdefault("called", True), ({}, {}))[1],
    )
    engines = [(0, _FakeEngine("80.48.5.52"))]
    # flag OFF → must NOT call the helper
    R._allocate_rollout_engine_addr_and_ports_normal(
        args=SimpleNamespace(vllm_data_parallel_external_lb=False, rollout_num_gpus_per_engine=8, num_gpus_per_node=8, vllm_dp_size=1),
        rollout_engines=engines,
    )
    assert "called" not in seen
    # flag ON → routes to helper
    R._allocate_rollout_engine_addr_and_ports_normal(
        args=SimpleNamespace(vllm_data_parallel_external_lb=True),
        rollout_engines=engines,
    )
    assert seen.get("called") is True
