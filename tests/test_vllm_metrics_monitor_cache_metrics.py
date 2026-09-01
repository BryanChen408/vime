"""CPU contracts for per-engine KV and prefix-cache metrics."""

from __future__ import annotations

import pytest

from scripts import vllm_metrics_monitor_v2 as monitor


NUM_GPUS = 0


def test_engine_snapshot_exposes_real_kv_and_prefix_hit(monkeypatch):
    key = "127.0.0.1:19999"
    monkeypatch.setitem(monitor.engines, key, monitor._blank_engine(f"http://{key}/metrics"))
    monkeypatch.setattr(monitor.time, "time", lambda: 100.0)

    engine = monitor.update_engine(
        key,
        {
            "vllm:generation_tokens_total": 1234.0,
            "vllm:num_requests_running": 3.0,
            "vllm:num_requests_waiting": 1.0,
            "vllm:kv_cache_usage_perc": 0.625,
            "vllm:prefix_cache_hits_total": 750.0,
            "vllm:prefix_cache_queries_total": 1000.0,
        },
    )

    assert engine["kv_usage"] == pytest.approx(0.625)
    assert engine["prefix_hit"] == pytest.approx(0.75)

    with monitor.app.test_client() as client:
        payload = client.get("/api/metrics").get_json()
    exported = next(item for item in payload["engines"] if item["key"] == key)
    assert exported["kv_usage"] == pytest.approx(0.625)
    assert exported["prefix_hit"] == pytest.approx(0.75)
    assert exported["prefix_cache_hits_total"] == pytest.approx(750.0)
    assert exported["prefix_cache_queries_total"] == pytest.approx(1000.0)


def test_prefix_hit_is_unknown_until_vllm_reports_queries(monkeypatch):
    key = "127.0.0.1:19998"
    monkeypatch.setitem(monitor.engines, key, monitor._blank_engine(f"http://{key}/metrics"))

    engine = monitor.update_engine(
        key,
        {
            "vllm:kv_cache_usage_perc": 0.0,
            "vllm:prefix_cache_hits_total": 0.0,
            "vllm:prefix_cache_queries_total": 0.0,
        },
    )

    assert engine["kv_usage"] == 0.0
    assert engine["prefix_hit"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
