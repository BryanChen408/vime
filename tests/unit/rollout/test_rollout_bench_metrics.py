"""rollout_bench/* 指标与 "Rollout Benchmark Result" 打印。

背景：打印块(rollout.py:_log_rollout_data)早就存在，但没有任何代码生成
``rollout_bench/*`` 键 → ``if rollout_bench_metrics:`` 恒为假 → 实跑日志里
(train_qwen36_polar_20260808-131159.log)一条 benchmark 都没有。这里锁住三件事：

1. 指标确实被生成，且键名与打印块里写死的那 16 个 label 对得上；
2. 引擎没上报 timing 时返回 ``{}`` —— 否则打印块会输出一整屏 0，比不打印更误导；
3. 吞吐除以 **wall clock**(rollout_time)而不是 per-request latency 之和。
   请求是并发的，累加 latency 会把分母放大约等于并发度倍，吞吐被系统性低估。
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_VIME_ROOT = Path(__file__).resolve().parents[3]
if str(_VIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIME_ROOT))

from vime.utils.types import Sample  # noqa: E402


def _load_rollout_module():
    """按路径加载 rollout.py，避免 import vime.ray.rollout 拉起 ray/vllm 依赖链。"""
    return importlib.import_module("vime.ray.rollout")


def _make_sample(*, ttft=500.0, tpots=(30.0, 32.0), itls=(25.0, 27.0), prompt=2000, out=128, resp_len=128):
    s = Sample()
    s.response_length = resp_len
    s.status = Sample.Status.COMPLETED
    s.reward = 0.2  # _compute_zero_std_metrics 会 round() 它，不能是 None
    s.benchmark_info.ttft_ms = ttft
    s.benchmark_info.tpot_ms = list(tpots)
    s.benchmark_info.itl_ms = list(itls)
    s.benchmark_info.total_input_tokens = prompt
    s.benchmark_info.total_output_tokens = out
    return s


@pytest.fixture
def mod():
    return _load_rollout_module()


@pytest.fixture
def args():
    from types import SimpleNamespace

    return SimpleNamespace(
        rollout_num_gpus=8,
        vllm_speculative_config=None,
        advantage_estimator="grpo",
        reward_key=None,  # get_reward_value 用它决定 reward 是标量还是 dict
        log_reward_category=None,
    )


@pytest.mark.unit
def test_metrics_generated_with_expected_keys(mod, args):
    samples = [_make_sample(ttft=400.0 + 100 * i, out=100 + 10 * i) for i in range(4)]
    m = mod._compute_rollout_bench_metrics(args, samples, rollout_time=10.0)

    for stat in ("ttft", "tpot", "itl"):
        for agg in ("mean", "median", "p99", "min", "max"):
            assert f"{stat}_{agg}_ms" in m, f"缺少 {stat}_{agg}_ms"
    for key in (
        "total_input_tokens",
        "total_generated_tokens",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
    ):
        assert key in m, f"缺少 {key}"


@pytest.mark.unit
def test_returns_empty_when_engine_reported_no_timings(mod, args):
    """引擎不上报 timing → 返回 {}，打印块保持静默(而不是打印一屏 0)。"""
    blank = [_make_sample(ttft=0.0, tpots=(), itls=(), prompt=0, out=0) for _ in range(3)]
    assert mod._compute_rollout_bench_metrics(args, blank, rollout_time=10.0) == {}
    assert mod._compute_rollout_bench_metrics(args, [], rollout_time=10.0) == {}


@pytest.mark.unit
def test_throughput_uses_wall_clock_not_summed_latencies(mod, args):
    """8 个请求并发跑完 4 秒 → 2 req/s。若按 latency 之和(8×2s=16s)算会得到 0.5。"""
    samples = [_make_sample(out=100) for _ in range(8)]
    for s in samples:
        s.benchmark_info.request_latency_ms = 2000.0  # 每个请求各自 2s

    m = mod._compute_rollout_bench_metrics(args, samples, rollout_time=4.0)

    assert m["request_throughput"] == pytest.approx(2.0)
    assert m["output_throughput"] == pytest.approx(800 / 4.0)
    assert m["total_token_throughput"] == pytest.approx((8 * 2000 + 800) / 4.0)


@pytest.mark.unit
def test_zero_rollout_time_omits_throughput_without_dividing_by_zero(mod, args):
    m = mod._compute_rollout_bench_metrics(args, [_make_sample()], rollout_time=0.0)
    assert "request_throughput" not in m
    assert "ttft_mean_ms" in m  # 延迟类指标不依赖 rollout_time，仍应产出


@pytest.mark.unit
def test_latency_aggregates_are_correct(mod, args):
    s1 = _make_sample(ttft=100.0, tpots=(10.0, 20.0), itls=(5.0,))
    s2 = _make_sample(ttft=300.0, tpots=(30.0,), itls=(15.0,))
    m = mod._compute_rollout_bench_metrics(args, [s1, s2], rollout_time=1.0)

    assert m["ttft_mean_ms"] == pytest.approx(200.0)
    assert m["ttft_min_ms"] == pytest.approx(100.0)
    assert m["ttft_max_ms"] == pytest.approx(300.0)
    # tpot 跨 sample 摊平: [10, 20, 30]
    assert m["tpot_mean_ms"] == pytest.approx(20.0)
    assert m["itl_mean_ms"] == pytest.approx(10.0)


@pytest.mark.unit
def test_bench_keys_survive_prefixing_at_top_level(mod, args):
    """dict_add_prefix 后必须是 rollout_bench/xxx，不能变成 rollout/rollout_bench/xxx。

    打印块过滤的是 ``k.startswith("rollout_bench/")``；早先把 bench 指标塞进
    compute_metrics_from_samples 会被二次加前缀成 rollout/rollout_bench/*，过滤不到。
    """
    m = mod._compute_rollout_bench_metrics(args, [_make_sample()], rollout_time=5.0)
    prefixed = mod.dict_add_prefix(m, "rollout_bench/")
    assert prefixed, "指标为空"
    assert all(k.startswith("rollout_bench/") for k in prefixed)
    assert not any(k.startswith("rollout/rollout_bench/") for k in prefixed)
    assert {k for k, v in prefixed.items() if k.startswith("rollout_bench/")}


@pytest.mark.unit
def test_compute_metrics_from_samples_does_not_emit_bench_keys(mod, args):
    """bench 指标只能来自顶层那次调用，避免重复/双前缀。"""
    out = mod.compute_metrics_from_samples(args, [_make_sample()])
    assert not [k for k in out if "rollout_bench" in k]


@pytest.mark.unit
def test_benchmark_info_roundtrip():
    s = _make_sample(ttft=680.23, tpots=(45.1, 46.2), itls=(33.8,))
    restored = Sample.from_dict(s.to_dict())
    assert restored.benchmark_info.ttft_ms == pytest.approx(680.23)
    assert restored.benchmark_info.tpot_ms == pytest.approx([45.1, 46.2])
    assert restored.benchmark_info.itl_ms == pytest.approx([33.8])
    assert restored.benchmark_info.total_input_tokens == 2000


@pytest.mark.unit
def test_update_from_meta_info_populates_benchmark_info(args):
    s = Sample()
    s.update_from_meta_info(
        args,
        {
            "finish_reason": {"type": "stop"},
            "prompt_tokens": 1234,
            "completion_tokens": 56,
            "ttft_ms": 512.5,
            "tpot_ms": [40.0, 41.0],
            "itl_ms": [30.0],
            "request_latency_ms": 2600.0,
        },
    )
    assert s.benchmark_info.ttft_ms == pytest.approx(512.5)
    assert s.benchmark_info.tpot_ms == pytest.approx([40.0, 41.0])
    assert s.benchmark_info.total_input_tokens == 1234
    assert s.benchmark_info.total_output_tokens == 56
    assert s.benchmark_info.request_latency_ms == pytest.approx(2600.0)
    assert s.status == Sample.Status.COMPLETED


@pytest.mark.unit
def test_update_from_meta_info_without_timings_leaves_bench_empty(args):
    """老引擎/无 metrics 字段时不应炸，且 benchmark_info 保持空。"""
    s = Sample()
    s.update_from_meta_info(args, {"finish_reason": {"type": "stop"}, "prompt_tokens": 10, "completion_tokens": 2})
    assert s.benchmark_info.ttft_ms == 0.0
    assert s.benchmark_info.tpot_ms == []


@pytest.mark.unit
def test_benchmark_info_accumulates_across_partial_rollout_calls(args):
    """partial rollout 下同一 sample 多次 update → 必须累加，不能被最后一次覆盖。

    早先 add() 用的是赋值，3 次 1000ms 的调用只会留下 1000ms，token 数也只剩最后一段。
    """
    s = Sample()
    for _ in range(3):
        s.update_from_meta_info(
            args,
            {
                "finish_reason": {"type": "length"},
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "request_latency_ms": 1000.0,
            },
        )
    assert s.benchmark_info.total_output_tokens == 150
    assert s.benchmark_info.total_input_tokens == 300
    assert s.benchmark_info.request_latency_ms == pytest.approx(3000.0)


@pytest.mark.unit
def test_ttft_keeps_first_nonzero_across_calls(args):
    """TTFT 是"首个 token 的延迟"，只发生一次 —— 后续续写不该覆盖它，也不该累加。"""
    s = Sample()
    s.update_from_meta_info(args, {"finish_reason": {"type": "length"}, "ttft_ms": 0.0})
    s.update_from_meta_info(args, {"finish_reason": {"type": "length"}, "ttft_ms": 500.0})
    s.update_from_meta_info(args, {"finish_reason": {"type": "stop"}, "ttft_ms": 900.0})
    assert s.benchmark_info.ttft_ms == pytest.approx(500.0)


@pytest.mark.unit
def test_tpot_itl_lists_extend_not_replace(args):
    s = Sample()
    s.update_from_meta_info(args, {"finish_reason": {"type": "length"}, "tpot_ms": [10.0], "itl_ms": [5.0]})
    s.update_from_meta_info(args, {"finish_reason": {"type": "stop"}, "tpot_ms": [20.0], "itl_ms": [7.0]})
    assert s.benchmark_info.tpot_ms == pytest.approx([10.0, 20.0])
    assert s.benchmark_info.itl_ms == pytest.approx([5.0, 7.0])


@pytest.mark.unit
def test_benchmark_block_actually_prints(mod, args, caplog, monkeypatch):
    """端到端走 _log_rollout_data，确认 "Rollout Benchmark Result" 真的出现在日志里。

    这是本次改动的验收点：改前该块因 rollout_bench_metrics 恒空而从不执行。
    """
    monkeypatch.setattr(mod.logging_utils, "log", lambda *a, **k: None)
    args.custom_rollout_log_function_path = None
    args.load_debug_rollout_data = False
    args.wandb_always_use_train_step = False

    samples = [_make_sample(ttft=400.0 + 50 * i, out=120) for i in range(4)]
    for s in samples:
        s.benchmark_info.request_latency_ms = 2500.0

    with caplog.at_level("INFO", logger=mod.logger.name):
        mod._log_rollout_data(0, args, samples, None, rollout_time=8.0)

    text = caplog.text
    assert "Rollout Benchmark Result" in text
    assert "Mean TTFT (ms):" in text
    assert "Total generated tokens:" in text
    assert "Output token throughput (tok/s):" in text
    # 未生成的两项应被跳过而不是打印 0
    assert "Peak concurrent requests:" not in text
    assert "Peak output token throughput" not in text


@pytest.mark.unit
def test_benchmark_block_silent_when_no_timings(mod, args, caplog, monkeypatch):
    """引擎无计时数据时整块静默 —— 不能打印一屏 0 误导读者。"""
    monkeypatch.setattr(mod.logging_utils, "log", lambda *a, **k: None)
    args.custom_rollout_log_function_path = None
    args.load_debug_rollout_data = False
    args.wandb_always_use_train_step = False

    blank = [_make_sample(ttft=0.0, tpots=(), itls=(), prompt=0, out=0) for _ in range(3)]

    with caplog.at_level("INFO", logger=mod.logger.name):
        mod._log_rollout_data(1, args, blank, None, rollout_time=5.0)

    assert "Rollout Benchmark Result" not in caplog.text
