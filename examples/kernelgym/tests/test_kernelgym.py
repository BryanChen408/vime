"""CPU-only unit tests for the kernelgym example.

These exercise the pure logic (reward strategies, kernel extraction, payload
building, env stepping with a mocked verifier, data conversion) with no NPU and
no running kernelGym service. Run from the vime repo root::

    python -m pytest examples/kernelgym/tests/test_kernelgym.py -q
"""

from __future__ import annotations

import asyncio

import pytest

from examples.kernelgym import env_kernelgym, kernelgym_client, prepare_data, reward
from examples.kernelgym.env_kernelgym import KernelGymEnv, KernelGymRunConfig
from examples.kernelgym.reward import RewardConfig

REF_CODE = "import torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def forward(self, x):\n        return x\n"
GOOD_KERNEL = "```triton\nclass ModelNew(nn.Module):\n    def forward(self, x):\n        return x\n```"


# --------------------------------------------------------------------------- #
# reward strategies
# --------------------------------------------------------------------------- #
def test_weighted_correct_and_fast():
    cfg = RewardConfig(reward_func_name="calculate_reward_weighted")
    out = reward.calculate_reward_weighted(
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 2.0}, cfg
    )
    assert out["reward"] == pytest.approx(1.0)
    assert out["success"] is True


def test_weighted_correct_no_speedup():
    cfg = RewardConfig()
    out = reward.calculate_reward_weighted(
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 1.0}, cfg
    )
    # 0.5*correct + 0.5*(1.0 >= 1.01 -> False)
    assert out["reward"] == pytest.approx(0.5)


def test_weighted_failure_and_decoy_use_penalty():
    cfg = RewardConfig(penalty_score=-0.2)
    failed = reward.calculate_reward_weighted({"status": "failed", "error_message": "boom"}, cfg)
    assert failed["reward"] == pytest.approx(-0.2)
    assert failed["success"] is False
    decoy = reward.calculate_reward_weighted(
        {"status": "completed", "decoy_kernel": True, "correctness": True, "speedup": 9.0}, cfg
    )
    assert decoy["reward"] == pytest.approx(-0.2)
    assert decoy["decoy_kernel"] is True


def test_speedup_strategy_clamps():
    cfg = RewardConfig(
        reward_func_name="calculate_reward_speedup",
        init_correct_weight=0.0,
        init_performance_weight=1.0,
        speedup_reward_upper_bound=3.0,
    )
    out = reward.calculate_reward_speedup(
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 10.0}, cfg
    )
    assert out["reward"] == pytest.approx(3.0)  # clamped to upper bound


def test_like_kernel_ladder():
    cfg = RewardConfig()
    base = {"status": "completed", "compiled": True, "correctness": True}
    assert reward.calculate_reward_like_kernel({**base, "speedup": 3.5}, cfg)["reward"] == 1.0
    assert reward.calculate_reward_like_kernel({**base, "speedup": 1.3}, cfg)["reward"] == 0.4
    not_compiled = reward.calculate_reward_like_kernel(
        {"status": "completed", "compiled": False, "correctness": False}, cfg
    )
    assert not_compiled["reward"] == pytest.approx(cfg.compilation_fail_penalty)


def test_coverage_bonus_added_when_enabled():
    cfg = RewardConfig(coverage_enable=True, coverage_reward_type="number_coverage", coverage_weight=0.25)
    out = reward.calculate_reward_weighted(
        {
            "status": "completed",
            "compiled": True,
            "correctness": True,
            "speedup": 2.0,
            "metadata": {"num_custom_kernels": 4, "num_total_kernels": 4},
        },
        cfg,
    )
    # weighted base 1.0 + 0.25 * (4/4)
    assert out["reward"] == pytest.approx(1.25)


def test_reward_config_from_nested_mapping():
    cfg = RewardConfig.from_mapping(
        {
            "reward_func_name": "calculate_reward_speedup",
            "reward_policy": {"penalties": {"penalty_score": -0.3, "compilation_fail": -0.7}},
            "coverage_reward": {"enable": True, "weight": 0.5},
        }
    )
    assert cfg.reward_func_name == "calculate_reward_speedup"
    assert cfg.penalty_score == pytest.approx(-0.3)
    assert cfg.compilation_fail_penalty == pytest.approx(-0.7)
    assert cfg.coverage_enable is True
    assert cfg.coverage_weight == pytest.approx(0.5)


def test_dispatch_unknown_falls_back_to_weighted():
    cfg = RewardConfig(reward_func_name="does_not_exist")
    out = reward.compute_reward_summary(
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 2.0}, cfg
    )
    assert out["reward"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# kernel extraction / preflight / payload
# --------------------------------------------------------------------------- #
def test_extract_kernel_prefers_last_fenced_block():
    text = "thinking...\n```python\nx = 1\n```\nmore\n```triton\nclass ModelNew(nn.Module):\n    pass\n```"
    code = env_kernelgym.extract_kernel_code(text)
    assert "class ModelNew(nn.Module)" in code
    assert "x = 1" not in code


def test_patch_bare_model_new():
    assert env_kernelgym.patch_model_new("class ModelNew:\n    pass") == "class ModelNew(nn.Module):\n    pass"


def test_preflight_validate():
    ok, missing = kernelgym_client.preflight_validate(REF_CODE, "class ModelNew(nn.Module):\n    pass", "Model")
    assert ok and missing == ""
    bad_ok, bad_missing = kernelgym_client.preflight_validate(REF_CODE, "class Other:\n    pass", "Model")
    assert not bad_ok and "ModelNew" in bad_missing


def test_build_eval_payload_validation_forces_decoy():
    payload = kernelgym_client.build_eval_payload(
        task_id="t1", reference_code=REF_CODE, kernel_code="k", entry_point="Model",
        backend="triton", is_valid=True, detect_decoy_kernel=False,
    )
    assert payload["detect_decoy_kernel"] is True  # forced on under validation
    assert payload["task_id"] == "t1"
    assert payload["backend"] == "triton"
    assert payload["eval_tag"] == "train"  # eval_tag is set by the env, default here


# --------------------------------------------------------------------------- #
# env stepping with a mocked verifier
# --------------------------------------------------------------------------- #
def _make_env(cfg: KernelGymRunConfig) -> KernelGymEnv:
    env = KernelGymEnv(
        reference_code=REF_CODE,
        entry_point="Model",
        backend="triton",
        problem_id="p1",
        initial_messages=None,
        cfg=cfg,
    )
    env.reset()
    return env


def test_env_step_correct_stops_and_rewards(monkeypatch):
    async def fake_eval(server_url, payload, **kwargs):
        return {"status": "completed", "compiled": True, "correctness": True, "speedup": 2.5}

    monkeypatch.setattr(env_kernelgym, "evaluate_kernel", fake_eval)
    env = _make_env(KernelGymRunConfig(max_turns=3, stop_on_correct=True))
    obs, done, info = asyncio.run(env.step(GOOD_KERNEL))
    assert done is True
    assert env.final_reward == pytest.approx(1.0)
    assert "speedup" in obs["obs_str"]
    assert env.last_eval_info["correctness"] is True


def test_env_multiturn_best_aggregation(monkeypatch):
    seq = [
        {"status": "completed", "compiled": True, "correctness": False, "speedup": 0.0},
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 1.5},
    ]

    async def fake_eval(server_url, payload, **kwargs):
        return seq[env.turn - 1]

    env = _make_env(KernelGymRunConfig(max_turns=3, stop_on_correct=True, reward_aggregation="best"))
    monkeypatch.setattr(env_kernelgym, "evaluate_kernel", fake_eval)

    obs1, done1, _ = asyncio.run(env.step(GOOD_KERNEL))
    assert done1 is False  # wrong, keep going
    obs2, done2, _ = asyncio.run(env.step(GOOD_KERNEL))
    assert done2 is True  # correct -> stop
    assert env.final_reward == pytest.approx(1.0)  # best of [0.0, 1.0]
    assert env.turn == 2


def test_env_preflight_failure_gives_penalty(monkeypatch):
    async def fail_if_called(server_url, payload, **kwargs):  # pragma: no cover
        raise AssertionError("evaluate_kernel should not be called when preflight fails")

    monkeypatch.setattr(env_kernelgym, "evaluate_kernel", fail_if_called)
    env = _make_env(KernelGymRunConfig(max_turns=1, reward=RewardConfig(penalty_score=0.0)))
    # No `class ModelNew(nn.Module)` -> preflight fails, no HTTP call.
    obs, done, info = asyncio.run(env.step("```triton\nclass Wrong:\n    pass\n```"))
    assert done is True  # max_turns reached
    assert env.rewards[0] == pytest.approx(0.0)
    assert "validation failed" in obs["obs_str"].lower()


def test_env_max_turns_truncates(monkeypatch):
    async def fake_eval(server_url, payload, **kwargs):
        return {"status": "completed", "compiled": True, "correctness": False, "speedup": 0.0}

    monkeypatch.setattr(env_kernelgym, "evaluate_kernel", fake_eval)
    env = _make_env(KernelGymRunConfig(max_turns=2, stop_on_correct=True))
    _, done1, _ = asyncio.run(env.step(GOOD_KERNEL))
    assert done1 is False
    _, done2, _ = asyncio.run(env.step(GOOD_KERNEL))
    assert done2 is True  # hit max_turns
    assert env.turn == 2


def test_build_initial_messages_has_system_and_reference():
    env = _make_env(KernelGymRunConfig())
    msgs = env.build_initial_messages()
    assert msgs[0]["role"] == "system"
    assert "Triton" in msgs[0]["content"]
    assert any(REF_CODE.strip()[:30] in m["content"] for m in msgs if m["role"] == "user")


# --------------------------------------------------------------------------- #
# data conversion
# --------------------------------------------------------------------------- #
def test_convert_record_basic():
    rec = prepare_data.convert_record(
        {"task": {"problem_id": "p_conv", "reference_code": REF_CODE}, "backend": "cuda"}
    )
    assert rec is not None
    assert rec["label"] == "p_conv"
    assert rec["metadata"]["backend"] == "cuda"
    assert rec["metadata"]["entry_point"] == "Model"  # default
    assert REF_CODE.strip()[:20] in rec["prompt"]


def test_convert_record_skips_empty_reference():
    assert prepare_data.convert_record({"task": {"problem_id": "x", "reference_code": "   "}}) is None
