"""Training-side backstop that drops non-agent-side Polar traces.

Polar's ``record_filters`` is the primary gate, but it used to fire only when the
bare-request payload matched one hardcoded document, so the same shape carrying
other file bodies became a trainable trace. This backstop keeps a stale polar
deployment from feeding those into GRPO. Shapes below mirror real rollout dumps:
across 1526 dumped samples the predicate caught 38/38 bad traces with 0 of 1488
legitimate traces dropped.
"""

from __future__ import annotations

from typing import Any

from vime_bridge.adapter import _is_non_agent_side_trace


class _Trace:
    def __init__(
        self,
        prompt_messages: Any,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.prompt_messages = prompt_messages
        self.tools = tools


_HARNESS_SYSTEM = {
    "role": "system",
    "content": "You are Claude Code, Anthropic's official CLI for Claude.",
}
_CMAKE_BODY = (
    "# Copyright (c) 2025 Huawei Technologies Co., Ltd.\n"
    "function(ascendc_compile_kernel target_name)\n"
    "endfunction()\n"
)


def test_drops_bare_user_prompt_carrying_a_file_body() -> None:
    trace = _Trace([{"role": "user", "content": _CMAKE_BODY}])

    assert _is_non_agent_side_trace(trace) is True


def test_drops_bare_prompt_regardless_of_payload() -> None:
    for body in (
        "# Triton Ascend 基础知识参考手册\n\n本文档汇集 Triton Ascend 编程要点。",
        "数值验证失败: 测试第 1/3 组输入...",
        "---\nname: ascend-kernel-developer\n---\n",
        "#!/usr/bin/env python3\n# -*- coding: UTF-8 -*-\n",
    ):
        assert _is_non_agent_side_trace(_Trace([{"role": "user", "content": body}])) is True, body


def test_keeps_main_agent_prompt() -> None:
    trace = _Trace([_HARNESS_SYSTEM, {"role": "user", "content": "Implement 3_Add."}])

    assert _is_non_agent_side_trace(trace) is False


def test_keeps_subagent_prompt_with_its_own_system_message() -> None:
    """A dispatched subagent's first turn: one user message, but a system prompt."""
    trace = _Trace(
        [
            {"role": "system", "content": "You are a file search specialist. READ-ONLY MODE."},
            {"role": "user", "content": "Locate kernel/op_host/*.cpp"},
        ]
    )

    assert _is_non_agent_side_trace(trace) is False


def test_keeps_bare_prompt_that_still_carries_a_tool_schema() -> None:
    trace = _Trace(
        [{"role": "user", "content": _CMAKE_BODY}],
        tools=[{"type": "function", "function": {"name": "Read"}}],
    )

    assert _is_non_agent_side_trace(trace) is False


def test_keeps_multi_turn_prompt_without_a_system_message() -> None:
    trace = _Trace(
        [
            {"role": "user", "content": "Fix the CopyOut bug."},
            {"role": "assistant", "content": "Reading the kernel."},
            {"role": "tool", "content": "1\t__aicore__ inline void CopyOut(...)"},
        ]
    )

    assert _is_non_agent_side_trace(trace) is False


def test_tool_call_count_is_not_a_signal() -> None:
    """Legit trailing turns answer in prose and call nothing — must survive.

    Real dumps contain such traces woken by a ``<task-notification>`` or an
    output-token-limit resume; keying off tool-call count would drop them.
    """
    trace = _Trace(
        [
            _HARNESS_SYSTEM,
            {"role": "user", "content": "Implement the operator."},
            {"role": "assistant", "content": "Generation phase exhausted (5/5)."},
            {"role": "user", "content": "<task-notification>...</task-notification>"},
        ]
    )

    assert _is_non_agent_side_trace(trace) is False


def test_missing_or_malformed_prompt_is_not_dropped() -> None:
    for prompt in ([], None, "", [{"role": "assistant", "content": "hi"}], [None]):
        assert _is_non_agent_side_trace(_Trace(prompt)) is False, prompt


def test_absent_tools_attribute_is_tolerated() -> None:
    class _Bare:
        prompt_messages = [{"role": "user", "content": _CMAKE_BODY}]

    assert _is_non_agent_side_trace(_Bare()) is True
