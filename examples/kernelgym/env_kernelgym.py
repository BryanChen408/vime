"""KernelGym multi-turn interaction environment for vime.

Ports rllm-lilac's ``KernelGymEnv`` + ``KernelAgent`` into vime's custom-rollout
paradigm. One :class:`KernelGymEnv` drives a single prompt's multi-turn episode:

    turn 0: model sees the PyTorch reference  -> emits a Triton ``ModelNew``
    env.step: extract kernel -> POST to kernelGym -> compile/correctness/perf
              -> reward + human-readable feedback
    turn 1+: model sees the feedback          -> emits an improved ``ModelNew``
    ... until correct (configurable) or max_turns.

The env is async-native (``await env.step(...)``) because kernel evaluation is a
remote HTTP round-trip; the companion ``rollout.generate`` awaits it. The final
trajectory reward is exposed via :pyattr:`final_reward` and the rollout writes it
onto ``sample.reward`` (vime's vllm_rollout leaves an already-filled reward
untouched).

Wire-up (see ``kernelgym_config.yaml``)::

    --custom-generate-function-path examples.kernelgym.rollout.generate
    --custom-config-path examples/kernelgym/kernelgym_config.yaml
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from .kernelgym_client import (
    KERNEL_CODE_IMPORT_PREFIX,
    build_eval_payload,
    evaluate_kernel,
    preflight_validate,
)
from .reward import RewardConfig, compute_reward_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates (migrated from rllm KernelAgent)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are looking at this PyTorch code and thinking it could be optimized with Triton. You need to create a Triton version with the `ModelNew`. This triton version must be execution on Ascend NPU platforms.

Please firstly analyze this code and think hard how you can optimize it. YOU MUST wrap your final code in a ```triton ... ``` code block. No other code block markers are acceptable.

**Please output and show your thinking, plan, analysis etc., before your coding, which should be as more as possible.**

Here's the PyTorch code:
"""

INITIAL_USER_TEMPLATE = """
```python
{reference_code}
```
"""

REVISION_USER_TEMPLATE = """\
Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{feedback}

Return an improved Triton implementation named `ModelNew` as a single ```triton``` block. Let's think step by step.
"""


# ---------------------------------------------------------------------------
# Kernel-code extraction (migrated from rllm extract_kernel_code)
# ---------------------------------------------------------------------------
def extract_kernel_code(solution_str: str) -> str:
    """Extract the kernel implementation from a model response.

    Tries explicit markers first, then the last fenced code block (which covers
    the ```triton ...``` block the system prompt asks for), and finally falls
    back to the whole response.
    """
    patterns = [
        r"# Kernel Implementation\s*\n(.*?)(?=# End|$)",
        r"```python\s*# Kernel\s*\n(.*?)```",
        r"# Your implementation:\s*\n(.*?)(?=# End|$)",
        r"# Generated kernel:\s*\n(.*?)(?=# End|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Last fenced block of any language (```triton / ```python / ```).
    code_blocks = re.findall(r"```(?:\w+)?\s*\n?(.*?)```", solution_str, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    return solution_str.strip()


def patch_model_new(kernel_code: str) -> str:
    """rllm Patch 1: a bare ``class ModelNew:`` must inherit ``nn.Module``."""
    if "class ModelNew:" in kernel_code:
        kernel_code = kernel_code.replace("class ModelNew:", "class ModelNew(nn.Module):")
    return kernel_code


# ---------------------------------------------------------------------------
# Run configuration (parsed from args / custom-config yaml)
# ---------------------------------------------------------------------------
@dataclass
class KernelGymRunConfig:
    server_url: str = "http://127.0.0.1:10907"
    max_turns: int = 4
    # Stop the episode as soon as a correct kernel is produced (saves eval
    # cost). When False, run all max_turns regardless.
    stop_on_correct: bool = True
    # How to reduce per-turn rewards into one trajectory reward: best | last.
    reward_aggregation: str = "best"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    server_timeout: int = 600  # kernelGym-side per-task timeout (payload.timeout)
    client_timeout: int = 900  # client-side total wait (>= server_timeout)
    request_timeout: int = 120  # per single HTTP request
    max_retries: int = 2
    poll_interval: float = 1.0
    # Validation mode forces decoy-kernel detection on (matches rllm is_valid).
    is_valid: bool = False
    enable_profiling: bool = True
    verbose_errors: bool = True
    detect_decoy_kernel: bool = True
    # How the reference is executed on the server: "pytorch" | "compile" | None.
    reference_backend: str | None = None
    # Default kernel backend; overridden per-sample by metadata["backend"].
    backend: str = "triton"
    feedback_max_chars: int = 4000
    reward: RewardConfig = field(default_factory=RewardConfig)

    _SIMPLE_KEYS = (
        "server_url",
        "stop_on_correct",
        "reward_aggregation",
        "num_correct_trials",
        "num_perf_trials",
        "server_timeout",
        "client_timeout",
        "request_timeout",
        "max_retries",
        "poll_interval",
        "is_valid",
        "enable_profiling",
        "verbose_errors",
        "detect_decoy_kernel",
        "reference_backend",
        "backend",
        "feedback_max_chars",
    )

    @staticmethod
    def from_args(args: Any) -> "KernelGymRunConfig":
        raw = getattr(args, "kernelgym", None) or {}
        try:
            raw = dict(raw)
        except (TypeError, ValueError):
            raw = {}

        cfg = KernelGymRunConfig()
        # max_turns is shared with the rollout loop, so prefer the top-level arg.
        max_turns = getattr(args, "max_turns", None)
        if max_turns is None:
            max_turns = raw.get("max_turns")
        if max_turns is not None:
            cfg.max_turns = int(max_turns)

        for key in KernelGymRunConfig._SIMPLE_KEYS:
            if key in raw and raw[key] is not None:
                setattr(cfg, key, raw[key])

        cfg.reward = RewardConfig.from_mapping(raw.get("reward"))

        # Honor the timeout invariant: client wait must cover server execution.
        if cfg.client_timeout < cfg.server_timeout:
            logger.warning(
                "kernelgym client_timeout (%ss) < server_timeout (%ss); raising client_timeout",
                cfg.client_timeout,
                cfg.server_timeout,
            )
            cfg.client_timeout = cfg.server_timeout
        return cfg


# ---------------------------------------------------------------------------
# Feedback formatting
# ---------------------------------------------------------------------------
def format_feedback(summary: dict, feedback_max_chars: int) -> str:
    status = summary.get("status", "unknown")
    compiled = summary.get("compiled", False)
    correctness = summary.get("correctness", False)
    speedup = summary.get("speedup", 0.0)
    error = summary.get("error") or ""
    if error and len(str(error)) > feedback_max_chars:
        error = str(error)[:feedback_max_chars] + "... [truncated]"
    lines = [
        f"- status: {status}",
        f"- compiled: {compiled}",
        f"- correctness: {correctness}",
        f"- speedup: {speedup}",
    ]
    if error:
        lines.append(f"- error: {error}")
    if summary.get("decoy_kernel"):
        lines.append("- decoy_kernel: true (your kernel only forwarded to the reference; write a real kernel)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class KernelGymEnv:
    """Single-episode async environment for kernel generation."""

    def __init__(
        self,
        *,
        reference_code: str,
        entry_point: str,
        backend: str,
        problem_id: str,
        initial_messages: list[dict] | None,
        cfg: KernelGymRunConfig,
        train_id: str = "",
    ) -> None:
        self.reference_code = reference_code or ""
        self.entry_point = entry_point or "Model"
        self.backend = backend or cfg.backend
        self.problem_id = problem_id or "task"
        self._initial_messages = initial_messages
        self.cfg = cfg
        self.train_id = train_id

        self.session_uuid = uuid.uuid4().hex[:8]
        self.turn = 0
        self.rewards: list[float] = []
        self.eval_infos: list[dict] = []

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self.session_uuid = uuid.uuid4().hex[:8]
        self.turn = 0
        self.rewards = []
        self.eval_infos = []

    async def close(self) -> None:
        # The HTTP session is process-wide and shared; nothing to release here.
        return

    # -- prompts ------------------------------------------------------------
    def build_initial_messages(self) -> list[dict]:
        """System prompt + the dataset's initial user turn (reference code)."""
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        if self._initial_messages:
            return [system_msg] + [dict(m) for m in self._initial_messages]
        user_msg = {
            "role": "user",
            "content": INITIAL_USER_TEMPLATE.format(reference_code=self.reference_code),
        }
        return [system_msg, user_msg]

    def format_observation(self, observation: dict) -> dict:
        """Turn a step observation into the next user message."""
        feedback = (observation or {}).get("obs_str", "")
        return {"role": "user", "content": REVISION_USER_TEMPLATE.format(feedback=feedback)}

    # -- the core step ------------------------------------------------------
    async def step(self, response_text: str) -> tuple[dict, bool, dict]:
        """Evaluate one model response. Returns (observation, done, info)."""
        self.turn += 1
        kernel_code = patch_model_new(extract_kernel_code(response_text))
        full_kernel = KERNEL_CODE_IMPORT_PREFIX + kernel_code

        ok, missing = preflight_validate(self.reference_code, full_kernel, self.entry_point)
        if not ok:
            result = {
                "status": "failed",
                "error_message": f"Client validation failed: missing {missing}",
                "compiled": False,
                "correctness": False,
                "speedup": 0.0,
            }
        else:
            eval_tag = "validation" if self.cfg.is_valid else "train"
            task_id = f"{self.problem_id}_{self.session_uuid}|{self.turn}"
            payload = build_eval_payload(
                task_id=task_id,
                reference_code=self.reference_code,
                kernel_code=full_kernel,
                entry_point=self.entry_point,
                backend=self.backend,
                train_id=self.train_id,
                eval_tag=eval_tag,
                num_correct_trials=self.cfg.num_correct_trials,
                num_perf_trials=self.cfg.num_perf_trials,
                server_timeout=self.cfg.server_timeout,
                is_valid=self.cfg.is_valid,
                enable_profiling=self.cfg.enable_profiling,
                verbose_errors=self.cfg.verbose_errors,
                detect_decoy_kernel=self.cfg.detect_decoy_kernel,
                reference_backend=self.cfg.reference_backend,
            )
            try:
                result = await evaluate_kernel(
                    self.cfg.server_url,
                    payload,
                    client_timeout=self.cfg.client_timeout,
                    request_timeout=self.cfg.request_timeout,
                    max_retries=self.cfg.max_retries,
                    poll_interval=self.cfg.poll_interval,
                )
            except Exception as exc:  # noqa: BLE001 - never let a bad eval kill the rollout
                logger.warning("kernelgym evaluate failed problem=%s: %s", self.problem_id, exc)
                result = {"status": "failed", "error_message": f"client exception: {exc}"}

        summary = compute_reward_summary(result, self.cfg.reward)
        reward = float(summary.get("reward", self.cfg.reward.penalty_score))
        self.rewards.append(reward)
        self.eval_infos.append(summary)

        is_correct = bool(summary.get("correctness", False))
        done = (self.cfg.stop_on_correct and is_correct) or self.turn >= self.cfg.max_turns
        observation = {
            "obs_str": format_feedback(summary, self.cfg.feedback_max_chars),
            "server_result": summary,
        }
        info = {"reward": reward, "summary": summary, "turn": self.turn}
        return observation, done, info

    # -- outputs ------------------------------------------------------------
    @property
    def final_reward(self) -> float:
        if not self.rewards:
            return float(self.cfg.reward.penalty_score)
        if self.cfg.reward_aggregation == "last":
            return float(self.rewards[-1])
        return float(max(self.rewards))

    @property
    def last_eval_info(self) -> dict:
        if not self.eval_infos:
            return {"speedup": 0.0, "correctness": False, "compiled": False, "success": False}
        m = self.eval_infos[-1]
        return {
            "speedup": float(m.get("speedup", 0.0) or 0.0),
            "correctness": bool(m.get("correctness", False)),
            "compiled": bool(m.get("compiled", False)),
            "success": bool(m.get("success", False)),
        }

    @property
    def best_eval_info(self) -> dict:
        """Eval info of the best-reward turn (for logging the trajectory peak)."""
        if not self.eval_infos:
            return {"speedup": 0.0, "correctness": False, "compiled": False, "success": False}
        best_idx = max(range(len(self.rewards)), key=lambda i: self.rewards[i])
        m = self.eval_infos[best_idx]
        return {
            "speedup": float(m.get("speedup", 0.0) or 0.0),
            "correctness": bool(m.get("correctness", False)),
            "compiled": bool(m.get("compiled", False)),
            "success": bool(m.get("success", False)),
        }


# ---------------------------------------------------------------------------
# Factory used by rollout.generate via args.rollout_interaction_env_path
# ---------------------------------------------------------------------------
def _extract_task_fields(sample: Any) -> dict:
    """Pull reference_code / entry_point / backend / problem_id from a Sample.

    Primary source is ``sample.metadata`` (written by prepare_data.py). Falls
    back to ``sample.label`` for problem_id.
    """
    metadata = getattr(sample, "metadata", None) or {}
    reference_code = metadata.get("reference_code", "")
    entry_point = metadata.get("entry_point") or "Model"
    backend = metadata.get("backend") or "triton"
    problem_id = metadata.get("problem_id")
    if not problem_id:
        label = getattr(sample, "label", None)
        problem_id = str(label) if label is not None else "task"
    initial_messages = None
    prompt = getattr(sample, "prompt", None)
    if isinstance(prompt, list):
        initial_messages = prompt
    return {
        "reference_code": reference_code,
        "entry_point": entry_point,
        "backend": backend,
        "problem_id": problem_id,
        "initial_messages": initial_messages,
    }


def build_env(sample: Any | None = None, args: Any | None = None, **_: Any) -> KernelGymEnv:
    """Construct a KernelGymEnv for one sample (mirrors geo3k ``build_env``)."""
    cfg = KernelGymRunConfig.from_args(args)
    fields = _extract_task_fields(sample) if sample is not None else {
        "reference_code": "",
        "entry_point": "Model",
        "backend": cfg.backend,
        "problem_id": "task",
        "initial_messages": None,
    }
    if not fields["reference_code"]:
        logger.warning("kernelgym build_env: sample has empty reference_code (problem=%s)", fields["problem_id"])
    train_id = str(getattr(args, "kernelgym_train_id", "") or "")
    return KernelGymEnv(cfg=cfg, train_id=train_id, **fields)
