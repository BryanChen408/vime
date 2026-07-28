"""Configuration helpers for Slime-driven Polar rollouts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from types import SimpleNamespace
from typing import Any

_PLACEHOLDER_RE = re.compile(r"{([^{}]+)}")


@dataclass(frozen=True, slots=True)
class PolarSlimeConfig:
    rollout_server_url: str
    submit_mode: str
    operator_profile: str | None
    task_template: dict[str, Any]
    task_id_template: str
    instruction_template: str | None
    reward_key: str
    max_concurrency: int
    max_session_concurrency: int
    max_async_level: int
    max_sessions_per_task: int | None
    max_off_policy_steps: int
    request_timeout: float | None
    callback_host: str
    scoring_mode: str
    min_complete_accept_fraction: float
    tokenizer_name_or_path: str | None
    add_generation_prompt: bool
    eval_dataset_name: str
    scheduler_mode: str
    max_active_sessions: int
    session_pool_pause_policy: str
    operator_tasks_dir: str | None = None
    session_pool_release_on_postrun: bool = False


def resolve_polar_slime_config(args: Any) -> PolarSlimeConfig:
    rollout_server_url = _first_configured(args, "polar_url", "polar_rollout_url")
    if rollout_server_url is None:
        raise ValueError(
            "Polar rollout URL is not configured. Set polar_url or polar_rollout_url "
            "in Slime's custom config YAML."
        )

    task_template = _load_polar_task_template(getattr(args, "polar_task_template", None))
    submit_mode = str(getattr(args, "polar_submit_mode", "") or "").strip().lower()
    if not submit_mode:
        submit_mode = "task_request" if task_template else "operator_samples"
    if submit_mode not in {"task_request", "operator_samples"}:
        raise ValueError("polar_submit_mode must be 'task_request' or 'operator_samples'")
    if submit_mode == "task_request" and "agent" not in task_template:
        raise ValueError("polar_task_template must include an agent spec")

    max_async_level = int(_first_configured(args, "rollout_max_async_level", "polar_max_async_level", default=2))
    if max_async_level <= 0:
        raise ValueError("rollout_max_async_level must be greater than 0")

    rollout_batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be greater than 0")

    group_size = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    if group_size <= 0:
        raise ValueError("n_samples_per_prompt must be greater than 0")

    update_weights_interval = int(getattr(args, "update_weights_interval", 1) or 1)
    if update_weights_interval <= 0:
        raise ValueError("update_weights_interval must be greater than 0")

    max_concurrency = rollout_batch_size * max_async_level
    max_session_concurrency = max_concurrency * group_size
    max_sessions_per_task = _resolve_max_sessions_per_task(args)
    scheduler_mode = str(_first_configured(args, "rollout_scheduler_mode", "polar_scheduler_mode", default="group")).strip().lower()
    if scheduler_mode not in {"group", "session_pool"}:
        raise ValueError("rollout_scheduler_mode must be 'group' or 'session_pool'")
    max_active_sessions = _resolve_max_active_sessions(args, default=max_session_concurrency)
    session_pool_pause_policy = str(
        _first_configured(
            args,
            "rollout_session_pool_pause_policy",
            "polar_session_pool_pause_policy",
            default="drain_open_groups",
        )
    ).strip().lower()
    if session_pool_pause_policy != "drain_open_groups":
        raise ValueError("rollout_session_pool_pause_policy must be 'drain_open_groups'")
    session_pool_release_on_postrun = _resolve_bool(
        args,
        "rollout_release_on_postrun",
        fallback_names=(
            "rollout_session_pool_release_on_postrun",
            "polar_session_pool_release_on_postrun",
        ),
        default=False,
    )
    max_off_policy_steps = max_async_level + update_weights_interval

    request_timeout = _first_configured(args, "rollout_request_timeout", "polar_request_timeout")
    if request_timeout is not None:
        request_timeout = float(request_timeout)
        if request_timeout <= 0:
            raise ValueError("rollout_request_timeout must be greater than 0")

    callback_host = str(getattr(args, "polar_callback_host", "127.0.0.1")).strip()
    if not callback_host:
        raise ValueError("polar_callback_host must be a non-empty host or IP")
    if callback_host in {"0.0.0.0", "::"}:
        raise ValueError("polar_callback_host must be reachable by the rollout server, not a wildcard bind address")

    scoring_mode = str(getattr(args, "polar_scoring_mode", "group")).strip().lower()
    if scoring_mode not in {"group", "individual"}:
        raise ValueError("polar_scoring_mode must be 'group' or 'individual'")

    min_complete_accept_fraction = float(
        _first_configured(
            args,
            "rollout_min_complete_accept_fraction",
            "polar_min_complete_accept_fraction",
            default=0.0,
        )
        or 0.0
    )
    if not 0.0 <= min_complete_accept_fraction <= 1.0:
        raise ValueError("polar_min_complete_accept_fraction must be between 0 and 1")

    return PolarSlimeConfig(
        rollout_server_url=str(rollout_server_url).rstrip("/"),
        submit_mode=submit_mode,
        operator_profile=_optional_text(getattr(args, "polar_profile", None)),
        operator_tasks_dir=_optional_text(
            getattr(args, "operator_tasks_dir", None)
            or getattr(args, "polar_tasks_dir", None)
        ),
        task_template=task_template,
        task_id_template=str(
            getattr(args, "polar_task_id_template", "polar-slime-{rollout_id}-{sample.group_index}")
        ),
        instruction_template=getattr(args, "polar_instruction_template", None),
        reward_key=str(
            getattr(args, "polar_reward_key", None)
            or getattr(args, "reward_key", None)
            or "score"
        ),
        max_concurrency=max_concurrency,
        max_session_concurrency=max_session_concurrency,
        max_async_level=max_async_level,
        max_sessions_per_task=max_sessions_per_task,
        max_off_policy_steps=max_off_policy_steps,
        request_timeout=request_timeout,
        callback_host=callback_host,
        scoring_mode=scoring_mode,
        min_complete_accept_fraction=min_complete_accept_fraction,
        tokenizer_name_or_path=getattr(args, "hf_checkpoint", None),
        add_generation_prompt=bool(getattr(args, "polar_add_generation_prompt", True)),
        eval_dataset_name=str(getattr(args, "polar_eval_dataset_name", "polar_eval")),
        scheduler_mode=scheduler_mode,
        max_active_sessions=max_active_sessions,
        session_pool_pause_policy=session_pool_pause_policy,
        session_pool_release_on_postrun=session_pool_release_on_postrun,
    )


def _resolve_max_sessions_per_task(args: Any) -> int | None:
    configured = getattr(args, "polar_max_sessions_per_task", None)
    if configured not in (None, ""):
        value = int(configured)
        if value <= 0:
            raise ValueError("polar_max_sessions_per_task must be greater than 0")
        return value
    return None


def _resolve_max_active_sessions(args: Any, *, default: int) -> int:
    configured = _first_configured(args, "rollout_max_active_sessions", "polar_max_active_sessions")
    if configured in (None, ""):
        return int(default)
    value = int(configured)
    if value <= 0:
        raise ValueError("rollout_max_active_sessions must be greater than 0")
    return value


def _resolve_bool(
    args: Any,
    name: str,
    *,
    fallback_names: tuple[str, ...] = (),
    default: bool,
) -> bool:
    configured = _first_configured(args, name, *fallback_names, default=default)
    if configured in (None, ""):
        return default
    if isinstance(configured, bool):
        return configured
    if isinstance(configured, int):
        return bool(configured)
    value = str(configured).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _first_configured(args: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(args, name, None)
        if value not in (None, ""):
            return value
    return default


def _load_polar_task_template(value: Any) -> dict[str, Any]:
    """Resolve ``--polar-task-template`` into a task-payload mapping.

    Accepts a path to an OmegaConf YAML/JSON file (the ``task_request`` submit
    mode, e.g. the SWE-Gym coding-agent pipeline), an already-parsed mapping
    (tests / programmatic callers), or ``None``/"" (``operator_samples`` mode).
    Mirrors how ``--eval-config`` is loaded in ``vime/utils/arguments.py``.
    """
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return deepcopy(value)
    if isinstance(value, str):
        from omegaconf import OmegaConf

        loaded = OmegaConf.to_container(OmegaConf.load(value), resolve=True)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"polar_task_template file {value!r} must contain a mapping"
            )
        return loaded
    raise ValueError("polar_task_template must be a file path or a mapping")


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _parse_device_pool(spec: Any) -> list[str]:
    if isinstance(spec, (list, tuple)):
        return [str(item).strip() for item in spec if str(item).strip()]

    text = str(spec or "").strip()
    if not text:
        return []
    if "-" in text and "," not in text:
        lo, hi = text.split("-", 1)
        return [str(device) for device in range(int(lo), int(hi) + 1)]
    return [part.strip() for part in text.split(",") if part.strip()]


def resolve_vllm_router_base_url(args: Any) -> str | None:
    """vime serves the OpenAI endpoint from vllm/vllm-ascend (not sglang)."""
    ip = getattr(args, "vllm_router_ip", None)
    port = getattr(args, "vllm_router_port", None)
    if ip in (None, "") or port in (None, ""):
        return None
    return f"http://{ip}:{port}"


def resolve_sglang_router_base_url(args: Any) -> str | None:
    ip = getattr(args, "sglang_router_ip", None)
    port = getattr(args, "sglang_router_port", None)
    if ip in (None, "") or port in (None, ""):
        # vime uses vllm, not sglang; fall back to the vllm router so any
        # existing {sglang.router_base_url} templates keep resolving.
        return resolve_vllm_router_base_url(args)
    return f"http://{ip}:{port}"


def render_task_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    sample: Any,
    instruction: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
) -> dict[str, Any]:
    context = _build_context(
        args=args,
        sample=sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=num_rollouts,
    )
    payload = _render_template_value(deepcopy(config.task_template), context)
    if not isinstance(payload, dict):
        raise ValueError("polar_task_template must render to a mapping")

    payload["task_id"] = str(_render_template_value(config.task_id_template, context))
    payload["instruction"] = instruction
    payload["num_samples"] = num_rollouts
    return payload


def render_instruction(
    *,
    args: Any,
    config: PolarSlimeConfig,
    sample: Any,
    prompt_text: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
) -> str:
    template = config.instruction_template
    if not template:
        return prompt_text
    context = _build_context(
        args=args,
        sample=sample,
        instruction=prompt_text,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=num_rollouts,
    )
    rendered = _render_template_value(template, context)
    if not isinstance(rendered, str):
        raise ValueError("polar_instruction_template must render to a string")
    return rendered


def _build_context(
    *,
    args: Any,
    sample: Any,
    instruction: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
) -> dict[str, Any]:
    args_namespace = SimpleNamespace(**vars(args)) if hasattr(args, "__dict__") else args
    metadata = deepcopy(getattr(sample, "metadata", None) or {})
    return {
        "args": args_namespace,
        "instruction": instruction,
        "num_rollouts": num_rollouts,
        "rollout_id": rollout_id,
        "sglang": SimpleNamespace(router_base_url=resolve_sglang_router_base_url(args)),
        "vllm": SimpleNamespace(router_base_url=resolve_vllm_router_base_url(args)),
        "sample": SimpleNamespace(
            prompt=deepcopy(getattr(sample, "prompt", "")),
            response=deepcopy(getattr(sample, "response", "")),
            label=getattr(sample, "label", None),
            metadata=_to_namespace(metadata),
            index=getattr(sample, "index", None),
            group_index=getattr(sample, "group_index", None),
            status=getattr(sample, "status", None),
        ),
        "task_position": task_position,
    }


def _render_template_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if match := re.fullmatch(r"{([^{}]+)}", value):
            resolved = deepcopy(_resolve_path(context, match.group(1)))
            return _from_namespace(resolved)

        def replace(match: re.Match[str]) -> str:
            resolved = _resolve_path(context, match.group(1))
            return "" if resolved is None else str(resolved)

        return _PLACEHOLDER_RE.sub(replace, value)

    if isinstance(value, list):
        return [_render_template_value(item, context) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _render_template_value(item, context)
            for key, item in value.items()
        }

    return value


def _resolve_path(context: dict[str, Any], path: str) -> Any:
    current: Any = context
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise ValueError(f"Unknown template variable: {path}")
            current = current[part]
            continue

        if hasattr(current, part):
            current = getattr(current, part)
            continue

        raise ValueError(f"Unknown template variable: {path}")
    return current


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _from_namespace(value: Any) -> Any:
    """Convert SimpleNamespace trees back to plain dicts for JSON serialization."""
    if isinstance(value, SimpleNamespace):
        return {k: _from_namespace(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _from_namespace(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_namespace(item) for item in value]
    return value
