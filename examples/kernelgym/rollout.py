"""KernelGym multi-turn rollout (text-only) for vime.

A trimmed-down sibling of ``examples/geo3k_vlm_multi_turn/rollout.py``: the same
prefix-stable multi-turn token bookkeeping over the vLLM render route, but with
all multimodal handling removed and a kernelGym verifier env in the loop.

Per turn:
  render(messages) -> /inference/v1/generate -> decode -> env.step (HTTP eval)
  -> append the feedback as the next user turn (prefix-stable) -> repeat.

After the episode, the trajectory reward (computed from kernelGym's
compile/correctness/speedup signal) is written onto ``sample.reward``. vime's
default ``vllm_rollout`` leaves an already-filled reward untouched, so no
``--custom-rm-path`` is needed.

Wire-up::

    --custom-generate-function-path examples.kernelgym.rollout.generate
    --custom-config-path examples/kernelgym/kernelgym_config.yaml
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from vime.rollout.vllm_rollout import (
    GenerateState,
    _build_inference_sampling_params,
    _coerce_flat_int_token_ids,
    _mm_render_response_to_generate_body,
)
from vime.utils.http_utils import post
from vime.utils.types import Sample

logger = logging.getLogger(__name__)

DEFAULT_ENV_MODULE = "examples.kernelgym.env_kernelgym"


def _load_env_module(env_path: str | None):
    """Load the interaction-env module from a dotted module path or a .py file."""
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"kernelgym_env_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def _abort(sample: Sample, reason: str, reward: float = 0.0) -> Sample:
    """Return a uniform, non-crashing sample when the rollout cannot proceed."""
    if not sample.tokens:
        sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = max(1, len(sample.loss_mask or []))
    if not sample.loss_mask:
        sample.loss_mask = [0] * len(sample.tokens)
    sample.reward = reward
    sample.status = Sample.Status.FAILED
    sample.metadata = {**(sample.metadata or {}), "kernelgym_abort_reason": reason}
    logger.warning("[kernelgym] rollout aborted: %s", reason)
    return sample


def _extract_tokens_and_logprobs(choice: dict) -> tuple[list[int], list[float]]:
    """Pull response token ids and per-token logprobs from a vLLM
    ``/inference/v1/generate`` choice (inlined from vllm_rollout's default path,
    whose private helper is not importable across vime versions)."""
    tokens = list(choice.get("token_ids") or [])
    logprobs: list[float] = []
    lp = choice.get("logprobs")
    if isinstance(lp, dict):
        content_items = lp.get("content") or []
        logprobs = [
            float(item.get("logprob", 0.0)) if isinstance(item, dict) else 0.0 for item in content_items
        ]
    if not logprobs:
        logprobs = [0.0] * len(tokens)
    return tokens, logprobs


async def generate(args: Any, sample: Sample, sampling_params: dict) -> Sample:
    """Custom multi-turn rollout that drives the KernelGymEnv verifier loop."""
    assert not args.partial_rollout, "Partial rollout is not supported for kernelgym rollouts."
    if getattr(args, "use_rollout_routing_replay", False):
        raise NotImplementedError(
            "kernelgym multi-turn rollout does not support --use-rollout-routing-replay; "
            "disable it for kernelgym training."
        )
    if args.max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")

    state = GenerateState(args)
    base_url = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"

    headers = None
    if getattr(args, "router_policy", None) == "consistent_hash":
        sample.session_id = sample.session_id or str(uuid.uuid4())
        headers = {"x-session-id": sample.session_id}

    env_module = _load_env_module(getattr(args, "rollout_interaction_env_path", None))
    build_env = getattr(env_module, "build_env", None)
    if not callable(build_env):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    env = build_env(sample=sample, args=args)

    messages = env.build_initial_messages()
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.tokens = list(sample.tokens) if sample.tokens else []

    sampling_params = sampling_params.copy()
    inference_sampling_params = _build_inference_sampling_params(sampling_params)
    max_response_budget = sampling_params.get("max_new_tokens")

    def remaining_budget() -> int | None:
        return None if max_response_budget is None else max_response_budget - sample.response_length

    async def render() -> dict:
        payload = {"model": args.hf_checkpoint, "messages": messages}
        render_data = await post(f"{base_url}/v1/chat/completions/render", payload, headers=headers)
        return _mm_render_response_to_generate_body(render_data, args.hf_checkpoint)

    def append_response_window(token_ids: list[int], loss_mask: list[int], log_probs: list[float] | None = None) -> None:
        if not token_ids:
            return
        if len(loss_mask) != len(token_ids):
            raise ValueError(f"loss_mask length {len(loss_mask)} != token_ids length {len(token_ids)}")
        sample.tokens.extend(token_ids)
        sample.loss_mask.extend(loss_mask)
        sample.rollout_log_probs.extend(log_probs if log_probs is not None else [0.0] * len(token_ids))
        sample.response_length += len(token_ids)

    def sampling_params_for_turn() -> dict | None:
        params = dict(inference_sampling_params)
        max_tokens = remaining_budget()
        if max_tokens is None:
            return params
        if max_tokens <= 0:
            return None
        params["max_tokens"] = max_tokens
        return params

    response_tokens: list[int] = []
    try:
        env.reset()
        pending_obs_offset: int | None = None
        rendered_body = await render()
        prompt_ids = _coerce_flat_int_token_ids(rendered_body.get("token_ids"))
        if not sample.tokens:
            sample.tokens = list(prompt_ids)
        if getattr(args, "rollout_max_context_len", None) is not None:
            max_response_budget = max(0, args.rollout_max_context_len - len(sample.tokens))

        for turn_idx in range(args.max_turns):
            input_ids = _coerce_flat_int_token_ids(rendered_body.get("token_ids"))

            # Account the just-appended feedback tokens against the budget.
            if pending_obs_offset is not None:
                obs_tokens = input_ids[pending_obs_offset:]
                remaining = remaining_budget()
                if remaining is not None and len(obs_tokens) > remaining:
                    append_response_window(obs_tokens[: max(remaining, 0)], [0] * max(remaining, 0))
                    sample.status = Sample.Status.TRUNCATED
                    break
                append_response_window(obs_tokens, [0] * len(obs_tokens))
                pending_obs_offset = None

            current_sampling_params = sampling_params_for_turn()
            if current_sampling_params is None:
                sample.status = Sample.Status.TRUNCATED
                break

            body = dict(rendered_body)
            body["sampling_params"] = current_sampling_params
            output = await post(f"{base_url}/inference/v1/generate", body, headers=headers)
            choice = output["choices"][0]
            finish_reason = choice.get("finish_reason") or "stop"
            new_tokens, new_logprobs = _extract_tokens_and_logprobs(choice)

            if not new_tokens and finish_reason in ("abort", "cancelled"):
                sample.status = Sample.Status.ABORTED
                break

            response_text = state.tokenizer.decode(new_tokens, skip_special_tokens=False) if new_tokens else ""
            train_tokens = list(new_tokens)
            train_logprobs = list(new_logprobs)
            train_loss_mask = [1] * len(train_tokens)

            # Append an artificial EOS after a stop string (mirrors geo3k).
            stop = current_sampling_params.get("stop")
            eos_token_id = getattr(state.tokenizer, "eos_token_id", None)
            if (
                stop
                and eos_token_id is not None
                and getattr(args, "append_eos_token_after_stop_str_in_multi_turn", True)
            ):
                stop_strings = (stop,) if isinstance(stop, str) else tuple(stop)
                already_has_eos = bool(train_tokens and train_tokens[-1] == eos_token_id)
                if stop_strings and response_text.endswith(stop_strings) and not already_has_eos:
                    train_tokens.append(int(eos_token_id))
                    train_logprobs.append(0.0)
                    train_loss_mask.append(0)

            response_tokens.extend(new_tokens)
            append_response_window(train_tokens, train_loss_mask, train_logprobs)
            messages.append({"role": "assistant", "content": response_text})

            if finish_reason == "length":
                sample.status = Sample.Status.TRUNCATED
                break
            if finish_reason in ("abort", "cancelled"):
                sample.status = Sample.Status.ABORTED
                break

            # Verify the kernel and decide whether to continue revising.
            observation, done, _info = await env.step(response_text)
            if done:
                sample.status = Sample.Status.COMPLETED
                break
            if turn_idx + 1 >= args.max_turns:
                sample.status = Sample.Status.TRUNCATED
                break

            messages.append(env.format_observation(observation))
            render_prefix_len = len(input_ids) + len(new_tokens)
            pending_obs_offset = render_prefix_len
            rendered_body = await render()
            rendered_ids = _coerce_flat_int_token_ids(rendered_body.get("token_ids"))
            is_prefix_stable = rendered_ids[:pending_obs_offset] == sample.tokens[:pending_obs_offset]
            sample.metadata["multiturn_render"] = {
                "prefix_stable": is_prefix_stable,
                "prefix_len": pending_obs_offset,
                "sample_len": len(sample.tokens),
                "rendered_len": len(rendered_ids),
            }
            if not is_prefix_stable:
                raise RuntimeError(
                    "Full conversation render is not prefix-stable with the generated token stream: "
                    f"{sample.metadata['multiturn_render']}"
                )

        sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
        sample.response_length = len(sample.loss_mask)
        # Fill the trajectory reward from the kernelGym verifier.
        sample.reward = env.final_reward
        sample.metadata = {
            **(sample.metadata or {}),
            "kernelgym_reward": sample.reward,
            "kernelgym_best": env.best_eval_info,
            "kernelgym_last": env.last_eval_info,
            "kernelgym_num_turns": env.turn,
        }
        if sample.status == Sample.Status.PENDING:
            sample.status = Sample.Status.COMPLETED
        return sample
    except Exception as exc:  # noqa: BLE001 - isolate a bad trajectory from the batch
        logger.warning("[kernelgym] rollout exception (problem=%s): %s", getattr(env, "problem_id", "?"), exc)
        try:
            fallback = env.final_reward
        except Exception:
            fallback = 0.0
        return _abort(sample, f"exception:{type(exc).__name__}", reward=fallback)
    finally:
        try:
            await env.close()
        except Exception:
            pass
