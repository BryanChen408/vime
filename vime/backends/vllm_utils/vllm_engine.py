"""Ray actor and launch helpers for vLLM OpenAI HTTP rollout.

Per-Ray-actor ``server_args`` dict is built via :func:`_compute_server_args`,
then :func:`build_vllm_cmd_and_env` turns it into ``vllm serve`` CLI + subprocess env.
:class:`VLLMEngine` manages the runtime HTTP control plane.
User-facing vLLM knobs remain on ``train.py`` as ``--vllm-*`` (see ``arguments.py``).
"""

from __future__ import annotations

import base64
import dataclasses
import ipaddress
import logging
import multiprocessing
import os
import time
from argparse import BooleanOptionalAction
from typing import Any
from urllib.parse import quote

import cloudpickle
import requests

from vime.backends.vllm_utils.arguments import SKIPPED_DESTS, get_vllm_cli_action_table
from vime.ray.ray_actor import RayActor
from vime.utils.http_utils import get_host_info
from vime.utils.common import is_npu

logger = logging.getLogger(__name__)

_spawn_ctx = multiprocessing.get_context("spawn")

# Fields checked against external ``GET /server_info``.
EXTERNAL_ENGINE_CHECK_FIELDS = ("tp_size", "pp_size", "dp_size", "nnodes")

_REDACTED_FLAGS = frozenset({"--hf-token"})

# vLLM sleep/wake only supports these tags (``cuda_graph`` is not supported).
_VLLM_WAKE_TAGS = frozenset({"weights", "kv_cache"})

_PRIMITIVE_TYPES = (str, int, float, bool)


def _format_v6_uri(addr: str | None) -> str | None:
    if not addr or addr.startswith("["):
        return addr
    try:
        if ipaddress.ip_address(addr).version == 6:
            return f"[{addr}]"
    except ValueError:
        pass
    return addr


def _response_json(response: requests.Response) -> dict:
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        e.add_note(f"{response.text=}")
        raise
    # vLLM sleep/wake endpoints may return 200 with an empty body.
    if not response.content or not response.content.strip():
        return {"ok": True}
    return response.json()


def get_base_gpu_id(args, rank):
    """First local GPU index on this node for rollout engine *rank* (colocate vs actor[/critic]-offset layout)."""
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
        if args.use_critic:
            num_critic_gpus = args.critic_num_gpus_per_node * args.critic_num_nodes
            start_index = (num_actor_gpus + num_critic_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    if is_npu():
        cvd = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    else:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


@dataclasses.dataclass(frozen=True)
class VllmEngineTopology:
    """Per-Ray-actor placement for one slice of a logical rollout engine."""

    nnodes: int
    node_rank: int
    local_num_gpus: int
    tensor_parallel_size: int
    pipeline_parallel_size: int

    @property
    def headless(self) -> bool:
        return self.node_rank != 0

    @property
    def multi_node(self) -> bool:
        return self.nnodes > 1


def _get_vllm_pp_size(args) -> int:
    return int(getattr(args, "vllm_pipeline_parallel_size", 1) or 1)


def _get_vllm_dp_size(args) -> int:
    return int(getattr(args, "vllm_dp_size", None) or getattr(args, "vllm_data_parallel_size", 1) or 1)


def _resolve_vllm_parallel_sizes(args, *, gpus_per_engine: int) -> tuple[int, int]:
    # Derive TP per-engine from THIS engine's GPU count (matches upstream slime's
    # sglang_engine: tp = _gpus_per_engine // pp). Deliberately does NOT consult a global
    # ``args.vllm_tp_size``: validate_args used to set that from the *global*
    # rollout_num_gpus_per_engine, which shadowed this per-engine value and made a
    # heterogeneous per-group engine (e.g. a tp=2 group) launch with the global TP —
    # desyncing the weight-transfer rendezvous (the 300s "3/4 clients joined" hang).
    pp = _get_vllm_pp_size(args)
    dp = _get_vllm_dp_size(args)
    # [DP #4 P1 / external-LB] each engine slot IS one DP rank (occupies tp*pp cards); the
    #   DP width lives across the N slots, not inside one slot, so tp must NOT divide by dp
    #   here — dp is emitted as --data-parallel-size on each slot (§19). This restores
    #   slime's sglang_engine ``tp = gpus_per_engine // pp``. Non-external-LB (single engine
    #   / ③ replica dp_size=1) keeps ``// (pp*dp)`` and is byte-identical to before.
    if getattr(args, "vllm_data_parallel_external_lb", False):
        divisor, divisor_desc = pp, f"vllm_pipeline_parallel_size ({pp})"
    else:
        divisor = pp * dp
        divisor_desc = f"vllm_pipeline_parallel_size * vllm_data_parallel_size ({pp} * {dp} = {pp * dp})"
    if gpus_per_engine % divisor != 0:
        raise ValueError(
            f"num_gpus_per_engine ({gpus_per_engine}) must be divisible by {divisor_desc}"
        )
    tp = gpus_per_engine // divisor
    return tp, pp


def compute_vllm_engine_topology(
    args,
    global_rank: int,
    *,
    num_gpus_per_engine: int | None = None,
) -> VllmEngineTopology:
    """Compute nnodes / node_rank / local GPU slice for one Ray actor."""
    gpus_per_engine = num_gpus_per_engine if num_gpus_per_engine is not None else args.rollout_num_gpus_per_engine
    nnodes = max(1, gpus_per_engine // args.num_gpus_per_node)
    node_rank = global_rank % nnodes
    if nnodes == 1:
        local_num_gpus = min(args.num_gpus_per_node, gpus_per_engine)
    else:
        if gpus_per_engine % nnodes != 0:
            raise ValueError(
                f"rollout_num_gpus_per_engine ({gpus_per_engine}) must be divisible by the number of "
                f"nodes per engine ({nnodes})"
            )
        local_num_gpus = gpus_per_engine // nnodes
    tp, pp = _resolve_vllm_parallel_sizes(args, gpus_per_engine=gpus_per_engine)
    return VllmEngineTopology(
        nnodes=nnodes,
        node_rank=node_rank,
        local_num_gpus=local_num_gpus,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
    )


def parse_dist_init_addr(dist_init_addr: str) -> tuple[str, int]:
    """Split ``host:port`` (IPv6-safe) into master host and port."""
    ip_part, port_part = dist_init_addr.rsplit(":", 1)
    host = _format_v6_uri(ip_part) or ip_part
    return host.strip("[]"), int(port_part)


def append_vllm_distributed_launch_flags(
    cmd: list[str],
    topology: VllmEngineTopology,
    master: tuple[str, int],
    args,
) -> None:
    """Append vLLM multi-node flags when ``topology.multi_node`` (no-op for single-node)."""
    if not topology.multi_node:
        return
    master_host, master_port = master
    cmd += [
        "--nnodes",
        str(topology.nnodes),
        "--node-rank",
        str(topology.node_rank),
        "--master-addr",
        master_host,
        "--master-port",
        str(master_port),
    ]
    if not _user_overrode(args, "vllm_data_parallel_backend"):
        cmd += ["--data-parallel-backend", "mp"]
    if not _user_overrode(args, "vllm_distributed_executor_backend"):
        cmd += ["--distributed-executor-backend", "mp"]
    if topology.headless:
        cmd.append("--headless")


def _user_overrode(args, dest: str) -> bool:
    user_provided: set[str] = getattr(args, "_vllm_user_provided", set())
    if dest in user_provided:
        return True
    entry = get_vllm_cli_action_table().get(dest)
    if entry is None:
        return False
    _, action = entry
    return getattr(args, dest, action.default) != action.default


def _apply_vllm_overrides(args, server_args: dict[str, Any], vllm_overrides: dict | None, rank: int) -> None:
    """Merge per-group ``overrides`` from rollout YAML into ``args`` / ``server_args``."""
    if not vllm_overrides:
        return
    for key, value in vllm_overrides.items():
        normalized = key.replace("-", "_")
        if normalized == "model_path":
            server_args["model_path"] = value
            continue
        if normalized.startswith("disaggregation"):
            logger.debug("vllm_overrides: skipping unsupported key %s (rank=%s)", key, rank)
            continue
        dest = normalized if normalized.startswith("vllm_") else f"vllm_{normalized}"
        if hasattr(args, dest):
            logger.info("vllm_overrides: %s=%r (rank=%s)", dest, value, rank)
            setattr(args, dest, value)
            continue
        if normalized in server_args:
            server_args[normalized] = value
            continue
        logger.debug("vllm_overrides: unrecognized key %s (rank=%s)", key, rank)


class _RobustJsonEncoder:
    @staticmethod
    def default(obj):
        import enum
        from pathlib import Path

        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, enum.Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def serialize_for_cli(value) -> str | None:
    import json

    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if dataclasses.is_dataclass(value) or isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, default=_RobustJsonEncoder.default)
        except (TypeError, ValueError) as exc:
            logger.debug("JSON serialization failed for %r: %s", type(value).__name__, exc)
            return None
    return None


def _serialize_weight_transfer_config(value) -> str:
    serialized = serialize_for_cli(value)
    if serialized is None:
        import json

        return json.dumps({"backend": str(value)})
    return serialized


def _resolve_mooncake_connector_name() -> str:
    """Pick the newest Mooncake connector name supported by the installed vLLM."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

        if "MooncakeConnectorV1" in KVConnectorFactory._registry:
            return "MooncakeConnectorV1"
        if "MooncakeConnector" in KVConnectorFactory._registry:
            return "MooncakeConnector"
    except Exception as exc:
        logger.debug("Failed to inspect KV connector registry for Mooncake alias: %s", exc)
    return "MooncakeConnectorV1"


def _build_mooncake_kv_transfer_config(server_args: dict[str, Any]) -> dict[str, Any]:
    worker_type = server_args["worker_type"]
    tp_size = server_args["tp_size"]
    bootstrap_port = server_args["disaggregation_bootstrap_port"]
    assert worker_type in ("prefill", "decode"), f"Unsupported worker_type for Mooncake: {worker_type}"
    assert bootstrap_port is not None, "Mooncake PD workers require disaggregation_bootstrap_port"
    role = "kv_producer" if worker_type == "prefill" else "kv_consumer"
    kv_rank = 0 if worker_type == "prefill" else 1

    topology = server_args.get("pd_topology") or {}
    prefill_topology = topology.get("prefill") or {}
    decode_topology = topology.get("decode") or {}
    default_dp = int(server_args.get("dp_size") or 1)
    extra_config = {
        "prefill": {
            "dp_size": int(prefill_topology.get("dp_size") or (default_dp if worker_type == "prefill" else 1)),
            "tp_size": int(prefill_topology.get("tp_size") or (tp_size if worker_type == "prefill" else 1)),
        },
        "decode": {
            "dp_size": int(decode_topology.get("dp_size") or (default_dp if worker_type == "decode" else 1)),
            "tp_size": int(decode_topology.get("tp_size") or (tp_size if worker_type == "decode" else 1)),
        },
    }
    if is_npu():
        extra_config["use_ascend_direct"] = True
        # [2026-08-14] ADXL 建链必需的本地 RoCE 端点资源,缺了 PD 一定跑不起来。
        #   vllm-ascend 的 setup_ascend_local_comm_res(utils.py:568)只认这里的
        #   ascend_local_comm_res_path,取不到就**静默 return**(:583),于是
        #   ASCEND_LOCAL_COMM_RES 永不设置 → Mooncake 初始化 ADXL 时 getenv 拿到 NULL、
        #   不传 adxl.LocalCommRes → Connect() 返回 adxl::PARAM_INVALID(103900,~1ms 即返回,
        #   不是超时)→ Mooncake transfer ret=-1 → decode EngineCore 死 →
        #   proxy preflight 永远 ready 不了 → rollout 全量 drop。全链路无一句告警。
        #   目录里按**物理卡号**放 ub_endpoint_npu_<dev>.json(样例见 /workspace/pd_16/local_comm_res)。
        local_comm_res_path = os.environ.get("ASCEND_LOCAL_COMM_RES_PATH")
        if local_comm_res_path:
            extra_config["ascend_local_comm_res_path"] = local_comm_res_path
        elif os.environ.get("HCCL_INTRA_ROCE_ENABLE") == "1":
            # 只有强制 RoCE 时才缺资源;默认 HCCS 路径不需要,别误报。
            logger.warning(
                "HCCL_INTRA_ROCE_ENABLE=1 强制节点内走 RoCE,但 ASCEND_LOCAL_COMM_RES_PATH 未设置 → "
                "ADXL 拿不到本地 RoCE 端点资源,PD KV 传输会以 Connect status=103900(PARAM_INVALID)/"
                "transfer ret=-1 失败。要么置 0 回落 HCCS(节点内 P/D 推荐,更快),"
                "要么提供 ub_endpoint_npu_<物理卡号>.json 目录。"
            )
    return {
        "kv_connector": _resolve_mooncake_connector_name(),
        "kv_buffer_device": "npu" if is_npu() else "cuda",
        "kv_role": role,
        "kv_parallel_size": 1,
        "kv_port": bootstrap_port,
        "kv_rank": kv_rank,
        "kv_connector_extra_config": extra_config,
    }


def _forward_vllm_cli_args(args, cmd: list[str]) -> None:
    """Append user ``--vllm-*`` overrides not already set by the orchestrator."""
    fixed = {flag for flag in cmd if isinstance(flag, str) and flag.startswith("--")}
    raw_values: dict[str, str] = getattr(args, "_vllm_raw_values", {})

    for vime_dest, (vllm_flag, action) in get_vllm_cli_action_table().items():
        if vime_dest in SKIPPED_DESTS:
            continue
        if vllm_flag in fixed:
            continue
        if not hasattr(args, vime_dest):
            continue
        value = getattr(args, vime_dest)
        default = action.default
        if value == default or value is None:
            continue

        if isinstance(action, BooleanOptionalAction):
            cmd.append(vllm_flag if value else f"--no-{vllm_flag[2:]}")
            continue
        if action.nargs == 0:
            cmd.append(vllm_flag)
            continue
        if action.nargs == "?" and action.const is not None and value == action.const:
            cmd.append(vllm_flag)
            continue
        if action.nargs in ("+", "*") or (action.nargs not in (None, "?") and isinstance(value, (list, tuple))):
            if not isinstance(value, (list, tuple)):
                value = [value]
            if not value:
                continue
            if not all(isinstance(v, _PRIMITIVE_TYPES) for v in value):
                logger.debug("Skipping %s: list contains non-primitive items (%r)", vllm_flag, value)
                continue
            cmd.append(vllm_flag)
            cmd.extend(str(v) for v in value)
            continue
        if not isinstance(value, _PRIMITIVE_TYPES):
            raw = raw_values.get(vime_dest)
            if raw is not None:
                cmd.extend([vllm_flag, raw])
                continue
        serialized = serialize_for_cli(value)
        if serialized is None:
            logger.debug(
                "Skipping forward of %s: parsed value %r (%s) cannot be serialized",
                vllm_flag,
                value,
                type(value).__name__,
            )
            continue
        cmd.extend([vllm_flag, serialized])


def redact_cmd_for_log(cmd: list[str]) -> str:
    """Stringify ``cmd`` for logging, redacting credential flags."""
    parts: list[str] = []
    redact_next = False
    for token in cmd:
        if redact_next:
            parts.append("***")
            redact_next = False
            continue
        parts.append(token)
        if isinstance(token, str) and token in _REDACTED_FLAGS:
            redact_next = True
    return " ".join(parts)


def build_vllm_subprocess_env(server_args: dict[str, Any]) -> dict[str, str]:
    """Child-process environment for ``vllm serve``."""
    args = server_args["args"]
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    # [NPU] Drop a parent-inherited PYTORCH_NPU_ALLOC_CONF (the Docker image sets
    # expandable_segments:True for training). vllm-ascend manages this itself and
    # MUST NOT have expandable_segments under sleep mode — its CaMemAllocator
    # memory pool asserts against it (camem.py). Letting vllm-ascend's own
    # sleep-mode-aware logic (platform.py) re-add it only when sleep mode is off.
    # [vime 2026-07-15 EXPERIMENTAL] 默认仍 pop(安全)。VIME_VLLM_KEEP_EXPANDABLE=1 时保留,
    #   让 vLLM 子进程也带 expandable(须配合 camem.py 已放开 assert;sleep/wake 可能出错,仅实验)。
    if os.environ.get("VIME_VLLM_KEEP_EXPANDABLE", "0") != "1":
        env.pop("PYTORCH_NPU_ALLOC_CONF", None)
    env.pop("TORCHDYNAMO_DISABLE", None)
    # torch.compile; the vLLM subprocess MUST have torch.compile enabled for
    # cudagraph capture.
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    if is_npu():
        env["ASCEND_RT_VISIBLE_DEVICES"] = server_args["visible_devices"]
    else:
        env["CUDA_VISIBLE_DEVICES"] = server_args["visible_devices"]
    env.setdefault("VLLM_SERVER_DEV_MODE", "1")
    if getattr(args, "vllm_enable_deterministic_inference", False):
        env["VLLM_BATCH_INVARIANT"] = "1"
    if getattr(args, "colocate", False):
        import vime

        vime_root = os.path.dirname(os.path.dirname(os.path.abspath(vime.__file__)))
        existing_pp = env.get("PYTHONPATH", "")
        if vime_root not in {p for p in existing_pp.split(os.pathsep) if p}:
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [vime_root, existing_pp]))
        env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    return env


def build_vllm_cmd_and_env(server_args: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Translate ``server_args`` to ``vllm serve`` argv and subprocess environment."""
    args = server_args["args"]
    topology: VllmEngineTopology = server_args["topology"]
    env = build_vllm_subprocess_env(server_args)
    host_for_subprocess = (server_args["host"] or "127.0.0.1").strip("[]")

    cmd = [
        "vllm",
        "serve",
        str(server_args["model_path"]),
        "--tensor-parallel-size",
        str(server_args["tp_size"]),
        "--port",
        str(server_args["port"]),
        "--host",
        host_for_subprocess,
        "--seed",
        str(server_args["seed"]),
        "--trust-remote-code",
    ]

    if server_args["pp_size"] > 1:
        cmd += ["--pipeline-parallel-size", str(server_args["pp_size"])]

    if topology.multi_node:
        if server_args["master_addr"] is None or server_args["master_port"] is None:
            raise ValueError("master_addr/master_port required for multi-node vLLM engine")
        append_vllm_distributed_launch_flags(
            cmd,
            topology,
            (server_args["master_addr"], server_args["master_port"]),
            args,
        )

    # [DP #4 P1 / external-LB] Inject the per-rank DP flags. --data-parallel-size and
    #   --data-parallel-external-lb are auto-forwarded by _forward_vllm_cli_args (user set them
    #   via --vllm-*), so only rank/address/rpc-port — which are per-slot and runtime-allocated
    #   (§19 #3) — are added here. executor-backend MUST stay mp: external_launcher would make
    #   vLLM multiply world_size by dp and break the weight-sync rank math (§18.1). OFF path
    #   (data_parallel_rank None) adds nothing → byte-identical.
    if getattr(args, "vllm_data_parallel_external_lb", False) and server_args.get("data_parallel_rank") is not None:
        cmd += ["--data-parallel-rank", str(server_args["data_parallel_rank"])]
        if server_args.get("data_parallel_address") is not None:
            cmd += ["--data-parallel-address", str(server_args["data_parallel_address"])]
        if server_args.get("data_parallel_rpc_port") is not None:
            cmd += ["--data-parallel-rpc-port", str(server_args["data_parallel_rpc_port"])]
        if not _user_overrode(args, "vllm_distributed_executor_backend") and "--distributed-executor-backend" not in cmd:
            cmd += ["--distributed-executor-backend", "mp"]

    if getattr(args, "fp16", False):
        cmd += ["--dtype", "float16"]

    if getattr(args, "offload_rollout", False) and not getattr(args, "vllm_enable_sleep_mode", False):
        cmd += ["--enable-sleep-mode"]
        args.vllm_enable_sleep_mode = True

    if (
        getattr(args, "rollout_max_context_len", None) is not None
        and getattr(args, "vllm_max_model_len", None) is None
    ):
        cmd += ["--max-model-len", str(args.rollout_max_context_len)]

    if getattr(args, "use_rollout_routing_replay", False):
        cmd += ["--enable-return-routed-experts"]

    # gpu_memory_utilization: no vime-forced default. In colocate, training and rollout do not
    # occupy the GPU simultaneously (sleep/offload cycles), so vLLM's own default is fine. A user
    # value passed via --vllm-gpu-memory-utilization is auto-forwarded by _forward_vllm_cli_args.

    # 2) logprobs_mode: vllm's raw_logprobs are pre-temperature, while Megatron
    #    replay compares against rollout-temperature-scaled logprobs.
    if not _user_overrode(args, "vllm_logprobs_mode"):
        cmd += ["--logprobs-mode", "processed_logprobs"]

    # 3) weight_transfer_config: vllm default None disables /init_weight_transfer_engine,
    #    so vime's weight sync would fail.
    #    - Colocated mode: use IPC backend. UpdateWeightFromTensor calls
    #      IPCWeightTransferEngine.trainer_send_weights and passes an empty init_info
    #      dict, which is the correct signature for the IPC backend.
    #    - Non-colocated mode: use NCCL backend. Weight sync goes through
    #      update_weights_from_distributed; the vLLM engine still needs
    #      init_weight_transfer_engine to succeed (with NCCL the caller must supply
    #      master_address, master_port, rank_offset, and world_size separately).
    #    - 混合(share)布局:按节点分叉 —— 与 actor 同节点的共卡引擎走 IPC,
    #      其余(专用/远程)引擎走 NCCL/HCCL。共卡引擎不进 HCCL 域,规避
    #      "同域两 rank 同卡" 的 RanktableDetect 拒绝。
    #    Users who pass ``--vllm-weight-transfer-config`` explicitly are honored.
    _colocated_engine = getattr(args, "colocate", False)
    _spec = getattr(args, "resource_layout_spec", None)
    if not _colocated_engine and _spec is not None and getattr(_spec, "rollout_has_share", False):
        _actor_nodes = {item.node for item in _spec.actor}
        _colocated_engine = host_for_subprocess in _actor_nodes
    if _user_overrode(args, "vllm_weight_transfer_config"):
        cmd += [
            "--weight-transfer-config",
            _serialize_weight_transfer_config(args.vllm_weight_transfer_config),
        ]
    elif _colocated_engine:
        # npu_ipc 而非 ipc:vllm-ascend 注册的 NPU 原生 IPC 引擎(torch_npu 的
        # _share_npu_/rebuild_npu_tensor);vllm core 的 "ipc" 是 CUDA 语义。
        # vime 的 update_weights_chunk 实际绕过 transfer engine,这里主要是构造期
        # 与生命周期空操作不出错。
        cmd += ["--weight-transfer-config", '{"backend":"npu_ipc"}']
    else:
        cmd += ["--weight-transfer-config", '{"backend":"nccl"}']

    worker_type = server_args.get("worker_type", "regular")
    disagg_backend = getattr(args, "disaggregation_backend", "nixl")

    if _colocated_engine and "--worker-extension-cls" not in cmd:
        cmd += [
            "--worker-extension-cls",
            "vime.backends.megatron_utils.update_weight.update_weight_from_tensor.vLLMColocateWorkerExtension",
        ]

    if worker_type in ("prefill", "decode") and topology.node_rank == 0:
        if disagg_backend == "mooncake":
            env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(server_args["disaggregation_bootstrap_port"])
            cmd += ["--kv-transfer-config", serialize_for_cli(_build_mooncake_kv_transfer_config(server_args))]
        else:
            env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = host_for_subprocess
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(server_args["disaggregation_bootstrap_port"])

    _tp = os.environ.get("VLLM_TOOL_CALL_PARSER")
    if _tp and "--tool-call-parser" not in cmd:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", _tp]
    _rp = os.environ.get("VLLM_REASONING_PARSER")
    if _rp and "--reasoning-parser" not in cmd:
        cmd += ["--reasoning-parser", _rp]
    _forward_vllm_cli_args(args, cmd)
    logger.info("Launching vLLM server: %s", redact_cmd_for_log(cmd))
    return cmd, env


def _exec_vllm_cmd(cmd: list[str], env: dict[str, str]) -> None:
    """Entry point for multiprocessing child process."""
    os.execvpe(cmd[0], cmd, env)


def _normalize_vllm_wake_tags(tags: list[str] | None) -> list[str] | None:
    if not tags:
        return tags
    normalized = [t for t in tags if t in _VLLM_WAKE_TAGS]
    dropped = set(tags) - set(normalized)
    if dropped:
        logger.debug("vLLM wake_up: dropped tags not supported by vLLM: %s", sorted(dropped))
    return normalized or None


def launch_server_process(server_args: dict) -> multiprocessing.Process:
    """Spawn ``vllm serve`` from a :func:`_compute_server_args` dict."""
    cmd, env = build_vllm_cmd_and_env(server_args)
    p = _spawn_ctx.Process(target=_exec_vllm_cmd, args=(cmd, env))
    p.start()
    return p


def _wait_worker_process_alive(process: multiprocessing.Process, timeout_s: float = 300.0) -> None:
    """Non-head nodes have no HTTP health endpoint; ensure the subprocess stays up."""
    start = time.time()
    while process.is_alive():
        if time.time() - start > timeout_s:
            return
        time.sleep(2)
    raise RuntimeError(f"vLLM worker process exited unexpectedly with code {process.exitcode}")


def _wait_server_healthy(base_url: str, process: multiprocessing.Process | None) -> None:
    """Wait until the vLLM server responds on ``GET /health`` (no time limit, SGLang-style).

    Loops until /health returns 200, or — for a managed subprocess — until it dies (fail fast via
    ``process.is_alive()``). There is no overall deadline, so a slow-but-healthy startup (a large
    MoE / DP engine loading + compiling + capturing CUDA graphs across replicas) is never
    spuriously timed out. The per-probe ``timeout=3`` bounds each individual request so a single
    stuck socket cannot wedge the loop. In external mode (``process is None``) there is no liveness
    signal, so a permanently unreachable URL loops indefinitely by design (the external engine is
    caller-managed). Mirrors slime's SGLang backend _wait_server_healthy.
    """
    while True:
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

        if process is not None and not process.is_alive():
            raise RuntimeError(f"vLLM server exited unexpectedly with code {process.exitcode}")
        time.sleep(2)


class VLLMEngine(RayActor):
    """Ray actor for vLLM OpenAI HTTP rollout (connect or spawn local ``vllm serve``)."""

    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        model_path: str | None = None,
        vllm_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.model_path = model_path or args.hf_checkpoint
        self.vllm_overrides = vllm_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine
        self.process: multiprocessing.Process | None = None
        self._weight_version: str | None = None
        self.node_rank = 0
        self._topology: VllmEngineTopology | None = None
        self._server_args: dict | None = None

    def _http_base(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
        data_parallel_size=None,
        data_parallel_rank=None,
        data_parallel_address=None,
        data_parallel_rpc_port=None,
    ):
        # ``nccl_port`` is allocated by rollout but unused by vLLM (rendezvous uses ``dist_init_addr``).
        del nccl_port

        gpus_per_engine = self.num_gpus_per_engine or self.args.rollout_num_gpus_per_engine
        host = host or get_host_info()[1]

        self._server_args = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            host,
            port,
            worker_type=self.worker_type,
            base_gpu_id=self.base_gpu_id,
            vllm_overrides=self.vllm_overrides,
            num_gpus_per_engine=gpus_per_engine,
            disaggregation_bootstrap_port=disaggregation_bootstrap_port,
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_address=data_parallel_address,
            data_parallel_rpc_port=data_parallel_rpc_port,
        )
        self._topology = self._server_args["topology"]
        self.node_rank = self._topology.node_rank

        # rollout always passes the resolved router (engine.init(router_ip=self.router_ip, ...))
        # and _start_router always returns a real address, so no fallback to args is needed.
        self.router_ip = router_ip
        self.router_port = router_port
        self.server_host = self._server_args["host"]
        self.server_port = port
        self.disaggregation_bootstrap_port = disaggregation_bootstrap_port

        if self.worker_type != "regular":
            logger.info(
                "vLLMEngine: worker_type=%s affects PD/disaggregation launch wiring.",
                self.worker_type,
            )

        if self.args.rollout_external:
            # Only the HTTP-owning head node (node_rank 0) can hit /health and
            # /server_info. Headless workers (node_rank>0) expose no HTTP endpoint,
            # so skip the check for them — mirrors the head/worker split in _init_normal.
            if self.node_rank == 0:
                self._init_external()
            else:
                logger.info(
                    "External vLLM headless worker (rank=%s node_rank=%s): skip HTTP health/config "
                    "check (only node_rank 0 owns HTTP).",
                    self.rank,
                    self.node_rank,
                )
        else:
            self._init_normal()

        if self.node_rank == 0 and self.router_ip and self.router_port:
            self._register_worker_with_router()

    def _register_worker_with_router(self) -> None:
        worker_url = self._http_base()
        payload = {"url": worker_url, "worker_type": self.worker_type}
        response = requests.post(
            f"http://{self.router_ip}:{self.router_port}/workers",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

    def _deregister_worker_from_router(self) -> None:
        if self.node_rank != 0 or not self.router_ip or not self.router_port:
            return
        worker_url = self._http_base()
        try:
            all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers", timeout=30).json()[
                "workers"
            ]
            for worker in all_workers:
                if worker["url"] == worker_url:
                    response = requests.delete(
                        f"http://{self.router_ip}:{self.router_port}/workers/{quote(worker_url, safe='')}",
                        timeout=30,
                    )
                    response.raise_for_status()
                    return
            logger.warning("Worker %s not found in vllm-router during shutdown.", worker_url)
        except Exception as e:
            logger.warning("Failed to list/remove worker on vllm-router: %s", e)

    def _init_external(self) -> None:
        logger.info("Use external vLLM engine (rank=%s) at %s:%s", self.rank, self.server_host, self.server_port)
        _wait_server_healthy(self._http_base(), process=None)
        self._sanity_check_external_server_args()

    def _sanity_check_external_server_args(self) -> None:
        """Strictly verify an external engine's parallel config matches what we expect; raise on mismatch.

        Replaces the previous warn-only check, which (a) compared against the *global*
        ``rollout_num_gpus_per_engine`` — wrong for heterogeneous / multi-node groups — and
        (b) only logged a warning, so a misconfigured external engine sailed through and then
        hung the weight-sync rendezvous ~300s later with no clear error.

        We now compare every field in ``EXTERNAL_ENGINE_CHECK_FIELDS`` against the per-engine
        expectation in ``self._server_args`` and raise immediately on mismatch. A field that the
        engine's ``/server_info`` does not report (``actual is None``) is skipped rather than
        treated as a mismatch (e.g. vLLM ``parallel_config`` may not surface ``nnodes``), so the
        check stays strict for reported fields without false-failing on unreported ones.
        """
        response = requests.get(f"{self._http_base()}/server_info", params={"config_format": "json"}, timeout=30)
        body = _response_json(response)
        parallel_cfg = body.get("vllm_config", {}).get("parallel_config", {})
        if not parallel_cfg:
            raise RuntimeError(f"External vLLM /server_info missing vllm_config.parallel_config: {body}")
        actual = {
            "tp_size": parallel_cfg.get("tensor_parallel_size"),
            "pp_size": parallel_cfg.get("pipeline_parallel_size"),
            "dp_size": parallel_cfg.get("data_parallel_size"),
            "nnodes": parallel_cfg.get("nnodes"),
        }
        expect = {name: self._server_args.get(name) for name in EXTERNAL_ENGINE_CHECK_FIELDS}
        for name in EXTERNAL_ENGINE_CHECK_FIELDS:
            actual_value = actual.get(name)
            if actual_value is None:
                logger.debug("External vLLM /server_info did not report %s; skipping that check.", name)
                continue
            if actual_value != expect.get(name):
                raise AssertionError(
                    f"External vLLM server arg mismatch: {name}: expect={expect.get(name)} "
                    f"actual={actual_value} (full expect={expect} actual={actual})"
                )

    def _init_normal(self) -> None:
        topology = self._topology
        assert topology is not None and self._server_args is not None
        logger.info(
            "Launch vLLM OpenAI api_server at: %s:%s (rank=%s node_rank=%s/%s)",
            self.server_host,
            self.server_port,
            self.rank,
            topology.node_rank,
            topology.nnodes,
        )
        self.process = launch_server_process(self._server_args)
        if topology.node_rank == 0:
            _wait_server_healthy(self._http_base(), process=self.process)
        else:
            _wait_worker_process_alive(self.process)

    def _make_request(self, endpoint: str, payload: dict | None = None) -> dict | None:
        """Control-plane POST returning parsed JSON (mirrors SGLang's ``_make_request``).

        The single choke point for control-plane POSTs: headless workers (node_rank>0) own no
        HTTP server, so they no-op to None; otherwise POST and parse via the shared
        ``_response_json`` (also reused by the query-param endpoints /sleep, /wake_up, ...).
        """
        if self.node_rank != 0:
            return None
        url = f"{self._http_base()}/{endpoint.lstrip('/')}"
        return _response_json(requests.post(url, json=payload or {}))

    def _post_vllm_update_weights_http(self, update_info: dict) -> dict:
        """POST ``/update_weights`` with ``{"update_info": ...}`` (vLLM RLHF control plane).

        Caller must invoke ``start_weight_update`` / ``finish_weight_update`` around a batch of
        ``/update_weights`` calls (see ``UpdateWeightFromTensor`` / ``UpdateWeightFromDistributed``).
        """
        return self._make_request(
            "update_weights",
            {"update_info": update_info},
        )

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Return True if ``GET /health`` succeeds."""
        if self.node_rank != 0:
            return True
        response = requests.get(f"{self._http_base()}/health", timeout=timeout)
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        *,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ipc_handles: list[dict] | None = None,
        weight_version: str | None = None,
        flush_cache: bool = False,
    ) -> dict | None:
        """POST ``IPCWeightTransferUpdateInfo`` (names / dtype_names / shapes /
        ipc_handles) to ``/update_weights``; record ``weight_version`` only on
        success. ``ipc_handles`` are base64-cloudpickle'd (rebuild_fn closures).
        """
        if self.node_rank != 0:
            return None

        payload: dict = {"names": names, "dtype_names": dtype_names, "shapes": shapes}
        if ipc_handles is not None:
            payload["ipc_handles_pickled"] = base64.b64encode(cloudpickle.dumps(ipc_handles)).decode("utf-8")
        if flush_cache:
            self.flush_cache()

        response = self._make_request(
            "collective_rpc",
            {"method": "update_weights_chunk", "kwargs": {"update_info": payload}},
        )
        if weight_version is not None:
            self._weight_version = str(weight_version)
        return response
    
    def update_weights_chunk(self, update_info: dict) -> dict:
        """POST ``/update_weights_chunk`` with a single named-tensor chunk.

        Mirrors the SkyRL ``RemoteInferenceClient.update_weights_chunk`` API.
        Must be called between :meth:`start_weight_update` and
        :meth:`finish_weight_update`.

        Unlike :meth:`update_weights`, ``update_info`` is the *inner* payload
        dict (``names``, ``dtype_names``, ``shapes``, and one of
        ``ipc_handles`` / ``ipc_handles_pickled`` for IPC, or ``packed`` for
        NCCL) — **not** wrapped in ``{"update_info": ...}``.

        If ``ipc_handles`` are present (raw CUDA callables produced by
        ``reduce_tensor``), they are serialised with cloudpickle + base64 so
        vLLM can deserialise them when
        ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` is set.
        """
        if self.node_rank != 0:
            return {"ok": True, "skipped": True}

        import base64

        import cloudpickle

        payload = dict(update_info)
        if payload.get("ipc_handles") is not None:
            payload["ipc_handles_pickled"] = base64.b64encode(cloudpickle.dumps(payload.pop("ipc_handles"))).decode(
                "utf-8"
            )
        response = self._make_request(
            "collective_rpc",
            {"method": "update_weights_chunk", "kwargs": {"update_info": payload}},
        )
        return response

    def flush_cache(self):
        """Clear prefix cache via ``POST /reset_prefix_cache``."""
        if self.node_rank != 0:
            return
        params = {"reset_running_requests": False}
        requests.post(f"{self._http_base()}/reset_prefix_cache", params=params).raise_for_status()

    def get_url(self):
        """Worker HTTP base URL, or ``None`` when ``node_rank != 0``."""
        if self.node_rank != 0:
            return None
        return self._http_base()

    def get_pd_endpoint(self):
        """Return ``(url, bootstrap_port)`` for PD workers, or ``None`` on headless ranks."""
        if self.node_rank != 0:
            return None
        return self._http_base(), self.disaggregation_bootstrap_port

    def shutdown(self):
        logger.info("Shutdown vLLM engine %s:%s...", self.server_host, self.server_port)
        self._deregister_worker_from_router()
        if self.args.rollout_external:
            return

        if self.process is None or not self.process.is_alive():
            return
        pid = self.process.pid
        try:
            from vllm.utils.system_utils import kill_process_tree

            kill_process_tree(pid)
        except Exception as e:
            logger.warning("vLLM kill_process_tree failed (%s); terminate root only.", e)
            if self.process.is_alive():
                self.process.terminate()
                try:
                    self.process.join(timeout=15)
                except Exception:
                    pass
                if self.process.is_alive():
                    self.process.kill()
        try:
            self.process.join(timeout=30)
        except Exception:
            pass
        self.process = None

    def get_weight_version(self) -> str | None:
        """Return the version recorded by the last successful weight transfer.

        Raises ``RuntimeError`` if no weight transfer has recorded a version
        yet — we don't fall back to a ``/v1/models`` lookup, which would
        return the model path string and never match the trainer's integer
        counter (i.e. produce a misleading "mismatch" downstream).
        Worker ranks (``node_rank != 0``) short-circuit per the class idiom.
        """
        if self.node_rank != 0:
            return None
        if self._weight_version is None:
            raise RuntimeError(
                "VLLMEngine.get_weight_version called before any successful "
                "weight transfer recorded a version (update_weights_from_tensor "
                "/ update_weights_from_distributed never reached their "
                "post-POST version write)."
            )
        return self._weight_version

    def release_memory_occupation(self, level: int = 1):
        """Flush prefix cache, then ``POST /sleep?level={level}``.

        level=1 权重 park 到 host pinned(70G/引擎驻留);level=2 直接丢弃(仅
        重映射空壳)。训推分时(共卡)场景用 level=2:权重随后必由 reload 会话
        全量覆写,驻留纯属浪费(slime 09450c1b 语义)。历史上 level=2 的两个
        障碍已消除:① wake 后 param 丢 weight_loader —— 由 _VLLMHijack 在
        start_weight_update 重补丁;② wake 转置与 layerwise copy-back 冲突 ——
        已按 LAYERWISE_INFO 门控跳过(vllm-ascend-023 560b39d6)。
        """
        if self.node_rank != 0:
            return None
        self.flush_cache()
        response = requests.post(
            f"{self._http_base()}/sleep",
            params={"level": level},
        )
        return _response_json(response)

    def resume_memory_occupation(self, tags: list[str] | None = None):
        """``POST /wake_up`` with vLLM-supported wake tags."""
        if self.node_rank != 0:
            return None
        tags = _normalize_vllm_wake_tags(tags)
        # vLLM ``POST /wake_up`` uses ``query_params.getlist("tags")``, not JSON.
        # Omit params when ``tags`` is empty so the server wakes all tags (see api_router.wake_up).
        wake_params: list[tuple[str, str]] | None = [("tags", t) for t in tags] if tags else None
        response = requests.post(
            f"{self._http_base()}/wake_up",
            params=wake_params,
            timeout=30,
        )
        return _response_json(response)

    def init_weight_transfer_engine(self, payload: dict) -> dict:
        """``POST /init_weight_transfer_engine`` with a caller-supplied payload (IPC path).

        For IPC mode the payload is ``{"init_info": {}}``; for NCCL use
        ``init_weights_update_group`` which constructs the payload from typed args.
        """
        last_error = None
        for attempt in range(1, 4):
            try:
                # NOTE: _make_request takes no timeout kwarg (the upstream `timeout=init_timeout_s`
                # was doubly broken: undefined name + unsupported kwarg). The 3-attempt retry below
                # provides resilience for the first-call IPC setup handshake.
                return self._make_request("init_weight_transfer_engine", payload)
            except Exception as e:
                last_error = e
                if attempt < 3:
                    logger.warning("init_weight_transfer_engine attempt %s/3 failed: %s", attempt, e)
                    time.sleep(2 * attempt)
        raise RuntimeError(f"vLLM init_weight_transfer_engine failed: {last_error}") from last_error

    def start_weight_update(self, is_checkpoint_format: bool = False) -> dict:
        """``POST /start_weight_update`` — signals vLLM to enter IPC weight-update mode."""
        # [WSYNC-DBG 2026-07-14] 定位 2-engine "already active":打印本引擎身份 + 实际要 POST 的 http_base。
        #   两条日志 http_base 相同但 rank 不同 → B 的 base 指到了 A 的端口(2-DP 端口分配 bug);
        #   rank/base 两次完全相同 → _begin 列表里有重复句柄。ray 前缀里的 (VLLMEngine pid=) 即哪个引擎。
        logger.info("[WSYNC-DBG] start_weight_update entry rank=%s node_rank=%s http_base=%s ckpt=%s",
                    getattr(self, "rank", "?"), getattr(self, "node_rank", "?"),
                    self._http_base(), is_checkpoint_format)
        return self._make_request("start_weight_update", {"is_checkpoint_format": is_checkpoint_format})

    def finish_weight_update(self) -> dict:
        """``POST /finish_weight_update`` — signals vLLM to exit IPC weight-update mode.

        Purely a state-machine bookend now; ``_weight_version`` is recorded by
        ``update_weights_from_tensor`` (the IPC data-carrying RPC), matching slime's
        single-RPC version-with-data semantics.
        """
        return self._make_request("finish_weight_update", {})

    def check_weights(self, action: str):
        """No vLLM ``weights_checker`` route; return a placeholder dict."""
        del action
        return {"ok": True, "supported": False, "note": "vLLM has no weights_checker endpoint."}

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        """Call ``POST /init_weight_transfer_engine`` with an ``init_info`` block.

        ``group_name`` / ``backend`` are accepted for a uniform caller signature but are not sent to vLLM.
        Always uses the vllm-native weight transfer engine; reload-on-continue fallback is no longer supported.
        """
        del group_name, backend
        payload = {
            "init_info": {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
            }
        }
        last_error = None
        for attempt in range(1, 4):
            try:
                return self._make_request("init_weight_transfer_engine", payload)
            except Exception as e:
                last_error = e
                if attempt < 3:
                    logger.warning("init_weight_transfer_engine attempt %s/3 failed: %s", attempt, e)
                    time.sleep(2 * attempt)
        raise RuntimeError(f"vLLM init_weight_transfer_engine failed: {last_error}") from last_error

    def destroy_weights_update_group(self, group_name):
        """No vLLM destroy call; return ``None``."""
        del group_name
        return None

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
        packed: bool = True,
    ):
        """NCCL path: ``POST /update_weights`` with packed tensor metadata.

        Payload matches vLLM NCCL weight transfer (see upstream rlhf_http_nccl example).
        """
        del group_name
        if weight_version is not None:
            self._weight_version = str(weight_version)
        if flush_cache:
            self.flush_cache()
        dtype_names = [str(d).replace("torch.", "") for d in dtypes]
        update_info = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": [list(s) for s in shapes],
            "packed": bool(packed),
        }
        return self._post_vllm_update_weights_http(update_info)

    def update_weights_from_disk(self, model_path: str, load_format: str | None = None):
        """``POST /collective_rpc`` with ``reload_weights`` and ``weights_path``."""
        if self.node_rank != 0:
            return
        del load_format
        response = requests.post(
            f"{self._http_base()}/collective_rpc",
            json={
                "method": "reload_weights",
                "kwargs": {"weights_path": model_path, "is_checkpoint_format": True},
            },
            timeout=600,
        )
        return _response_json(response)

    def pause_generation(self):
        """``POST /pause`` with mode="abort": abort in-flight requests at weight sync so
        they return finish_reason="abort" (consumer marks the session ERROR — slime
        option-3, prevents mixed-weight trajectories reaching the trainer). See
        docs/design/vime_abort_on_weight_sync.md. Returns the ``requests.Response``."""
        logger.info("[WSYNC-DBG] pause_generation called node_rank=%s", self.node_rank)
        if self.node_rank != 0:
            return None
        response = requests.post(
            f"{self._http_base()}/pause",
            params={"mode": "abort", "clear_cache": "false"},
            json={},
            timeout=120,
        )
        response.raise_for_status()
        return response

    def continue_generation(self):
        """``POST /resume`` to continue generation after pause."""
        logger.info("[WSYNC-DBG] continue_generation called node_rank=%s", self.node_rank)
        if self.node_rank != 0:
            return None
        response = requests.post(f"{self._http_base()}/resume", json={}, timeout=120)
        response.raise_for_status()
        logger.info("[WSYNC-DBG] continue_generation POST /resume -> %s", response.status_code)
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        """No vLLM HTTP hook for post-load processing; return a noop placeholder dict."""
        del restore_weights_before_load, post_process_quantization
        return {"ok": True, "noop": True, "note": "vLLM post_process is internal to load; no HTTP API."}

    def start_profile(
        self,
        output_dir: str | None = None,
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        """``POST /start_profile`` with an empty JSON body; kwargs are not forwarded and may be ignored by the server."""
        if self.node_rank != 0:
            return None
        if any(
            x is not None and x is not False
            for x in (
                output_dir,
                start_step,
                num_steps,
                activities,
                profile_by_stage,
                with_stack,
                record_shapes,
            )
        ):
            logger.warning("vLLM start_profile: extra kwargs may be ignored by server; see vLLM profiling docs.")
        response = requests.post(f"{self._http_base()}/start_profile", json={}, timeout=30)
        response.raise_for_status()
        return response

    def stop_profile(self):
        """POST ``/stop_profile`` to stop an active server-side profile."""
        if self.node_rank != 0:
            return None
        response = requests.post(f"{self._http_base()}/stop_profile", json={}, timeout=30)
        response.raise_for_status()
        return response

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return
        logger.info("Simulating crash on vLLM engine %s:%s...", self.server_host, self.server_port)
        self.shutdown()


def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    host,
    port,
    *,
    worker_type: str = "regular",
    base_gpu_id: int | None = None,
    vllm_overrides: dict | None = None,
    num_gpus_per_engine: int | None = None,
    disaggregation_bootstrap_port: int | None = None,
    data_parallel_size: int | None = None,
    data_parallel_rank: int | None = None,
    data_parallel_address: str | None = None,
    data_parallel_rpc_port: int | None = None,
) -> dict[str, Any]:
    """Build per-actor launch config for ``launch_server_process``."""
    gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    if gpus_per_engine > args.num_gpus_per_node and gpus_per_engine % args.num_gpus_per_node != 0:
        raise ValueError(
            "vLLM multi-node rollout requires rollout_num_gpus_per_engine to be divisible by "
            f"num_gpus_per_node, got rollout_num_gpus_per_engine={gpus_per_engine} "
            f"num_gpus_per_node={args.num_gpus_per_node}."
        )

    topology = compute_vllm_engine_topology(args, rank, num_gpus_per_engine=gpus_per_engine)
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)

    master_addr: str | None = None
    master_port: int | None = None
    if topology.multi_node:
        if not dist_init_addr:
            raise ValueError("dist_init_addr is required when launching a multi-node vLLM engine")
        master_addr, master_port = parse_dist_init_addr(dist_init_addr)

    pd_topology = None
    if vllm_overrides:
        raw_pd_topology = vllm_overrides.get("pd_topology")
        if isinstance(raw_pd_topology, dict):
            pd_topology = raw_pd_topology

    server_args = {
        "args": args,
        "rank": rank,
        "worker_type": worker_type,
        "model_path": args.hf_checkpoint,
        "host": _format_v6_uri(host),
        "port": port,
        "master_addr": master_addr,
        "master_port": master_port,
        "dist_init_addr": dist_init_addr,
        "nnodes": topology.nnodes,
        "node_rank": topology.node_rank,
        "topology": topology,
        "visible_devices": ",".join(str(base + i) for i in range(topology.local_num_gpus)),
        "tp_size": topology.tensor_parallel_size,
        "pp_size": topology.pipeline_parallel_size,
        "dp_size": _get_vllm_dp_size(args),
        "pd_topology": pd_topology,
        "seed": getattr(args, "seed", 1234) + rank,
        "disaggregation_bootstrap_port": disaggregation_bootstrap_port,
        # [DP #3/#4 P1 / external-LB] per-rank DP wiring (None on non-external-LB path).
        "data_parallel_size": data_parallel_size,
        "data_parallel_rank": data_parallel_rank,
        "data_parallel_address": data_parallel_address,
        "data_parallel_rpc_port": data_parallel_rpc_port,
    }
    _apply_vllm_overrides(args, server_args, vllm_overrides, rank)
    return server_args
