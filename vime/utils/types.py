from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


@dataclass
class Sample:
    """The sample generated"""

    group_index: int | None = None
    index: int | None = None
    # Id of the rollout this sample came from. Defaults to ``None`` and the
    # downstream pipeline falls back to ``index`` (so the default rollout
    # path, where one execution = one training sample, sees rollout_id ==
    # index). Compact / subagent paths that split one rollout execution into
    # multiple training samples should set the same ``rollout_id`` on every
    # sibling, so loss aggregation averages within the rollout instead of
    # over-counting it.
    rollout_id: int | None = None
    # prompt
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    multimodal_inputs: dict[str, Any] | None = None  # raw multimodal data, e.g. images, videos, etc.
    multimodal_train_inputs: dict[str, Any] | None = None  # processed multimodal data, e.g. pixel_values, etc.
    # response
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None  # Log probabilities from rollout engine
    rollout_routed_experts: list[list[int]] | None = None  # Routed experts from rollout engine
    remove_sample: bool = False
    teacher_log_probs: list[float] | None = None  # Log probabilities from teacher model for OPD

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: dict | None = None

    # Session ID for consistent hashing routing (used when router policy is consistent_hashing)
    session_id: str | None = None

    non_generation_time: float = 0.0  # time spent in non-generation steps

    @dataclass
    class SpecInfo:
        spec_accept_token_num: int = 0
        spec_draft_token_num: int = 0
        spec_verify_ct: int = 0
        completion_token_num: int = 0

        @property
        def spec_accept_rate(self) -> float:
            return self.spec_accept_token_num / self.spec_draft_token_num if self.spec_draft_token_num > 0 else 0.0

        @property
        def spec_accept_length(self) -> float:
            return self.completion_token_num / self.spec_verify_ct if self.spec_verify_ct > 0 else 0.0

        def add(self, meta_info: dict):
            self.spec_accept_token_num += meta_info.get("spec_accept_token_num", 0)
            self.spec_draft_token_num += meta_info.get("spec_draft_token_num", 0)
            self.spec_verify_ct += meta_info.get("spec_verify_ct", 0)
            self.completion_token_num += meta_info.get("completion_tokens", 0)

        def to_dict(self):
            return {
                "spec_accept_token_num": self.spec_accept_token_num,
                "spec_draft_token_num": self.spec_draft_token_num,
                "spec_verify_ct": self.spec_verify_ct,
                "completion_token_num": self.completion_token_num,
            }

        @staticmethod
        def from_dict(data: dict):
            info = Sample.SpecInfo()
            info.spec_accept_token_num = data.get("spec_accept_token_num", 0)
            info.spec_draft_token_num = data.get("spec_draft_token_num", 0)
            info.spec_verify_ct = data.get("spec_verify_ct", 0)
            info.completion_token_num = data.get("completion_token_num", 0)
            return info

    spec_info: SpecInfo = field(default_factory=SpecInfo)

    @dataclass
    class PrefixCacheInfo:
        cached_tokens: int = 0
        total_prompt_tokens: int = 0

        @property
        def prefix_cache_hit_rate(self) -> float:
            return self.cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens > 0 else 0.0

        def add(self, meta_info: dict):
            self.cached_tokens += meta_info.get("cached_tokens", 0)
            # new_tokens = input_tokens - cached_tokens
            self.total_prompt_tokens += meta_info.get("prompt_tokens", 0)

        def to_dict(self):
            return {
                "cached_tokens": self.cached_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
            }

        @staticmethod
        def from_dict(data: dict):
            info = Sample.PrefixCacheInfo()
            info.cached_tokens = data.get("cached_tokens", 0)
            info.total_prompt_tokens = data.get("total_prompt_tokens", 0)
            return info

    prefix_cache_info: PrefixCacheInfo = field(default_factory=PrefixCacheInfo)

    @dataclass
    class BenchmarkInfo:
        """Benchmark metrics from vLLM inference"""
        ttft_ms: float = 0.0  # Time to first token (milliseconds)
        tpot_ms: list[float] = field(default_factory=list)  # Time per output token (milliseconds)
        itl_ms: list[float] = field(default_factory=list)  # Inter-token latency (milliseconds)
        total_input_tokens: int = 0
        total_output_tokens: int = 0
        request_latency_ms: float = 0.0  # Total request latency

        def add(self, meta_info: dict):
            """累加一次 generate 调用的计时。

            partial rollout 下同一个 sample 会多次调用 ``update_from_meta_info``，
            所以这里必须累加而不是赋值（与 ``PrefixCacheInfo.add`` 一致）：
            latency/token 求和，TTFT 取第一次的非零值（真正的首 token 延迟只发生一次）。

            ``ttft_ms`` / ``tpot_ms`` / ``itl_ms`` 目前不会被填充：本版 vLLM 的响应体
            不含计时字段，且 rollout 走非流式 post()。字段保留是为了让打印块在
            以后能拿到真值时直接生效，不必再改结构。
            """
            ttft = meta_info.get("ttft_ms", 0.0) or 0.0
            if ttft > 0 and self.ttft_ms == 0.0:
                self.ttft_ms = ttft
            self.tpot_ms.extend(meta_info.get("tpot_ms") or [])
            self.itl_ms.extend(meta_info.get("itl_ms") or [])
            self.total_input_tokens += meta_info.get("prompt_tokens", 0)
            self.total_output_tokens += meta_info.get("completion_tokens", 0)
            self.request_latency_ms += meta_info.get("request_latency_ms", 0.0) or 0.0

        def to_dict(self):
            return {
                "ttft_ms": self.ttft_ms,
                "tpot_ms": self.tpot_ms,
                "itl_ms": self.itl_ms,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "request_latency_ms": self.request_latency_ms,
            }

        @staticmethod
        def from_dict(data: dict):
            info = Sample.BenchmarkInfo()
            info.ttft_ms = data.get("ttft_ms", 0.0)
            info.tpot_ms = data.get("tpot_ms", [])
            info.itl_ms = data.get("itl_ms", [])
            info.total_input_tokens = data.get("total_input_tokens", 0)
            info.total_output_tokens = data.get("total_output_tokens", 0)
            info.request_latency_ms = data.get("request_latency_ms", 0.0)
            return info

    benchmark_info: BenchmarkInfo = field(default_factory=BenchmarkInfo)

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        value["spec_info"] = self.spec_info.to_dict()
        value["prefix_cache_info"] = self.prefix_cache_info.to_dict()
        value["benchmark_info"] = self.benchmark_info.to_dict()
        return value

    @staticmethod
    def from_dict(data: dict):
        data = dict(data)
        data["status"] = Sample.Status(data["status"])
        data["spec_info"] = Sample.SpecInfo.from_dict(data.get("spec_info", {}))
        data["prefix_cache_info"] = Sample.PrefixCacheInfo.from_dict(data.get("prefix_cache_info", {}))
        data["benchmark_info"] = Sample.BenchmarkInfo.from_dict(data.get("benchmark_info", {}))

        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {k: v for k, v in data.items() if k in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]

    @property
    def effective_response_length(self):
        return sum(self.loss_mask) if self.loss_mask is not None else self.response_length

    def update_from_meta_info(self, args, meta_info: dict):
        """
        Update the sample with new information from meta_info returned by the rollout engine.
        And extract
        """
        if args.vllm_speculative_config:
            # cannot directly use spec info from vLLM because of partial rollout.
            self.spec_info.add(meta_info=meta_info)

        # Collect prefix cache statistics
        self.prefix_cache_info.add(meta_info=meta_info)

        # Collect benchmark statistics
        self.benchmark_info.add(meta_info=meta_info)

        if "weight_version" in meta_info:
            self.weight_versions.append(meta_info["weight_version"])

        match meta_info["finish_reason"]["type"]:
            case "length":
                self.status = Sample.Status.TRUNCATED
            case "abort":
                self.status = Sample.Status.ABORTED
            case "stop":
                self.status = Sample.Status.COMPLETED


@dataclass(frozen=True)
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int


# A dict-based batch produced along the rollout -> training path
# In Megatron backend, several fields are converted to torch.Tensor lists on GPU
# before being consumed by data iterators (see megatron_utils.actor._get_rollout_data).
RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]


@dataclass
class MultimodalType:
    name: str  # Type identifier used in message content (e.g., "image")
    placeholder: str  # Placeholder token in conversation messages (e.g., "<image>")


class MultimodalTypes:
    IMAGE = MultimodalType(name="image", placeholder="<image>")
    VIDEO = MultimodalType(name="video", placeholder="<video>")
    AUDIO = MultimodalType(name="audio", placeholder="<audio>")

    @classmethod
    def all(cls) -> list[MultimodalType]:
        return [cls.IMAGE, cls.VIDEO, cls.AUDIO]

    @classmethod
    def get(cls, name: str) -> MultimodalType | None:
        return next((m for m in cls.all() if m.name == name), None)
