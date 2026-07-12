"""Polar HTTP wire models used by Slime without importing the Polar server."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SessionStatus(StrEnum):
    REGISTERED = "REGISTERED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    POST_RUN = "POST_RUN"
    BUILDING = "BUILDING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


class Trace(BaseModel):
    prompt_ids: list[int] = Field(default_factory=list)
    response_ids: list[int] = Field(default_factory=list)
    loss_mask: list[int] = Field(default_factory=list)
    prompt_messages: list[dict[str, Any]] = Field(default_factory=list)
    response_messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    response_logprobs: list[float] | None = None
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("loss_mask")
    @classmethod
    def _validate_loss_mask_values(cls, value: list[int]) -> list[int]:
        out: list[int] = []
        for item in value:
            mask_value = int(item)
            if mask_value not in (0, 1):
                raise ValueError("loss_mask values must be 0 or 1")
            out.append(mask_value)
        return out

    @model_validator(mode="after")
    def _validate_response_lengths(self) -> "Trace":
        if self.loss_mask and len(self.loss_mask) != len(self.response_ids):
            raise ValueError("loss_mask length must match response_ids length")
        if self.response_logprobs is not None and len(self.response_logprobs) != len(self.response_ids):
            raise ValueError("response_logprobs length must match response_ids length")
        return self


class Trajectory(BaseModel):
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    traces: list[Trace] = Field(default_factory=list)
    error: str | None = None


class SessionTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    register_to_init_queue_ms: float = 0.0
    init_ms: float = 0.0
    run_ms: float = 0.0
    postrun_ms: float = 0.0


class SessionResult(BaseModel):
    session_id: str
    task_id: str
    status: SessionStatus
    trajectory: Trajectory
    timing: SessionTiming = Field(default_factory=SessionTiming)
    node_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    task_id: str
    status: str
    results: list[SessionResult]
    result_paths: list[str] = Field(default_factory=list)


class TaskStatus(BaseModel):
    task_id: str
    status: str
    total_sessions: int
    completed_sessions: int
    results: list[SessionResult] = Field(default_factory=list)
    result_paths: list[str] = Field(default_factory=list)
