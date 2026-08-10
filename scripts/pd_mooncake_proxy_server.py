# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mooncake PD proxy for vime rollout.

Aligned with the vllm-ascend PD proxy flow:
- startup waits for prefiller/decoder ``/health`` to become ready
- proxy sends a minimal placeholder ``kv_transfer_params`` to prefiller
- prefiller returns authoritative ``kv_transfer_params`` in its JSON response
- proxy forwards that blob unchanged to decoder
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


@dataclass
class DecodeSelection:
    request_id: str
    server_idx: int
    priority_score: float
    client_info: dict[str, Any]


@dataclass
class ReadinessStatus:
    phase: str = "starting"
    last_error: str | None = None
    attempts: int = 0
    last_request_id: str | None = None
    last_success_at: float | None = None


class ProxyState:
    def __init__(self, prefill_clients: list[dict[str, Any]], decode_clients: list[dict[str, Any]]):
        self.prefill_clients = prefill_clients
        self.decode_clients = decode_clients
        self.req_id_lock = asyncio.Lock()
        self.prefill_index = 0
        self.decode_active_tokens = [0.0 for _ in decode_clients]

    async def next_request_id(self) -> str:
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def next_prefill_client(self) -> dict[str, Any]:
        if not self.prefill_clients:
            raise RuntimeError("No prefill servers available")
        client_info = self.prefill_clients[self.prefill_index % len(self.prefill_clients)]
        self.prefill_index += 1
        return client_info

    def calculate_request_score(
        self,
        api: str,
        req_data: dict[str, Any],
        request_length: int,
    ) -> float:
        if api == "/inference/v1/generate":
            sampling_params = req_data.get("sampling_params") or {}
            max_tokens = sampling_params.get("max_tokens", 16)
            ignore_eos = sampling_params.get("ignore_eos", False)
        else:
            max_tokens = req_data.get("max_completion_tokens", req_data.get("max_tokens", 16))
            ignore_eos = req_data.get("ignore_eos", False)
        if ignore_eos:
            return request_length + max_tokens
        return request_length + 0.5 * max_tokens

    def select_decode_client(
        self,
        session_id: str | None,
        priority_score: float,
        request_id: str,
    ) -> DecodeSelection:
        if not self.decode_clients:
            raise RuntimeError("No decode servers available")
        if session_id:
            idx = int(hashlib.md5(session_id.encode("utf-8")).hexdigest(), 16) % len(self.decode_clients)
        else:
            idx = min(range(len(self.decode_clients)), key=lambda i: self.decode_active_tokens[i])
        self.decode_active_tokens[idx] += priority_score
        return DecodeSelection(
            request_id=request_id,
            server_idx=idx,
            priority_score=priority_score,
            client_info=self.decode_clients[idx],
        )

    def release_decode_client(self, selection: DecodeSelection) -> None:
        self.decode_active_tokens[selection.server_idx] -= selection.priority_score


app = FastAPI()
proxy_state: ProxyState | None = None


PRECHECK_INTERVAL_SECONDS = float(os.environ.get("PD_MOONCAKE_PREFLIGHT_INTERVAL_SECONDS", "2.0"))
PRECHECK_BACKOFF_MAX_SECONDS = float(os.environ.get("PD_MOONCAKE_PREFLIGHT_BACKOFF_MAX_SECONDS", "10.0"))


def _set_readiness_status(
    ready: asyncio.Event,
    *,
    phase: str,
    last_error: str | None = None,
    attempts: int | None = None,
    last_request_id: str | None = None,
    last_success_at: float | None = None,
) -> None:
    status = app.state.readiness_status
    status.phase = phase
    status.last_error = last_error
    if attempts is not None:
        status.attempts = attempts
    if last_request_id is not None:
        status.last_request_id = last_request_id
    if last_success_at is not None:
        status.last_success_at = last_success_at
    if phase == "ready":
        ready.set()
    else:
        ready.clear()


async def _wait_for_backends_ready(
    prefill_clients: list[dict[str, Any]],
    decode_clients: list[dict[str, Any]],
    ready: asyncio.Event,
) -> None:
    attempts = 0
    sleep_seconds = PRECHECK_INTERVAL_SECONDS
    while True:
        request_id = f"pd-preflight-{uuid.uuid4()}"
        try:
            _set_readiness_status(
                ready,
                phase="healthcheck",
                last_error=None,
                attempts=attempts,
                last_request_id=request_id,
            )
            for client_info in prefill_clients:
                response = await client_info["client"].get("/health")
                response.raise_for_status()
            for client_info in decode_clients:
                response = await client_info["client"].get("/health")
                response.raise_for_status()

            _set_readiness_status(
                ready,
                phase="preflight",
                last_error=None,
                attempts=attempts,
                last_request_id=request_id,
            )
            for prefill_client_info in prefill_clients:
                for decode_client_info in decode_clients:
                    await _run_pd_preflight(
                        prefill_client_info,
                        decode_client_info,
                        request_id,
                    )

            _set_readiness_status(
                ready,
                phase="ready",
                last_error=None,
                attempts=attempts + 1,
                last_request_id=request_id,
                last_success_at=time.time(),
            )
            logger.info(
                "All Mooncake PD backend instances passed health and PD preflight "
                "(prefillers=%s decoders=%s request_id=%s attempts=%s)",
                [c["url"] for c in prefill_clients],
                [c["url"] for c in decode_clients],
                request_id,
                attempts + 1,
            )
            return
        except Exception as exc:
            attempts += 1
            _set_readiness_status(
                ready,
                phase="preflight_failed",
                last_error=str(exc),
                attempts=attempts,
                last_request_id=request_id,
            )
            logger.warning(
                "Mooncake PD preflight attempt failed attempt=%s request_id=%s error=%s",
                attempts,
                request_id,
                exc,
            )
            await asyncio.sleep(sleep_seconds)
            sleep_seconds = min(sleep_seconds * 1.5, PRECHECK_BACKOFF_MAX_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global proxy_state
    prefill_clients: list[dict[str, Any]] = []
    decode_clients: list[dict[str, Any]] = []
    app.state.ready = asyncio.Event()
    app.state.readiness_status = ReadinessStatus()

    for url in global_args.prefill:
        prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
                    trust_env=False,
                ),
                "url": url,
            }
        )

    for url in global_args.decode:
        decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
                    trust_env=False,
                ),
                "url": url,
            }
        )

    proxy_state = ProxyState(prefill_clients, decode_clients)
    ready_task = asyncio.create_task(_wait_for_backends_ready(prefill_clients, decode_clients, app.state.ready))

    try:
        yield
    finally:
        ready_task.cancel()
        for client_info in prefill_clients:
            await client_info["client"].aclose()
        for client_info in decode_clients:
            await client_info["client"].aclose()


app.router.lifespan_context = lifespan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--prefill",
        nargs="+",
        action="append",
        dest="prefill_raw",
        metavar=("URL", "legacy_bootstrap_port"),
    )
    parser.add_argument("--decode", nargs=1, action="append", dest="decode_raw", metavar=("URL",))
    args = parser.parse_args()
    args.prefill = _parse_prefill_urls(args.prefill_raw)
    args.decode = _parse_decode_urls(args.decode_raw)
    return args


def _parse_prefill_urls(prefill_list):
    if not prefill_list:
        return []
    prefill_urls = []
    for prefill_args in prefill_list:
        url = prefill_args[0]
        if len(prefill_args) > 1:
            logger.warning(
                "Ignoring legacy Mooncake bootstrap port args %s for prefiller %s; "
                "proxy now uses prefiller-returned kv_transfer_params",
                prefill_args[1:],
                url,
            )
        prefill_urls.append(url)
    return prefill_urls


def _parse_decode_urls(decode_list):
    if not decode_list:
        return []
    return [url[0] for url in decode_list]


def _build_preflight_request() -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }


def _make_internal_headers(request_id: str) -> dict[str, str]:
    headers = {"X-Request-Id": request_id}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    return headers


async def _run_pd_preflight(
    prefill_client_info: dict[str, Any],
    decode_client_info: dict[str, Any],
    request_id: str,
) -> None:
    api = "/v1/chat/completions"
    req_data = _build_preflight_request()
    headers = _make_internal_headers(request_id)

    prefill_payload = _build_prefill_request(api, req_data, request_id)
    prefill_response = await prefill_client_info["client"].post(
        api,
        json=prefill_payload,
        headers=headers,
    )
    try:
        prefill_response.raise_for_status()
        kv_transfer_params = _extract_prefill_kv_transfer_params(
            prefill_response.json()
        )
    finally:
        await prefill_response.aclose()

    decode_payload = _build_decode_payload(req_data, kv_transfer_params)
    decode_response = await decode_client_info["client"].post(
        api,
        json=decode_payload,
        headers=headers,
    )
    try:
        decode_response.raise_for_status()
        body = decode_response.json()
    finally:
        await decode_response.aclose()

    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            "PD preflight decode response missing choices; "
            f"body_type={type(body).__name__}"
        )


async def listen_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def _make_headers(request: Request, request_id: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"X-Request-Id": request_id}
    if extra:
        headers.update(extra)
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    elif os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    return headers


def _build_prefill_request(api: str, req_data: dict[str, Any], request_id: str) -> dict[str, Any]:
    del request_id
    payload = dict(req_data)
    payload["stream"] = False
    payload.pop("stream_options", None)
    if api == "/inference/v1/generate":
        sampling_params = dict(payload.get("sampling_params") or {})
        sampling_params.pop("logprobs", None)
        sampling_params.pop("top_logprobs", None)
        sampling_params["max_tokens"] = 1
        sampling_params["min_tokens"] = 1
        payload["sampling_params"] = sampling_params
    else:
        payload.pop("logprobs", None)
        payload.pop("top_logprobs", None)
        payload["max_tokens"] = 1
        payload["min_tokens"] = 1
        if "max_completion_tokens" in payload:
            payload["max_completion_tokens"] = 1
        if "min_completion_tokens" in payload:
            payload["min_completion_tokens"] = 1
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    return payload


def _extract_prefill_kv_transfer_params(prefill_body: dict[str, Any]) -> dict[str, Any]:
    kv_transfer_params = prefill_body.get("kv_transfer_params") or {}
    if not kv_transfer_params:
        raise RuntimeError(
            "Prefiller response missing kv_transfer_params; "
            "expected vllm-ascend PD Mooncake handoff fields"
        )

    required_fields = (
        "remote_engine_id",
        "remote_request_id",
        "remote_host",
        "remote_port",
    )
    missing = [field for field in required_fields if kv_transfer_params.get(field) is None]
    if missing:
        raise RuntimeError(
            "Prefiller response kv_transfer_params missing required fields "
            f"{missing}; got keys={sorted(kv_transfer_params.keys())}"
        )
    return dict(kv_transfer_params)


async def _send_prefill(
    prefill_client_info: dict[str, Any],
    api: str,
    req_data: dict[str, Any],
    request: Request,
    request_id: str,
) -> dict[str, Any]:
    payload = _build_prefill_request(api, req_data, request_id)
    headers = _make_headers(request, request_id)
    logger.info(
        "PD-DIAG prefill_request api=%s request_id=%s original_keys=%s payload_keys=%s has_messages=%s has_prompt=%s has_sampling_params=%s has_return_token_ids=%s has_return_logprobs=%s",
        api,
        request_id,
        sorted(req_data.keys()),
        sorted(payload.keys()),
        "messages" in req_data,
        "prompt" in req_data,
        "sampling_params" in req_data,
        "return_token_ids" in req_data,
        "return_logprobs" in req_data,
    )
    response = await prefill_client_info["client"].post(api, json=payload, headers=headers)
    try:
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        logger.info(
            "PD-DIAG prefill_response api=%s request_id=%s response_keys=%s choices=%s choice0_keys=%s has_token_ids=%s has_logprobs=%s finish_reason=%r has_kv_transfer_params=%s",
            api,
            request_id,
            sorted(body.keys()) if isinstance(body, dict) else type(body).__name__,
            len(choices) if isinstance(choices, list) else None,
            sorted(first_choice.keys()) if isinstance(first_choice, dict) else None,
            isinstance(first_choice, dict) and ("token_ids" in first_choice),
            isinstance(first_choice, dict) and ("logprobs" in first_choice),
            first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
            isinstance(body, dict) and ("kv_transfer_params" in body),
        )
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = response.text
        except Exception:
            detail = "<unreadable response body>"
        logger.error(
            "Prefiller request failed with %s for %s%s; payload=%s; response=%s",
            response.status_code,
            prefill_client_info["url"],
            api,
            payload,
            detail,
        )
        raise exc
    finally:
        await response.aclose()
    return _extract_prefill_kv_transfer_params(body)


DECODE_MAX_TOKENS_CAP = int(os.environ.get("PD_MOONCAKE_DECODE_MAX_TOKENS_CAP", "32768"))


def _build_decode_payload(req_data: dict[str, Any], kv_transfer_params: dict[str, Any]) -> dict[str, Any]:
    payload = dict(req_data)
    if "sampling_params" in payload and isinstance(payload["sampling_params"], dict):
        payload["sampling_params"] = dict(payload["sampling_params"])

    if "max_tokens" in payload:
        payload["max_tokens"] = min(int(payload["max_tokens"]), DECODE_MAX_TOKENS_CAP)
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = min(int(payload["max_completion_tokens"]), DECODE_MAX_TOKENS_CAP)
    if "min_tokens" in payload:
        payload["min_tokens"] = min(int(payload["min_tokens"]), payload.get("max_tokens", DECODE_MAX_TOKENS_CAP))
    if "min_completion_tokens" in payload:
        payload["min_completion_tokens"] = min(
            int(payload["min_completion_tokens"]),
            payload.get("max_completion_tokens", DECODE_MAX_TOKENS_CAP),
        )
    if "sampling_params" in payload and isinstance(payload["sampling_params"], dict):
        if "max_tokens" in payload["sampling_params"]:
            payload["sampling_params"]["max_tokens"] = min(
                int(payload["sampling_params"]["max_tokens"]),
                DECODE_MAX_TOKENS_CAP,
            )
        if "min_tokens" in payload["sampling_params"]:
            payload["sampling_params"]["min_tokens"] = min(
                int(payload["sampling_params"]["min_tokens"]),
                payload["sampling_params"].get("max_tokens", DECODE_MAX_TOKENS_CAP),
            )

    payload["kv_transfer_params"] = dict(kv_transfer_params)
    return payload


async def _post_decode(selection: DecodeSelection, api: str, req_data: dict[str, Any], request: Request) -> JSONResponse:
    response = await selection.client_info["client"].post(
        api,
        json=req_data,
        headers=_make_headers(request, selection.request_id),
    )
    try:
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        logger.info(
            "PD-DIAG decode_response api=%s request_id=%s response_keys=%s choices=%s choice0_keys=%s has_token_ids=%s has_logprobs=%s finish_reason=%r",
            api,
            selection.request_id,
            sorted(body.keys()) if isinstance(body, dict) else type(body).__name__,
            len(choices) if isinstance(choices, list) else None,
            sorted(first_choice.keys()) if isinstance(first_choice, dict) else None,
            isinstance(first_choice, dict) and ("token_ids" in first_choice),
            isinstance(first_choice, dict) and ("logprobs" in first_choice),
            first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
        )
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = response.text
        except Exception:
            detail = "<unreadable response body>"
        logger.error(
            "Decoder request failed with %s for %s%s; payload=%s; response=%s",
            response.status_code,
            selection.client_info["url"],
            api,
            req_data,
            detail,
        )
        raise exc
    finally:
        await response.aclose()
    return JSONResponse(status_code=response.status_code, content=body)


async def _stream_decode(selection: DecodeSelection, api: str, req_data: dict[str, Any], request: Request):
    async with selection.client_info["client"].stream(
        "POST",
        api,
        json=req_data,
        headers=_make_headers(request, selection.request_id),
    ) as response:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = await response.aread()
                detail = detail.decode("utf-8", errors="replace")
            except Exception:
                detail = "<unreadable response body>"
            logger.error(
                "Decoder streaming request failed with %s for %s%s; payload=%s; response=%s",
                response.status_code,
                selection.client_info["url"],
                api,
                req_data,
                detail,
            )
            raise exc
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_request(api: str, request: Request):
    assert proxy_state is not None
    if not app.state.ready.is_set():
        raise HTTPException(status_code=503, detail="Service Unavailable")

    req_data = await request.json()
    req_body = await request.body()
    session_id = request.headers.get("x-session-id")
    request_id = await proxy_state.next_request_id()

    prefill_client_info = proxy_state.next_prefill_client()
    kv_transfer_params = await _send_prefill(prefill_client_info, api, req_data, request, request_id)

    priority_score = proxy_state.calculate_request_score(api, req_data, len(req_body))
    selection = proxy_state.select_decode_client(session_id, priority_score, request_id)
    decode_payload = _build_decode_payload(req_data, kv_transfer_params)

    logger.info(
        "PD-DIAG decode_request api=%s request_id=%s original_keys=%s payload_keys=%s has_messages=%s has_prompt=%s has_sampling_params=%s has_return_token_ids=%s has_return_logprobs=%s kv_keys=%s session_id=%s",
        api,
        request_id,
        sorted(req_data.keys()),
        sorted(decode_payload.keys()),
        "messages" in decode_payload,
        "prompt" in decode_payload,
        "sampling_params" in decode_payload,
        "return_token_ids" in decode_payload,
        "return_logprobs" in decode_payload,
        sorted(kv_transfer_params.keys()),
        session_id,
    )
    try:
        if req_data.get("stream"):

            async def generate_stream():
                try:
                    async for chunk in _stream_decode(selection, api, decode_payload, request):
                        yield chunk
                finally:
                    proxy_state.release_decode_client(selection)

            media_type = "text/event-stream" if api != "/inference/v1/generate" else "application/json"
            return StreamingResponse(generate_stream(), media_type=media_type)
        return await _post_decode(selection, api, decode_payload, request)
    finally:
        if not req_data.get("stream"):
            proxy_state.release_decode_client(selection)


@app.post("/inference/v1/generate")
@with_cancellation
async def handle_generate(request: Request):
    return await _handle_request("/inference/v1/generate", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await _handle_request("/v1/chat/completions", request)


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await _handle_request("/v1/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    assert proxy_state is not None
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefill_clients),
        "decode_instances": len(proxy_state.decode_clients),
    }


@app.get("/health")
async def health():
    assert proxy_state is not None
    return {
        "status": "ok" if app.state.ready.is_set() else "starting",
        "ready": app.state.ready.is_set(),
        "prefill_instances": len(proxy_state.prefill_clients),
        "decode_instances": len(proxy_state.decode_clients),
    }


@app.get("/ready")
async def ready():
    assert proxy_state is not None
    status = app.state.readiness_status
    return {
        "status": "ok" if app.state.ready.is_set() else status.phase,
        "ready": app.state.ready.is_set(),
        "prefill_instances": len(proxy_state.prefill_clients),
        "decode_instances": len(proxy_state.decode_clients),
        "readiness_phase": status.phase,
        "preflight_attempts": status.attempts,
        "last_preflight_request_id": status.last_request_id,
        "last_preflight_error": status.last_error,
        "last_success_at": status.last_success_at,
    }


if __name__ == "__main__":
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
