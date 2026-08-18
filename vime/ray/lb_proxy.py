"""Load-balancing proxy that fronts the rollout engines.

Adapted from vLLM's ``toy_proxy_server.py`` (SPDX-License-Identifier: Apache-2.0).

It stands in for the Rust vllm-router, which parses requests into typed models and drops
vLLM extensions such as ``return_token_ids`` along the way. This one forwards the request
dict verbatim, so token-level fields survive the round trip.

Requests carrying an ``x-session-id`` header are hashed onto a fixed engine, which keeps a
sample's turns on the same prefix cache. Everything else goes to the least loaded engine.

Run it standalone with::

    python -m vime.ray.lb_proxy --host 0.0.0.0 --port 9000 \
        --dp-hosts 127.0.0.1 127.0.0.1 --dp-ports 8100 8101
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 0.2
# vLLM's own default for max_tokens, used to score a request that does not set one.
_DEFAULT_MAX_TOKENS = 16
# Without ignore_eos a request usually stops well before max_tokens, so only part of the
# budget is charged against the engine it lands on.
_EXPECTED_COMPLETION_RATIO = 0.5


class ServerState:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0


@dataclass
class InstanceInfo:
    request_id: str
    server_idx: int
    priority_score: float
    server_state: ServerState


class ProxyState:
    """Tracks in-flight token load per engine so requests can be spread across them."""

    def __init__(self, server_instances):
        self.dp_servers: list[ServerState] = [ServerState(host, port) for host, port in server_instances]
        self.session_servers: dict[str, int] = {}

    async def aclose(self):
        for server in self.dp_servers:
            await server.client.aclose()

    def select_server(self, token_count: float) -> int:
        """Charge the request to the least loaded engine and return its index."""
        if not self.dp_servers:
            raise RuntimeError("no rollout engines registered")
        idx = min(range(len(self.dp_servers)), key=lambda i: self.dp_servers[i].active_tokens)
        self.dp_servers[idx].active_tokens += token_count
        return idx

    def select_server_by_session(self, session_id: str, token_count: float) -> int:
        """Place a new session by load, then pin its later turns to that engine."""
        idx = self.session_servers.get(session_id)
        if idx is None:
            idx = min(range(len(self.dp_servers)), key=lambda i: self.dp_servers[i].active_tokens)
            self.session_servers[session_id] = idx
        self.dp_servers[idx].active_tokens += token_count
        return idx

    def release_server(self, idx: int, token_count: float):
        self.dp_servers[idx].active_tokens -= token_count

    def score_request(self, request_length: int, max_tokens: int, ignore_eos: bool) -> float:
        if ignore_eos:
            return request_length + max_tokens
        return request_length + _EXPECTED_COMPLETION_RATIO * max_tokens


async def _forward_upstream_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int,
    base_delay: float,
) -> tuple[httpx.Response | None, tuple[int, bytes] | None]:
    """Forward to an engine, retrying transport errors and non-2xx replies.

    Returns either ``(response, None)`` with headers already received and the body left for
    the caller to stream, or ``(None, (status, body))`` once the retries are spent.

    An upstream 4xx/5xx must reach the client as itself. Swallowing it into an empty 200
    leaves the caller parsing an empty body, which surfaces as a decode error rather than an
    HTTP failure, and a session that cannot be marked failed stalls the rollout that waits
    on it.
    """
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    for attempt in range(1, max_retries + 1):
        request = client.build_request("POST", endpoint, json=req_data, headers=headers)
        try:
            response = await client.send(request, stream=True)
        except httpx.RequestError as exc:
            if attempt < max_retries:
                logger.warning("Attempt %s failed connecting %s: %s", attempt, endpoint, exc)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            logger.error("All %s attempts failed connecting %s: %s", max_retries, endpoint, exc)
            return None, (502, _error_body(f"upstream connect failed after {max_retries} attempts: {exc}"))

        if response.status_code < 400:
            return response, None

        body = await response.aread()
        await response.aclose()
        if attempt < max_retries:
            logger.warning("Attempt %s got HTTP %s from %s", attempt, response.status_code, endpoint)
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            continue
        logger.error(
            "All %s attempts got HTTP %s from %s; propagating it unchanged",
            max_retries,
            response.status_code,
            endpoint,
        )
        return None, (response.status_code, body)

    return None, (502, _error_body("upstream retry exhausted"))


def _error_body(message: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": "upstream_error"}}).encode()


async def _listen_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    """Abandon the handler as soon as the client hangs up."""

    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(_listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def build_app(server_instances, *, max_retries=DEFAULT_MAX_RETRIES, retry_delay=DEFAULT_RETRY_DELAY) -> FastAPI:
    """Build the proxy over ``(host, port)`` pairs, one per rollout engine."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.proxy = ProxyState(server_instances)
        logger.info("LB proxy fronting %s engines", len(app.state.proxy.dp_servers))
        yield
        await app.state.proxy.aclose()

    app = FastAPI(lifespan=lifespan)

    def _select_instance(proxy: ProxyState, req_data: Any, request_length: int, session_id: str | None):
        score = proxy.score_request(
            request_length,
            max_tokens=req_data.get("max_tokens", _DEFAULT_MAX_TOKENS),
            ignore_eos=req_data.get("ignore_eos", False),
        )
        if session_id:
            server_idx = proxy.select_server_by_session(session_id, score)
        else:
            server_idx = proxy.select_server(score)
        return InstanceInfo(
            request_id=str(uuid.uuid4()),
            server_idx=server_idx,
            priority_score=score,
            server_state=proxy.dp_servers[server_idx],
        )

    async def _handle_completions(endpoint: str, request: Request):
        proxy: ProxyState = request.app.state.proxy
        req_data = await request.json()
        request_length = len(await request.body())
        instance = _select_instance(proxy, req_data, request_length, request.headers.get("x-session-id"))

        released = False

        def release_once():
            nonlocal released
            if not released:
                released = True
                proxy.release_server(instance.server_idx, instance.priority_score)

        try:
            response, error = await _forward_upstream_with_retry(
                instance.server_state.client,
                endpoint,
                req_data,
                request_id=instance.request_id,
                max_retries=max_retries,
                base_delay=retry_delay,
            )
        except BaseException:
            # Includes the CancelledError raised when the client hangs up mid-flight; without
            # this the request's tokens would stay charged to the engine forever.
            release_once()
            raise

        if error is not None:
            release_once()
            status_code, body = error
            return Response(content=body, status_code=status_code, media_type="application/json")

        async def stream_upstream():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            except Exception as exc:
                logger.error(
                    "Streaming from %s interrupted after it began, request %s: %s",
                    instance.server_state.url,
                    instance.request_id,
                    exc,
                )
            finally:
                await response.aclose()
                release_once()

        return StreamingResponse(stream_upstream(), media_type="application/json")

    @app.post("/v1/completions")
    @with_cancellation
    async def handle_completions(request: Request):
        return await _handle_completions("/completions", request)

    @app.post("/v1/chat/completions")
    @with_cancellation
    async def handle_chat_completions(request: Request):
        return await _handle_completions("/chat/completions", request)

    def _status(request: Request):
        return {"status": "ok", "dp_instances": len(request.app.state.proxy.dp_servers)}

    # /health is what a vLLM worker exposes and what the operator probes; a router that only
    # answered /healthcheck would be marked down and never sent any work.
    @app.get("/health")
    async def health(request: Request):
        return _status(request)

    @app.get("/healthcheck")
    async def healthcheck(request: Request):
        return _status(request)

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dp-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--dp-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY)
    args = parser.parse_args(argv)
    if len(args.dp_hosts) != len(args.dp_ports):
        raise ValueError(
            f"--dp-hosts and --dp-ports must have the same length, got {len(args.dp_hosts)} and {len(args.dp_ports)}"
        )
    return args


def main(argv=None):
    import uvicorn

    args = parse_args(argv)
    app = build_app(
        list(zip(args.dp_hosts, args.dp_ports)),
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
