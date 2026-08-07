import importlib.util
import sys
from pathlib import Path

import pytest


class _FakeResponse:
    def __init__(self, *, json_data: dict | None = None, status_code: int = 200):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = str(self._json_data)

    async def aclose(self):
        return None

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            req = httpx.Request("POST", "http://example.test")
            resp = httpx.Response(self.status_code, request=req, text=self.text)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=req, response=resp
            )


class _FakeAsyncClient:
    def __init__(self, responses=None, get_responses=None):
        self._responses = list(responses or [])
        self._get_responses = list(get_responses or [])
        self.calls = []
        self.get_calls = []

    async def post(self, api, json=None, headers=None):
        self.calls.append({"api": api, "json": json, "headers": headers})
        if not self._responses:
            raise AssertionError("No fake responses remaining")
        return self._responses.pop(0)

    async def get(self, api):
        self.get_calls.append(api)
        if not self._get_responses:
            raise AssertionError("No fake GET responses remaining")
        return self._get_responses.pop(0)


def _load_proxy_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "pd_mooncake_proxy_server.py"
    module_name = "test_pd_mooncake_proxy_server_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_build_prefill_request_chat_uses_ascend_placeholder_contract():
    mod = _load_proxy_module()

    payload = mod._build_prefill_request(
        "/v1/chat/completions",
        {
            "model": "demo",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "logprobs": True,
            "top_logprobs": 5,
            "max_tokens": 32,
            "min_tokens": 5,
        },
        "req-1",
    )

    assert payload["stream"] is False
    assert "stream_options" not in payload
    assert "logprobs" not in payload
    assert "top_logprobs" not in payload
    assert payload["max_tokens"] == 1
    assert payload["min_tokens"] == 1
    assert payload["kv_transfer_params"] == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }


@pytest.mark.unit
def test_build_prefill_request_generate_uses_ascend_placeholder_contract():
    mod = _load_proxy_module()

    payload = mod._build_prefill_request(
        "/inference/v1/generate",
        {
            "model": "demo",
            "prompt": "hello",
            "sampling_params": {
                "max_tokens": 32,
                "min_tokens": 7,
                "temperature": 0.7,
                "logprobs": 1,
                "top_logprobs": 4,
            },
        },
        "req-2",
    )

    assert payload["sampling_params"]["max_tokens"] == 1
    assert payload["sampling_params"]["min_tokens"] == 1
    assert payload["sampling_params"]["temperature"] == 0.7
    assert "logprobs" not in payload["sampling_params"]
    assert "top_logprobs" not in payload["sampling_params"]
    assert payload["kv_transfer_params"]["remote_host"] is None
    assert payload["kv_transfer_params"]["remote_port"] is None


@pytest.mark.unit
def test_extract_prefill_kv_transfer_params_requires_ascend_fields():
    mod = _load_proxy_module()

    body = {
        "kv_transfer_params": {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_engine_id": "engine-1",
            "remote_request_id": "req-remote-1",
            "remote_host": "80.48.5.56",
            "remote_port": 20001,
            "remote_block_ids": [1, 2, 3],
        }
    }

    kv = mod._extract_prefill_kv_transfer_params(body)
    assert kv["remote_engine_id"] == "engine-1"
    assert kv["remote_request_id"] == "req-remote-1"
    assert kv["remote_host"] == "80.48.5.56"
    assert kv["remote_port"] == 20001
    assert kv["remote_block_ids"] == [1, 2, 3]


@pytest.mark.unit
def test_extract_prefill_kv_transfer_params_rejects_missing_required_fields():
    mod = _load_proxy_module()

    with pytest.raises(RuntimeError, match="missing required fields"):
        mod._extract_prefill_kv_transfer_params(
            {
                "kv_transfer_params": {
                    "remote_engine_id": "engine-1",
                    "remote_host": "80.48.5.56",
                }
            }
        )


@pytest.mark.unit
def test_build_decode_payload_forwards_prefiller_kv_transfer_params_verbatim():
    mod = _load_proxy_module()

    req_data = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
    kv_transfer_params = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_engine_id": "engine-2",
        "remote_request_id": "req-remote-2",
        "remote_host": "80.48.5.56",
        "remote_port": 20002,
        "remote_block_ids": [5, 6],
        "remote_pcp_size": 1,
    }

    payload = mod._build_decode_payload(req_data, kv_transfer_params)

    assert payload["messages"] == req_data["messages"]
    assert payload["kv_transfer_params"] == kv_transfer_params
    assert "remote_bootstrap_addr" not in payload["kv_transfer_params"]


@pytest.mark.unit
def test_run_pd_preflight_strips_logprobs_and_requires_decode_choices():
    mod = _load_proxy_module()

    prefill_client = _FakeAsyncClient(
        [
            _FakeResponse(
                json_data={
                    "kv_transfer_params": {
                        "remote_engine_id": "engine-1",
                        "remote_request_id": "req-remote-1",
                        "remote_host": "80.48.5.56",
                        "remote_port": 20001,
                    }
                }
            )
        ]
    )
    decode_client = _FakeAsyncClient(
        [_FakeResponse(json_data={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]})]
    )

    import asyncio

    asyncio.run(
        mod._run_pd_preflight(
            {"client": prefill_client, "url": "http://prefill"},
            {"client": decode_client, "url": "http://decode"},
            "pd-preflight-1",
        )
    )

    prefill_call = prefill_client.calls[0]
    decode_call = decode_client.calls[0]
    assert prefill_call["json"]["max_tokens"] == 1
    assert prefill_call["json"]["messages"] == [{"role": "user", "content": "ping"}]
    assert "logprobs" not in prefill_call["json"]
    assert "top_logprobs" not in prefill_call["json"]
    assert decode_call["json"]["kv_transfer_params"]["remote_port"] == 20001
    assert decode_call["json"]["messages"] == [{"role": "user", "content": "ping"}]


@pytest.mark.unit
def test_run_pd_preflight_rejects_decode_response_without_choices():
    mod = _load_proxy_module()

    prefill_client = _FakeAsyncClient(
        [
            _FakeResponse(
                json_data={
                    "kv_transfer_params": {
                        "remote_engine_id": "engine-1",
                        "remote_request_id": "req-remote-1",
                        "remote_host": "80.48.5.56",
                        "remote_port": 20001,
                    }
                }
            )
        ]
    )
    decode_client = _FakeAsyncClient([_FakeResponse(json_data={"choices": []})])

    import asyncio

    with pytest.raises(RuntimeError, match="missing choices"):
        asyncio.run(
            mod._run_pd_preflight(
                {"client": prefill_client, "url": "http://prefill"},
                {"client": decode_client, "url": "http://decode"},
                "pd-preflight-2",
            )
        )


@pytest.mark.unit
def test_wait_for_backends_ready_records_preflight_failure_state_then_succeeds():
    mod = _load_proxy_module()

    import asyncio

    ready = asyncio.Event()
    mod.app.state.ready = ready
    mod.app.state.readiness_status = mod.ReadinessStatus()

    prefill_client = _FakeAsyncClient(
        responses=[
            _FakeResponse(
                json_data={
                    "kv_transfer_params": {
                        "remote_engine_id": "engine-1",
                        "remote_request_id": "req-remote-1",
                        "remote_host": "80.48.5.56",
                        "remote_port": 20001,
                    }
                }
            ),
            _FakeResponse(
                json_data={
                    "kv_transfer_params": {
                        "remote_engine_id": "engine-1",
                        "remote_request_id": "req-remote-2",
                        "remote_host": "80.48.5.56",
                        "remote_port": 20001,
                    }
                }
            ),
        ],
        get_responses=[_FakeResponse(), _FakeResponse()],
    )
    decode_client = _FakeAsyncClient(
        responses=[
            _FakeResponse(json_data={"choices": []}),
            _FakeResponse(json_data={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]}),
        ],
        get_responses=[_FakeResponse(), _FakeResponse()],
    )

    import asyncio

    asyncio.run(
        mod._wait_for_backends_ready(
            [{"client": prefill_client, "url": "http://prefill"}],
            [{"client": decode_client, "url": "http://decode"}],
            ready,
        )
    )

    status = mod.app.state.readiness_status
    assert ready.is_set() is True
    assert status.phase == "ready"
    assert status.attempts == 2
    assert status.last_error is None
    assert status.last_success_at is not None
